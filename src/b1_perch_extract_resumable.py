"""B1' — RESUMABLE + LOGGED Perch-V2 CPU-ONNX extraction (recovery rewrite).

Replaces the opaque, non-resumable `b1_perch_extract.py --mode all` run that died
on a network drop after 76 min with zero recovered work (it accumulated the whole
embedding array in RAM and only np.save'd at the very end, with no progress logs).

WINDOWS-PER-CLIP POLICY (unchanged, documented):
  ONE representative 5 s window per clip = the CENTER window (load_center_window,
  identical to the original script and to b1_build_wav_cache). This is the correct
  policy for a CLASSIFICATION OOF anchor: the bronze pipeline scores clips with a
  single Perch embedding per clip (ProtoSSM prototype match), and the running matrix
  invariant (exp-plan-v4) treats each clip as one example with one anchor score.
  => exactly N=35,549 Perch inferences, NOT (windows x N). Processing every 5 s
  window of every clip would be ~5-20x more inferences with no benefit to the
  clip-level OOF and is explicitly NOT done.

RESUMABILITY:
  - The clip list (folds.csv order) is split into fixed shards of --shard_size.
  - Each shard writes its own files atomically (tmp -> rename) on completion:
        shards/emb_<sid>.npy        float16 [shard_n, 1536]
        shards/lab_<sid>.npy        float16 [shard_n, 14795]   (if --cache_label_head)
        shards/done_<sid>.flag
  - On restart, any shard whose done flag + .npy files exist is SKIPPED.
  - After all shards exist, --mode assemble concatenates them (folds.csv order) into
        perch_cache/embeddings.npy   float16 [N,1536]
        perch_cache/label_head.npy   float16 [N,14795]
    and writes oof_targets.npy + oof_meta.csv (filename,fold,author).
  - `--mode all` does extract-then-assemble; safe to re-run after a crash.

LOGGING: prints progress every --log_every clips with clips/s rate + ETA, plus a
  per-shard line. Run under tmux/nohup with `python -u`.

float16 disk budget (N=35,549): embeddings ~0.11 GB, label_head ~1.05 GB. Tiny.
"""
import argparse
import ast
import os
import time
import numpy as np
import pandas as pd

SR = 32000
WIN = 5 * SR
EMB_DIM = 1536
N_LABEL_HEAD = 14795


def load_taxonomy_order(data_dir):
    tax = pd.read_csv(os.path.join(data_dir, "taxonomy.csv"))
    classes = tax["primary_label"].astype(str).tolist()
    assert len(classes) == 234, f"expected 234 classes, got {len(classes)}"
    return classes


def build_targets(df, class_order):
    idx = {c: i for i, c in enumerate(class_order)}
    Y = np.zeros((len(df), len(class_order)), dtype=np.float32)
    for r, row in enumerate(df.itertuples(index=False)):
        p = str(getattr(row, "primary_label"))
        if p in idx:
            Y[r, idx[p]] = 1.0
        sec = getattr(row, "secondary_labels", None)
        if isinstance(sec, str) and sec.startswith("["):
            try:
                for s in ast.literal_eval(sec):
                    s = str(s)
                    if s in idx:
                        Y[r, idx[s]] = 1.0
            except Exception:
                pass
    return Y


def load_center_window(path, sr=SR, win=WIN):
    """Center 5 s window, pad/truncate to `win`. IDENTICAL to original extract."""
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) == 0:
        return np.zeros(win, dtype=np.float32)
    if len(y) >= win:
        start = (len(y) - win) // 2
        y = y[start:start + win]
    else:
        y = np.pad(y, (0, win - len(y)))
    return y.astype(np.float32)


def make_session(onnx_path, n_threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if n_threads:
        so.intra_op_num_threads = n_threads
    return ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def run_perch(sess, batch_audio, out_names, want_label_head):
    outs = sess.run(None, {"inputs": batch_audio})
    d = dict(zip(out_names, outs))
    emb = d["embedding"]
    lab = d["label"] if (want_label_head and "label" in d) else None
    return emb, lab


def merged_df(args):
    df = pd.read_csv(args.folds_csv)  # filename, fold  (defines canonical row order)
    meta = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    meta = meta[["filename", "primary_label", "secondary_labels", "author"]]
    df = df.merge(meta, on="filename", how="left")
    assert df["author"].notna().all(), "author join produced NaNs -> grouping broken"
    return df


def extract(args):
    df = merged_df(args)
    N = len(df)
    audio_root = os.path.join(args.data_dir, "train_audio")
    shard_dir = os.path.join(args.out_dir, "perch_cache", "shards")
    os.makedirs(shard_dir, exist_ok=True)

    shard_size = args.shard_size
    n_shards = (N + shard_size - 1) // shard_size
    print(f"[extract] N={N} clips, shard_size={shard_size} -> {n_shards} shards "
          f"| bs={args.batch_size} threads={args.threads} "
          f"label_head={args.cache_label_head}", flush=True)

    # which shards are already done?
    todo = []
    for sid in range(n_shards):
        emb_f = os.path.join(shard_dir, f"emb_{sid:04d}.npy")
        flag_f = os.path.join(shard_dir, f"done_{sid:04d}.flag")
        lab_ok = (not args.cache_label_head) or \
                 os.path.exists(os.path.join(shard_dir, f"lab_{sid:04d}.npy"))
        if os.path.exists(emb_f) and os.path.exists(flag_f) and lab_ok:
            continue
        todo.append(sid)
    print(f"[extract] {n_shards - len(todo)}/{n_shards} shards already done; "
          f"{len(todo)} remaining", flush=True)
    if not todo:
        print("[extract] nothing to do.", flush=True)
        return

    sess = make_session(args.onnx, args.threads)
    out_names = [o.name for o in sess.get_outputs()]

    t0 = time.time()
    clips_since = 0
    clips_total_run = 0
    for k, sid in enumerate(todo):
        lo = sid * shard_size
        hi = min(lo + shard_size, N)
        rows = df.iloc[lo:hi]
        sn = len(rows)
        emb = np.zeros((sn, EMB_DIM), dtype=np.float16)
        lab = np.zeros((sn, N_LABEL_HEAD), dtype=np.float16) if args.cache_label_head else None
        fns = rows["filename"].tolist()
        for j in range(0, sn, args.batch_size):
            batch_fns = fns[j:j + args.batch_size]
            batch = np.stack([load_center_window(os.path.join(audio_root, fn))
                              for fn in batch_fns]).astype(np.float32)
            e, l = run_perch(sess, batch, out_names, args.cache_label_head)
            emb[j:j + len(batch_fns)] = e.astype(np.float16)
            if lab is not None:
                lab[j:j + len(batch_fns)] = l.astype(np.float16)
            clips_since += len(batch_fns)
            clips_total_run += len(batch_fns)
            if clips_since >= args.log_every:
                el = time.time() - t0
                rate = clips_total_run / el
                remaining = sum(min((s + 1) * shard_size, N) - s * shard_size
                                for s in todo[k:]) - (j + len(batch_fns))
                eta = remaining / max(rate, 1e-6)
                print(f"[extract] shard {sid} ({k+1}/{len(todo)}) "
                      f"{clips_total_run} clips this run | {rate:.1f} clip/s | "
                      f"eta {eta/60:.1f} min", flush=True)
                clips_since = 0
        # atomic write of this shard
        ef = os.path.join(shard_dir, f"emb_{sid:04d}.npy")
        np.save(ef + ".tmp.npy", emb); os.replace(ef + ".tmp.npy", ef)
        if lab is not None:
            lf = os.path.join(shard_dir, f"lab_{sid:04d}.npy")
            np.save(lf + ".tmp.npy", lab); os.replace(lf + ".tmp.npy", lf)
        open(os.path.join(shard_dir, f"done_{sid:04d}.flag"), "w").write("ok")
        print(f"[extract] WROTE shard {sid:04d} rows[{lo}:{hi}] -> {ef}", flush=True)
    dt = time.time() - t0
    print(f"[extract] run done: {clips_total_run} clips in {dt/60:.1f} min "
          f"({clips_total_run/max(dt,1e-6):.1f} clip/s)", flush=True)


def assemble(args):
    df = merged_df(args)
    N = len(df)
    shard_dir = os.path.join(args.out_dir, "perch_cache", "shards")
    shard_size = args.shard_size
    n_shards = (N + shard_size - 1) // shard_size
    missing = [sid for sid in range(n_shards)
               if not os.path.exists(os.path.join(shard_dir, f"done_{sid:04d}.flag"))]
    if missing:
        print(f"[assemble] NOT ready: {len(missing)} shards missing "
              f"(first few {missing[:5]}). Re-run extract.", flush=True)
        return False

    embs = np.zeros((N, EMB_DIM), dtype=np.float16)
    lab = np.zeros((N, N_LABEL_HEAD), dtype=np.float16) if args.cache_label_head else None
    for sid in range(n_shards):
        lo = sid * shard_size
        e = np.load(os.path.join(shard_dir, f"emb_{sid:04d}.npy"))
        embs[lo:lo + len(e)] = e
        if lab is not None:
            l = np.load(os.path.join(shard_dir, f"lab_{sid:04d}.npy"))
            lab[lo:lo + len(l)] = l

    cache_dir = os.path.join(args.out_dir, "perch_cache")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "embeddings.npy"), embs)
    if lab is not None:
        np.save(os.path.join(cache_dir, "label_head.npy"), lab)
    class_order = load_taxonomy_order(args.data_dir)
    Y = build_targets(df, class_order)
    np.save(os.path.join(args.out_dir, "oof_targets.npy"), Y)
    df[["filename", "fold", "author"]].to_csv(
        os.path.join(args.out_dir, "oof_meta.csv"), index=False)
    print(f"[assemble] DONE embeddings.npy {embs.shape} "
          f"{'+ label_head.npy ' + str(lab.shape) if lab is not None else ''} "
          f"| oof_targets {Y.shape} | oof_meta rows {len(df)}", flush=True)
    assert np.isfinite(embs.astype(np.float32)).all(), "non-finite embeddings"
    print("[assemble] embeddings finite OK", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extract", "assemble", "all"], default="all")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/SHC/birdclef2026_clef/data"))
    ap.add_argument("--folds_csv", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--shard_size", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--cache_label_head", action="store_true")
    args = ap.parse_args()

    if args.folds_csv is None:
        cand = os.path.join(os.path.dirname(args.data_dir.rstrip("/")), "folds.csv")
        args.folds_csv = cand if os.path.exists(cand) else os.path.join(args.data_dir, "folds.csv")
    if args.onnx is None:
        args.onnx = os.path.join(args.data_dir, "perch_onnx", "perch_v2_no_dft.onnx")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "oof")
    print(f"[cfg] folds_csv={args.folds_csv}\n[cfg] onnx={args.onnx}\n"
          f"[cfg] out_dir={args.out_dir} mode={args.mode}", flush=True)

    if args.mode in ("extract", "all"):
        extract(args)
    if args.mode in ("assemble", "all"):
        assemble(args)


if __name__ == "__main__":
    main()
