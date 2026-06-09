"""Decoded-waveform cache (32 kHz, 5 s center window, float16) — RESUMABLE + LOGGED.

Same product as b1_build_wav_cache.py (a single contiguous wav_cache.npy the CNN /
PANNs sidecar loaders read instead of re-decoding ogg every epoch), but:
  - sharded (--shard_size) with per-shard done flags so a crash/network drop loses
    at most one in-flight shard, not the whole 11 GB pass;
  - logs progress every --log_every clips with clip/s + ETA;
  - disk guard aborts before exceeding ~92% of /home.

Each shard decodes its clip range with a ProcessPool, then atomically writes:
    wav_shards/wav_<sid>.npy   float16 [shard_n, 160000]
    wav_shards/done_<sid>.flag
--mode assemble concatenates shards (folds.csv order) into:
    wav_cache.npy        float16 [N, 160000]   (~11.4 GB)
    wav_cache_index.csv  row,filename,fold
    wav_cache_meta.json
and validates a random cached clip == fresh decode (float16 round-trip).

Usage:
  python -u src/b1_build_wav_cache_resumable.py --mode all --workers 24
  python -u src/b1_build_wav_cache_resumable.py --mode assemble   # after shards exist
  python -u src/b1_build_wav_cache_resumable.py --validate_only
"""
import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

SR = 32000
WIN = 5 * SR  # 160000


def load_center_window(path, sr=SR, win=WIN):
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


def _decode_one(args):
    i, path = args
    try:
        return i, load_center_window(path).astype(np.float16), None
    except Exception as e:
        return i, np.zeros(WIN, dtype=np.float16), f"{path}: {e}"


def _paths(args):
    if args.folds_csv is None:
        cand = os.path.join(os.path.dirname(args.data_dir.rstrip("/")), "folds.csv")
        args.folds_csv = cand if os.path.exists(cand) else os.path.join(args.data_dir, "folds.csv")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "wav_cache")
    os.makedirs(args.out_dir, exist_ok=True)
    return (os.path.join(args.out_dir, "wav_cache.npy"),
            os.path.join(args.out_dir, "wav_cache_index.csv"),
            os.path.join(args.out_dir, "wav_cache_meta.json"),
            os.path.join(args.out_dir, "wav_shards"),
            os.path.join(args.data_dir, "train_audio"))


def disk_guard(out_dir, need_bytes, cap=92.0):
    total, used, free = shutil.disk_usage(out_dir)
    proj = 100.0 * (used + need_bytes) / total
    print(f"[disk] free={free/1e9:.1f}GB need={need_bytes/1e9:.2f}GB "
          f"projected_used={proj:.1f}% (cap {cap}%)", flush=True)
    if proj > cap:
        raise SystemExit(f"[ABORT] would push /home to {proj:.1f}% (>{cap}%).")


def build(args):
    cache_path, index_path, meta_path, shard_dir, audio_root = _paths(args)
    os.makedirs(shard_dir, exist_ok=True)
    df = pd.read_csv(args.folds_csv)
    N = len(df)
    paths = [os.path.join(audio_root, fn) for fn in df["filename"]]
    shard_size = args.shard_size
    n_shards = (N + shard_size - 1) // shard_size
    # final cache + remaining shards both need space; guard on the larger (final).
    disk_guard(args.out_dir, N * WIN * 2)

    todo = [sid for sid in range(n_shards)
            if not os.path.exists(os.path.join(shard_dir, f"done_{sid:04d}.flag"))]
    print(f"[build] N={N} shard_size={shard_size} -> {n_shards} shards; "
          f"{n_shards - len(todo)} done, {len(todo)} remaining; workers={args.workers}",
          flush=True)

    t0 = time.time()
    done_clips = 0
    n_err = 0
    errs = []
    for k, sid in enumerate(todo):
        lo = sid * shard_size
        hi = min(lo + shard_size, N)
        sn = hi - lo
        buf = np.zeros((sn, WIN), dtype=np.float16)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_decode_one, (i - lo, paths[i])) for i in range(lo, hi)]
            for fut in as_completed(futs):
                j, y, err = fut.result()
                buf[j] = y
                if err:
                    n_err += 1
                    if len(errs) < 20:
                        errs.append(err)
                done_clips += 1
                if done_clips % args.log_every == 0:
                    el = time.time() - t0
                    rate = done_clips / el
                    remaining = sum(min((s + 1) * shard_size, N) - s * shard_size
                                    for s in todo[k:]) - (done_clips - sum(
                        min((s + 1) * shard_size, N) - s * shard_size for s in todo[:k]))
                    eta = max(remaining, 0) / max(rate, 1e-6)
                    print(f"[build] shard {sid} ({k+1}/{len(todo)}) "
                          f"{done_clips} clips this run | {rate:.1f} clip/s | "
                          f"eta {eta/60:.1f} min | errs={n_err}", flush=True)
        wf = os.path.join(shard_dir, f"wav_{sid:04d}.npy")
        np.save(wf + ".tmp.npy", buf); os.replace(wf + ".tmp.npy", wf)
        open(os.path.join(shard_dir, f"done_{sid:04d}.flag"), "w").write("ok")
        print(f"[build] WROTE shard {sid:04d} rows[{lo}:{hi}] -> {wf}", flush=True)
    dt = time.time() - t0
    print(f"[build] run done: {done_clips} clips in {dt/60:.1f} min; errors={n_err}",
          flush=True)
    if errs:
        print(f"[build] first errors: {errs}", flush=True)


def assemble(args):
    cache_path, index_path, meta_path, shard_dir, audio_root = _paths(args)
    df = pd.read_csv(args.folds_csv)
    N = len(df)
    paths = [os.path.join(audio_root, fn) for fn in df["filename"]]
    shard_size = args.shard_size
    n_shards = (N + shard_size - 1) // shard_size
    missing = [sid for sid in range(n_shards)
               if not os.path.exists(os.path.join(shard_dir, f"done_{sid:04d}.flag"))]
    if missing:
        print(f"[assemble] NOT ready: {len(missing)} shards missing "
              f"(first {missing[:5]}). Re-run build.", flush=True)
        return False
    disk_guard(args.out_dir, N * WIN * 2)
    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16,
                                      shape=(N, WIN))
    for sid in range(n_shards):
        lo = sid * shard_size
        b = np.load(os.path.join(shard_dir, f"wav_{sid:04d}.npy"))
        cache[lo:lo + len(b)] = b
    cache.flush(); del cache
    df_idx = df.copy(); df_idx.insert(0, "row", range(N))
    df_idx[["row", "filename", "fold"]].to_csv(index_path, index=False)
    json.dump({"n": N, "win": WIN, "sr": SR, "dtype": "float16",
               "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(meta_path, "w"), indent=2)
    print(f"[assemble] DONE wav_cache.npy [{N},{WIN}] + index + meta", flush=True)
    return validate(cache_path, df, paths)


def validate(cache_path, df, paths, n_check=3):
    cache = np.load(cache_path, mmap_mode="r")
    print(f"[validate] cache shape {cache.shape} dtype {cache.dtype}", flush=True)
    rng = np.random.RandomState(0)
    rows = rng.choice(len(df), size=min(n_check, len(df)), replace=False)
    ok = True
    for r in rows:
        fresh = load_center_window(paths[r]).astype(np.float16).astype(np.float32)
        cached = np.asarray(cache[r]).astype(np.float32)
        max_abs = float(np.max(np.abs(fresh - cached)))
        same = max_abs < 1e-3
        ok = ok and same
        print(f"[validate] row {r} {df['filename'].iloc[r]}: "
              f"max|fresh-cached|={max_abs:.2e} -> {'OK' if same else 'MISMATCH'}",
              flush=True)
    print(f"[validate] {'ALL OK' if ok else 'FAILED'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build", "assemble", "all"], default="all")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/SHC/birdclef2026_clef/data"))
    ap.add_argument("--folds_csv", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=2000)
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()
    if args.validate_only:
        cache_path, _, _, _, audio_root = _paths(args)
        df = pd.read_csv(args.folds_csv)
        paths = [os.path.join(audio_root, fn) for fn in df["filename"]]
        return validate(cache_path, df, paths)
    if args.mode in ("build", "all"):
        build(args)
    if args.mode in ("assemble", "all"):
        assemble(args)


if __name__ == "__main__":
    main()
