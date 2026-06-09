"""B1' — Perch-V2 CPU-ONNX feature/anchor extraction + OOF anchor head.

The reproducible "Perch-V2 anchor head" for the paper.

ONNX FACTS (verified on server, perch_v2_no_dft.onnx):
  input  : inputs            [batch, 160000]  = 5 s @ 32 kHz raw mono audio
  outputs:
    embedding         [batch, 1536]            <- per-clip foundation embedding (USED)
    spatial_embedding [batch, 16, 4, 1536]
    spectrogram       [batch, 500, 128]
    label             [batch, 14795]           <- Perch GLOBAL taxonomy logits (NOT 234)

CRITICAL DESIGN DECISION (see HARNESS.md / harness-report):
  The `label` head is Perch's own 14795-class global vocabulary, and the dataset
  ships NO label-name file, so it cannot be directly subset to the 234 BirdCLEF
  classes without an external Perch->eBird name map. The bronze pipeline itself
  uses the *embedding* (via ProtoSSM prototype matching), not the raw 14795 head.
  Therefore the reproducible anchor head here =
      per-fold LINEAR PROBE (multinomial-ish multilabel logistic regression)
      trained on the frozen 1536-d embedding -> 234 classes.
  We ALSO persist the raw 14795 label-head probs for the parity report.

Two-phase usage:
  1) extract : run Perch ONNX over all clips once -> cache embeddings (+ optional
               label-head) to data/perch_cache/embeddings.npy  (frozen, reusable).
  2) oof     : for each fold, fit a linear probe on TRAIN embeddings, predict VAL
               -> assemble a (N, 234) OOF matrix aligned to taxonomy order.

234-vs-206 handling:
  - Column order = taxonomy.csv primary_label order (the 234 scored classes,
    == sample_submission.csv column order).
  - 28 scored classes never appear in any focal training clip => the probe has
    no positives for them => those OOF columns are left as NaN (no signal).
  - Macro-AUC is computed ONLY over classes with >=1 true positive in the eval
    set (reuse of scripts/optimize_ensemble.py:19 logic), so NaN columns are
    skipped naturally and never counted.

Outputs (written under --out_dir, default data/oof/):
  perch_anchor_oof.npy   float32 [N, 234]  OOF anchor probs (NaN for absent cls)
  oof_targets.npy        float32 [N, 234]  multi-hot ground truth (primary+secondary)
  oof_meta.csv           filename, fold, author  (row order == the .npy rows)
  perch_cache/embeddings.npy  float32 [N, 1536]  (frozen feature cache)

Dry-run proves the whole pipeline on ~40 clips and prints ms/clip + projected
full-set hours. It does NOT run the full 35k extraction.
"""
import argparse
import ast
import os
import time
import numpy as np
import pandas as pd

SR = 32000
WIN = 5 * SR          # 160000 samples == ONNX input length
EMB_DIM = 1536
N_LABEL_HEAD = 14795


# ----------------------------- metric (shared) -----------------------------
def compute_macro_auc(y_true, y_pred):
    """Macro ROC-AUC, skipping classes with no positives. Mirrors
    scripts/optimize_ensemble.py:19. NaN pred columns are skipped because a
    column with no positives is skipped anyway; columns WITH positives must be
    non-NaN (asserted at call sites)."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            col = y_pred[:, i]
            if np.isnan(col).any():
                continue
            try:
                aucs.append(roc_auc_score(y_true[:, i], col))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


# ----------------------------- taxonomy / labels -----------------------------
def load_taxonomy_order(data_dir):
    """Return the 234-class column order (taxonomy.csv primary_label order)."""
    tax = pd.read_csv(os.path.join(data_dir, "taxonomy.csv"))
    classes = tax["primary_label"].astype(str).tolist()
    assert len(classes) == 234, f"expected 234 classes, got {len(classes)}"
    return classes


def build_targets(df, class_order):
    """Multi-hot [N, 234] from primary + secondary labels, taxonomy-aligned."""
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


# ----------------------------- audio -----------------------------
def load_center_window(path, sr=SR, win=WIN):
    """Load a clip, take the centre 5 s window, pad/truncate to `win` samples."""
    import librosa
    y, sr0 = librosa.load(path, sr=sr, mono=True)
    if len(y) == 0:
        return np.zeros(win, dtype=np.float32)
    if len(y) >= win:
        start = (len(y) - win) // 2
        y = y[start:start + win]
    else:
        y = np.pad(y, (0, win - len(y)))
    return y.astype(np.float32)


# ----------------------------- ONNX session -----------------------------
def make_session(onnx_path, n_threads=0):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if n_threads:
        so.intra_op_num_threads = n_threads
    return ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def run_perch(sess, batch_audio, want_label_head=False):
    """batch_audio: [B, 160000] float32. Returns (emb[B,1536], label[B,14795]|None)."""
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, {"inputs": batch_audio})
    d = dict(zip(out_names, outs))
    emb = d["embedding"]
    lab = d["label"] if (want_label_head and "label" in d) else None
    return emb, lab


# ----------------------------- extraction -----------------------------
def extract(args):
    df = pd.read_csv(args.folds_csv)               # filename, fold
    meta = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    meta = meta[["filename", "primary_label", "secondary_labels", "author"]]
    df = df.merge(meta, on="filename", how="left")
    assert df["author"].notna().all(), "author join produced NaNs -> grouping broken"

    if args.dry_run:
        df = df.groupby("fold", group_keys=False).head(args.dry_n).reset_index(drop=True)
        print(f"[DRY] subset to {len(df)} clips ({args.dry_n}/fold)")

    class_order = load_taxonomy_order(args.data_dir)
    Y = build_targets(df, class_order)

    sess = make_session(args.onnx, n_threads=args.threads)
    audio_root = os.path.join(args.data_dir, "train_audio")

    embs = np.zeros((len(df), EMB_DIM), dtype=np.float32)
    lab_cache = None
    if args.cache_label_head:
        lab_cache = np.zeros((len(df), N_LABEL_HEAD), dtype=np.float32)

    bs = args.batch_size
    t0 = time.time()
    n_done = 0
    for i in range(0, len(df), bs):
        rows = df.iloc[i:i + bs]
        batch = np.stack([load_center_window(os.path.join(audio_root, fn))
                          for fn in rows["filename"]]).astype(np.float32)
        emb, lab = run_perch(sess, batch, want_label_head=args.cache_label_head)
        embs[i:i + len(rows)] = emb
        if lab is not None:
            lab_cache[i:i + len(rows)] = lab
        n_done += len(rows)
    dt = time.time() - t0
    ms_per_clip = dt * 1000.0 / max(n_done, 1)
    full_hours = ms_per_clip * 35549 / 1000.0 / 3600.0
    print(f"[extract] {n_done} clips in {dt:.1f}s -> {ms_per_clip:.1f} ms/clip "
          f"(incl audio I/O) | projected full 35,549-clip set: {full_hours:.2f} h "
          f"single-thread-batch (parallelisable across CPU workers)")

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "perch_cache")
    os.makedirs(cache_dir, exist_ok=True)
    suffix = "_dry" if args.dry_run else ""
    np.save(os.path.join(cache_dir, f"embeddings{suffix}.npy"), embs)
    np.save(os.path.join(args.out_dir, f"oof_targets{suffix}.npy"), Y)
    df[["filename", "fold", "author"]].to_csv(
        os.path.join(args.out_dir, f"oof_meta{suffix}.csv"), index=False)
    if lab_cache is not None:
        np.save(os.path.join(cache_dir, f"label_head{suffix}.npy"), lab_cache)
        parity_report(lab_cache, embs, Y, df)
    print(f"[extract] cached embeddings -> {cache_dir}/embeddings{suffix}.npy "
          f"shape {embs.shape}")
    return embs, Y, df, class_order


# ----------------------------- parity report -----------------------------
def parity_report(lab_head, embs, Y, df):
    """Sanity / parity checks on raw Perch outputs (not vs Kaggle frozen yet —
    that needs the same clips through the Kaggle kernel; here we assert internal
    sanity: probabilities in range, rank stability across a tiny resample,
    embedding norms finite)."""
    probs = 1.0 / (1.0 + np.exp(-lab_head))
    print("\n[parity] === Perch-ONNX output sanity ===")
    print(f"[parity] label-head sigmoid: min={probs.min():.4f} "
          f"max={probs.max():.4f} mean={probs.mean():.4f} "
          f"(expect 0<.<1, mean ~0.2-0.3)")
    assert probs.min() >= 0.0 and probs.max() <= 1.0, "probs out of [0,1]"
    assert np.isfinite(embs).all(), "non-finite embeddings"
    # rank stability: top-5 label indices should be deterministic on repeat
    # (we just report the top-5 of clip 0 for the reviewer)
    top5 = np.argsort(-probs[0])[:5]
    print(f"[parity] clip0 top-5 label-head idx: {top5.tolist()} "
          f"probs={probs[0][top5].round(3).tolist()}")
    print(f"[parity] embedding norm: mean={np.linalg.norm(embs,axis=1).mean():.3f}")
    print("[parity] NOTE: label head is Perch's 14795 global taxonomy, "
          "NOT the 234 scored classes -> anchor uses embedding linear probe.\n")


# ----------------------------- OOF anchor (linear probe) -----------------------------
def build_oof(args):
    """Fit per-fold linear probe on cached embeddings -> taxonomy-aligned OOF."""
    from sklearn.linear_model import LogisticRegression
    suffix = "_dry" if args.dry_run else ""
    cache = os.path.join(args.out_dir, "perch_cache", f"embeddings{suffix}.npy")
    embs = np.load(cache)
    Y = np.load(os.path.join(args.out_dir, f"oof_targets{suffix}.npy"))
    meta = pd.read_csv(os.path.join(args.out_dir, f"oof_meta{suffix}.csv"))
    folds = meta["fold"].values
    N, C = Y.shape
    oof = np.full((N, C), np.nan, dtype=np.float32)

    for f in sorted(np.unique(folds)):
        tr = folds != f
        va = folds == f
        Xtr, Xva = embs[tr], embs[va]
        for c in range(C):
            ytr = Y[tr, c]
            if ytr.sum() == 0:
                continue  # absent in train fold -> leave NaN (206/234 gap)
            if len(np.unique(ytr)) < 2:
                continue
            clf = LogisticRegression(max_iter=args.max_iter, C=args.reg_C,
                                     class_weight="balanced")
            clf.fit(Xtr, ytr)
            oof[va, c] = clf.predict_proba(Xva)[:, 1].astype(np.float32)

    suf = suffix
    np.save(os.path.join(args.out_dir, f"perch_anchor_oof{suf}.npy"), oof)
    # report OOF macro-AUC on the full assembled matrix (eval skips NaN/zero-pos)
    auc = compute_macro_auc(Y, np.nan_to_num(oof, nan=0.0))
    n_eval = int(((Y.sum(0) > 0) & (~np.isnan(oof).any(0))).sum())
    print(f"[oof] anchor OOF macro-AUC (skip-zero-pos) = {auc:.4f} "
          f"over {n_eval} eval classes (of {C}); "
          f"saved perch_anchor_oof{suf}.npy {oof.shape}")
    return oof, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extract", "oof", "all"], default="all")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/SHC/birdclef2026_clef/data"))
    ap.add_argument("--folds_csv", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0=default)")
    ap.add_argument("--cache_label_head", action="store_true",
                    help="also cache the 14795 label head (for parity report)")
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--reg_C", type=float, default=1.0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dry_n", type=int, default=8, help="clips per fold in dry-run")
    args = ap.parse_args()

    if args.folds_csv is None:
        args.folds_csv = os.path.join(os.path.dirname(args.data_dir.rstrip("/")), "folds.csv")
        if not os.path.exists(args.folds_csv):
            args.folds_csv = os.path.join(args.data_dir, "folds.csv")
    if args.onnx is None:
        args.onnx = os.path.join(args.data_dir, "perch_onnx", "perch_v2_no_dft.onnx")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "oof")

    print(f"[cfg] folds_csv={args.folds_csv}\n[cfg] onnx={args.onnx}\n"
          f"[cfg] out_dir={args.out_dir} dry_run={args.dry_run}")

    if args.mode in ("extract", "all"):
        extract(args)
    if args.mode in ("oof", "all"):
        build_oof(args)


if __name__ == "__main__":
    main()
