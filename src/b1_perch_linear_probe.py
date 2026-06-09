"""A2 anchor — Perch linear probe: per-fold logistic regression on cached 1536-d
Perch embeddings -> (N,234) OOF. The HEADLINE supervised anchor (so the sidecar is
not a straw baseline). Identical estimator to b1_perch_extract.build_oof, packaged
as a standalone step that consumes the frozen embedding cache (no Perch re-run).

Reads:  <out_dir>/perch_cache/embeddings.npy [N,1536]
        <out_dir>/oof_targets.npy            [N,234]
        <out_dir>/oof_meta.csv               (filename, fold, author) row-aligned
Writes: <out_dir>/perch_anchor_oof.npy       [N,234]  (NaN for classes absent-in-train)

Usage: python src/b1_perch_linear_probe.py --max_iter 1000 --reg_C 1.0
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def macro_auc(y_true, y_pred):
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
    return float(np.mean(aucs)) if aucs else 0.0, len(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.expanduser("~/SHC/birdclef2026_clef/data/oof"))
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--reg_C", type=float, default=1.0)
    args = ap.parse_args()

    embs = np.load(os.path.join(args.out_dir, "perch_cache", "embeddings.npy"))
    Y = np.load(os.path.join(args.out_dir, "oof_targets.npy"))
    meta = pd.read_csv(os.path.join(args.out_dir, "oof_meta.csv"))
    folds = meta["fold"].values
    assert len(embs) == len(Y) == len(folds), \
        f"row mismatch emb={len(embs)} Y={len(Y)} folds={len(folds)}"
    N, C = Y.shape
    print(f"[probe] embeddings {embs.shape} targets {Y.shape} folds {np.unique(folds)}")
    oof = np.full((N, C), np.nan, dtype=np.float32)

    for f in sorted(np.unique(folds)):
        tr = folds != f
        va = folds == f
        # Per-fold standardization: scaler FIT ONLY on train rows, applied to val.
        # No leakage (val fold never seen by the scaler or any classifier).
        scaler = StandardScaler().fit(embs[tr].astype(np.float32))
        Xtr = scaler.transform(embs[tr].astype(np.float32))
        Xva = scaler.transform(embs[va].astype(np.float32))
        n_fit = 0
        for c in range(C):
            ytr = Y[tr, c]
            if ytr.sum() == 0 or len(np.unique(ytr)) < 2:
                continue
            clf = LogisticRegression(max_iter=args.max_iter, C=args.reg_C,
                                     class_weight="balanced")
            clf.fit(Xtr, ytr)
            oof[va, c] = clf.predict_proba(Xva)[:, 1].astype(np.float32)
            n_fit += 1
        print(f"[probe] fold {f}: fit {n_fit} classes, {va.sum()} val rows", flush=True)

    np.save(os.path.join(args.out_dir, "perch_anchor_oof.npy"), oof)
    au, k = macro_auc(Y, np.nan_to_num(oof, nan=0.0))
    print(f"[probe] linear-probe pooled macro-AUC (skip-zero-pos) = {au:.4f} "
          f"over {k} eval classes; saved perch_anchor_oof.npy {oof.shape}")


if __name__ == "__main__":
    main()
