"""Merge per-(model,fold,seed) OOF blocks into full (N,234) matrices + sanity AUC.

For each (model, seed): scatter every fold's VAL-rows OOF block back into a full
(N,234) matrix using the saved {tag}_fold{f}_seed{s}_rows.npy row indices (which are
positions into folds.csv / oof_targets order). Writes:
  {model}_seed{s}_oof.npy   float32 [N,234]
and prints that matrix's pooled macro-AUC (skip-zero-pos) against oof_targets.npy.

Also prints, per model, the mean pooled macro-AUC across seeds, and reports the two
anchors (zero-shot, linear-probe) if their OOFs are present. Does NOT run B2.

Usage:
  python src/b1_matrix_merge.py --models cnn,pann --seeds 41,42,43,44,45
"""
import argparse
import os
import numpy as np
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
    ap.add_argument("--models", default="cnn,pann")
    ap.add_argument("--seeds", default="41,42,43,44,45")
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()

    models = args.models.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    folds = [int(x) for x in args.folds.split(",")]

    Y = np.load(os.path.join(args.out_dir, "oof_targets.npy"))
    N, C = Y.shape
    print(f"[merge] targets {Y.shape}")

    # ---- anchors (informational) ----
    for name, fn in [("anchor_zeroshot", "perch_zeroshot_oof.npy"),
                     ("anchor_linear_probe", "perch_anchor_oof.npy")]:
        p = os.path.join(args.out_dir, fn)
        if os.path.exists(p):
            a = np.load(p)
            n = min(len(a), len(Y))
            au, k = macro_auc(Y[:n], np.nan_to_num(a[:n], nan=0.0))
            print(f"[anchor] {name:22s} pooled macro-AUC = {au:.4f} ({k} eval cls) <- {fn}")
        else:
            print(f"[anchor] {name:22s} MISSING ({fn})")

    # ---- sidecars: merge folds per (model,seed) ----
    for model in models:
        seed_aucs = []
        for s in seeds:
            full = np.full((N, C), np.nan, dtype=np.float32)
            n_rows = 0
            missing = []
            for f in folds:
                tag = f"{model}_fold{f}_seed{s}"
                blk_p = os.path.join(args.out_dir, f"{tag}.npy")
                row_p = os.path.join(args.out_dir, f"{tag}_rows.npy")
                if not (os.path.exists(blk_p) and os.path.exists(row_p)):
                    missing.append(f); continue
                blk = np.load(blk_p)
                rows = np.load(row_p)
                full[rows] = blk
                n_rows += len(rows)
            if missing:
                print(f"[merge] {model} seed{s}: MISSING folds {missing} -> partial")
            outp = os.path.join(args.out_dir, f"{model}_seed{s}_oof.npy")
            np.save(outp, full)
            au, k = macro_auc(Y, np.nan_to_num(full, nan=0.0))
            seed_aucs.append(au)
            print(f"[merge] {model} seed{s}: {n_rows} val rows scattered, "
                  f"pooled macro-AUC = {au:.4f} ({k} eval cls) -> {os.path.basename(outp)}")
        if seed_aucs:
            print(f"[merge] === {model}: mean pooled macro-AUC across "
                  f"{len(seed_aucs)} seeds = {np.mean(seed_aucs):.4f} "
                  f"+/- {np.std(seed_aucs):.4f} (min {min(seed_aucs):.4f}, "
                  f"max {max(seed_aucs):.4f}) ===")


if __name__ == "__main__":
    main()
