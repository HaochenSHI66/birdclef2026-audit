"""Task 4+5: merge sidecar per-run OOFs into per-seed + seed-ensemble (N,234)
matrices, then compute POOLED macro-AUC (macro ROC-AUC skipping zero-positive
classes) on a COMMON evaluable-class mask across the compared models, reusing the
b2_oracle_headroom metric fns. Prints actual numbers, #eval classes, per-seed spread.

Compared models: perch_anchor (linear-probe), perch_zeroshot, cnn per-seed +
seed-ensemble, pann per-seed + seed-ensemble.

No retraining; pure post-hoc on cached OOFs.
"""
import argparse
import os
import sys
import json
import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from b2_oracle_headroom import common_eval_mask, macro_auc_on_mask, per_class_auc_on_mask  # noqa


def load_anchor(out_dir, fn):
    p = os.path.join(out_dir, fn)
    if not os.path.exists(p):
        return None
    return np.load(p).astype(np.float64)


def merge_sidecar(out_dir, model, seeds, folds, N, C):
    """Return dict seed->full(N,C) (NaN where unscored) + 'ensemble' mean over seeds.
    Also returns missing report."""
    per_seed = {}
    missing = {}
    for s in seeds:
        full = np.full((N, C), np.nan, dtype=np.float64)
        miss = []
        n_rows = 0
        for f in folds:
            tag = f"{model}_fold{f}_seed{s}"
            blk_p = os.path.join(out_dir, f"{tag}.npy")
            row_p = os.path.join(out_dir, f"{tag}_rows.npy")
            if not (os.path.exists(blk_p) and os.path.exists(row_p)):
                miss.append(f)
                continue
            blk = np.load(blk_p).astype(np.float64)
            rows = np.load(row_p)
            full[rows] = blk
            n_rows += len(rows)
        per_seed[s] = full
        if miss:
            missing[s] = miss
        # save per-seed merged (for downstream B2)
        np.save(os.path.join(out_dir, f"{model}_seed{s}_oof.npy"),
                full.astype(np.float32))
    # seed-ensemble = nanmean over seeds (all rows scored in every seed -> plain mean)
    stack = np.stack([per_seed[s] for s in seeds], axis=0)
    with np.errstate(invalid="ignore"):
        ens = np.nanmean(stack, axis=0)
    np.save(os.path.join(out_dir, f"{model}_ensemble_oof.npy"),
            ens.astype(np.float32))
    return per_seed, ens, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",
                    default=os.path.expanduser("~/SHC/birdclef2026_clef/data/oof"))
    ap.add_argument("--seeds", default="41,42,43,44,45")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    folds = [int(x) for x in args.folds.split(",")]

    Y = np.load(os.path.join(args.out_dir, "oof_targets.npy")).astype(np.float64)
    N, C = Y.shape
    print(f"[load] targets {Y.shape}  pos_classes={int((Y.sum(0)>0).sum())}")

    # ---- anchors ----
    anchor_lp = load_anchor(args.out_dir, "perch_anchor_oof.npy")
    anchor_zs = load_anchor(args.out_dir, "perch_zeroshot_oof.npy")
    gates = []
    if anchor_lp is None:
        gates.append("perch_anchor_oof.npy MISSING")
    if anchor_zs is None:
        gates.append("perch_zeroshot_oof.npy MISSING")

    # ---- sidecars ----
    cnn_seeds, cnn_ens, cnn_miss = merge_sidecar(args.out_dir, "cnn", seeds, folds, N, C)
    pann_seeds, pann_ens, pann_miss = merge_sidecar(args.out_dir, "pann", seeds, folds, N, C)
    n_present_cnn = sum(1 for s in seeds for f in folds
                        if os.path.exists(os.path.join(args.out_dir, f"cnn_fold{f}_seed{s}.npy")))
    n_present_pann = sum(1 for s in seeds for f in folds
                         if os.path.exists(os.path.join(args.out_dir, f"pann_fold{f}_seed{s}.npy")))
    print(f"[merge] cnn runs present {n_present_cnn}/25  missing={cnn_miss}")
    print(f"[merge] pann runs present {n_present_pann}/25  missing={pann_miss}")
    if n_present_cnn != 25:
        gates.append(f"cnn runs {n_present_cnn}/25")
    if n_present_pann != 25:
        gates.append(f"pann runs {n_present_pann}/25")

    # NaN sanity on the rows each model scored
    def nan_report(name, M):
        # full coverage expected for anchors+sidecar merges except unmapped/absent classes
        all_nan_rows = int(np.isnan(M).all(1).sum())
        return all_nan_rows

    # ---- common mask across ALL compared models ----
    models = {}
    if anchor_lp is not None:
        models["perch_anchor_linear_probe"] = anchor_lp
    if anchor_zs is not None:
        models["perch_zeroshot"] = anchor_zs
    models["cnn_ensemble"] = cnn_ens
    models["pann_ensemble"] = pann_ens
    for s in seeds:
        models[f"cnn_seed{s}"] = cnn_seeds[s]
        models[f"pann_seed{s}"] = pann_seeds[s]

    pred_list = list(models.values())
    common_mask = common_eval_mask(Y, pred_list)
    n_common = int(common_mask.sum())
    print(f"\n[mask] COMMON evaluable classes across all compared models = {n_common}")

    # Also report each model's NATIVE eval coverage (self mask), informative
    res = {"n_common_eval_classes": n_common, "N": int(N), "C": int(C)}
    res["pos_classes"] = int((Y.sum(0) > 0).sum())
    res["macro_auc_common_mask"] = {}
    res["native_eval_classes"] = {}
    res["all_nan_rows"] = {}
    print(f"\n{'model':30s} {'macroAUC(common)':>16s} {'nativeEvalCls':>13s} {'allNaNrows':>11s}")
    for name, M in models.items():
        au = macro_auc_on_mask(Y, M, common_mask)
        self_mask = common_eval_mask(Y, [M])
        res["macro_auc_common_mask"][name] = au
        res["native_eval_classes"][name] = int(self_mask.sum())
        res["all_nan_rows"][name] = nan_report(name, M)
        print(f"{name:30s} {au:16.4f} {int(self_mask.sum()):13d} {nan_report(name,M):11d}")

    # per-seed spread
    cnn_seed_aucs = [res["macro_auc_common_mask"][f"cnn_seed{s}"] for s in seeds]
    pann_seed_aucs = [res["macro_auc_common_mask"][f"pann_seed{s}"] for s in seeds]
    res["cnn_seed_spread"] = {
        "mean": float(np.mean(cnn_seed_aucs)), "std": float(np.std(cnn_seed_aucs)),
        "min": float(np.min(cnn_seed_aucs)), "max": float(np.max(cnn_seed_aucs)),
        "per_seed": {str(s): cnn_seed_aucs[i] for i, s in enumerate(seeds)}}
    res["pann_seed_spread"] = {
        "mean": float(np.mean(pann_seed_aucs)), "std": float(np.std(pann_seed_aucs)),
        "min": float(np.min(pann_seed_aucs)), "max": float(np.max(pann_seed_aucs)),
        "per_seed": {str(s): pann_seed_aucs[i] for i, s in enumerate(seeds)}}

    print(f"\n[spread] CNN  per-seed (common mask): "
          f"mean {res['cnn_seed_spread']['mean']:.4f} "
          f"std {res['cnn_seed_spread']['std']:.4f} "
          f"min {res['cnn_seed_spread']['min']:.4f} "
          f"max {res['cnn_seed_spread']['max']:.4f} "
          f"ensemble {res['macro_auc_common_mask']['cnn_ensemble']:.4f}")
    print(f"[spread] PANN per-seed (common mask): "
          f"mean {res['pann_seed_spread']['mean']:.4f} "
          f"std {res['pann_seed_spread']['std']:.4f} "
          f"min {res['pann_seed_spread']['min']:.4f} "
          f"max {res['pann_seed_spread']['max']:.4f} "
          f"ensemble {res['macro_auc_common_mask']['pann_ensemble']:.4f}")

    res["gates"] = gates
    if gates:
        print(f"\n[GATE] {gates}")
    else:
        print("\n[GATE] none")

    print("\n" + json.dumps(res, indent=2, default=float))
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(res, fh, indent=2, default=float)
        print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
