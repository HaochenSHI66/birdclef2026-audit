"""Merge BEATs per-fold/seed OOF shards into per-seed full (N,234) + 5-seed
ensemble `beats_ensemble_oof.npy`, identically to CNN/PANN (reuses merge_sidecar
from b1_pooled_auc_compare.py). Then report BEATs pooled macro-AUC (ensemble +
per-seed range) on the COMMON evaluable-class mask across all four model families,
and sanity-check 25/25 present, 0 all-NaN rows.
"""
import os, sys, json
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from b2_oracle_headroom import common_eval_mask, macro_auc_on_mask
from b1_pooled_auc_compare import merge_sidecar

OOF = os.path.expanduser("~/SHC/birdclef2026_clef/data/oof")
seeds = [41, 42, 43, 44, 45]
folds = [0, 1, 2, 3, 4]

Y = np.load(os.path.join(OOF, "oof_targets.npy")).astype(np.float64)
N, C = Y.shape

# count present
n_present = sum(1 for s in seeds for f in folds
                if os.path.exists(os.path.join(OOF, f"beats_fold{f}_seed{s}.npy"))
                and os.path.exists(os.path.join(OOF, f"beats_fold{f}_seed{s}_rows.npy")))
print(f"[beats] shards present {n_present}/25")
assert n_present == 25, f"BEATs incomplete: {n_present}/25"

beats_seeds, beats_ens, beats_miss = merge_sidecar(OOF, "beats", seeds, folds, N, C)
print(f"[beats] missing per seed: {beats_miss}")

# all-NaN row check (every row should be scored in every fold)
ens_allnan = int(np.isnan(beats_ens).all(1).sum())
print(f"[beats] ensemble all-NaN rows = {ens_allnan}")
for s in seeds:
    an = int(np.isnan(beats_seeds[s]).all(1).sum())
    print(f"[beats] seed{s} all-NaN rows = {an}")

# pooled macro-AUC on COMMON mask across all 4 families' ensembles + anchors
perch_lp = np.load(os.path.join(OOF, "perch_anchor_oof.npy")).astype(np.float64)
perch_zs = np.load(os.path.join(OOF, "perch_zeroshot_oof.npy")).astype(np.float64)
cnn = np.load(os.path.join(OOF, "cnn_ensemble_oof.npy")).astype(np.float64)
pann = np.load(os.path.join(OOF, "pann_ensemble_oof.npy")).astype(np.float64)

models = {"perch_lp": perch_lp, "perch_zs": perch_zs, "cnn_ens": cnn,
          "pann_ens": pann, "beats_ens": beats_ens}
for s in seeds:
    models[f"beats_seed{s}"] = beats_seeds[s]

mask = common_eval_mask(Y, list(models.values()))
print(f"\n[mask] common eval classes across all (incl BEATs seeds) = {int(mask.sum())}")
out = {"n_common_eval_classes": int(mask.sum()), "macro_auc_common": {}}
for nm, M in models.items():
    au = macro_auc_on_mask(Y, M, mask)
    out["macro_auc_common"][nm] = au
    print(f"  {nm:14s} {au:.4f}")

beats_seed_aucs = [out["macro_auc_common"][f"beats_seed{s}"] for s in seeds]
out["beats_seed_spread"] = {
    "mean": float(np.mean(beats_seed_aucs)), "std": float(np.std(beats_seed_aucs)),
    "min": float(np.min(beats_seed_aucs)), "max": float(np.max(beats_seed_aucs)),
    "per_seed": {str(s): beats_seed_aucs[i] for i, s in enumerate(seeds)}}
out["beats_ensemble_macro_auc"] = out["macro_auc_common"]["beats_ens"]
out["beats_all_nan_rows_ensemble"] = ens_allnan
out["beats_shards_present"] = n_present
print(f"\n[beats] ENSEMBLE common-mask macro-AUC = {out['beats_ensemble_macro_auc']:.4f}")
print(f"[beats] per-seed: mean {out['beats_seed_spread']['mean']:.4f} "
      f"std {out['beats_seed_spread']['std']:.4f} "
      f"range [{out['beats_seed_spread']['min']:.4f}, {out['beats_seed_spread']['max']:.4f}]")

with open(os.path.join(OOF, "beats_merge_report.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=float)
print(f"[done] wrote {os.path.join(OOF, 'beats_merge_report.json')}")
