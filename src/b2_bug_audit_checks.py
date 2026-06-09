"""B2 oracle-headroom AUDIT checks. Read-only on the real OOFs.

Checks:
  (1) choice-count vs mask: does the held-out selector write/score off-mask classes?
      Report C (full), n_common_mask, sum(choice_counts), and whether scored macro-AUC
      changes if we ZERO the oracle preds on off-mask classes (must be identical).
  (2) force-best-single: candidate set = {best_single only} -> held-out headroom MUST == 0.
      Also all-identical-candidates -> 0.
  (3) ZS routing audit: row-alignment of ZS OOF + per-class selection index sanity.
  (4) null center: report null mean/median/p5/p95 for one ZS and one LP cell.
"""
import os, sys
import numpy as np
import pandas as pd

D = os.path.expanduser("~/SHC/birdclef2026_clef/data/oof")
SRC = os.path.expanduser("~/SHC/birdclef2026_clef/src")
sys.path.insert(0, SRC)
from b2_headline_driver import (
    heldout_oracle_nested, vec_macro_auc_common, vec_auc_per_class,
    permute_within_author,
)
from b2_oracle_headroom import common_eval_mask, macro_auc_on_mask


def L(name):
    return np.load(os.path.join(D, name)).astype(np.float64)


perch_lp = L("perch_anchor_oof.npy")
perch_zs = L("perch_zeroshot_oof.npy")
cnn = L("cnn_ensemble_oof.npy")
pann = L("pann_ensemble_oof.npy")
y = L("oof_targets.npy")
meta = pd.read_csv(os.path.join(D, "oof_meta.csv"))
authors = meta["author"].values
folds = meta["fold"].values

keep = ~(np.isnan(perch_lp).all(1) | np.isnan(perch_zs).all(1)
         | np.isnan(cnn).all(1) | np.isnan(pann).all(1))
perch_lp, perch_zs, cnn, pann, y = perch_lp[keep], perch_zs[keep], cnn[keep], pann[keep], y[keep]
authors, folds = authors[keep], folds[keep]
N, C = y.shape
print(f"[load] N={N} C={C} authors={len(set(authors))}")

# ============ CHECK 1: choice-count vs mask ============
print("\n=== CHECK 1: 1170 vs ~196 ===")
models = {"perch_zeroshot": perch_zs, "cnn": cnn}
ho_orc, ho_chosen, cand_names = heldout_oracle_nested(models, y, folds)
all_arrs = list(models.values())
mask = common_eval_mask(y, all_arrs + [ho_orc])
nfolds = len(set(folds))
print(f"C(full)={C}  nfolds={nfolds}  C*nfolds={C*nfolds}")
print(f"chosen.shape={ho_chosen.shape}  sum(choice_counts)={ho_chosen.size}")
print(f"n_common_mask={int(mask.sum())}")
# Does scoring change if we NaN-out off-mask oracle columns? It must NOT, because
# macro_auc_on_mask only averages over mask classes.
ho_orc_masked = ho_orc.copy()
offmask = ~mask
ho_orc_masked[:, offmask] = np.nan
auc_full = macro_auc_on_mask(y, ho_orc, mask)
auc_masked = macro_auc_on_mask(y, ho_orc_masked, mask)
print(f"macro_auc(oracle, mask)         = {auc_full:.8f}")
print(f"macro_auc(oracle off-mask NaN'd)= {auc_masked:.8f}")
print(f"identical? {abs(auc_full-auc_masked) < 1e-12}")
# How many of the 1170 choices are for off-mask classes (benign extras)?
onmask_choices = int(mask.sum()) * nfolds
print(f"on-mask choices counted in macro-AUC = {int(mask.sum())} classes x {nfolds} folds = {onmask_choices}")
print(f"off-mask choices logged but NEVER scored = {ho_chosen.size - onmask_choices}")

# ============ CHECK 2: force best-single -> headroom 0 ============
print("\n=== CHECK 2: force-best-single == 0 ===")
# best single in the ZS+CNN cell is cnn (per JSON). Candidate set = {cnn only}.
single = {"cnn": cnn}
orc1, chosen1, cn1 = heldout_oracle_nested(single, y, folds)
# candidate dict adds 'mean' = mean of [cnn] = cnn, so candidates are {cnn, mean==cnn}
m1 = common_eval_mask(y, [cnn, orc1])
d_single = macro_auc_on_mask(y, orc1, m1) - macro_auc_on_mask(y, cnn, m1)
print(f"candidates={cn1}")
print(f"headroom(oracle - cnn) with single-model candidate set = {d_single:.3e}")
print(f"== 0 exactly? {d_single == 0.0}  (|d|<1e-12? {abs(d_single)<1e-12})")
# all-identical candidates: anchor + dup(anchor)
dup = {"a": cnn, "b": cnn.copy()}
orc2, _, cn2 = heldout_oracle_nested(dup, y, folds)
m2 = common_eval_mask(y, [cnn, orc2])
d_dup = macro_auc_on_mask(y, orc2, m2) - macro_auc_on_mask(y, cnn, m2)
print(f"headroom with identical-2-candidate set = {d_dup:.3e}  == 0? {abs(d_dup)<1e-12}")

# ============ CHECK 3: ZS routing / alignment audit ============
print("\n=== CHECK 3: ZS routing & row alignment ===")
# 3a. Is ZS row-aligned to targets? A correctly-aligned ZS should have a high
# per-class AUC on the SAME rows; a misaligned (shuffled) ZS would collapse to ~0.5.
auc_zs, valid_zs = vec_auc_per_class(y, perch_zs)
print(f"ZS per-class AUC: median={np.nanmedian(auc_zs[valid_zs]):.4f} "
      f"mean={np.nanmean(auc_zs[valid_zs]):.4f} valid_classes={int(valid_zs.sum())}")
# Sanity: a randomly row-permuted ZS should give ~0.5 (confirms alignment matters)
rng = np.random.default_rng(0)
perm = rng.permutation(N)
auc_zs_shuf, _ = vec_auc_per_class(y, perch_zs[perm])
print(f"ZS row-SHUFFLED per-class AUC median={np.nanmedian(auc_zs_shuf[valid_zs]):.4f} (should be ~0.5)")
# 3b. Selection-index sanity: for ZS+CNN, on a held-out fold, recompute the per-class
# argmax independently and confirm it equals what heldout_oracle wrote.
models_zc = {"perch_zeroshot": perch_zs, "cnn": cnn}
orc_zc, chosen_zc, cn_zc = heldout_oracle_nested(models_zc, y, folds)
# manual recompute for fold 0
f0 = sorted(set(folds))[0]
sel = folds != f0
cand = {"perch_zeroshot": perch_zs, "cnn": cnn, "mean": 0.5*(perch_zs+cnn)}
cand_arrs = list(cand.values()); cand_nm = list(cand.keys())
auc_stack = np.full((len(cand_arrs), C), -np.inf)
for k, M in enumerate(cand_arrs):
    a, _ = vec_auc_per_class(y[sel], M[sel])
    auc_stack[k] = np.where(np.isnan(a), -np.inf, a)
manual_best = np.argmax(auc_stack, axis=0)
written = chosen_zc[0]  # fold 0 row of chosen
mismatch = sum(1 for c in range(C) if cand_nm[manual_best[c]] != written[c]
               and np.isfinite(auc_stack[:, c].max()))
print(f"fold0 selection-index mismatches (manual argmax vs written) = {mismatch} (should be 0)")
# 3c. Is the negative headroom driven by inner-vs-outer AUC instability of ZS?
# regret = how often the inner-fold winner is NOT the outer-fold winner.
ho_zs_auc, _ = vec_auc_per_class(y, perch_zs)
ho_cnn_auc, _ = vec_auc_per_class(y, cnn)
print(f"global per-class: ZS>CNN in {int((ho_zs_auc>ho_cnn_auc).sum())} classes, "
      f"CNN>ZS in {int((ho_cnn_auc>ho_zs_auc).sum())} classes")

# ============ CHECK 4: null center ============
print("\n=== CHECK 4: null distribution center ===")
def null_dist(anchor, sidecars, best_single, n_rep=8, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_rep):
        pm = {"anchor": anchor}
        for i, sc in enumerate(sidecars):
            pm[f"sc{i}"] = permute_within_author(sc, authors, rng)
        orc, _, _ = heldout_oracle_nested(pm, y, folds)
        (a_best, a_orc), nc = vec_macro_auc_common(y, [best_single, orc])
        if nc and np.isfinite(a_orc) and np.isfinite(a_best):
            vals.append(a_orc - a_best)
    return np.array(vals)

for label, anc, scs, bs in [
    ("LP+CNN", perch_lp, [cnn], cnn),
    ("ZS+CNN", perch_zs, [cnn], cnn),
]:
    nd = null_dist(anc, scs, bs, n_rep=8, seed=0)
    print(f"{label}: null mean={nd.mean():+.5f} median={np.median(nd):+.5f} "
          f"p5={np.percentile(nd,5):+.5f} p95={np.percentile(nd,95):+.5f} n={len(nd)}")
print("\n[done]")
