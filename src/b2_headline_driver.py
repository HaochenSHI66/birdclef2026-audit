"""B2' HEADLINE driver — oracle-headroom audit over the REAL BirdCLEF++ OOFs.

Runs the full anchor x combo matrix required by exp-plan-v4 B2':
  anchors: Perch linear-probe (A2, HEADLINE) and Perch zero-shot (A1)
  combos per anchor: {anchor+CNN, anchor+PANN, anchor+CNN+PANN}

For each cell, all scored with macro ROC-AUC (skip-zero-positive) on a COMMON
evaluable-class mask shared across every compared model:

  1. per-model macro-AUC (anchor, each sidecar)
  2. ORACLE TRIPLET:
       (a) apparent in-sample oracle (selection-biased, diagnostic)
       (b) held-out nested deployable per-class selector
       (c) optimism gap = (a) - (b)
     primary endpoint = held_out_oracle - best_single_model  (>= 0 by construction
     of the held-out selector candidate set including each base model).
  3. calibrated UNCLAMPED nested fusion vs best single
  4. author-clustered bootstrap 95% CI + MDE on held-out headroom dAUC
     (delta = held_out_oracle - best_single_model, common mask per replicate)
  5. NULL-HEADROOM permutation control: replace each sidecar with a label-preserving
     within-author permutation of itself, rerun held-out oracle + nested selector,
     => null distribution of fake headroom from selection+calibration noise. Compare
     observed headroom to the null. Also a "duplicate-anchor" fake-2nd-model control.
  6. complementarity (anchor vs each sidecar): Spearman/Pearson/Kuncheva-Q/double-fault

Reuses the locked/fixed primitives from b2_oracle_headroom.py.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b2_oracle_headroom import (
    common_eval_mask, per_class_auc_on_mask, macro_auc_on_mask,
    calibrated_unclamped_fusion, complementarity,
)
from sklearn.metrics import roc_auc_score


# ---------- N-model generalised oracles ----------
def _candidate_dict(models):
    """models: ordered dict name->array. Candidates = each model + the unweighted mean."""
    cands = dict(models)
    arrs = list(models.values())
    cands["mean"] = np.mean(arrs, axis=0)
    return cands


def heldout_oracle_nested(models, y, folds):
    """HELD-OUT nested deployable per-class selector over N base models + their mean.
    For each outer fold f, pick per-class best candidate by AUC on folds != f, apply
    to fold f only. Returns (oracle[N,C], chosen[F,C], cand_names)."""
    names = list(models.keys())
    any_arr = next(iter(models.values()))
    N, C = any_arr.shape
    cands = _candidate_dict(models)
    cand_names = list(cands.keys())
    cand_arrs = list(cands.values())
    oracle = np.full((N, C), np.nan, dtype=np.float64)
    uf = sorted(np.unique(folds))
    chosen = np.full((len(uf), C), "none", dtype=object)
    for fi, f in enumerate(uf):
        sel = folds != f
        eva = folds == f
        ysel = y[sel]
        # (n_cand, C) AUC matrix on the selection set, vectorized over classes.
        auc_mat = np.full((len(cand_arrs), C), -np.inf)
        for k, M in enumerate(cand_arrs):
            a, _ = vec_auc_per_class(ysel, M[sel])
            auc_mat[k] = np.where(np.isnan(a), -np.inf, a)
        best_k = np.argmax(auc_mat, axis=0)              # per class
        none_mask = ~np.isfinite(auc_mat.max(axis=0))    # no candidate scorable
        best_k[none_mask] = 0                            # fallback = first base model
        for c in range(C):
            k = best_k[c]
            chosen[fi, c] = cand_names[k]
            oracle[eva, c] = cand_arrs[k][eva, c]
    return oracle, chosen, cand_names


def insample_oracle_nested(models, y):
    """Apparent in-sample oracle (selection-biased, diagnostic) over N models + mean."""
    names = list(models.keys())
    any_arr = next(iter(models.values()))
    N, C = any_arr.shape
    cands = _candidate_dict(models)
    oracle = np.full((N, C), np.nan, dtype=np.float64)
    chosen = np.array(["none"] * C, dtype=object)
    for c in range(C):
        if y[:, c].sum() <= 0 or len(np.unique(y[:, c])) < 2:
            continue
        best_auc, best_name = -np.inf, None
        for nm, M in cands.items():
            col = M[:, c]
            if np.isnan(col).any():
                continue
            try:
                a = roc_auc_score(y[:, c], col)
            except Exception:
                continue
            if a > best_auc:
                best_auc, best_name = a, nm
        if best_name is not None:
            oracle[:, c] = cands[best_name][:, c]
            chosen[c] = best_name
    return oracle, chosen


# ---------- multi-model nested calibrated fusion ----------
def calibrated_fusion_multi(models, y, folds):
    """Per-class logistic blend of all N model logits, nested by outer fold."""
    from sklearn.linear_model import LogisticRegression
    names = list(models.keys())
    arrs = [models[n] for n in names]
    N, C = arrs[0].shape
    eps = 1e-6
    def logit(p):
        p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))
    L = [logit(a) for a in arrs]
    fused = np.full((N, C), np.nan, dtype=np.float64)
    uf = sorted(np.unique(folds))
    for f in uf:
        tr = folds != f; outer = folds == f
        for c in range(C):
            cols_tr = [a[tr, c] for a in arrs]
            y_tr = y[tr, c]
            valid = ~np.any([np.isnan(ct) for ct in cols_tr], axis=0)
            if y_tr[valid].sum() < 2 or len(np.unique(y_tr[valid])) < 2:
                continue
            X = np.stack([Lk[tr, c][valid] for Lk in L], 1)
            try:
                clf = LogisticRegression(max_iter=500, C=1.0)
                clf.fit(X, y_tr[valid])
            except Exception:
                continue
            Xo = np.stack([Lk[outer, c] for Lk in L], 1)
            ok = ~np.isnan(Xo).any(1)
            pred = np.full(outer.sum(), np.nan)
            if ok.any():
                pred[ok] = clf.predict_proba(Xo[ok])[:, 1]
            fused[outer, c] = pred
    return fused


# ---------- vectorized per-class AUC (Mann-Whitney rank-sum, all classes at once) ----------
def vec_auc_per_class(y, p):
    """Per-class ROC-AUC for ALL classes at once via the rank-sum identity.
    y,p: (n, C). Returns (auc[C], valid[C]); valid=False where a class lacks both
    labels or has any NaN pred on these rows (matches sklearn's defined cases).
    AUC = (sum_ranks(pos) - n_pos*(n_pos+1)/2) / (n_pos*n_neg), ranks are average
    ranks of p within each class (ties -> 0.5, identical to sklearn)."""
    n, C = y.shape
    npos = y.sum(0)
    nneg = n - npos
    finite = ~np.isnan(p).any(0)
    valid = (npos > 0) & (nneg > 0) & finite
    auc = np.full(C, np.nan)
    cols = np.where(valid)[0]
    if len(cols) == 0:
        return auc, valid
    pc = np.ascontiguousarray(p[:, cols], dtype=np.float32)   # f32 sort = ~2x faster
    M = pc.shape[1]
    order = np.argsort(pc, axis=0, kind="quicksort")          # ties handled explicitly below
    ps = np.take_along_axis(pc, order, axis=0)                # sorted values per col
    pos = np.arange(n)[:, None]                               # (n,1) 0-based positions
    # tie-corrected average rank for each SORTED row, vectorized over columns.
    # is_new[i]=True at the start of a tie run. cumulative-max of (start position)
    # gives the FIRST index of the current run; reverse cumulative-min of (1+last
    # position) gives the LAST index. avg position = (first+last)/2, +1 -> 1-based rank.
    is_new = np.ones((n, M), dtype=bool)
    is_new[1:, :] = ps[1:, :] != ps[:-1, :]
    first_pos = np.where(is_new, pos, -1)
    first_pos = np.maximum.accumulate(first_pos, axis=0)      # first index of run
    is_end = np.ones((n, M), dtype=bool)
    is_end[:-1, :] = ps[1:, :] != ps[:-1, :]                  # True at last row of run
    last_pos = np.where(is_end, pos, n)
    last_pos = np.minimum.accumulate(last_pos[::-1, :], axis=0)[::-1, :]  # last idx of run
    avg_rank_sorted = (first_pos + last_pos) / 2.0 + 1.0      # 1-based average rank
    ranks = np.empty((n, M), dtype=np.float64)
    np.put_along_axis(ranks, order, avg_rank_sorted, axis=0)
    yc = y[:, cols]
    sum_pos_rank = (ranks * yc).sum(0)
    npc = npos[cols]; nnc = nneg[cols]
    auc[cols] = (sum_pos_rank - npc * (npc + 1) / 2.0) / (npc * nnc)
    return auc, valid


def vec_macro_auc_common(y, preds):
    """Macro AUC for each pred matrix in preds on the COMMON mask (class evaluable
    in EVERY pred). Returns list of macro AUCs aligned to preds (NaN-safe)."""
    aucs, valids = [], []
    for p in preds:
        a, v = vec_auc_per_class(y, p)
        aucs.append(a); valids.append(v)
    common = np.all(valids, axis=0)
    if not common.any():
        return [np.nan] * len(preds), 0
    out = [float(np.nanmean(a[common])) for a in aucs]
    return out, int(common.sum())


# ---------- bootstrap: delta vs an arbitrary reference matrix ----------
def clustered_bootstrap_pair(ref, test, y, authors, n_boot=1000, seed=0):
    """Bootstrap dAUC = macro(test) - macro(ref), resample AUTHORS, common mask per
    replicate. Uses the vectorized rank-sum AUC for speed."""
    rng = np.random.default_rng(seed)
    # encode authors as integer codes; build a ragged CSR-style index so a resample of
    # authors maps to rows with a single vectorized gather (no per-author Python loop).
    codes, _ = pd.factorize(authors)
    A = codes.max() + 1
    order = np.argsort(codes, kind="mergesort")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(A), side="left")
    ends = np.searchsorted(sorted_codes, np.arange(A), side="right")
    counts = ends - starts
    deltas = []
    for _ in range(n_boot):
        samp = rng.integers(0, A, size=A)                 # resample author codes
        total = int(counts[samp].sum())
        rows = np.empty(total, dtype=np.int64)
        off = 0
        # gather the contiguous blocks for the sampled authors
        for a in samp:
            s, e = starts[a], ends[a]
            rows[off:off + (e - s)] = order[s:e]
            off += e - s
        yb = y[rows]
        (auc_ref, auc_test), nc = vec_macro_auc_common(yb, [ref[rows], test[rows]])
        if nc == 0 or not np.isfinite(auc_ref) or not np.isfinite(auc_test):
            continue
        deltas.append(auc_test - auc_ref)
    deltas = np.array(deltas)
    if len(deltas) == 0:
        return {"delta_mean": np.nan, "ci95": [np.nan, np.nan], "se": np.nan,
                "MDE_approx": np.nan, "n_boot": 0, "n_authors": int(A)}
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    se = float(deltas.std(ddof=1))
    return {"delta_mean": float(deltas.mean()), "ci95": [float(lo), float(hi)],
            "se": se, "MDE_approx": float(1.96 * se),
            "n_boot": int(len(deltas)), "n_authors": int(A)}


def clustered_bootstrap_multi(ref, tests, y, authors, n_boot=1000, seed=0):
    """Resample AUTHORS once per replicate; compute dAUC = macro(test) - macro(ref)
    for EACH test matrix in `tests` (dict name->array). Common mask per replicate is
    shared across {ref} + all tests so every delta in a replicate is scored on the
    identical class set. Returns {name: {delta_mean, ci95, se, MDE_approx, n_boot, n_authors}}."""
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(authors)
    A = codes.max() + 1
    order = np.argsort(codes, kind="mergesort")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(A), side="left")
    ends = np.searchsorted(sorted_codes, np.arange(A), side="right")
    counts = ends - starts
    names = list(tests.keys())
    deltas = {nm: [] for nm in names}
    for _ in range(n_boot):
        samp = rng.integers(0, A, size=A)
        total = int(counts[samp].sum())
        rows = np.empty(total, dtype=np.int64)
        off = 0
        for a in samp:
            s, e = starts[a], ends[a]
            rows[off:off + (e - s)] = order[s:e]
            off += e - s
        yb = y[rows]
        mats = [ref[rows]] + [tests[nm][rows] for nm in names]
        aucs, nc = vec_macro_auc_common(yb, mats)
        if nc == 0 or not np.isfinite(aucs[0]):
            continue
        for k, nm in enumerate(names):
            if np.isfinite(aucs[k + 1]):
                deltas[nm].append(aucs[k + 1] - aucs[0])
    out = {}
    for nm in names:
        d = np.array(deltas[nm])
        if len(d) == 0:
            out[nm] = {"delta_mean": np.nan, "ci95": [np.nan, np.nan], "se": np.nan,
                       "MDE_approx": np.nan, "n_boot": 0, "n_authors": int(A)}
            continue
        lo, hi = np.percentile(d, [2.5, 97.5])
        se = float(d.std(ddof=1))
        out[nm] = {"delta_mean": float(d.mean()), "ci95": [float(lo), float(hi)],
                   "se": se, "MDE_approx": float(1.96 * se),
                   "n_boot": int(len(d)), "n_authors": int(A)}
    return out


# ---------- null-headroom: within-author permutation of sidecars ----------
def permute_within_author(M, authors, rng):
    """Row-permute M within each author group (preserves the marginal score
    distribution per author, destroys row-label alignment)."""
    out = np.empty_like(M)
    for a in np.unique(authors):
        idx = np.where(authors == a)[0]
        perm = rng.permutation(idx)
        out[idx] = M[perm]
    return out


def null_headroom(anchor, sidecars, y, folds, authors, best_single, n_rep=50, seed=0):
    """For each replicate: replace EACH sidecar with a within-author permutation of
    itself, recompute the held-out oracle over {anchor, permuted-sidecars}, and the
    headroom = macro(oracle) - macro(best_single_observed). Distribution of fake
    headroom from selection noise. Common mask vs the observed best_single matrix."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_rep):
        perm_models = {"anchor": anchor}
        for i, sc in enumerate(sidecars):
            perm_models[f"sc{i}"] = permute_within_author(sc, authors, rng)
        orc, _, _ = heldout_oracle_nested(perm_models, y, folds)
        (a_best, a_orc), nc = vec_macro_auc_common(y, [best_single, orc])
        if nc == 0 or not np.isfinite(a_orc) or not np.isfinite(a_best):
            continue
        null.append(a_orc - a_best)
    null = np.array(null)
    if len(null) == 0:
        return {"null_mean": np.nan, "null_p95": np.nan, "null_max": np.nan, "n_rep": 0}
    return {"null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "null_max": float(null.max()), "null_std": float(null.std(ddof=1)),
            "n_rep": int(len(null))}


def dup_anchor_control(anchor, y, folds, authors):
    """Fake-2nd-model control: duplicate the anchor as the only sidecar. Any held-out
    'headroom' here is pure selection+mean-averaging noise (the mean of anchor with
    itself == anchor, so headroom should be ~0)."""
    models = {"anchor": anchor, "dup": anchor.copy()}
    orc, _, _ = heldout_oracle_nested(models, y, folds)
    mask = common_eval_mask(y, [anchor, orc])
    return macro_auc_on_mask(y, orc, mask) - macro_auc_on_mask(y, anchor, mask)


# ---------- run one anchor x combo cell ----------
def run_cell(anchor_name, anchor, sidecar_specs, y, folds, authors, n_boot, n_null, seed):
    """sidecar_specs: list of (name, array). Returns a result dict for this cell."""
    models = {anchor_name: anchor}
    for nm, arr in sidecar_specs:
        models[nm] = arr

    # per-model macro-AUC on the common mask across ALL models in this cell
    all_arrs = list(models.values())
    base_mask = common_eval_mask(y, all_arrs)
    per_model = {nm: macro_auc_on_mask(y, arr, base_mask) for nm, arr in models.items()}
    best_single_name = max(per_model, key=per_model.get)
    best_single = models[best_single_name]

    # oracles + fusion
    ho_orc, ho_chosen, cand_names = heldout_oracle_nested(models, y, folds)
    is_orc, is_chosen = insample_oracle_nested(models, y)
    fused = calibrated_fusion_multi(models, y, folds)

    # headline common mask across everything compared
    mask = common_eval_mask(y, all_arrs + [ho_orc, is_orc, fused])
    res = {
        "anchor": anchor_name,
        "sidecars": [nm for nm, _ in sidecar_specs],
        "candidate_set": cand_names,
        "n_common_eval_classes": int(mask.sum()),
        "per_model_macro_auc": {nm: macro_auc_on_mask(y, arr, mask) for nm, arr in models.items()},
        "best_single_model": best_single_name,
        "best_single_macro_auc": macro_auc_on_mask(y, best_single, mask),
        "apparent_oracle_macro_auc": macro_auc_on_mask(y, is_orc, mask),
        "heldout_oracle_macro_auc": macro_auc_on_mask(y, ho_orc, mask),
        "calibrated_unclamped_fusion_macro_auc": macro_auc_on_mask(y, fused, mask),
        "heldout_choice_counts": {k: int((ho_chosen == k).sum()) for k in cand_names + ["none"]},
        "apparent_choice_counts": {k: int((is_chosen == k).sum()) for k in cand_names + ["none"]},
    }
    res["optimism_gap"] = res["apparent_oracle_macro_auc"] - res["heldout_oracle_macro_auc"]
    res["heldout_minus_best_single_point"] = res["heldout_oracle_macro_auc"] - res["best_single_macro_auc"]
    res["fusion_minus_best_single_point"] = res["calibrated_unclamped_fusion_macro_auc"] - res["best_single_macro_auc"]

    # PRIMARY endpoint: held-out oracle vs best single (full n_boot, 2-matrix/rep).
    res["headroom_heldout_vs_best"] = clustered_bootstrap_pair(
        best_single, ho_orc, y, authors, n_boot=n_boot, seed=seed)
    # SECONDARY: fusion vs best single (fewer reps; cheaper, still author-clustered).
    res["fusion_vs_best"] = clustered_bootstrap_pair(
        best_single, fused, y, authors, n_boot=max(100, n_boot // 2), seed=seed + 1)

    # null-headroom + dup-anchor controls (vs observed best single)
    sidecar_arrs = [arr for _, arr in sidecar_specs]
    res["null_headroom"] = null_headroom(
        anchor, sidecar_arrs, y, folds, authors, best_single, n_rep=n_null, seed=seed)
    res["dup_anchor_control_headroom"] = float(dup_anchor_control(anchor, y, folds, authors))

    # complementarity: anchor vs each sidecar
    res["complementarity"] = {nm: complementarity(anchor, arr, y) for nm, arr in sidecar_specs}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_null", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = args.oof_dir
    def L(name):
        return np.load(os.path.join(d, name)).astype(np.float64)
    perch_lp = L("perch_anchor_oof.npy")     # A2 linear-probe (HEADLINE)
    perch_zs = L("perch_zeroshot_oof.npy")   # A1 zero-shot
    cnn = L("cnn_ensemble_oof.npy")
    pann = L("pann_ensemble_oof.npy")
    y = L("oof_targets.npy")
    meta = pd.read_csv(os.path.join(d, "oof_meta.csv"))
    authors = meta["author"].values
    folds = meta["fold"].values
    for arr in (perch_lp, perch_zs, cnn, pann):
        assert arr.shape == y.shape, "shape mismatch"

    # rows scored by all four base models (drop all-NaN-in-any rows)
    keep = ~(np.isnan(perch_lp).all(1) | np.isnan(perch_zs).all(1)
             | np.isnan(cnn).all(1) | np.isnan(pann).all(1))
    if not keep.all():
        print(f"[run] restricting to {int(keep.sum())}/{len(keep)} rows scored by all base models")
        perch_lp, perch_zs, cnn, pann, y = perch_lp[keep], perch_zs[keep], cnn[keep], pann[keep], y[keep]
        authors, folds = authors[keep], folds[keep]

    anchors = {"perch_linear_probe": perch_lp, "perch_zeroshot": perch_zs}
    combos = [
        ("+CNN", [("cnn", cnn)]),
        ("+PANN", [("pann", pann)]),
        ("+CNN+PANN", [("cnn", cnn), ("pann", pann)]),
    ]
    results = {"rows": int(len(y)), "n_authors": int(len(set(authors))),
               "fold_sizes": {int(f): int((folds == f).sum()) for f in sorted(set(folds))},
               "cells": {}}
    for an, anchor in anchors.items():
        for cname, specs in combos:
            key = f"{an}{cname}"
            print(f"[cell] {key} ...", flush=True)
            results["cells"][key] = run_cell(
                an, anchor, specs, y, folds, authors, args.n_boot, args.n_null, args.seed)
            r = results["cells"][key]
            print(f"   best_single={r['best_single_model']} {r['best_single_macro_auc']:.4f} "
                  f"| heldout_oracle={r['heldout_oracle_macro_auc']:.4f} "
                  f"(d={r['heldout_minus_best_single_point']:+.4f}) "
                  f"| fusion={r['calibrated_unclamped_fusion_macro_auc']:.4f} "
                  f"| apparent={r['apparent_oracle_macro_auc']:.4f} gap={r['optimism_gap']:.4f}",
                  flush=True)

    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"[done] wrote {args.out_json}")
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
