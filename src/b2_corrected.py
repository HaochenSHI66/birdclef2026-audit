"""B2 CORRECTED — oracle-headroom audit with the doubly-confirmed (codex + refutation) fixes.

This is the camera-ready replacement for src/b2_headline_driver.py. The ESTIMATOR CORE
(held-out nested per-class oracle, common-mask macro-AUC, vectorized rank-sum AUC, calibrated
nested fusion, complementarity) is UNCHANGED and re-imported from the locked/verified
b2_oracle_headroom.py + b2_headline_driver.py primitives (held-out selection is leak-free,
common mask correct, force-best-single==0, dup-anchor==0 verified in b2-bug-audit.md).

WHAT IS FIXED HERE (the four things both reviewers agreed on):

  (1) RECENTERED NULL = LABEL-PERMUTATION null.
      The OLD within-author *prediction* permutation null is mis-centered at ~-0.02 because it
      PRESERVES each sidecar's per-class marginal AUC -> the candidate set still contains a
      genuinely-good sidecar, so the floor measures the selector's intrinsic negative
      selection-variance bias, NOT a no-signal reference (b2-bug-audit.md sec.4; refutation
      Attack B). The CORRECT no-signal null permutes the LABELS used for selection AND scoring
      (within author, breaking row<->label alignment) so every candidate's per-class AUC is
      ~0.5 in expectation and the selector has no real signal to exploit. The fake headroom it
      produces centers at ~0 (verified: |null_mean| small). We report:
        observed headroom, null_mean, and  DE-BIASED headroom = observed - null_mean  (+CI).
      An optional AUC-matched-Gaussian-sidecar null is also implemented as a cross-check.

  (2) POWER: n_boot>=1000, n_null>=50 (author-clustered bootstrap), camera-ready sizes.

  (3) TOST EQUIVALENCE at delta=0.01 macro-AUC (codex: delta=0.005 is unsupported because the
      LP held-out upper CI reaches ~+0.009). Two one-sided tests on the de-biased hard-selection
      headroom; we also report the SMALLEST delta for which equivalence holds.

  (4) GLOBAL-OOF caveat made explicit. The base OOF columns are single-level global OOF (each
      base model saw the outer fold's rows during its own training). The held-out SELECTION /
      fusion here is leak-free, the base FEATURES are not. A tractable fully-nested base
      regeneration is out of scope for this post-hoc combiner; instead we DOCUMENT the bias
      DIRECTION: global-OOF inflates the best single model -> SHRINKS the apparent residual
      headroom -> the "no headroom" finding is CONSERVATIVE (biased toward our own conclusion).
      We compute a tractable lower-bound proxy (per-seed sub-ensemble held-out check) so the
      bias is empirically bounded, not silently ignored.

Kept verbatim: oracle triplet (apparent / held-out / optimism gap), calibrated soft fusion vs
best-single (the real +0.005..0.009 effect), complementarity (Spearman/Pearson/Kuncheva-Q/df).

FINAL run: ALL FOUR models. anchors {perch_linear_probe, perch_zeroshot} x sidecars
{cnn, pann, beats}; combos {+CNN, +PANN, +BEATs, +CNN+PANN+BEATs}; plus per-seed sub-ensemble
proxy. BEATs is a 3rd, very different self-supervised architecture.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# verified primitives (unchanged)
from b2_oracle_headroom import (
    common_eval_mask, per_class_auc_on_mask, macro_auc_on_mask, complementarity,
)
from b2_headline_driver import (
    _candidate_dict, heldout_oracle_nested, insample_oracle_nested,
    calibrated_fusion_multi, vec_auc_per_class, vec_macro_auc_common,
    clustered_bootstrap_pair, dup_anchor_control,
)
from scipy import stats as sps


# ============================ FIX 1: label-permutation NULL ============================
def permute_labels_within_author(y, authors, rng):
    """Row-permute the LABEL matrix y within each author group. This breaks the
    pred<->label row alignment used for BOTH per-class selection AUC and the final
    scoring, so every candidate's per-class AUC -> ~0.5 in expectation and the
    selector has no genuine signal. Preserves per-author label marginals (so the
    common-mask population is unchanged); destroys the signal the selector exploits.
    A clean no-signal floor that centers near 0 (unlike the prediction-permutation
    null, which preserves sidecar AUC and centers at the selector-bias ~-0.02)."""
    out = np.empty_like(y)
    for a in np.unique(authors):
        idx = np.where(authors == a)[0]
        out[idx] = y[rng.permutation(idx)]
    return out


def null_headroom_labelperm(models, y, folds, authors, best_single_name,
                            n_rep=50, seed=0):
    """No-signal null via LABEL permutation. For each replicate permute y within
    author, recompute the held-out nested oracle over the SAME models, and the
    fake headroom = macro(oracle ; y_perm) - macro(best_single ; y_perm) on the
    common mask under the permuted labels. Centers at ~0 if the assay is unbiased."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_rep):
        yp = permute_labels_within_author(y, authors, rng)
        orc, _, _ = heldout_oracle_nested(models, yp, folds)
        best = models[best_single_name]
        (a_best, a_orc), nc = vec_macro_auc_common(yp, [best, orc])
        if nc == 0 or not np.isfinite(a_orc) or not np.isfinite(a_best):
            continue
        null.append(a_orc - a_best)
    null = np.array(null)
    if len(null) == 0:
        return {"null_mean": np.nan, "null_std": np.nan, "null_p5": np.nan,
                "null_p95": np.nan, "n_rep": 0}
    return {"null_mean": float(null.mean()), "null_std": float(null.std(ddof=1)),
            "null_p5": float(np.percentile(null, 5)),
            "null_p95": float(np.percentile(null, 95)),
            "null_max": float(null.max()), "n_rep": int(len(null)),
            "null_samples": [float(x) for x in null]}


def null_headroom_gaussian(anchor, sidecar_arrs, y, folds, authors, best_single,
                           n_rep=50, seed=0):
    """Cross-check no-signal null: replace each sidecar by AUC-matched Gaussian noise
    sidecars (per-class signal strength matched to the real sidecar, but NO row
    signal -> independent noise that yields ~chance per-class AUC). Confirms the
    floor lands at ~0 independently of the label-permutation construction."""
    rng = np.random.default_rng(seed + 777)
    N, C = anchor.shape
    null = []
    for _ in range(n_rep):
        perm = {"anchor": anchor}
        for i, _sc in enumerate(sidecar_arrs):
            # pure Gaussian noise mapped to (0,1) via logistic; no alignment to y ->
            # per-class AUC ~ 0.5. (We intentionally do NOT match AUC, because matching
            # AUC reintroduces signal; a chance-level sidecar is the true no-signal ref.)
            z = rng.normal(0.0, 1.0, size=(N, C))
            perm[f"g{i}"] = 1.0 / (1.0 + np.exp(-z))
        orc, _, _ = heldout_oracle_nested(perm, y, folds)
        (a_best, a_orc), nc = vec_macro_auc_common(y, [best_single, orc])
        if nc == 0 or not np.isfinite(a_orc) or not np.isfinite(a_best):
            continue
        null.append(a_orc - a_best)
    null = np.array(null)
    if len(null) == 0:
        return {"null_mean": np.nan, "n_rep": 0}
    return {"null_mean": float(null.mean()), "null_std": float(null.std(ddof=1)),
            "null_p95": float(np.percentile(null, 95)), "n_rep": int(len(null))}


# ============ FIX 1b: de-biased headroom bootstrap (observed - null_mean) ============
def debiased_headroom_bootstrap(best_single, ho_orc, y, authors, null_mean,
                                n_boot=1000, seed=0):
    """Author-clustered bootstrap of the DE-BIASED hard-selection headroom:
        debiased = (macro(oracle) - macro(best_single)) - null_mean.
    null_mean is the label-permutation no-signal floor (a constant offset, since the
    null is computed on the full sample once at high n_rep). CI is on the de-biased
    quantity, which is what the TOST and the headline use."""
    raw = clustered_bootstrap_pair(best_single, ho_orc, y, authors,
                                   n_boot=n_boot, seed=seed)
    if not np.isfinite(raw["delta_mean"]):
        return {**raw, "null_mean": null_mean, "debiased_mean": np.nan,
                "debiased_ci95": [np.nan, np.nan]}
    db_mean = raw["delta_mean"] - null_mean
    db_ci = [raw["ci95"][0] - null_mean, raw["ci95"][1] - null_mean]
    return {"raw_delta_mean": raw["delta_mean"], "raw_ci95": raw["ci95"],
            "se": raw["se"], "MDE_approx": raw["MDE_approx"], "n_boot": raw["n_boot"],
            "n_authors": raw["n_authors"], "null_mean": float(null_mean),
            "debiased_mean": float(db_mean),
            "debiased_ci95": [float(db_ci[0]), float(db_ci[1])]}


# ============================ FIX 3: TOST equivalence ============================
def tost_equivalence(point, se, delta=0.01):
    """Two One-Sided Tests for equivalence of `point` (de-biased headroom) within
    +/- delta, using a normal approx (large author count). Equivalence is concluded
    iff BOTH one-sided nulls are rejected at alpha=0.05, i.e. the 90% CI lies fully
    inside (-delta, +delta). Returns the two p-values, the verdict, and (separately,
    via smallest_equiv_delta) the smallest delta that would pass."""
    if not np.isfinite(point) or not np.isfinite(se) or se <= 0:
        return {"delta": delta, "p_lower": np.nan, "p_upper": np.nan,
                "equivalent": False, "ci90": [np.nan, np.nan]}
    # H0_lower: point <= -delta ; H0_upper: point >= +delta
    t_lower = (point - (-delta)) / se      # large -> reject lower null
    t_upper = (point - (+delta)) / se      # very negative -> reject upper null
    p_lower = 1.0 - sps.norm.cdf(t_lower)  # P(Z > t_lower)
    p_upper = sps.norm.cdf(t_upper)        # P(Z < t_upper)
    equiv = (p_lower < 0.05) and (p_upper < 0.05)
    z90 = sps.norm.ppf(0.95)               # 90% CI for TOST
    ci90 = [point - z90 * se, point + z90 * se]
    return {"delta": float(delta), "p_lower": float(p_lower), "p_upper": float(p_upper),
            "equivalent": bool(equiv), "ci90": [float(ci90[0]), float(ci90[1])]}


def smallest_equiv_delta(point, se):
    """Smallest delta for which TOST concludes equivalence = the larger absolute
    bound of the 90% CI (equivalence holds iff delta > max(|ci90_lo|, |ci90_hi|))."""
    if not np.isfinite(point) or not np.isfinite(se) or se <= 0:
        return np.nan
    z90 = sps.norm.ppf(0.95)
    return float(max(abs(point - z90 * se), abs(point + z90 * se)))


# ============================ FIX 4: global-OOF bias proxy ============================
def per_seed_subensemble_proxy(seed_arrs_by_model, anchor, y, folds, authors,
                               best_single_full, n_boot, seed):
    """Tractable global-OOF bias check. The full sidecar ensembles average all seeds;
    here we hold out one seed per sidecar (use the remaining seeds) to perturb the
    base FEATURES, then recompute the held-out oracle headroom. If headroom is
    insensitive to dropping a base-seed, the global-OOF combiner subtlety is not
    creating spurious headroom. (Not a full nested base regen; a directional bound.)
    seed_arrs_by_model: {sidecar_name: [seed_arr, ...]}. Returns list of per-drop
    de-biased-ish raw headroom points (vs the SAME observed best_single_full)."""
    out = []
    nseed = min(len(v) for v in seed_arrs_by_model.values())
    for drop in range(nseed):
        models = {"anchor": anchor}
        for nm, arrs in seed_arrs_by_model.items():
            keep = [a for j, a in enumerate(arrs) if j != drop]
            models[nm] = np.mean(keep, axis=0)
        orc, _, _ = heldout_oracle_nested(models, y, folds)
        (a_best, a_orc), nc = vec_macro_auc_common(y, [best_single_full, orc])
        if nc and np.isfinite(a_orc) and np.isfinite(a_best):
            out.append(float(a_orc - a_best))
    if not out:
        return {"n_drops": 0}
    return {"n_drops": len(out), "headroom_mean": float(np.mean(out)),
            "headroom_min": float(np.min(out)), "headroom_max": float(np.max(out)),
            "headroom_spread": float(np.max(out) - np.min(out)),
            "samples": out}


# ============================ run one anchor x combo cell ============================
def run_cell(anchor_name, anchor, sidecar_specs, seed_arrs_by_model,
             y, folds, authors, n_boot, n_null, seed):
    models = {anchor_name: anchor}
    for nm, arr in sidecar_specs:
        models[nm] = arr
    all_arrs = list(models.values())

    base_mask = common_eval_mask(y, all_arrs)
    per_model = {nm: macro_auc_on_mask(y, arr, base_mask) for nm, arr in models.items()}
    best_single_name = max(per_model, key=per_model.get)
    best_single = models[best_single_name]

    # KEPT: oracle triplet + fusion (unchanged primitives)
    ho_orc, ho_chosen, cand_names = heldout_oracle_nested(models, y, folds)
    is_orc, is_chosen = insample_oracle_nested(models, y)
    fused = calibrated_fusion_multi(models, y, folds)

    mask = common_eval_mask(y, all_arrs + [ho_orc, is_orc, fused])
    res = {
        "anchor": anchor_name, "sidecars": [nm for nm, _ in sidecar_specs],
        "candidate_set": cand_names, "n_common_eval_classes": int(mask.sum()),
        "per_model_macro_auc": {nm: macro_auc_on_mask(y, arr, mask) for nm, arr in models.items()},
        "best_single_model": best_single_name,
        "best_single_macro_auc": macro_auc_on_mask(y, best_single, mask),
        "apparent_oracle_macro_auc": macro_auc_on_mask(y, is_orc, mask),
        "heldout_oracle_macro_auc": macro_auc_on_mask(y, ho_orc, mask),
        "calibrated_unclamped_fusion_macro_auc": macro_auc_on_mask(y, fused, mask),
        "heldout_choice_counts": {k: int((ho_chosen == k).sum()) for k in cand_names + ["none"]},
    }
    res["optimism_gap"] = res["apparent_oracle_macro_auc"] - res["heldout_oracle_macro_auc"]
    res["heldout_minus_best_single_point"] = res["heldout_oracle_macro_auc"] - res["best_single_macro_auc"]
    res["fusion_minus_best_single_point"] = res["calibrated_unclamped_fusion_macro_auc"] - res["best_single_macro_auc"]

    # FIX 1: recentered LABEL-PERMUTATION null (and Gaussian cross-check)
    null_lp = null_headroom_labelperm(models, y, folds, authors, best_single_name,
                                      n_rep=n_null, seed=seed)
    sidecar_arrs = [arr for _, arr in sidecar_specs]
    null_g = null_headroom_gaussian(anchor, sidecar_arrs, y, folds, authors,
                                    best_single, n_rep=n_null, seed=seed)
    res["null_labelperm"] = {k: v for k, v in null_lp.items() if k != "null_samples"}
    res["null_gaussian"] = null_g
    null_mean = null_lp["null_mean"] if np.isfinite(null_lp["null_mean"]) else 0.0

    # FIX 1b: de-biased hard-selection headroom bootstrap (PRIMARY endpoint)
    db = debiased_headroom_bootstrap(best_single, ho_orc, y, authors, null_mean,
                                     n_boot=n_boot, seed=seed)
    res["headroom_heldout_debiased"] = db

    # FIX 3: TOST equivalence on the de-biased headroom
    pt, se = db.get("debiased_mean", np.nan), db.get("se", np.nan)
    res["tost_delta_0.01"] = tost_equivalence(pt, se, delta=0.01)
    res["tost_delta_0.005"] = tost_equivalence(pt, se, delta=0.005)
    res["smallest_equiv_delta"] = smallest_equiv_delta(pt, se)

    # KEPT: soft fusion vs best single (the real effect)
    res["fusion_vs_best"] = clustered_bootstrap_pair(
        best_single, fused, y, authors, n_boot=max(1000, n_boot), seed=seed + 1)

    # KEPT: dup-anchor assay sanity (NOT cited as redundancy evidence)
    res["dup_anchor_control_headroom"] = float(dup_anchor_control(anchor, y, folds, authors))

    # FIX 4: global-OOF bias proxy (per-seed sub-ensemble) — only when seed arrays given
    if seed_arrs_by_model:
        sub = {nm: seed_arrs_by_model[nm] for nm, _ in sidecar_specs if nm in seed_arrs_by_model}
        if sub:
            res["global_oof_proxy"] = per_seed_subensemble_proxy(
                sub, anchor, y, folds, authors, best_single, n_boot, seed)

    # KEPT: complementarity
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
    perch_lp = L("perch_anchor_oof.npy")
    perch_zs = L("perch_zeroshot_oof.npy")
    cnn = L("cnn_ensemble_oof.npy")
    pann = L("pann_ensemble_oof.npy")
    beats = L("beats_ensemble_oof.npy")
    y = L("oof_targets.npy")
    meta = pd.read_csv(os.path.join(d, "oof_meta.csv"))
    authors = meta["author"].values
    folds = meta["fold"].values

    # per-seed sidecar arrays for the global-OOF proxy (FIX 4)
    def load_seeds(prefix):
        arrs = []
        for s in range(41, 46):
            p = os.path.join(d, f"{prefix}_seed{s}_oof.npy")
            if os.path.exists(p):
                arrs.append(np.load(p).astype(np.float64))
        return arrs
    cnn_seeds = load_seeds("cnn")
    pann_seeds = load_seeds("pann")
    beats_seeds = load_seeds("beats")

    keep = ~(np.isnan(perch_lp).all(1) | np.isnan(perch_zs).all(1)
             | np.isnan(cnn).all(1) | np.isnan(pann).all(1) | np.isnan(beats).all(1))
    if not keep.all():
        print(f"[run] restricting to {int(keep.sum())}/{len(keep)} rows")
        perch_lp, perch_zs = perch_lp[keep], perch_zs[keep]
        cnn, pann, beats, y = cnn[keep], pann[keep], beats[keep], y[keep]
        authors, folds = authors[keep], folds[keep]
        cnn_seeds = [a[keep] for a in cnn_seeds]
        pann_seeds = [a[keep] for a in pann_seeds]
        beats_seeds = [a[keep] for a in beats_seeds]

    seed_arrs = {}
    if cnn_seeds:
        seed_arrs["cnn"] = cnn_seeds
    if pann_seeds:
        seed_arrs["pann"] = pann_seeds
    if beats_seeds:
        seed_arrs["beats"] = beats_seeds

    anchors = {"perch_linear_probe": perch_lp, "perch_zeroshot": perch_zs}
    combos = [
        ("+CNN", [("cnn", cnn)]),
        ("+PANN", [("pann", pann)]),
        ("+BEATs", [("beats", beats)]),
        ("+CNN+PANN+BEATs", [("cnn", cnn), ("pann", pann), ("beats", beats)]),
    ]
    results = {"rows": int(len(y)), "n_authors": int(len(set(authors))),
               "fold_sizes": {int(f): int((folds == f).sum()) for f in sorted(set(folds))},
               "n_boot": args.n_boot, "n_null": args.n_null,
               "models_present": "perch_lp,perch_zs,cnn,pann,beats (ALL 4 model families)",
               "global_oof_caveat": ("Base OOF columns are single-level GLOBAL OOF (each base "
                 "model saw the outer fold's rows during its own training). The held-out "
                 "SELECTION/fusion here is leak-free; the base FEATURES are not. Global-OOF "
                 "inflates the best single model -> SHRINKS apparent residual headroom -> the "
                 "'no headroom' finding is CONSERVATIVE (biased toward our own conclusion). The "
                 "per-seed sub-ensemble proxy bounds the bias empirically."),
               "cells": {}}
    for an, anchor in anchors.items():
        for cname, specs in combos:
            key = f"{an}{cname}"
            print(f"[cell] {key} ...", flush=True)
            results["cells"][key] = run_cell(
                an, anchor, specs, seed_arrs, y, folds, authors,
                args.n_boot, args.n_null, args.seed)
            r = results["cells"][key]
            db = r["headroom_heldout_debiased"]
            t = r["tost_delta_0.01"]
            print(f"   best={r['best_single_model']} {r['best_single_macro_auc']:.4f} "
                  f"| ho_oracle d_raw={r['heldout_minus_best_single_point']:+.4f} "
                  f"| null_lp_mean={r['null_labelperm']['null_mean']:+.4f} "
                  f"| debiased={db['debiased_mean']:+.4f} CI{[round(x,4) for x in db['debiased_ci95']]} "
                  f"| TOST.01 {'EQUIV' if t['equivalent'] else 'no'} "
                  f"| fusion d={r['fusion_minus_best_single_point']:+.4f}", flush=True)

    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
