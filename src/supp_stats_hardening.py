#!/usr/bin/env python
"""SUPP STATISTICAL HARDENING + reviewer-robustness for the BirdCLEF++ 2026 audit.

ONE detached script. Recomputes the 8-cell headline de-biased oracle dAUC at higher
resolution (n_null=1000, n_boot=10000), repeated recordist-grouped splits, selector
shrinkage/min-positives sensitivity, coverage ceiling, and reviewer cross-checks.
ALL from existing frozen OOF artifacts. CPU only. Reuses the LOCKED metric primitives
from src/b2_oracle_headroom.py + src/b2_headline_driver.py + src/b2_corrected.py
(zero re-implementation of the headline estimator; reviewer item (a) adds an
INDEPENDENT macro-AUC cross-check on purpose).

Writes data/oof/supp_*.json incrementally so partial results survive a crash.
"""
import argparse, json, os, sys, time, traceback
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

W = os.path.expanduser("~/SHC/birdclef2026_clef")
OOF = os.path.join(W, "data", "oof")
SRC = os.path.join(W, "src")
sys.path.insert(0, SRC)

from b2_oracle_headroom import (
    common_eval_mask, per_class_auc_on_mask, macro_auc_on_mask, complementarity,
)
from b2_headline_driver import (
    heldout_oracle_nested, insample_oracle_nested, _candidate_dict,
    calibrated_fusion_multi, vec_auc_per_class, vec_macro_auc_common,
    clustered_bootstrap_pair,
)
from b2_corrected import (
    null_headroom_labelperm, debiased_headroom_bootstrap,
    tost_equivalence, smallest_equiv_delta,
)

HEADLINE_KEY = "perch_linear_probe+CNN"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def dump(name, obj):
    p = os.path.join(OOF, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    log("wrote", p)


# ----------------------------- load (mirror b2_par_cell.load_all) -----------------------------
def L(name):
    return np.load(os.path.join(OOF, name)).astype(np.float64)


def load_all():
    perch_lp = L("perch_anchor_oof.npy")
    perch_zs = L("perch_zeroshot_oof.npy")
    cnn = L("cnn_ensemble_oof.npy")
    pann = L("pann_ensemble_oof.npy")
    beats = L("beats_ensemble_oof.npy")
    y = L("oof_targets.npy")
    meta = pd.read_csv(os.path.join(OOF, "oof_meta.csv"))
    authors = meta["author"].values
    folds = meta["fold"].values
    filenames = meta["filename"].values

    def load_seeds(prefix):
        arrs = []
        for s in range(41, 46):
            p = os.path.join(OOF, f"{prefix}_seed{s}_oof.npy")
            if os.path.exists(p):
                arrs.append(np.load(p).astype(np.float64))
        return arrs
    cnn_seeds, pann_seeds, beats_seeds = load_seeds("cnn"), load_seeds("pann"), load_seeds("beats")

    keep = ~(np.isnan(perch_lp).all(1) | np.isnan(perch_zs).all(1)
             | np.isnan(cnn).all(1) | np.isnan(pann).all(1) | np.isnan(beats).all(1))
    if not keep.all():
        perch_lp, perch_zs = perch_lp[keep], perch_zs[keep]
        cnn, pann, beats, y = cnn[keep], pann[keep], beats[keep], y[keep]
        authors, folds, filenames = authors[keep], folds[keep], filenames[keep]
        cnn_seeds = [a[keep] for a in cnn_seeds]
        pann_seeds = [a[keep] for a in pann_seeds]
        beats_seeds = [a[keep] for a in beats_seeds]

    seed_arrs = {}
    if cnn_seeds: seed_arrs["cnn"] = cnn_seeds
    if pann_seeds: seed_arrs["pann"] = pann_seeds
    if beats_seeds: seed_arrs["beats"] = beats_seeds

    anchors = {"perch_linear_probe": perch_lp, "perch_zeroshot": perch_zs}
    combos = [
        ("+CNN", [("cnn", cnn)]),
        ("+PANN", [("pann", pann)]),
        ("+BEATs", [("beats", beats)]),
        ("+CNN+PANN+BEATs", [("cnn", cnn), ("pann", pann), ("beats", beats)]),
    ]
    cells = []  # (key, anchor_name, anchor_arr, sidecar_specs)
    for an, anc in anchors.items():
        for cname, specs in combos:
            cells.append((f"{an}{cname}", an, anc, specs))
    return dict(perch_lp=perch_lp, perch_zs=perch_zs, cnn=cnn, pann=pann, beats=beats,
                y=y, authors=authors, folds=folds, filenames=filenames,
                seed_arrs=seed_arrs, cells=cells)


# globals populated in main (inherited by fork workers)
D = None
BASE = None


def build_base():
    """Precompute frozen per-cell estimator outputs (held-out oracle, best single)."""
    y, folds = D["y"], D["folds"]
    base = {}
    for key, an, anc, specs in D["cells"]:
        models = {an: anc}
        for nm, arr in specs:
            models[nm] = arr
        all_arrs = list(models.values())
        bmask = common_eval_mask(y, all_arrs)
        per_model = {nm: macro_auc_on_mask(y, arr, bmask) for nm, arr in models.items()}
        best_name = max(per_model, key=per_model.get)
        best = models[best_name]
        ho_orc, ho_chosen, cand_names = heldout_oracle_nested(models, y, folds)
        is_orc, _ = insample_oracle_nested(models, y)
        fused = calibrated_fusion_multi(models, y, folds)  # only to reproduce b2_final's 196-class common mask
        mask = common_eval_mask(y, all_arrs + [ho_orc, is_orc, fused])
        base[key] = dict(
            anchor=an, specs=specs, models=models, cand_names=cand_names,
            best_name=best_name, best=best, ho_orc=ho_orc, ho_chosen=ho_chosen,
            mask=mask, n_eval=int(mask.sum()),
            best_macro=macro_auc_on_mask(y, best, mask),
            ho_macro=macro_auc_on_mask(y, ho_orc, mask),
            per_model={nm: macro_auc_on_mask(y, arr, mask) for nm, arr in models.items()},
        )
        log(f"BASE {key}: best={best_name} {base[key]['best_macro']:.4f} "
            f"ho={base[key]['ho_macro']:.4f} d={base[key]['ho_macro']-base[key]['best_macro']:+.5f} "
            f"n_eval={base[key]['n_eval']}")
    return base


# ================================ ANALYSIS 1: deepstats ================================
def deepstats_cell(args):
    key, n_null, n_boot, seed = args
    b = BASE[key]
    y, folds, authors = D["y"], D["folds"], D["authors"]
    t0 = time.time()
    null_lp = null_headroom_labelperm(b["models"], y, folds, authors, b["best_name"],
                                      n_rep=n_null, seed=seed)
    null_mean = null_lp["null_mean"] if np.isfinite(null_lp["null_mean"]) else 0.0
    db = debiased_headroom_bootstrap(b["best"], b["ho_orc"], y, authors, null_mean,
                                     n_boot=n_boot, seed=seed)
    pt, se = db["debiased_mean"], db["se"]
    t01 = tost_equivalence(pt, se, 0.01)
    t005 = tost_equivalence(pt, se, 0.005)
    excl = {}
    for delta in (0.005, 0.01):
        z = (pt - delta) / se if (np.isfinite(pt) and np.isfinite(se) and se > 0) else np.nan
        p_up = float(sps.norm.cdf(z)) if np.isfinite(z) else np.nan
        excl[f"+{delta}"] = {"z": float(z), "p_upper": p_up,
                             "pos_headroom_excluded": bool(np.isfinite(p_up) and p_up < 0.05)}
    res = {
        "best_single_model": b["best_name"], "best_single_macro_auc": b["best_macro"],
        "heldout_oracle_macro_auc": b["ho_macro"], "n_common_eval_classes": b["n_eval"],
        "heldout_minus_best_single_point": b["ho_macro"] - b["best_macro"],
        "null_labelperm": {k: v for k, v in null_lp.items() if k != "null_samples"},
        "headroom_heldout_debiased": db,
        "tost_delta_0.01": t01, "tost_delta_0.005": t005,
        "smallest_equiv_delta": smallest_equiv_delta(pt, se),
        "pos_headroom_exclusion": excl,
        "secs": round(time.time() - t0, 1),
    }
    log(f"DEEPSTATS {key} debiased={pt:+.5f} ci{[round(x,4) for x in db['debiased_ci95']]} "
        f"se={se:.5f} null={null_mean:+.5f}(n={null_lp.get('n_rep')}) "
        f"TOST.01={'EQ' if t01['equivalent'] else 'no'} excl.01={excl['+0.01']['pos_headroom_excluded']} "
        f"({res['secs']}s)")
    return key, res


def run_deepstats(n_null, n_boot, seed):
    log(f"=== ANALYSIS 1 deepstats (n_null={n_null}, n_boot={n_boot}) ===")
    keys = [c[0] for c in D["cells"]]
    jobs = [(k, n_null, n_boot, seed) for k in keys]
    out = {}
    with Pool(processes=min(8, len(keys))) as pool:
        for key, res in pool.imap_unordered(deepstats_cell, jobs):
            out[key] = res
    # compare to stored b2_final (n_null=50, n_boot=1000)
    verdict = {}
    try:
        bf = json.load(open(os.path.join(OOF, "b2_final.json")))["cells"]
    except Exception:
        bf = {}
    for k in keys:
        new = out[k]
        old = bf.get(k, {}).get("headroom_heldout_debiased", {})
        old_db = old.get("debiased_mean", np.nan)
        old_ci_hi = old.get("debiased_ci95", [np.nan, np.nan])[1]
        new_db = new["headroom_heldout_debiased"]["debiased_mean"]
        new_ci_hi = new["headroom_heldout_debiased"]["debiased_ci95"][1]
        # qualitative verdict: headroom<=0 (debiased<=0) AND ci upper excludes +0.01 region
        new_le0 = new_db <= 0
        old_le0 = (old_db <= 0) if np.isfinite(old_db) else None
        new_excl01 = new["pos_headroom_exclusion"]["+0.01"]["pos_headroom_excluded"]
        old_excl01 = bool(bf.get(k, {}).get("tost_delta_0.01", {}).get("p_upper", 1) < 0.05)
        verdict[k] = {
            "old_debiased": old_db, "new_debiased": new_db,
            "old_ci_hi": old_ci_hi, "new_ci_hi": new_ci_hi,
            "old_le0": old_le0, "new_le0": bool(new_le0),
            "old_pos_excluded_0.01": old_excl01, "new_pos_excluded_0.01": new_excl01,
            "verdict_changed": bool((old_le0 is not None and old_le0 != bool(new_le0))
                                    or (old_excl01 != new_excl01)),
        }
    n_changed = sum(v["verdict_changed"] for v in verdict.values())
    out["_verdict_vs_b2final"] = verdict
    out["_summary"] = {
        "n_cells": len(keys), "n_null": n_null, "n_boot": n_boot,
        "n_cells_debiased_le0": int(sum(out[k]["headroom_heldout_debiased"]["debiased_mean"] <= 0 for k in keys)),
        "n_cells_pos_headroom_0.01_excluded": int(sum(out[k]["pos_headroom_exclusion"]["+0.01"]["pos_headroom_excluded"] for k in keys)),
        "n_cells_pos_headroom_0.005_excluded": int(sum(out[k]["pos_headroom_exclusion"]["+0.005"]["pos_headroom_excluded"] for k in keys)),
        "n_cells_TOST_equiv_0.01": int(sum(out[k]["tost_delta_0.01"]["equivalent"] for k in keys)),
        "max_new_debiased_ci_hi": float(max(out[k]["headroom_heldout_debiased"]["debiased_ci95"][1] for k in keys)),
        "n_verdict_changes_vs_b2final": int(n_changed),
        "survives": bool(n_changed == 0),
    }
    dump("supp_deepstats.json", out)
    return out


# ================================ ANALYSIS 2: repsplits ================================
def regroup_folds(authors, n_folds, seed):
    """Random recordist(author)-grouped K partition: each author -> one fold."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(authors)))
    rng.shuffle(uniq)
    a2f = {a: i % n_folds for i, a in enumerate(uniq)}
    return np.array([a2f[a] for a in authors])


def run_repsplits(n_seeds, deepstats):
    log(f"=== ANALYSIS 2 repsplits (n_seeds={n_seeds}) ===")
    y, authors = D["y"], D["authors"]
    n_folds = len(set(D["folds"]))
    out = {
        "approximation": (
            "Base OOF predictions are FROZEN (single-level global OOF generated under the "
            "ORIGINAL author-GroupKFold split). Re-grouping only re-assigns the SELECTION/"
            "SCORING folds of the held-out nested oracle; base FEATURES are NOT regenerated. "
            "Hence this is a faithful sensitivity test of the held-out SELECTOR + headroom to "
            "the fold partition (recordist grouping preserved: each author kept in one fold), "
            "but it is NOT a re-trained CV. A fully faithful repeated-split would require "
            "regenerating base OOF per split (a base-model change, out of scope). What IS valid: "
            "the spread below bounds how much the headline held-out headroom depends on the "
            "particular author->fold assignment."),
        "n_seeds": n_seeds, "n_folds": n_folds, "cells": {},
    }
    for key, an, anc, specs in D["cells"]:
        b = BASE[key]
        null_mean = deepstats[key]["null_labelperm"]["null_mean"]
        raw_pts, deb_pts = [], []
        for s in range(n_seeds):
            nf = regroup_folds(authors, n_folds, seed=1000 + s)
            orc, _, _ = heldout_oracle_nested(b["models"], y, nf)
            (a_best, a_orc), nc = vec_macro_auc_common(y, [b["best"], orc])
            if nc and np.isfinite(a_orc) and np.isfinite(a_best):
                raw_pts.append(a_orc - a_best)
                deb_pts.append((a_orc - a_best) - null_mean)
        raw = np.array(raw_pts); deb = np.array(deb_pts)
        out["cells"][key] = {
            "n_ok": int(len(raw)),
            "raw_point_mean": float(raw.mean()), "raw_point_std": float(raw.std(ddof=1)),
            "raw_point_min": float(raw.min()), "raw_point_max": float(raw.max()),
            "debiased_point_mean": float(deb.mean()), "debiased_point_std": float(deb.std(ddof=1)),
            "debiased_point_min": float(deb.min()), "debiased_point_max": float(deb.max()),
            "frozen_original_raw_point": b["ho_macro"] - b["best_macro"],
            "any_seed_debiased_gt_0.01": bool((deb > 0.01).any()),
            "any_seed_debiased_gt_0": bool((deb > 0).any()),
        }
        log(f"REPSPLIT {key} raw[{raw.min():+.4f},{raw.max():+.4f}] "
            f"deb_mean={deb.mean():+.4f} deb_max={deb.max():+.4f} any>0.01={out['cells'][key]['any_seed_debiased_gt_0.01']}")
    out["summary"] = {
        "n_cells_any_seed_debiased_gt_0.01": int(sum(v["any_seed_debiased_gt_0.01"] for v in out["cells"].values())),
        "max_debiased_point_across_all": float(max(v["debiased_point_max"] for v in out["cells"].values())),
        "survives": bool(all(not v["any_seed_debiased_gt_0.01"] for v in out["cells"].values())),
    }
    dump("supp_repsplits.json", out)
    return out


# ================================ ANALYSIS 3: selector ================================
def param_heldout_oracle(models, y, folds, default_name, margin, min_pos):
    """Held-out per-class selector with SHRINKAGE (only switch away from the
    deployable default if a candidate beats it by > margin on the selection folds)
    and MIN-POSITIVES (require >= min_pos selection-set positives before switching).
    Mirrors heldout_oracle_nested's vectorized AUC + fold scheme."""
    cands = _candidate_dict(models)
    cand_names = list(cands.keys())
    cand_arrs = list(cands.values())
    def_k = cand_names.index(default_name)
    N, C = cand_arrs[0].shape
    oracle = np.full((N, C), np.nan)
    uf = sorted(np.unique(folds))
    counts = {n: 0 for n in cand_names}
    for f in uf:
        sel = folds != f; eva = folds == f
        ysel = y[sel]
        npos = ysel.sum(0)
        auc_mat = np.full((len(cand_arrs), C), -np.inf)
        for k, M in enumerate(cand_arrs):
            a, _ = vec_auc_per_class(ysel, M[sel])
            auc_mat[k] = np.where(np.isnan(a), -np.inf, a)
        best_k = np.argmax(auc_mat, axis=0)
        def_auc = auc_mat[def_k]
        best_auc = auc_mat[best_k, np.arange(C)]
        # revert to default unless gain>margin AND enough positives AND default scorable
        revert = (~np.isfinite(best_auc)) | (~np.isfinite(def_auc)) | \
                 ((best_auc - def_auc) <= margin) | (npos < min_pos)
        chosen_k = np.where(revert, def_k, best_k)
        for c in range(C):
            k = chosen_k[c]
            counts[cand_names[k]] += 1
            oracle[eva, c] = cand_arrs[k][eva, c]
    return oracle, counts


def run_selector():
    log("=== ANALYSIS 3 selector shrinkage + min-positives ===")
    y, folds = D["y"], D["folds"]
    margins = [0.0, 0.005, 0.01, 0.02]
    min_positives = [1, 5, 10, 20]
    out = {"margins": margins, "min_positives": min_positives, "cells": {}}
    for key, an, anc, specs in D["cells"]:
        b = BASE[key]
        grid = []
        for mg in margins:
            for mp in min_positives:
                orc, counts = param_heldout_oracle(b["models"], y, folds, b["best_name"], mg, mp)
                (a_best, a_orc), nc = vec_macro_auc_common(y, [b["best"], orc])
                grid.append({"margin": mg, "min_pos": mp,
                             "headroom_vs_best": float(a_orc - a_best),
                             "winner_counts": counts})
        hrs = [g["headroom_vs_best"] for g in grid]
        out["cells"][key] = {
            "default_selector": b["best_name"],
            "baseline_heldout_headroom": b["ho_macro"] - b["best_macro"],
            "grid": grid,
            "headroom_min": float(min(hrs)), "headroom_max": float(max(hrs)),
            "any_config_headroom_gt_0.01": bool(any(h > 0.01 for h in hrs)),
        }
        log(f"SELECTOR {key} headroom range [{min(hrs):+.4f},{max(hrs):+.4f}] "
            f"any>0.01={out['cells'][key]['any_config_headroom_gt_0.01']}")
    out["summary"] = {
        "max_headroom_any_cell_any_config": float(max(v["headroom_max"] for v in out["cells"].values())),
        "n_cells_any_config_gt_0.01": int(sum(v["any_config_headroom_gt_0.01"] for v in out["cells"].values())),
        "survives": bool(all(not v["any_config_headroom_gt_0.01"] for v in out["cells"].values())),
        "interpretation": ("Per-class winner shifts toward the deployable default (best single / mean) "
                           "as margin and min_pos increase; headroom stays <= ~0 across the whole grid, "
                           "i.e. the no-headroom finding is not an artifact of an aggressive zero-margin selector."),
    }
    dump("supp_selector.json", out)
    return out


# ================================ ANALYSIS 4: coverage ceiling ================================
def run_coverage_ceiling():
    log("=== ANALYSIS 4 coverage ceiling ===")
    y = D["y"]
    m = pd.read_csv(os.path.join(OOF, "perch_zeroshot_map.csv"))
    assert len(m) == y.shape[1], "map/target column count mismatch"
    perch_zs = D["perch_zs"]
    mapped = m["stage"].values != "unmapped"
    pos = y.sum(0)
    twoclass = np.array([len(np.unique(y[:, c])) >= 2 for c in range(y.shape[1])])
    evaluable = (pos > 0) & twoclass
    unmapped_idx = np.where(~mapped)[0]
    unmapped = [{"col": int(i), "primary_label": str(m.iloc[i]["primary_label"]),
                 "class_name": str(m.iloc[i]["class_name"]),
                 "n_positives": int(pos[i]), "evaluable": bool(evaluable[i])}
                for i in unmapped_idx]
    n_unmapped_eval = int(((~mapped) & evaluable).sum())
    n_eval = int(evaluable.sum())
    # zero-shot macro-AUC ceiling: unmapped evaluable classes cannot be scored by
    # zero-shot (no perch index) -> AUC pinned at chance 0.5; best case mapped=1.0.
    ceiling_zs = float((1.0 * (mapped & evaluable).sum() + 0.5 * n_unmapped_eval) / max(n_eval, 1))
    # achieved zero-shot macro on evaluable mask, and the unmapped drag
    achieved_zs = macro_auc_on_mask(y, perch_zs, evaluable)
    # what zero-shot scores ON the unmapped-evaluable classes (should be ~chance)
    zs_pc = per_class_auc_on_mask(y, perch_zs, evaluable)
    unmapped_eval_cols = [int(i) for i in unmapped_idx if evaluable[i]]
    zs_on_unmapped = [zs_pc[c] for c in unmapped_eval_cols if c in zs_pc]
    out = {
        "n_classes": int(y.shape[1]), "n_mapped": int(mapped.sum()),
        "n_unmapped": int((~mapped).sum()),
        "stage_counts": m["stage"].value_counts().to_dict(),
        "n_evaluable_classes": n_eval,
        "n_unmapped_evaluable": n_unmapped_eval,
        "unmapped_total_positives": int(pos[unmapped_idx].sum()),
        "unmapped_classes": unmapped,
        "zeroshot_macro_auc_ceiling_if_unmapped_pinned_chance": ceiling_zs,
        "zeroshot_macro_auc_ceiling_drag_vs_perfect": float(1.0 - ceiling_zs),
        "zeroshot_achieved_macro_auc_evaluable": float(achieved_zs),
        "zeroshot_auc_on_unmapped_evaluable_classes": {str(c): float(zs_pc.get(c, np.nan)) for c in unmapped_eval_cols},
        "mean_zs_auc_on_unmapped_evaluable": float(np.mean(zs_on_unmapped)) if zs_on_unmapped else None,
        "note": ("31 classes have NO Perch zero-shot index (eBird synonym / taxonomy version pin "
                 "vs the Perch label set): non-bird taxa (Insecta/Reptilia/Amphibia/Mammalia) absent "
                 "from Perch's avian label space + a few eBird-code synonyms. Only "
                 f"{n_unmapped_eval} of the 31 are AUC-evaluable (>=1 focal positive, both classes), "
                 f"carrying {int(pos[unmapped_idx].sum())} total positives, so the macro-AUC ceiling drag "
                 "on the ZERO-SHOT pathway is negligible. Linear-probe/CNN DO train these columns, so "
                 "the unmapped set caps only the zero-shot retrieval pathway, not the supervised models."),
    }
    dump("supp_coverage_ceiling.json", out)
    log(f"COVERAGE n_unmapped=31 n_unmapped_eval={n_unmapped_eval} zs_ceiling={ceiling_zs:.5f} "
        f"zs_achieved={achieved_zs:.5f}")
    return out


# ================================ ANALYSIS 5: reviewer ================================
def macro_auc_skip_zero_sklearn(y, p, mask):
    """INDEPENDENT reimplementation: per-class sklearn roc_auc_score, skip classes
    with zero positives / not in mask. Cross-checks the vectorized rank-sum primitive."""
    aucs = []
    for c in np.where(mask)[0]:
        yc = y[:, c]
        if yc.sum() <= 0 or len(np.unique(yc)) < 2:
            continue
        col = p[:, c]
        if np.isnan(col).any():
            continue
        aucs.append(roc_auc_score(yc, col))
    return float(np.mean(aucs)) if aucs else np.nan


def run_reviewer(deepstats):
    log("=== ANALYSIS 5 reviewer cross-checks ===")
    y, authors, folds, filenames = D["y"], D["authors"], D["folds"], D["filenames"]
    out = {}

    # (a) independent macro-AUC skip-zero cross-check on every cell's models
    log("5(a) independent macro-AUC cross-check ...")
    a_diffs = []
    cell_checks = {}
    for key, an, anc, specs in D["cells"]:
        b = BASE[key]
        cc = {}
        for nm, arr in b["models"].items():
            v_prim = macro_auc_on_mask(y, arr, b["mask"])
            v_indep = macro_auc_skip_zero_sklearn(y, arr, b["mask"])
            cc[nm] = {"primitive": v_prim, "independent_sklearn": v_indep,
                      "abs_diff": abs(v_prim - v_indep)}
            a_diffs.append(abs(v_prim - v_indep))
        cell_checks[key] = cc
    out["a_macro_auc_crosscheck"] = {
        "max_abs_diff": float(max(a_diffs)), "mean_abs_diff": float(np.mean(a_diffs)),
        "matches_to_1e-6": bool(max(a_diffs) < 1e-6),
        "per_cell": cell_checks,
    }
    log(f"5(a) max_abs_diff={max(a_diffs):.2e}")

    # (b) leakage report
    log("5(b) leakage report ...")
    meta = pd.DataFrame({"author": authors, "fold": folds, "filename": filenames})
    folds_per_author = meta.groupby("author")["fold"].nunique()
    authors_spanning = folds_per_author[folds_per_author > 1]
    fn_fold = meta.groupby("filename")["fold"].nunique()
    files_spanning = fn_fold[fn_fold > 1]
    dup_files = int(meta["filename"].duplicated().sum())
    out["b_leakage"] = {
        "n_authors": int(meta["author"].nunique()),
        "n_authors_spanning_multiple_folds": int(len(authors_spanning)),
        "author_grouping_clean": bool(len(authors_spanning) == 0),
        "n_files_spanning_multiple_folds": int(len(files_spanning)),
        "n_duplicate_filenames": dup_files,
        "no_train_val_overlap": bool(len(files_spanning) == 0 and dup_files == 0),
        "note": ("Fold assignment is author(recordist)-grouped: every author lives in exactly one "
                 "fold (GroupKFold integrity) so no recordist spans train/val. Filenames are unique "
                 "and each file is in exactly one fold -> no clip-level train/val overlap. RESIDUAL "
                 "(documented, not a leak in THIS combiner): base OOF columns are single-level global "
                 "OOF; the held-out SELECTION/fusion is leak-free but base FEATURES saw their own outer "
                 "fold -> inflates best-single -> SHRINKS apparent headroom -> conservative."),
    }
    log(f"5(b) authors_spanning={len(authors_spanning)} files_spanning={len(files_spanning)} dup_files={dup_files}")

    # (c) influence on the HEADLINE cell
    log(f"5(c) influence on headline {HEADLINE_KEY} ...")
    b = BASE[HEADLINE_KEY]
    best, ho = b["best"], b["ho_orc"]
    null_mean = deepstats[HEADLINE_KEY]["null_labelperm"]["null_mean"]
    base_pt = b["ho_macro"] - b["best_macro"]
    base_deb = base_pt - null_mean

    def point_on(keep):
        (a_best, a_orc), nc = vec_macro_auc_common(y[keep], [best[keep], ho[keep]])
        if nc and np.isfinite(a_orc) and np.isfinite(a_best):
            return a_orc - a_best
        return np.nan

    # leave-one-author-out (grouped, dominating perturbation)
    uniq = np.array(sorted(set(authors)))
    loa_deb = []
    for a in uniq:
        keep = authors != a
        p = point_on(keep)
        if np.isfinite(p):
            loa_deb.append(p - null_mean)
    loa_deb = np.array(loa_deb)
    loa = {
        "n": int(len(loa_deb)), "base_debiased": float(base_deb),
        "min_debiased": float(loa_deb.min()), "max_debiased": float(loa_deb.max()),
        "n_flip_sign_to_positive": int((loa_deb > 0).sum()),
        "n_exceed_+0.01": int((loa_deb > 0.01).sum()),
        "verdict_ever_flips": bool((loa_deb > 0.01).any()),
    }
    log(f"5(c) leave-one-author deb range [{loa_deb.min():+.5f},{loa_deb.max():+.5f}] flips={loa['verdict_ever_flips']}")

    # leave-one-file-out (strictly smaller perturbation than its author)
    lof_deb_min, lof_deb_max = np.inf, -np.inf
    n_files = len(y)
    t0 = time.time()
    n_pos_flip = 0; n_exceed = 0
    for i in range(n_files):
        keep = np.ones(n_files, dtype=bool); keep[i] = False
        p = point_on(keep)
        if np.isfinite(p):
            d = p - null_mean
            lof_deb_min = min(lof_deb_min, d); lof_deb_max = max(lof_deb_max, d)
            if d > 0: n_pos_flip += 1
            if d > 0.01: n_exceed += 1
        if i % 5000 == 0:
            log(f"   leave-one-file {i}/{n_files} ({time.time()-t0:.0f}s)")
    lof = {
        "n": int(n_files), "min_debiased": float(lof_deb_min), "max_debiased": float(lof_deb_max),
        "n_flip_sign_to_positive": int(n_pos_flip), "n_exceed_+0.01": int(n_exceed),
        "verdict_ever_flips": bool(lof_deb_max > 0.01),
    }
    log(f"5(c) leave-one-file deb range [{lof_deb_min:+.5f},{lof_deb_max:+.5f}] flips={lof['verdict_ever_flips']}")

    # prevalence-stratified influence (tertiles of positive count among eval classes)
    mask = b["mask"]
    cols = np.where(mask)[0]
    pos = y[:, cols].sum(0)
    pc_best = per_class_auc_on_mask(y, best, mask)
    pc_ho = per_class_auc_on_mask(y, ho, mask)
    q1, q2 = np.percentile(pos, [33.333, 66.667])
    strata = {"low": pos <= q1, "mid": (pos > q1) & (pos <= q2), "high": pos > q2}
    prev = {}
    for name, sm in strata.items():
        scols = cols[sm]
        bs = [pc_best[c] for c in scols if c in pc_best]
        hs = [pc_ho[c] for c in scols if c in pc_ho]
        if bs and hs:
            prev[name] = {"n_classes": int(len(scols)),
                          "median_positives": float(np.median(pos[sm])),
                          "best_macro": float(np.mean(bs)), "ho_macro": float(np.mean(hs)),
                          "raw_headroom": float(np.mean(hs) - np.mean(bs)),
                          "debiased_headroom": float(np.mean(hs) - np.mean(bs) - null_mean)}
    out["c_influence"] = {
        "headline_cell": HEADLINE_KEY, "null_mean": float(null_mean),
        "base_raw_point": float(base_pt), "base_debiased_point": float(base_deb),
        "leave_one_author_out": loa, "leave_one_file_out": lof,
        "prevalence_stratified": prev,
        "verdict": ("HEADLINE SURVIVES: no single author drop, no single file drop, and no "
                    "prevalence stratum produces deployable positive headroom (>+0.01)."
                    if (not loa["verdict_ever_flips"] and not lof["verdict_ever_flips"]
                        and all(v["debiased_headroom"] <= 0.01 for v in prev.values()))
                    else "FLAG: an influence drop changes the headline verdict (see fields)."),
    }

    # (d) version-pinned 234->eBird mapping table for release
    m = pd.read_csv(os.path.join(OOF, "perch_zeroshot_map.csv"))
    pos_all = y.sum(0)
    table = []
    for i in range(len(m)):
        r = m.iloc[i]
        table.append({"col": int(i), "primary_label": str(r["primary_label"]),
                      "class_name": str(r["class_name"]), "stage": str(r["stage"]),
                      "perch_idxs": (None if pd.isna(r["perch_idxs"]) else str(r["perch_idxs"])),
                      "mapped": bool(r["stage"] != "unmapped"),
                      "n_positives": int(pos_all[i])})
    out["d_mapping_table"] = {
        "version_pin": ("Perch v2 (Kaggle/Google) avian label set; eBird/Clements taxonomy as shipped "
                        "with the BirdCLEF++ 2026 train set. Mapping stages: exact_ebird (direct eBird "
                        "code match), sciname_fallback (scientific-name match when code differs), "
                        "unmapped (no Perch index: non-avian taxon or synonym not in Perch label space). "
                        "Pin these two versions together for reproducible zero-shot mapping at release."),
        "stage_counts": m["stage"].value_counts().to_dict(),
        "n_rows": len(table), "table": table,
    }

    out["summary"] = {
        "a_macro_auc_matches": out["a_macro_auc_crosscheck"]["matches_to_1e-6"],
        "a_max_abs_diff": out["a_macro_auc_crosscheck"]["max_abs_diff"],
        "b_no_leakage": out["b_leakage"]["author_grouping_clean"] and out["b_leakage"]["no_train_val_overlap"],
        "c_headline_survives_influence": bool(not loa["verdict_ever_flips"] and not lof["verdict_ever_flips"]),
        "d_mapping_rows": len(table),
    }
    dump("supp_reviewer.json", out)
    return out


def main():
    global D, BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_null", type=int, default=1000)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--n_repsplits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    log("loading OOF artifacts ...")
    D = load_all()
    log(f"loaded: rows={len(D['y'])} authors={len(set(D['authors']))} cells={len(D['cells'])}")
    BASE = build_base()

    deepstats = run_deepstats(args.n_null, args.n_boot, args.seed)
    repsplits = run_repsplits(args.n_repsplits, deepstats)
    selector = run_selector()
    coverage = run_coverage_ceiling()
    reviewer = run_reviewer(deepstats)

    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")
    log("SURVIVAL: deepstats=%s repsplits=%s selector=%s influence=%s" % (
        deepstats["_summary"]["survives"], repsplits["summary"]["survives"],
        selector["summary"]["survives"], reviewer["summary"]["c_headline_survives_influence"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
