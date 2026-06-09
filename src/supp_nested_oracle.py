"""SUPP #1 — nested-CV de-biased oracle headroom + soft fusion (ONE SEED).

Recomputes the HEADLINE de-biased held-out per-class oracle dAUC and the calibrated
soft-fusion delta under NESTED base OOFs, and compares to the GLOBAL one-seed baseline
(same seed, same everything except the outer-fold leakage in the base FEATURES).

GLOBAL one-seed base OOF: assembled from the existing {model}_fold{f}_seed42.npy.
NESTED base OOF: per outer fold f, the SELECTION-set rows (folds != f) use inner-OOF
preds (model trained WITHOUT f and WITHOUT the predicted fold), produced by
supp_nested_train.py; the APPLY rows (fold f) use the standard one-seed OOF. The
selector/fusion therefore never see base predictions trained on the outer fold.

For each cell (LP+CNN, LP+PANN, LP+CNN+PANN[, +BEATs]) we report, for BOTH the global
and nested base OOFs:
  best_single_macro, heldout_oracle_macro, raw heldout-minus-best, label-perm null_mean,
  DE-BIASED headroom mean + 95% CI (author-clustered bootstrap), and soft-fusion delta.
VERDICT field per cell: does de-biased dAUC <= 0 SURVIVE the nested OOFs?

Output: data/oof/supp_nestedcv.json
"""
import itertools, json, os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/SHC/birdclef2026_clef")
sys.path.insert(0, os.path.join(ROOT, "src"))
from b2_oracle_headroom import common_eval_mask, macro_auc_on_mask
from b2_headline_driver import (heldout_oracle_nested, calibrated_fusion_multi,
                                vec_auc_per_class, vec_macro_auc_common,
                                clustered_bootstrap_pair)
from b2_corrected import permute_labels_within_author
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OOF = os.path.join(ROOT, "data", "oof")
NES = os.path.join(OOF, "supp_nested")
SEED = 42
N_BOOT = 2000
N_NULL = 200
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ------------------------ assemble base matrices ------------------------
def assemble_global(model, N, C):
    """Global one-seed OOF (N,C) from {model}_fold{f}_seed{SEED}.npy + _rows.npy."""
    M = np.full((N, C), np.nan, np.float64)
    for f in range(5):
        p = os.path.join(OOF, f"{model}_fold{f}_seed{SEED}.npy")
        r = os.path.join(OOF, f"{model}_fold{f}_seed{SEED}_rows.npy")
        M[np.load(r)] = np.load(p).astype(np.float64)
    return M


def assemble_nested(model, N, C, folds):
    """sel[f] (N,C): rows in fold g (g!=f) = inner pred from pair {f,g} on fold g."""
    sel = {}
    ufolds = sorted(set(folds))
    for f in ufolds:
        Mf = np.full((N, C), np.nan, np.float64)
        for g in ufolds:
            if g == f:
                continue
            a, b = sorted((f, g))
            tag = f"{model}_ex{a}_{b}_seed{SEED}"
            pp = os.path.join(NES, f"{tag}_fold{g}.npy")
            rr = os.path.join(NES, f"{tag}_fold{g}_rows.npy")
            Mf[np.load(rr)] = np.load(pp).astype(np.float64)
        sel[f] = Mf
    return sel


def lp_probe(Xtr, ytr_mat, Xpred, C, max_iter=1000, reg_C=1.0):
    """Fit per-class balanced logistic on standardized embeddings, predict probs."""
    sc = StandardScaler().fit(Xtr)
    Xt = sc.transform(Xtr); Xp = sc.transform(Xpred)
    out = np.full((Xpred.shape[0], C), np.nan, np.float64)
    for c in range(C):
        yc = ytr_mat[:, c]
        if yc.sum() == 0 or len(np.unique(yc)) < 2:
            continue
        clf = LogisticRegression(max_iter=max_iter, C=reg_C, class_weight="balanced")
        clf.fit(Xt, yc)
        out[:, c] = clf.predict_proba(Xp)[:, 1]
    return out


def lp_nested_sel(embs, Y, folds, N, C):
    """LP nested selection mats: sel[f] rows of fold g = probe on S=folds\\{f,g} -> g."""
    ufolds = sorted(set(folds))
    sel = {f: np.full((N, C), np.nan, np.float64) for f in ufolds}
    for a, b in itertools.combinations(ufolds, 2):
        S = [x for x in ufolds if x not in (a, b)]
        trm = np.isin(folds, S)
        Xtr = embs[trm].astype(np.float32); ytr = Y[trm]
        for g in (a, b):
            f = b if g == a else a    # pair {f,g}; this serves outer fold f
            gm = folds == g
            pred = lp_probe(Xtr, ytr, embs[gm].astype(np.float32), C)
            sel[f][gm] = pred
        print(f"[lp-nested] pair {{ {a},{b} }} done", flush=True)
    return sel


# ------------------------ nested oracle / fusion / null ------------------------
def cand_dict(mats, names):
    d = {nm: mats[nm] for nm in names}
    d["mean"] = np.mean([mats[nm] for nm in names], axis=0)
    return d


def nested_oracle(apply_mats, sel_mats, names, y, folds):
    N, C = y.shape
    cand_names = names + ["mean"]
    oracle = np.full((N, C), np.nan, np.float64)
    chosen = []
    for f in sorted(set(folds)):
        sm = folds != f
        em = folds == f
        sel_c = cand_dict({nm: sel_mats[nm][f] for nm in names}, names)
        app_c = cand_dict({nm: apply_mats[nm] for nm in names}, names)
        auc_mat = np.full((len(cand_names), C), -np.inf)
        for k, cn in enumerate(cand_names):
            a, _ = vec_auc_per_class(y[sm], sel_c[cn][sm])
            auc_mat[k] = np.where(np.isnan(a), -np.inf, a)
        best_k = np.argmax(auc_mat, axis=0)
        none = ~np.isfinite(auc_mat.max(axis=0))
        best_k[none] = 0
        chosen.append(best_k)
        for c in range(C):
            oracle[em, c] = app_c[cand_names[best_k[c]]][em, c]
    return oracle, np.array(chosen)


def nested_fusion(apply_mats, sel_mats, names, y, folds):
    N, C = y.shape
    Lsel = {nm: logit(sel_mats[nm]) if not isinstance(sel_mats[nm], dict) else None for nm in names}
    fused = np.full((N, C), np.nan, np.float64)
    for f in sorted(set(folds)):
        tr = folds != f
        outer = folds == f
        Ls = {nm: logit(sel_mats[nm][f]) for nm in names}
        Lo = {nm: logit(apply_mats[nm]) for nm in names}
        for c in range(C):
            cols_tr = [Ls[nm][tr, c] for nm in names]
            ytr = y[tr, c]
            valid = ~np.any([np.isnan(ct) for ct in cols_tr], axis=0)
            if ytr[valid].sum() < 2 or len(np.unique(ytr[valid])) < 2:
                continue
            X = np.stack([ct[valid] for ct in cols_tr], 1)
            try:
                clf = LogisticRegression(max_iter=500, C=1.0)
                clf.fit(X, ytr[valid])
            except Exception:
                continue
            Xo = np.stack([Lo[nm][outer, c] for nm in names], 1)
            ok = ~np.isnan(Xo).any(1)
            pr = np.full(outer.sum(), np.nan)
            if ok.any():
                pr[ok] = clf.predict_proba(Xo[ok])[:, 1]
            fused[outer, c] = pr
    return fused


def nested_null_labelperm(apply_mats, sel_mats, names, y, folds, authors, best_name,
                          n_rep=N_NULL, seed=0):
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_rep):
        yp = permute_labels_within_author(y, authors, rng)
        orc, _ = nested_oracle(apply_mats, sel_mats, names, yp, folds)
        best = apply_mats[best_name]
        (a_best, a_orc), nc = vec_macro_auc_common(yp, [best, orc])
        if nc and np.isfinite(a_orc) and np.isfinite(a_best):
            null.append(a_orc - a_best)
    return float(np.mean(null)) if null else 0.0, len(null)


def global_null_labelperm(apply_mats, names, y, folds, authors, best_name,
                          n_rep=N_NULL, seed=0):
    rng = np.random.default_rng(seed)
    models = {nm: apply_mats[nm] for nm in names}
    null = []
    for _ in range(n_rep):
        yp = permute_labels_within_author(y, authors, rng)
        orc, _, _ = heldout_oracle_nested(models, yp, folds)
        best = apply_mats[best_name]
        (a_best, a_orc), nc = vec_macro_auc_common(yp, [best, orc])
        if nc and np.isfinite(a_orc) and np.isfinite(a_best):
            null.append(a_orc - a_best)
    return float(np.mean(null)) if null else 0.0, len(null)


def summarize(apply_mats, names, oracle, fused, y, folds, authors, null_mean, variant):
    mats = [apply_mats[nm] for nm in names]
    mask = common_eval_mask(y, mats + [oracle, fused])
    per_model = {nm: macro_auc_on_mask(y, apply_mats[nm], mask) for nm in names}
    best_name = max(per_model, key=per_model.get)
    best_macro = per_model[best_name]
    orc_macro = macro_auc_on_mask(y, oracle, mask)
    fus_macro = macro_auc_on_mask(y, fused, mask)
    raw = clustered_bootstrap_pair(apply_mats[best_name], oracle, y, authors,
                                   n_boot=N_BOOT, seed=0)
    deb_mean = raw["delta_mean"] - null_mean
    deb_ci = [raw["ci95"][0] - null_mean, raw["ci95"][1] - null_mean]
    fus = clustered_bootstrap_pair(apply_mats[best_name], fused, y, authors,
                                   n_boot=N_BOOT, seed=1)
    return {
        "variant": variant,
        "n_common_eval_classes": int(mask.sum()),
        "per_model_macro_auc": per_model,
        "best_single_model": best_name,
        "best_single_macro_auc": best_macro,
        "heldout_oracle_macro_auc": orc_macro,
        "heldout_minus_best_single_point": orc_macro - best_macro,
        "null_labelperm_mean": null_mean,
        "headroom_debiased_mean": float(deb_mean),
        "headroom_debiased_ci95": [float(deb_ci[0]), float(deb_ci[1])],
        "headroom_raw_mean": raw["delta_mean"], "headroom_raw_ci95": raw["ci95"],
        "se": raw["se"],
        "fusion_macro_auc": fus_macro,
        "fusion_minus_best_single_point": fus_macro - best_macro,
        "fusion_delta_ci95": fus["ci95"],
    }


def main():
    t0 = time.time()
    perch_lp = np.load(os.path.join(OOF, "perch_anchor_oof.npy")).astype(np.float64)
    y = np.load(os.path.join(OOF, "oof_targets.npy")).astype(np.float64)
    meta = pd.read_csv(os.path.join(OOF, "oof_meta.csv"))
    authors = meta["author"].values
    folds = meta["fold"].values
    embs = np.load(os.path.join(OOF, "perch_cache", "embeddings.npy"))
    N, C = y.shape
    assert perch_lp.shape == (N, C) and len(embs) == N, \
        f"shape {perch_lp.shape} {y.shape} embs {embs.shape}"
    print(f"[load] N={N} C={C} authors={len(set(authors))} folds={sorted(set(folds))}",
          flush=True)

    # which sidecars have nested files present
    have = {}
    for model in ["cnn", "pann", "beats"]:
        ok = all(os.path.exists(os.path.join(NES, f"{model}_ex{a}_{b}_seed{SEED}_fold{g}.npy"))
                 for a, b in itertools.combinations(sorted(set(folds)), 2) for g in (a, b))
        have[model] = ok
        print(f"[nested-files] {model}: {'present' if ok else 'MISSING'}", flush=True)

    # global one-seed apply mats + nested sel mats
    apply_mats = {"perch_linear_probe": perch_lp}
    sel_mats = {}
    print("[lp] computing nested LP selection probes (CPU)...", flush=True)
    sel_mats["perch_linear_probe"] = lp_nested_sel(embs, y, folds, N, C)
    for model in ["cnn", "pann", "beats"]:
        if not have[model]:
            continue
        apply_mats[model] = assemble_global(model, N, C)
        sel_mats[model] = assemble_nested(model, N, C, folds)
        print(f"[assemble] {model} global+nested ready", flush=True)

    cells = [("LP+CNN", ["perch_linear_probe", "cnn"]),
             ("LP+PANN", ["perch_linear_probe", "pann"]),
             ("LP+CNN+PANN", ["perch_linear_probe", "cnn", "pann"])]
    if have.get("beats"):
        cells.append(("LP+CNN+PANN+BEATs",
                      ["perch_linear_probe", "cnn", "pann", "beats"]))

    out = {"seed": SEED, "n_boot": N_BOOT, "n_null": N_NULL, "rows": int(N),
           "n_authors": int(len(set(authors))),
           "note": ("Nested base OOF: selection-set rows use inner-OOF preds trained "
                    "without the outer fold; apply rows use standard one-seed OOF. "
                    "Compared to global one-seed OOF (selection leaks outer fold)."),
           "cells": {}}
    for cname, names in cells:
        if not all(nm in apply_mats for nm in names):
            print(f"[skip] {cname} (missing model)", flush=True)
            continue
        print(f"\n[cell] {cname} {names}", flush=True)
        # GLOBAL
        gmodels = {nm: apply_mats[nm] for nm in names}
        g_orc, _, _ = heldout_oracle_nested(gmodels, y, folds)
        g_fus = calibrated_fusion_multi(gmodels, y, folds)
        gmask = common_eval_mask(y, [apply_mats[nm] for nm in names] + [g_orc, g_fus])
        g_per = {nm: macro_auc_on_mask(y, apply_mats[nm], gmask) for nm in names}
        g_best = max(g_per, key=g_per.get)
        g_null, _ = global_null_labelperm(apply_mats, names, y, folds, authors, g_best,
                                          seed=0)
        g_sum = summarize(apply_mats, names, g_orc, g_fus, y, folds, authors, g_null, "global")
        print(f"  GLOBAL deb={g_sum['headroom_debiased_mean']:+.5f} "
              f"CI{[round(x,4) for x in g_sum['headroom_debiased_ci95']]} "
              f"fus={g_sum['fusion_minus_best_single_point']:+.5f}", flush=True)
        # NESTED
        n_orc, _ = nested_oracle(apply_mats, sel_mats, names, y, folds)
        n_fus = nested_fusion(apply_mats, sel_mats, names, y, folds)
        nmask = common_eval_mask(y, [apply_mats[nm] for nm in names] + [n_orc, n_fus])
        n_per = {nm: macro_auc_on_mask(y, apply_mats[nm], nmask) for nm in names}
        n_best = max(n_per, key=n_per.get)
        n_null, _ = nested_null_labelperm(apply_mats, sel_mats, names, y, folds, authors,
                                          n_best, seed=0)
        n_sum = summarize(apply_mats, names, n_orc, n_fus, y, folds, authors, n_null, "nested")
        print(f"  NESTED deb={n_sum['headroom_debiased_mean']:+.5f} "
              f"CI{[round(x,4) for x in n_sum['headroom_debiased_ci95']]} "
              f"fus={n_sum['fusion_minus_best_single_point']:+.5f}", flush=True)
        survives = bool(n_sum["headroom_debiased_mean"] <= 0)
        out["cells"][cname] = {
            "global": g_sum, "nested": n_sum,
            "nested_minus_global_debiased": (n_sum["headroom_debiased_mean"]
                                             - g_sum["headroom_debiased_mean"]),
            "verdict_debiased_le0_survives_nested": survives,
        }
    out["overall_verdict_all_cells_survive"] = bool(
        all(v["verdict_debiased_le0_survives_nested"] for v in out["cells"].values()))
    out_path = os.path.join(OOF, "supp_nestedcv.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[done] {(time.time()-t0)/60:.1f} min -> {out_path}", flush=True)
    print("overall_survives =", out["overall_verdict_all_cells_survive"], flush=True)


if __name__ == "__main__":
    main()
