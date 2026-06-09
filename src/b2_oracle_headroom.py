"""B2' HEADLINE — oracle-blend headroom of adding the CNN sidecar to the Perch anchor.

Given two taxonomy-aligned (N, 234) OOF matrices on IDENTICAL rows:
    anchor  = Perch-V2 embedding linear-probe OOF  (b1_perch_extract.py)
    sidecar = author CNN OOF                        (b1_cnn_train.py)
plus oof_meta.csv (filename, fold, author) and the multi-hot targets (N, 234),
compute, per the locked plan (exp-plan-v2 B2'/B3'):

  Conditions (all scored with macro ROC-AUC on a COMMON evaluable-class mask):
    1. anchor-only
    2. sidecar-only
    3. calibrated UNCLAMPED fusion  -- per-class logistic blend, NESTED: the
       meta-learner for outer fold f is fit ONLY on folds != f (it never sees
       the outer fold's labels). Honest non-oracle blend.
    4. HELD-OUT (nested) per-class ORACLE best-of -- for each outer fold f,
       select per-class best of {anchor, sidecar, mean} using AUC on folds != f,
       apply to fold f only, pool outer preds. This is the DEFENSIBLE upper
       bound (no selection-on-evaluation-data bias).
    4b. in_sample_selection_biased  -- the OLD oracle that selects per class on
        the SAME labels it scores. Reported ONLY as an ex-post diagnostic.

  Primary endpoint:  dAUC = held_out_oracle - anchor, mean +/- 95% CI,
    bootstrap CLUSTERED BY author (resample authors, not rows), paired deltas
    over a COMMON evaluable class mask PER REPLICATE.
  Report Minimum Detectable Effect (MDE) from the bootstrap SE.

  Complementarity (B3'):  Spearman, Pearson, Kuncheva Q-statistic, double-fault.

Validates the math on small synthetic arrays first (--selftest), asserting
  held_out_oracle <= in_sample_oracle  AND  oracle >= max(anchor, sidecar) on
  the common mask, with finite deltas.

COMMON-MASK semantics (codex (e)): a class is evaluable iff it has >=1 positive
AND non-NaN predictions across ALL compared models on the rows being scored.
Anchor/sidecar/oracle deltas are then valid because every model is scored on the
identical class set.

RESIDUAL LIMITATION (codex (c)): the OOF feature columns themselves are produced
by base models that were trained on the outer fold's rows (global OOF). Fully
removing that requires regenerating inner-OOF base predictions inside each outer
training set (a base-model change, not a B2 change). The fusion/oracle SELECTION
here is leakage-free; the base FEATURES are not. Reported as a post-hoc combiner.
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score


# ----------------------------- common evaluable mask -----------------------------
def common_eval_mask(y_true, pred_list):
    """Boolean (C,) mask: class is evaluable iff it has >=1 positive AND every
    model in pred_list has all-finite (non-NaN) predictions for that class on
    these rows. Use the SAME mask for every model so deltas are valid."""
    C = y_true.shape[1]
    pos = (y_true.sum(0) > 0)
    finite = np.ones(C, dtype=bool)
    for P in pred_list:
        finite &= ~np.isnan(P).any(0)
    # need both classes present for a defined AUC
    twoclass = np.array([len(np.unique(y_true[:, c])) >= 2 for c in range(C)])
    return pos & finite & twoclass


def per_class_auc_on_mask(y_true, y_pred, mask):
    """{class_idx: auc} only for classes where mask is True."""
    out = {}
    for c in np.where(mask)[0]:
        try:
            out[int(c)] = roc_auc_score(y_true[:, c], y_pred[:, c])
        except Exception:
            pass
    return out


def macro_auc_on_mask(y_true, y_pred, mask):
    d = per_class_auc_on_mask(y_true, y_pred, mask)
    return float(np.mean(list(d.values()))) if d else 0.0


# Back-compat single-model helpers (per-model self-mask, used only standalone).
def per_class_auc(y_true, y_pred):
    out = {}
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() <= 0:
            continue
        col = y_pred[:, i]
        if np.isnan(col).any() or len(np.unique(y_true[:, i])) < 2:
            continue
        try:
            out[i] = roc_auc_score(y_true[:, i], col)
        except Exception:
            pass
    return out


def macro_auc(y_true, y_pred):
    d = per_class_auc(y_true, y_pred)
    return float(np.mean(list(d.values()))) if d else 0.0


# ----------------------------- nested calibrated fusion -----------------------------
def calibrated_unclamped_fusion(anchor, sidecar, y, folds, seed=0):
    """Per-class logistic blend of [anchor_logit, sidecar_logit], NESTED so the
    meta-learner for outer fold f is fit ONLY on folds != f and applied to fold f.
    The meta-learner never sees the outer fold's labels. Returns fused (N,234)
    (NaN where either input NaN, or class not fittable on the inner folds)."""
    from sklearn.linear_model import LogisticRegression
    N, C = anchor.shape
    fused = np.full((N, C), np.nan, dtype=np.float64)
    eps = 1e-6

    def logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    la, ls = logit(anchor), logit(sidecar)
    uf = sorted(np.unique(folds))
    for f in uf:
        inner_tr = folds != f      # meta-learner train: every fold but the outer
        outer = folds == f         # apply only here
        for c in range(C):
            a_tr, s_tr = anchor[inner_tr, c], sidecar[inner_tr, c]
            y_tr = y[inner_tr, c]
            valid = ~(np.isnan(a_tr) | np.isnan(s_tr))
            if y_tr[valid].sum() < 2 or len(np.unique(y_tr[valid])) < 2:
                continue
            X = np.stack([la[inner_tr, c][valid], ls[inner_tr, c][valid]], 1)
            try:
                clf = LogisticRegression(max_iter=500, C=1.0)
                clf.fit(X, y_tr[valid])
            except Exception:
                continue
            Xo = np.stack([la[outer, c], ls[outer, c]], 1)
            ok = ~np.isnan(Xo).any(1)
            pred = np.full(outer.sum(), np.nan)
            if ok.any():
                pred[ok] = clf.predict_proba(Xo[ok])[:, 1]
            fused[outer, c] = pred
    return fused


# ----------------------------- oracles -----------------------------
def heldout_oracle_best_of(anchor, sidecar, y, folds):
    """HELD-OUT / nested oracle (DEFENSIBLE upper bound).

    For each outer fold f: select the per-class best of {anchor, sidecar, mean}
    using AUC computed on folds != f, then apply that choice to fold f's rows
    only. Pool the outer-fold predictions. Selection never touches the rows it is
    later scored on -> no selection-on-evaluation-data bias.

    Returns (oracle[N,C], chosen[F,C])."""
    N, C = anchor.shape
    cands = {"anchor": anchor, "sidecar": sidecar, "mean": 0.5 * (anchor + sidecar)}
    oracle = np.full((N, C), np.nan, dtype=np.float64)
    uf = sorted(np.unique(folds))
    chosen = np.full((len(uf), C), "none", dtype=object)

    for fi, f in enumerate(uf):
        sel = folds != f   # select choice here
        eva = folds == f   # apply choice here
        for c in range(C):
            best_name, best_auc = None, -np.inf
            ysel = y[sel, c]
            for name, M in cands.items():
                col = M[sel, c]
                ok = ~np.isnan(col)
                if ysel[ok].sum() < 2 or len(np.unique(ysel[ok])) < 2:
                    continue
                try:
                    auc = roc_auc_score(ysel[ok], col[ok])
                except Exception:
                    continue
                if auc > best_auc:
                    best_auc, best_name = auc, name
            if best_name is None:
                best_name = "anchor"  # conservative fallback
            chosen[fi, c] = best_name
            oracle[eva, c] = cands[best_name][eva, c]
    return oracle, chosen


def in_sample_selection_biased(anchor, sidecar, y):
    """OLD oracle (DIAGNOSTIC ONLY, selection biased): per class picks whichever
    of {anchor, sidecar, mean} maximises that class's AUC on the SAME labels it is
    later scored on. Overstates headroom; report only as ex-post diagnostic."""
    N, C = anchor.shape
    cands = {"anchor": anchor, "sidecar": sidecar, "mean": 0.5 * (anchor + sidecar)}
    oracle = np.full((N, C), np.nan, dtype=np.float64)
    chosen = np.array(["none"] * C, dtype=object)
    for c in range(C):
        if y[:, c].sum() <= 0 or len(np.unique(y[:, c])) < 2:
            continue
        best_auc, best_name = -1.0, None
        for name, M in cands.items():
            col = M[:, c]
            if np.isnan(col).any():
                continue
            try:
                a = roc_auc_score(y[:, c], col)
            except Exception:
                continue
            if a > best_auc:
                best_auc, best_name = a, name
        if best_name is not None:
            oracle[:, c] = cands[best_name][:, c]
            chosen[c] = best_name
    return oracle, chosen


# ----------------------------- complementarity (B3') -----------------------------
def complementarity(anchor, sidecar, y):
    """Spearman/Pearson over flattened valid preds; Kuncheva Q-stat + double-fault
    over binarised correctness (pred>0.5 vs label) per class, averaged."""
    valid = ~(np.isnan(anchor) | np.isnan(sidecar)) & (y.sum(0, keepdims=True) > 0)
    a = anchor[valid]; s = sidecar[valid]
    sp = spearmanr(a, s).correlation if len(a) > 2 else np.nan
    pe = pearsonr(a, s)[0] if len(a) > 2 else np.nan

    qs, dfs = [], []
    for c in range(anchor.shape[1]):
        if y[:, c].sum() <= 0:
            continue
        ca, cs = anchor[:, c], sidecar[:, c]
        m = ~(np.isnan(ca) | np.isnan(cs))
        if m.sum() < 4:
            continue
        yt = y[m, c]
        oa = ((ca[m] > 0.5).astype(int) == yt).astype(int)
        os_ = ((cs[m] > 0.5).astype(int) == yt).astype(int)
        N11 = ((oa == 1) & (os_ == 1)).sum()
        N00 = ((oa == 0) & (os_ == 0)).sum()
        N10 = ((oa == 1) & (os_ == 0)).sum()
        N01 = ((oa == 0) & (os_ == 1)).sum()
        denom = N11 * N00 + N01 * N10
        if denom > 0:
            qs.append((N11 * N00 - N01 * N10) / denom)
        dfs.append(N00 / max(len(yt), 1))
    return {
        "spearman": float(sp), "pearson": float(pe),
        "kuncheva_Q": float(np.mean(qs)) if qs else np.nan,
        "double_fault": float(np.mean(dfs)) if dfs else np.nan,
    }


# ----------------------------- clustered bootstrap -----------------------------
def clustered_bootstrap_delta(anchor, oracle, y, authors, n_boot=1000, seed=0):
    """Bootstrap dAUC = macro(oracle) - macro(anchor), resampling AUTHORS, paired
    over a COMMON evaluable class mask recomputed PER REPLICATE (so anchor and
    oracle are always scored on the identical class set within a replicate)."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(authors)))
    idx_by_author = {a: np.where(authors == a)[0] for a in uniq}
    deltas = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_author[a] for a in samp])
        yb = y[rows]
        mask = common_eval_mask(yb, [anchor[rows], oracle[rows]])
        if not mask.any():
            continue
        d = macro_auc_on_mask(yb, oracle[rows], mask) - macro_auc_on_mask(yb, anchor[rows], mask)
        deltas.append(d)
    deltas = np.array(deltas)
    if len(deltas) == 0:
        return {"delta_mean": np.nan, "ci95": [np.nan, np.nan], "se": np.nan,
                "MDE_approx": np.nan, "n_boot": 0, "n_authors": len(uniq)}
    mean = float(deltas.mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    se = float(deltas.std(ddof=1))
    mde = 1.96 * se
    return {"delta_mean": mean, "ci95": [float(lo), float(hi)], "se": se,
            "MDE_approx": float(mde), "n_boot": int(len(deltas)), "n_authors": len(uniq)}


# ----------------------------- self-test -----------------------------
def selftest():
    print("[selftest] synthetic-array validation")
    rng = np.random.default_rng(0)
    N, C = 400, 12
    y = (rng.random((N, C)) < 0.3).astype(np.float32)
    anchor = np.clip(y * 0.7 + rng.normal(0, 0.3, (N, C)), 0, 1)
    sidecar = np.clip(y * 0.7 + rng.normal(0, 0.3, (N, C)), 0, 1)
    sidecar[:, :C // 2] = rng.random((N, C // 2))
    anchor[:, C // 2:] = rng.random((N, C - C // 2))
    authors = np.array([f"a{i % 20}" for i in range(N)])
    folds = np.array([i % 5 for i in range(N)])

    ho_oracle, ho_chosen = heldout_oracle_best_of(anchor, sidecar, y, folds)
    is_oracle, is_chosen = in_sample_selection_biased(anchor, sidecar, y)

    # common mask across anchor, sidecar, held-out oracle, in-sample oracle
    mask = common_eval_mask(y, [anchor, sidecar, ho_oracle, is_oracle])
    a_auc = macro_auc_on_mask(y, anchor, mask)
    s_auc = macro_auc_on_mask(y, sidecar, mask)
    ho_auc = macro_auc_on_mask(y, ho_oracle, mask)
    is_auc = macro_auc_on_mask(y, is_oracle, mask)
    print(f"[selftest] common-mask classes = {int(mask.sum())}/{C}")
    print(f"[selftest] anchor={a_auc:.4f} sidecar={s_auc:.4f} "
          f"held_out_oracle={ho_auc:.4f} in_sample_oracle={is_auc:.4f}")

    # ASSERT 1: held-out oracle <= in-sample oracle (selection bias inflates in-sample)
    assert ho_auc <= is_auc + 1e-9, \
        f"held_out_oracle ({ho_auc}) > in_sample_oracle ({is_auc})"
    # ASSERT 2: in-sample oracle per-class >= max(anchor,sidecar) on common mask
    pa = per_class_auc_on_mask(y, anchor, mask)
    ps = per_class_auc_on_mask(y, sidecar, mask)
    pis = per_class_auc_on_mask(y, is_oracle, mask)
    for c in pis:
        mx = max(pa.get(c, -1), ps.get(c, -1))
        assert pis[c] >= mx - 1e-9, f"in_sample_oracle<max at class {c}: {pis[c]} < {mx}"
    # ASSERT 3: in-sample oracle macro >= each component on the common mask
    assert is_auc >= a_auc - 1e-9 and is_auc >= s_auc - 1e-9, "oracle macro < component"
    print("[selftest] ASSERT held_out<=in_sample OK; oracle>=max(anchor,sidecar) OK")

    fused = calibrated_unclamped_fusion(anchor, sidecar, y, folds)
    fmask = common_eval_mask(y, [anchor, sidecar, fused])
    f_auc = macro_auc_on_mask(y, fused, fmask)
    print(f"[selftest] nested calibrated unclamped fusion macro={f_auc:.4f}")
    comp = complementarity(anchor, sidecar, y)
    print(f"[selftest] complementarity={ {k: round(v,3) for k,v in comp.items()} }")
    boot = clustered_bootstrap_delta(anchor, ho_oracle, y, authors, n_boot=200)
    print(f"[selftest] dAUC(held_out_oracle-anchor)={boot['delta_mean']:.4f} "
          f"CI95={[round(x,4) for x in boot['ci95']]} MDE~{boot['MDE_approx']:.4f}")
    # ASSERT 4: deltas finite
    assert np.isfinite(boot["delta_mean"]) and all(np.isfinite(boot["ci95"])), \
        "bootstrap deltas not finite"
    print("[selftest] ASSERT deltas finite OK")
    print("[selftest] ALL ASSERTS PASSED")


# ----------------------------- real run -----------------------------
def run(args):
    anchor = np.load(args.anchor).astype(np.float64)
    sidecar = np.load(args.sidecar).astype(np.float64)
    y = np.load(args.targets).astype(np.float64)
    meta = pd.read_csv(args.meta)
    authors = meta["author"].values
    folds = meta["fold"].values
    assert anchor.shape == sidecar.shape == y.shape, "matrix shape mismatch"

    # Restrict to rows scored by BOTH models (drop rows that are all-NaN in either,
    # e.g. partial dry OOFs where only some folds were emitted).
    keep = ~(np.isnan(anchor).all(1) | np.isnan(sidecar).all(1))
    if not keep.all():
        print(f"[run] restricting to {int(keep.sum())}/{len(keep)} rows scored by both models")
        anchor, sidecar, y = anchor[keep], sidecar[keep], y[keep]
        authors, folds = authors[keep], folds[keep]

    ho_oracle, ho_chosen = heldout_oracle_best_of(anchor, sidecar, y, folds)
    is_oracle, is_chosen = in_sample_selection_biased(anchor, sidecar, y)
    fused = calibrated_unclamped_fusion(anchor, sidecar, y, folds, seed=args.seed)

    # Headline numbers: common mask across all compared models.
    mask = common_eval_mask(y, [anchor, sidecar, fused, ho_oracle, is_oracle])

    res = {}
    res["n_common_eval_classes"] = int(mask.sum())
    res["anchor_macro_auc"] = macro_auc_on_mask(y, anchor, mask)
    res["sidecar_macro_auc"] = macro_auc_on_mask(y, sidecar, mask)
    res["calibrated_unclamped_macro_auc"] = macro_auc_on_mask(y, fused, mask)
    res["held_out_oracle_macro_auc"] = macro_auc_on_mask(y, ho_oracle, mask)
    res["in_sample_selection_biased_macro_auc"] = macro_auc_on_mask(y, is_oracle, mask)
    # held-out per-fold choice counts (flattened)
    res["held_out_oracle_choice_counts"] = {
        k: int((ho_chosen == k).sum()) for k in ["anchor", "sidecar", "mean", "none"]}
    res["in_sample_choice_counts"] = {
        k: int((is_chosen == k).sum()) for k in ["anchor", "sidecar", "mean", "none"]}
    # Primary endpoint: held-out oracle vs anchor, author-clustered, common mask/replicate.
    res["headroom_heldout"] = clustered_bootstrap_delta(
        anchor, ho_oracle, y, authors, n_boot=args.n_boot, seed=args.seed)
    res["complementarity"] = complementarity(anchor, sidecar, y)

    import json
    print(json.dumps(res, indent=2, default=float))
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(res, fh, indent=2, default=float)
        print(f"[done] wrote {args.out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--anchor"); ap.add_argument("--sidecar")
    ap.add_argument("--targets"); ap.add_argument("--meta")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    assert all([args.anchor, args.sidecar, args.targets, args.meta]), \
        "need --anchor --sidecar --targets --meta (or --selftest)"
    run(args)


if __name__ == "__main__":
    main()
