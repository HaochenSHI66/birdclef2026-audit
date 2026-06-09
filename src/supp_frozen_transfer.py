"""SUPP P7 — FROZEN focal->soundscape transfer (deployability test).

Tests whether a per-class HARD-SELECTION rule and a calibrated logistic-on-logits
FUSION rule, LEARNED on focal CV and FROZEN, actually help on the soundscape TARGET
domain. Directly probes the claim "fusion grows under shift (+0.009 focal CV ->
+0.037 soundscape LB)" with a proper FILE-CLUSTERED bootstrap. NO retraining.

Candidates (4 base models), all (.,234) taxonomy-aligned:
  perch_zs  : Perch-V2 zero-shot anchor
  cnn,pann,beats : fold-4 single checkpoints (trained folds 0-3, hold fold 4)

LEARN-ON-FOCAL set = the fold-4 held-out OOF of those SAME models (apples-to-apples
with the fold-4 checkpoints deployed on soundscape):
  perch_zs : perch_zeroshot_oof.npy[fold==4]
  cnn      : cnnrg_fold4_seed42.npy  (exact checkpoint that produced ss_cnnrg_preds)
  pann     : pann_fold4_seed42.npy   (standard fold-4 OOF; same config/seed as pannrg)
  beats    : beats_fold4_seed42.npy  (standard fold-4 OOF; same config/seed as beatsrg)

APPLY-ON-SOUNDSCAPE set (66 files / 1478 windows, labeled):
  perch_zs : ss_perch_labelhead.npy -> mapped max -> 234 (same map as focal anchor)
  cnn,pann,beats : ss_{m}rg_preds.npy , columns masked to trained classes only

Rules learned on focal, FROZEN, applied to soundscape:
  (a) hard selector : choice[c] = argmax_model focal-CV per-class AUC ; apply that
      model's soundscape column. (base-only; a +mean variant also recorded.)
  (b) fusion        : per-class LogisticRegression on [logit(model)] fit on focal,
      frozen coefs applied to soundscape logits. (== paper's calibrated stacker.)

Eval on soundscape vs soundscape labels, macro-AUC over a FIXED common evaluable
class mask; ΔAUC (selector-best_single) & (fusion-best_single) with FILE-CLUSTERED
bootstrap 95% CI (resample the 66 files). Verdict: do frozen selector / frozen fusion
HELP on the target domain? supports vs contradicts "fusion grows under shift".

Output: data/oof/supp_frozen_transfer.json
"""
import csv, json, os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/SHC/birdclef2026_clef")
sys.path.insert(0, os.path.join(ROOT, "src"))
from b2_oracle_headroom import common_eval_mask
from b2_headline_driver import vec_auc_per_class
from b3_domain_shift import load_perch_labels, build_map
from sklearn.linear_model import LogisticRegression

OOF = os.path.join(ROOT, "data", "oof")
RG = os.path.join(ROOT, "data", "oof_rg")
D = os.path.join(ROOT, "data")
EPS = 1e-6
N_BOOT = 2000
SEED = 0
MODELS = ["perch_zs", "cnn", "pann", "beats"]


def logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def macro_common(y, mats):
    """Macro-AUC for each matrix in mats over classes defined (AUC computable) in
    EVERY matrix. Returns (list_of_macros, common_bool[C], n_common)."""
    aucs = [vec_auc_per_class(y, m)[0] for m in mats]
    common = np.all([~np.isnan(a) for a in aucs], axis=0)
    if common.sum() == 0:
        return [np.nan] * len(mats), common, 0
    return [float(np.nanmean(a[common])) for a in aucs], common, int(common.sum())


def main():
    t0 = time.time()
    meta = pd.read_csv(os.path.join(OOF, "oof_meta.csv"))
    folds = meta["fold"].values
    f4 = folds == 4
    N = len(meta)

    # ----- targets / trained-class mask -----
    Yf_full = np.load(os.path.join(OOF, "oof_targets.npy")).astype(np.float64)
    tr_pos = (Yf_full[folds != 4].sum(0) > 0)          # classes sidecars trained on
    print(f"[load] N={N} fold4={int(f4.sum())} trained_classes={int(tr_pos.sum())}", flush=True)

    # ----- FOCAL fold-4 matched matrices (6922 rows) -----
    def scatter(path_pred, path_rows):
        M = np.full((N, 234), np.nan, np.float64)
        M[np.load(path_rows)] = np.load(path_pred).astype(np.float64)
        return M[f4]
    foc = {}
    foc["perch_zs"] = np.load(os.path.join(OOF, "perch_zeroshot_oof.npy")).astype(np.float64)[f4]
    foc["cnn"] = scatter(os.path.join(RG, "cnnrg_fold4_seed42.npy"),
                         os.path.join(RG, "cnnrg_fold4_seed42_rows.npy"))
    foc["pann"] = scatter(os.path.join(OOF, "pann_fold4_seed42.npy"),
                          os.path.join(OOF, "pann_fold4_seed42_rows.npy"))
    foc["beats"] = scatter(os.path.join(OOF, "beats_fold4_seed42.npy"),
                           os.path.join(OOF, "beats_fold4_seed42_rows.npy"))
    yf = Yf_full[f4]
    # mask sidecar untrained columns -> NaN (perch_zs left as-is; unmapped handled by M)
    for m in ["cnn", "pann", "beats"]:
        foc[m][:, ~tr_pos] = np.nan
    for m in MODELS:
        assert foc[m].shape == (int(f4.sum()), 234), (m, foc[m].shape)

    # ----- SOUNDSCAPE matrices (1478 windows) -----
    tax = pd.read_csv(os.path.join(D, "taxonomy.csv"))
    class_order = tax["primary_label"].astype(str).tolist()
    cls_idx = {c: i for i, c in enumerate(class_order)}
    lab = pd.read_csv(os.path.join(D, "train_soundscapes_labels.csv"))
    W = len(lab)
    files = lab["filename"].values
    Yss = np.zeros((W, 234), np.float64)
    for r, row in enumerate(lab.itertuples(index=False)):
        for c in str(row.primary_label).split(";"):
            c = c.strip()
            if c in cls_idx:
                Yss[r, cls_idx[c]] = 1.0
    # perch zero-shot soundscape (same mapping as focal anchor)
    ebird, inat = load_perch_labels(os.path.join(D, "perch_labels"))
    mapping = build_map(tax, ebird, inat)
    lh = np.load(os.path.join(OOF, "ss_perch_labelhead.npy"))
    probs = 1.0 / (1.0 + np.exp(-lh.astype(np.float64)))
    ss = {}
    pz = np.full((W, 234), np.nan, np.float64)
    for c, idxs in enumerate(mapping):
        if idxs:
            pz[:, c] = probs[:, idxs].max(axis=1)
    ss["perch_zs"] = pz
    ss["cnn"] = np.load(os.path.join(RG, "ss_cnnrg_preds.npy")).astype(np.float64)
    ss["pann"] = np.load(os.path.join(RG, "ss_pannrg_preds.npy")).astype(np.float64)
    ss["beats"] = np.load(os.path.join(RG, "ss_beatsrg_preds.npy")).astype(np.float64)
    for m in ["cnn", "pann", "beats"]:
        ss[m][:, ~tr_pos] = np.nan
    print(f"[ss] W={W} files={len(set(files))} ss_pos_classes={int((Yss.sum(0)>0).sum())}", flush=True)

    # ----- FIXED soundscape common evaluable mask over the 4 base models -----
    M = common_eval_mask(Yss, [ss[m] for m in MODELS])
    cols = np.where(M)[0]
    nM = int(M.sum())
    print(f"[mask] common evaluable classes on soundscape (4 base models) = {nM}", flush=True)
    assert nM > 0

    # ----- per-model soundscape macro-AUC over M ; best single -----
    ss_macro = {m: float(np.nanmean(vec_auc_per_class(Yss, ss[m])[0][M])) for m in MODELS}
    best_name = max(ss_macro, key=ss_macro.get)
    print(f"[best] soundscape per-model macro over M: "
          + " ".join(f"{m}={ss_macro[m]:.4f}" for m in MODELS)
          + f"  -> best_single={best_name}", flush=True)

    # ----- LEARN selector on focal: per-class argmax focal AUC (base-only + +mean) -----
    foc_auc = {m: vec_auc_per_class(yf, foc[m])[0] for m in MODELS}
    foc_mean = np.mean([foc[m] for m in MODELS], axis=0)
    foc_auc["mean"] = vec_auc_per_class(yf, foc_mean)[0]
    ss_mean = np.nanmean(np.stack([ss[m] for m in MODELS]), axis=0)  # nan-aware mean for fallback/+mean

    sel_base = np.full((W, 234), np.nan, np.float64)
    sel_mean = np.full((W, 234), np.nan, np.float64)
    choice_base, choice_mean = {}, {}
    for c in cols:
        # base-only
        cand = [(m, foc_auc[m][c]) for m in MODELS if np.isfinite(foc_auc[m][c])]
        pick = max(cand, key=lambda x: x[1])[0] if cand else best_name
        choice_base[int(c)] = pick
        sel_base[:, c] = ss[pick][:, c]
        # +mean variant
        cand2 = cand + ([("mean", foc_auc["mean"][c])] if np.isfinite(foc_auc["mean"][c]) else [])
        pick2 = max(cand2, key=lambda x: x[1])[0] if cand2 else best_name
        choice_mean[int(c)] = pick2
        sel_mean[:, c] = ss_mean[:, c] if pick2 == "mean" else ss[pick2][:, c]

    # ----- LEARN fusion on focal: per-class logistic on logits, FROZEN -> soundscape -----
    Lf = {m: logit(foc[m]) for m in MODELS}
    Ls = {m: logit(ss[m]) for m in MODELS}
    fus = np.full((W, 234), np.nan, np.float64)
    n_fit, n_fb = 0, 0
    for c in cols:
        ytr = yf[:, c]
        Xtr = np.stack([Lf[m][:, c] for m in MODELS], 1)
        valid = ~np.isnan(Xtr).any(1)
        if ytr[valid].sum() >= 2 and len(np.unique(ytr[valid])) >= 2:
            try:
                clf = LogisticRegression(max_iter=500, C=1.0)
                clf.fit(Xtr[valid], ytr[valid])
                Xo = np.stack([Ls[m][:, c] for m in MODELS], 1)
                fus[:, c] = clf.predict_proba(Xo)[:, 1]
                n_fit += 1
                continue
            except Exception:
                pass
        fus[:, c] = ss_mean[:, c]   # frozen label-free fallback when class unfittable on focal
        n_fb += 1
    print(f"[fusion] per-class stackers fit={n_fit} fallback(mean)={n_fb}", flush=True)

    best = ss[best_name]
    # ----- point estimates over M -----
    (m_best, m_selb, m_selm, m_fus), common0, nc0 = macro_common(
        Yss, [best, sel_base, sel_mean, fus])
    print(f"[point] best={m_best:.4f} sel_base={m_selb:.4f} sel_mean={m_selm:.4f} "
          f"fusion={m_fus:.4f} (common defined classes={nc0})", flush=True)

    # ----- FILE-CLUSTERED bootstrap (resample 66 files) -----
    uf = np.array(sorted(set(files)))
    idx_by_file = {f: np.where(files == f)[0] for f in uf}
    rng = np.random.default_rng(SEED)
    bcols = cols
    yb_cols = Yss[:, bcols]
    Bb, Sb, Sm, Fb = best[:, bcols], sel_base[:, bcols], sel_mean[:, bcols], fus[:, bcols]
    d_sel, d_selm, d_fus = [], [], []
    mac = {"best": [], "sel_base": [], "sel_mean": [], "fusion": []}
    for b in range(N_BOOT):
        samp = rng.choice(uf, len(uf), replace=True)
        rows = np.concatenate([idx_by_file[f] for f in samp])
        yy = yb_cols[rows]
        a_b = vec_auc_per_class(yy, Bb[rows])[0]
        a_sb = vec_auc_per_class(yy, Sb[rows])[0]
        a_sm = vec_auc_per_class(yy, Sm[rows])[0]
        a_f = vec_auc_per_class(yy, Fb[rows])[0]
        defc = ~np.isnan(a_b) & ~np.isnan(a_sb) & ~np.isnan(a_sm) & ~np.isnan(a_f)
        if defc.sum() == 0:
            continue
        mb = a_b[defc].mean(); msb = a_sb[defc].mean()
        msm = a_sm[defc].mean(); mf = a_f[defc].mean()
        mac["best"].append(mb); mac["sel_base"].append(msb)
        mac["sel_mean"].append(msm); mac["fusion"].append(mf)
        d_sel.append(msb - mb); d_selm.append(msm - mb); d_fus.append(mf - mb)
        if (b + 1) % 250 == 0:
            print(f"  [boot] {b+1}/{N_BOOT} dsel~{np.mean(d_sel):+.4f} "
                  f"dfus~{np.mean(d_fus):+.4f} ({time.time()-t0:.0f}s)", flush=True)

    def ci(x):
        x = np.array(x)
        return {"mean": float(x.mean()), "ci95": [float(np.percentile(x, 2.5)),
                float(np.percentile(x, 97.5))], "se": float(x.std(ddof=1)),
                "n_boot": int(len(x))}

    cisel, ciselm, cifus = ci(d_sel), ci(d_selm), ci(d_fus)

    def helps(c):
        return bool(c["ci95"][0] > 0)
    sel_helps = helps(cisel)
    fus_helps = helps(cifus)

    FOCAL_FUSION_CLAIM = 0.009    # paper's focal-CV fusion headroom (held-out OOF)
    LB_FUSION_CLAIM = 0.037       # paper's soundscape LB fusion delta (single point)
    # supports "fusion grows under shift" only if frozen fusion HELPS on target AND
    # the target delta is at least as large as the in-domain focal-CV delta.
    supports_grows = bool(fus_helps and cifus["mean"] >= FOCAL_FUSION_CLAIM)

    if fus_helps and supports_grows:
        verdict = ("SUPPORTS: frozen focal-learned fusion significantly helps on the "
                   "soundscape target (CI excludes 0) by >= the focal-CV delta -> "
                   "'fusion grows under shift' is corroborated by a proper file-bootstrap.")
    elif fus_helps and not supports_grows:
        verdict = ("PARTIAL: frozen fusion helps on target (CI>0) but the target delta is "
                   "NOT larger than the focal-CV delta -> does not establish 'grows under "
                   "shift'; keep only as a modest deployability note, drop the 'grows' framing.")
    else:
        verdict = ("CONTRADICTS: frozen focal-learned fusion does NOT significantly help on "
                   "the soundscape target (95% file-clustered CI includes 0) -> the "
                   "'+0.009 focal -> +0.037 soundscape, fusion grows under shift' claim is "
                   "NOT supported under a proper file-bootstrapped frozen-transfer eval. "
                   "DELETE the claim (do not soften).")

    out = {
        "task": "frozen_focal_to_soundscape_selector_fusion_transfer",
        "design": {
            "candidates": MODELS,
            "learn_on": "focal fold-4 held-out OOF of the same models (apples-to-apples "
                        "with the fold-4 checkpoints deployed on soundscape)",
            "apply_on": "66-file / 1478-window labeled soundscape",
            "selector": "per-class argmax of focal-CV per-class AUC (base-only primary; "
                        "+mean variant recorded), FROZEN, applied to soundscape columns",
            "fusion": "per-class LogisticRegression on [logit(model)] fit on focal, FROZEN, "
                      "applied to soundscape logits (paper's calibrated logit stacker)",
            "best_single": best_name,
            "mask": "FIXED soundscape common evaluable class mask (>=1 pos & >=1 neg & all "
                    "4 base models non-NaN); recomputed-per-replicate only for AUC-definedness",
            "bootstrap": f"file-clustered, resample {len(uf)} files, n_boot={N_BOOT}",
            "n_focal_rows": int(f4.sum()),
            "n_soundscape_windows": int(W),
            "n_soundscape_files": int(len(uf)),
            "n_common_eval_classes": nM,
            "fusion_classes_fit": n_fit, "fusion_classes_fallback_mean": n_fb,
            "artifact_provenance": {
                "perch_zs_focal": "perch_zeroshot_oof.npy[fold==4]",
                "cnn_focal": "oof_rg/cnnrg_fold4_seed42.npy (exact deployed checkpoint)",
                "pann_focal": "pann_fold4_seed42.npy (same config/seed as pannrg ckpt)",
                "beats_focal": "beats_fold4_seed42.npy (same config/seed as beatsrg ckpt)",
                "perch_zs_ss": "ss_perch_labelhead.npy mapped (max) -> 234",
                "cnn_ss": "oof_rg/ss_cnnrg_preds.npy",
                "pann_ss": "oof_rg/ss_pannrg_preds.npy",
                "beats_ss": "oof_rg/ss_beatsrg_preds.npy",
            },
        },
        "soundscape_macro_auc_over_mask": {
            "per_model": {m: round(ss_macro[m], 4) for m in MODELS},
            "best_single": round(m_best, 4),
            "frozen_hard_selector_base": round(m_selb, 4),
            "frozen_hard_selector_with_mean": round(m_selm, 4),
            "frozen_fusion": round(m_fus, 4),
            "n_common_defined_classes_point": nc0,
        },
        "delta_vs_best_single_file_clustered_bootstrap": {
            "frozen_hard_selector_base": cisel,
            "frozen_hard_selector_with_mean": ciselm,
            "frozen_fusion": cifus,
        },
        "selector_choice_counts_base": {
            k: sum(1 for v in choice_base.values() if v == k) for k in MODELS},
        "selector_choice_counts_with_mean": {
            k: sum(1 for v in choice_mean.values() if v == k) for k in MODELS + ["mean"]},
        "paper_claim_reference": {
            "focal_cv_fusion_delta": FOCAL_FUSION_CLAIM,
            "soundscape_lb_fusion_delta": LB_FUSION_CLAIM,
            "note": "Existing claim '+0.009 focal CV -> +0.037 soundscape LB' rests on "
                    "single LB points; this experiment re-tests it with a file-bootstrap.",
        },
        "verdict": {
            "frozen_selector_helps_target": sel_helps,
            "frozen_fusion_helps_target": fus_helps,
            "supports_fusion_grows_under_shift": supports_grows,
            "statement": verdict,
        },
        "runtime_min": round((time.time() - t0) / 60, 2),
    }
    out_path = os.path.join(OOF, "supp_frozen_transfer.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nWROTE", out_path, flush=True)
    print(json.dumps(out["soundscape_macro_auc_over_mask"], indent=2), flush=True)
    print(json.dumps(out["delta_vs_best_single_file_clustered_bootstrap"], indent=2), flush=True)
    print("VERDICT:", verdict, flush=True)
    print(f"[done] {out['runtime_min']:.2f} min", flush=True)


if __name__ == "__main__":
    main()
