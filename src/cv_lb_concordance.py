"""Task 1 — CV<->LB concordance (Spearman) on reproducible configs.

HONESTY FIRST. The 11 submitted Kaggle configs (SEED.md, 6 distinct private scores)
are MOSTLY the community EoS8 stack (Perch V2 + ProtoSSM + PowerOpt + ecological priors
+ taxonomy smoothing) and its sidecar variants, all clamped to ~0.006 movement. None of
those EoS8 variants are reproducible locally (no per-fold community anchor, C14). So a
1:1 CV<->LB rank correlation across all 11 configs is NOT possible.

What IS defensible: a SMALL set of configs for which we have BOTH (i) a known private-LB
score and (ii) a local pooled-OOF macro-AUC computed on the SAME competition metric, where
the local model is a faithful proxy of the submitted config. We enumerate the mapping
explicitly, mark each config reproducible y/n, and compute Spearman + Kendall over the
reproducible subset only. Given the tiny n this is a SANITY CHECK, not validation of local
effect estimates (late LB closed, C3; per exp-plan-v4 we explicitly do NOT claim LB
validates local effects).

Reproducible-subset mapping (best faith, documented):
  - "Perch standalone (anchor)"  <- private 0.70571 (submitted: "Perch V2 + own ProtoSSM
       standalone"). Local proxy = Perch ZERO-SHOT pooled CV (closest reproducible
       Perch-only head; ProtoSSM not reproducible, noted).
  - "Own CNN standalone"         <- private 0.78711 (submitted: "Own NFNet standalone").
       Local proxy = author-CNN ensemble pooled CV (own from-scratch acoustic CNN;
       NFNet vs timm arch differs, noted).
  - "Own ensemble standalone"    <- private 0.79142 (submitted: "Own 6-model ONNX
       ensemble"). Local proxy = CNN+PANN+BEATs mean pooled CV (own multi-model acoustic
       ensemble).
  - "Community EoS8 (best)"       <- private 0.94247. NOT reproducible (community stack).
       Local proxy = the best reproducible BLEND we have (calibrated mean of all locals)
       as an UPPER analog ONLY, flagged non-comparable; INCLUDED in an optional extended
       correlation but the headline rho uses the 3 strictly-own configs.

We report rho over (A) the 3 strictly-reproducible own configs and (B) the 4-point set
incl. the community analog, with the n and the caveat for each.

OUTPUT: data/oof/cv_lb_concordance.json
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score


def macro_auc(y, p):
    aucs = []
    for c in range(y.shape[1]):
        if y[:, c].sum() <= 0:
            continue
        col = p[:, c]
        if np.isnan(col).any() or len(np.unique(y[:, c])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y[:, c], col))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


def macro_auc_common(y, preds):
    """macro-AUC of each pred on a COMMON mask (apples-to-apples)."""
    C = y.shape[1]
    pos = y.sum(0) > 0
    finite = np.ones(C, bool)
    for p in preds:
        finite &= ~np.isnan(p).any(0)
    two = np.array([len(np.unique(y[:, c])) >= 2 for c in range(C)])
    mask = pos & finite & two
    outs = []
    for p in preds:
        a = []
        for c in np.where(mask)[0]:
            try:
                a.append(roc_auc_score(y[:, c], p[:, c]))
            except Exception:
                pass
        outs.append(float(np.mean(a)) if a else 0.0)
    return outs, int(mask.sum())


def main():
    data_dir = os.path.expanduser("~/SHC/birdclef2026_clef/data")
    oof = os.path.join(data_dir, "oof")
    y = np.load(os.path.join(oof, "oof_targets.npy")).astype(np.float64)

    zs = np.load(os.path.join(oof, "perch_zeroshot_oof.npy")).astype(np.float64)
    cnn = np.load(os.path.join(oof, "cnn_ensemble_oof.npy")).astype(np.float64)
    pann = np.load(os.path.join(oof, "pann_ensemble_oof.npy")).astype(np.float64)
    beats = np.load(os.path.join(oof, "beats_ensemble_oof.npy")).astype(np.float64)

    own_ens = np.nanmean(np.stack([cnn, pann, beats]), axis=0)
    # community analog = calibrated-ish blend of all locals (mean of zs + own_ens)
    comm_analog = np.nanmean(np.stack([zs, cnn, pann, beats]), axis=0)

    # native pooled macro-AUC (self-mask) for each config proxy
    cv = {
        "perch_standalone": macro_auc(y, np.nan_to_num(zs, nan=0.0)),
        "own_cnn_standalone": macro_auc(y, cnn),
        "own_ensemble_standalone": macro_auc(y, own_ens),
        "community_eos8_analog": macro_auc(y, np.nan_to_num(comm_analog, nan=0.0)),
    }

    lb = {
        "perch_standalone": 0.70571,
        "own_cnn_standalone": 0.78711,
        "own_ensemble_standalone": 0.79142,
        "community_eos8_analog": 0.94247,
    }
    reproducible = {
        "perch_standalone": True,
        "own_cnn_standalone": True,
        "own_ensemble_standalone": True,
        "community_eos8_analog": False,  # community stack not reproducible
    }

    # headline: strictly reproducible own configs
    own_keys = [k for k in cv if reproducible[k]]
    cv_own = [cv[k] for k in own_keys]
    lb_own = [lb[k] for k in own_keys]
    rho_own, p_own = spearmanr(cv_own, lb_own)
    tau_own, pt_own = kendalltau(cv_own, lb_own)

    # extended: include community analog (flagged non-comparable)
    all_keys = list(cv.keys())
    cv_all = [cv[k] for k in all_keys]
    lb_all = [lb[k] for k in all_keys]
    rho_all, p_all = spearmanr(cv_all, lb_all)
    tau_all, pt_all = kendalltau(cv_all, lb_all)

    out = {
        "task": "cv_lb_concordance",
        "feasible": "partial",
        "configs": [
            {"name": k, "local_cv_pooled_macro_auc": round(cv[k], 4),
             "private_lb": lb[k], "reproducible": reproducible[k]}
            for k in all_keys
        ],
        "headline_reproducible_only": {
            "n": len(own_keys),
            "configs": own_keys,
            "spearman_rho": round(float(rho_own), 4),
            "spearman_p": round(float(p_own), 4),
            "kendall_tau": round(float(tau_own), 4),
            "kendall_p": round(float(pt_own), 4),
        },
        "extended_with_community_analog": {
            "n": len(all_keys),
            "spearman_rho": round(float(rho_all), 4),
            "spearman_p": round(float(p_all), 4),
            "kendall_tau": round(float(tau_all), 4),
        },
        "caveat": (
            "n is TINY (3 strictly-reproducible configs; 4 incl. a non-comparable "
            "community analog). The 11 submitted configs are mostly the community EoS8 "
            "stack which is NOT reproducible locally (C14). Local proxies differ from "
            "the exact submitted models (Perch-ZS vs Perch+ProtoSSM; timm-CNN vs NFNet; "
            "local 3-model vs 6-model ONNX). Late LB is CLOSED (C3). Per exp-plan-v4 this "
            "is a SANITY CHECK ONLY; we explicitly do NOT claim LB validates local effect "
            "estimates. With n=3 a permutation test has only 3!=6 orderings (min two-sided "
            "p=0.33), so significance is unattainable by construction; the rho is "
            "descriptive ordering agreement, not an inferential claim."),
    }
    op = os.path.join(oof, "cv_lb_concordance.json")
    json.dump(out, open(op, "w"), indent=2)
    print("WROTE", op)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
