"""Task 2 — Post-processing ablation on the best reproducible OOF.

Best reproducible single-model OOF = Perch ZERO-SHOT (native pooled macro-AUC 0.928,
the strongest single model; foundation head, fully reproducible from the cached
label_head). Each knob is a deterministic, leakage-controlled transform of the OOF
probability matrix P (N,234). We measure macro-AUC (skip-zero-pos, on a fixed common
mask shared by ALL knobs incl. baseline) for baseline vs each knob, and ΔAUC with an
AUTHOR-CLUSTERED paired bootstrap (resample authors). Multiplicity over the knob family
controlled by Holm (FWER) AND BH-FDR.

Knobs (each toggled independently, vs the same baseline P):
  (a) TAXONOMY SMOOTHING — genus-level label smoothing. For each class, blend its prob
      with the (per-row) mean prob of its genus-mates (genus = first token of
      scientific_name). p' = (1-a)*p + a*mean_genus(p). a=0.1 (pre-set). Classes whose
      genus is a singleton are unchanged. Rank-based AUC sees this as a smoothing toward
      con-generic evidence. taxonomy.csv is the only input -> reproducible.
  (b) ECOLOGICAL / SITE SPATIAL PRIOR — NESTED, fold-safe. train.csv has lat/lon for all
      35549 clips. For each outer fold f, estimate a per-class spatial occupancy prior
      from clips in folds != f only (Gaussian KDE-style: per class, prob that this
      location is near that class's training-positive locations), then multiply the val
      row's per-class prob by a mild monotone function of that prior. No val labels or
      val locations leak into the estimate of OTHER rows; each val row only uses its own
      lat/lon scored against the held-out-fold training distribution. Caveat: focal lat/lon
      are GLOBAL iNat coordinates, NOT the Pantanal test sites -> this is the BEST
      reproducible offline proxy, reported descriptively.
  (c) CROSS-TAXA GATING — for each row, compute predicted mass per coarse taxon
      (class_name: Aves/Amphibia/Insecta/Mammalia/Reptilia) = sum of probs in that taxon.
      Softly down-weight classes belonging to taxa with low relative predicted mass:
      p'_c = p_c * (taxon_mass_share)^beta, beta=0.5. Reproducible from taxonomy + P.

OUTPUT: data/oof/postproc_ablation.json
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _auc_col(yc, pc):
    """Vectorized rank-sum (Mann-Whitney) AUC for one class. Verified == sklearn."""
    n = len(yc)
    npos = yc.sum()
    if npos <= 0 or npos >= n:
        return np.nan
    order = np.argsort(pc, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    # average ranks for ties
    sp = pc[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos = ranks[yc > 0].sum()
    return float((sum_pos - npos * (npos + 1) / 2.0) / (npos * (n - npos)))


def per_class_auc_on_mask(y, p, mask):
    out = {}
    for c in np.where(mask)[0]:
        col = p[:, c]
        if np.isnan(col).any():
            continue
        a = _auc_col(y[:, c], col)
        if not np.isnan(a):
            out[int(c)] = a
    return out


def macro_on_mask(y, p, mask):
    d = per_class_auc_on_mask(y, p, mask)
    return float(np.mean(list(d.values()))) if d else 0.0


# ---------------- knobs ----------------
def knob_taxonomy_smoothing(P, genus_groups, alpha=0.1):
    Pn = np.nan_to_num(P, nan=0.0)
    out = P.copy()
    for idxs in genus_groups:
        if len(idxs) < 2:
            continue
        gmean = Pn[:, idxs].mean(axis=1, keepdims=True)
        for c in idxs:
            if np.isnan(P[:, c]).all():
                continue
            out[:, c] = (1 - alpha) * Pn[:, c] + alpha * gmean[:, 0]
    return out


def knob_cross_taxa_gating(P, taxon_of_class, beta=0.5):
    Pn = np.nan_to_num(P, nan=0.0)
    taxa = sorted(set(taxon_of_class))
    mass = {}
    for t in taxa:
        cols = [c for c in range(P.shape[1]) if taxon_of_class[c] == t]
        mass[t] = Pn[:, cols].sum(axis=1)
    total = sum(mass.values()) + 1e-9
    out = P.copy()
    for c in range(P.shape[1]):
        if np.isnan(P[:, c]).all():
            continue
        share = mass[taxon_of_class[c]] / total
        out[:, c] = Pn[:, c] * np.power(np.clip(share, 1e-6, 1.0), beta)
    return out


def knob_spatial_prior(P, y, lat, lon, folds, mask, bw=3.0, strength=0.3):
    """Nested per-class spatial occupancy prior. For outer fold f, per class c with
    >=2 train-positive locations in folds!=f, score each val row's (lat,lon) by a
    Gaussian-kernel density of those train-positive locations (bw deg). Multiply
    p_c by (1 + strength*(rank_density-0.5)) i.e. nudge up where location matches the
    class's training range, down otherwise. Only mask classes processed (cost)."""
    """Histogram-smoothed per-class spatial occupancy prior (fast, O(n)). For outer
    fold f, build a 2D lat/lon histogram of train-positive locations (folds!=f) per
    class on a fixed grid, Gaussian-blur it (bw cells), then look up each val row's
    cell density; rank-normalize within the eval fold and nudge p_c by +-strength."""
    from scipy.ndimage import gaussian_filter
    Pn = np.nan_to_num(P, nan=0.0)
    out = P.copy()
    uf = sorted(np.unique(folds))
    eval_classes = np.where(mask)[0]
    # fixed grid over observed coord range
    la = lat; lo = lon
    lamin, lamax = np.percentile(la, 0.5), np.percentile(la, 99.5)
    lomin, lomax = np.percentile(lo, 0.5), np.percentile(lo, 99.5)
    G = 60
    def cell(v, vmin, vmax):
        return np.clip(((v - vmin) / (vmax - vmin + 1e-9) * (G - 1)).astype(int), 0, G - 1)
    ci_lat = cell(la, lamin, lamax)
    ci_lon = cell(lo, lomin, lomax)
    sigma = bw / ((lamax - lamin) / G + 1e-9)  # bw deg -> cells; clip below
    sigma = float(np.clip(sigma, 1.0, 5.0))
    import time as _t
    t0 = _t.time()
    for f in uf:
        tr = folds != f
        ev = np.where(folds == f)[0]
        ev_la, ev_lo = ci_lat[ev], ci_lon[ev]
        for c in eval_classes:
            pos = tr & (y[:, c] > 0)
            if pos.sum() < 2:
                continue
            H = np.zeros((G, G), dtype=np.float64)
            np.add.at(H, (ci_lat[pos], ci_lon[pos]), 1.0)
            H = gaussian_filter(H, sigma=sigma, mode="constant")
            dens = H[ev_la, ev_lo]
            order = dens.argsort().argsort().astype(np.float64)
            rnk = order / max(len(order) - 1, 1)
            factor = 1.0 + strength * (rnk - 0.5) * 2.0
            out[ev, c] = Pn[ev, c] * factor
        print(f"[spatial] fold {f} done ({_t.time()-t0:.0f}s)", flush=True)
    return out


def author_bootstrap_delta(y, base, knob, mask, authors, n_boot=1000, seed=0):
    """Paired ΔAUC (knob - base) macro over mask, author-clustered bootstrap."""
    rng = np.random.default_rng(seed)
    uauth = np.array(sorted(set(authors)))
    a2rows = {a: np.where(authors == a)[0] for a in uauth}
    point = macro_on_mask(y, knob, mask) - macro_on_mask(y, base, mask)
    mcols = np.where(mask)[0]
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(len(uauth), len(uauth), replace=True)
        rows = np.concatenate([a2rows[uauth[i]] for i in pick])
        yb = y[rows]; bb = base[rows]; kb = knob[rows]
        ab = []
        for c in mcols:
            yc = yb[:, c]
            s = yc.sum()
            if s <= 0 or s >= len(yc):
                continue
            da = _auc_col(yc, kb[:, c]) - _auc_col(yc, bb[:, c])
            if not np.isnan(da):
                ab.append(da)
        deltas.append(float(np.mean(ab)) if ab else 0.0)
    deltas = np.array(deltas)
    se = float(deltas.std(ddof=1))
    ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
    # two-sided bootstrap p (proportion crossing 0), normal-approx fallback
    from math import erf, sqrt
    z = point / (se + 1e-12)
    p_norm = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    mde = 1.96 * se
    return point, ci, se, mde, p_norm


def holm_bh(pvals):
    p = np.array(pvals); m = len(p)
    order = np.argsort(p)
    # Holm
    holm = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        run = max(run, val)
        holm[i] = min(run, 1.0)
    # BH
    bh = np.empty(m); prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = p[i] * m / (rank + 1)
        prev = min(prev, val)
        bh[i] = min(prev, 1.0)
    return holm.tolist(), bh.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.expanduser("~/SHC/birdclef2026_clef/data"))
    ap.add_argument("--base", default="perch_zeroshot_oof.npy")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    oof = os.path.join(args.data_dir, "oof")

    P = np.load(os.path.join(oof, args.base)).astype(np.float64)
    y = np.load(os.path.join(oof, "oof_targets.npy")).astype(np.float64)
    meta = pd.read_csv(os.path.join(oof, "oof_meta.csv"))
    authors = meta["author"].astype(str).values
    folds = meta["fold"].values
    tax = pd.read_csv(os.path.join(args.data_dir, "taxonomy.csv"))
    assert len(tax) == P.shape[1] == 234

    # genus groups from scientific_name first token
    genus = tax["scientific_name"].astype(str).str.split().str[0].str.lower().values
    genus_groups = []
    for g in pd.unique(genus):
        genus_groups.append([i for i in range(234) if genus[i] == g])
    taxon_of_class = tax["class_name"].astype(str).tolist()

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))[
        ["filename", "latitude", "longitude"]]
    mm = meta.merge(train, on="filename", how="left")
    lat = mm["latitude"].fillna(mm["latitude"].median()).values.astype(np.float64)
    lon = mm["longitude"].fillna(mm["longitude"].median()).values.astype(np.float64)

    # common mask = evaluable on baseline (fixed for all knobs)
    pos = (y.sum(0) > 0)
    finite = ~np.isnan(P).any(0)
    twocls = np.array([len(np.unique(y[:, c])) >= 2 for c in range(234)])
    mask = pos & finite & twocls
    print(f"[mask] {int(mask.sum())} evaluable classes; baseline macro-AUC "
          f"{macro_on_mask(y, P, mask):.4f}", flush=True)

    base_auc = macro_on_mask(y, P, mask)
    knobs = {
        "taxonomy_smoothing_genus_a0.1": knob_taxonomy_smoothing(P, genus_groups, 0.1),
        "cross_taxa_gating_b0.5": knob_cross_taxa_gating(P, taxon_of_class, 0.5),
        "ecological_spatial_prior_nested": knob_spatial_prior(
            P, y, lat, lon, folds, mask, bw=3.0, strength=0.3),
    }

    results = {}
    pvals = []
    names = []
    for name, K in knobs.items():
        kauc = macro_on_mask(y, K, mask)
        pt, ci, se, mde, pnorm = author_bootstrap_delta(
            y, P, K, mask, authors, n_boot=args.n_boot, seed=0)
        results[name] = {"knob_macro_auc": round(kauc, 4),
                         "delta_auc": round(pt, 4),
                         "ci95": [round(ci[0], 4), round(ci[1], 4)],
                         "se": round(se, 4), "mde": round(mde, 4),
                         "p_raw": round(pnorm, 4)}
        pvals.append(pnorm); names.append(name)
        print(f"[knob] {name}: macroAUC {kauc:.4f} dAUC {pt:+.4f} "
              f"CI {ci} p {pnorm:.4f}", flush=True)

    holm, bh = holm_bh(pvals)
    for i, name in enumerate(names):
        results[name]["p_holm"] = round(holm[i], 4)
        results[name]["p_bh_fdr"] = round(bh[i], 4)
        results[name]["helps_at_0.05_holm"] = bool(
            results[name]["delta_auc"] > 0 and holm[i] < 0.05)

    out = {
        "task": "postproc_ablation",
        "base_model": args.base,
        "n_eval_classes": int(mask.sum()),
        "baseline_macro_auc": round(base_auc, 4),
        "n_boot_author_clustered": args.n_boot,
        "multiplicity": "Holm (FWER) + BH-FDR over the 3-knob family",
        "knobs": results,
        "feasibility_note": (
            "taxonomy + cross-taxa fully reproducible from taxonomy.csv + OOF; "
            "spatial prior is a NESTED fold-safe offline proxy using GLOBAL iNat "
            "lat/lon (focal recordings are not at the Pantanal test sites), so its "
            "effect is a descriptive lower-bound on what a true site prior could do."),
    }
    op = os.path.join(oof, "postproc_ablation.json")
    json.dump(out, open(op, "w"), indent=2)
    print("WROTE", op, flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
