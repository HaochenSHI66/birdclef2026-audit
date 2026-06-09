"""Parallel re-run of SUPP #1 nested-CV oracle.

FAITHFUL parallelization of `clustered_bootstrap_pair`: the N_BOOT=1000 author
resamples are distributed across N_WORKERS processes. Each worker draws its own
independent author-resamples from a DETERMINISTIC spawned seed
(numpy SeedSequence(base_seed).spawn(n_workers)), recomputes the per-replicate
common mask + paired macro-AUC delta EXACTLY like the serial version, and returns
its chunk of deltas. The parent concatenates ALL 1000 deltas and computes
mean / percentile-CI[2.5,97.5] / SE(ddof=1) / MDE(1.96*SE) identically.

Same estimand, same total n_boot=1000 — only the RNG stream differs, so results
match the serial run to bootstrap noise. NO retraining; reuses cached nested OOFs.
Everything else (LP probes, oracle, fusion, null) is the unmodified module code.
"""
import os
# Parent runs the sequential sklearn LP probes / oracle / null -> let BLAS use a
# moderate thread pool so that phase is fast. The bootstrap WORKERS only do numpy
# argsort/reductions (NO BLAS) so 64 single-core workers do not oversubscribe.
_BLAS = os.environ.get("BLAS_THREADS", "16")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _BLAS)

import sys, time
import numpy as np
import pandas as pd
import multiprocessing as mp
from numpy.random import SeedSequence, default_rng

ROOT = os.path.expanduser("~/SHC/birdclef2026_clef")
sys.path.insert(0, os.path.join(ROOT, "src"))

import supp_nested_oracle as M               # the original module (unchanged on disk)
from b2_headline_driver import vec_macro_auc_common

N_WORKERS = int(os.environ.get("NW", "64"))

# globals inherited by fork()ed workers via copy-on-write (NO pickling of big arrays)
_G = {}


def _worker_chunk(args):
    """Run `n_rep` independent author-resamples; return list of paired deltas."""
    seed_state, n_rep = args
    ref = _G["ref"]; test = _G["test"]; y = _G["y"]
    order = _G["order"]; starts = _G["starts"]; ends = _G["ends"]
    counts = _G["counts"]; A = _G["A"]
    rng = default_rng(seed_state)
    out = []
    for _ in range(n_rep):
        samp = rng.integers(0, A, size=A)            # resample author codes
        total = int(counts[samp].sum())
        rows = np.empty(total, dtype=np.int64)
        off = 0
        for a in samp:                               # gather contiguous author blocks
            s, e = starts[a], ends[a]
            rows[off:off + (e - s)] = order[s:e]
            off += e - s
        yb = y[rows]
        (auc_ref, auc_test), nc = vec_macro_auc_common(yb, [ref[rows], test[rows]])
        if nc == 0 or not np.isfinite(auc_ref) or not np.isfinite(auc_test):
            continue
        out.append(auc_test - auc_ref)
    return out


def clustered_bootstrap_pair_par(ref, test, y, authors, n_boot=1000, seed=0):
    """Parallel drop-in for b2_headline_driver.clustered_bootstrap_pair.

    Identical CSR author index + identical per-replicate computation as the serial
    version; only the 1000 resamples are split across N_WORKERS processes, each with
    a deterministic spawned seed. Parent aggregates and computes the SAME stats."""
    t0 = time.time()
    codes, _ = pd.factorize(authors)
    A = codes.max() + 1
    order = np.argsort(codes, kind="mergesort")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(A), side="left")
    ends = np.searchsorted(sorted_codes, np.arange(A), side="right")
    counts = ends - starts

    _G.clear()
    _G.update(ref=ref, test=test, y=y, order=order, starts=starts, ends=ends,
              counts=counts, A=int(A))

    nw = min(N_WORKERS, n_boot)
    child_seeds = SeedSequence(seed).spawn(nw)
    base, rem = divmod(n_boot, nw)
    chunks = [base + (1 if i < rem else 0) for i in range(nw)]
    args = [(child_seeds[i], chunks[i]) for i in range(nw) if chunks[i] > 0]
    assert sum(c for _, c in args) == n_boot

    print(f"    [boot] seed={seed} n_boot={n_boot} -> {len(args)} workers "
          f"(chunks {min(chunks)}-{max(chunks)})", flush=True)
    results = []
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=nw) as pool:
        for sub in pool.imap_unordered(_worker_chunk, args):
            results.append(sub)
            done += 1
            if done % 8 == 0 or done == len(args):
                print(f"    [boot] {done}/{len(args)} chunks done "
                      f"({time.strftime('%H:%M:%S')})", flush=True)
    deltas = np.array([d for sub in results for d in sub])

    if len(deltas) == 0:
        return {"delta_mean": np.nan, "ci95": [np.nan, np.nan], "se": np.nan,
                "MDE_approx": np.nan, "n_boot": 0, "n_authors": int(A)}
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    se = float(deltas.std(ddof=1))
    print(f"    [boot] done {len(deltas)} deltas in {time.time()-t0:.1f}s", flush=True)
    return {"delta_mean": float(deltas.mean()), "ci95": [float(lo), float(hi)],
            "se": se, "MDE_approx": float(1.96 * se),
            "n_boot": int(len(deltas)), "n_authors": int(A)}


# ---------- parallel LP nested selection probes (deterministic -> faithful) ----------
# The 10 outer-fold pairs are independent; lp_probe (StandardScaler + lbfgs Logistic
# Regression) is deterministic, so the produced sel matrices are IDENTICAL to the
# serial lp_nested_sel. We use a 'spawn' pool (children call BLAS heavily -> fork is
# unsafe) and reduce per-child BLAS threads so all pairs run concurrently.
_LP = {}


def _lp_init():
    import numpy as _np
    import pandas as _pd
    OOF = os.path.join(ROOT, "data", "oof")
    _LP["embs"] = _np.load(os.path.join(OOF, "perch_cache", "embeddings.npy"))
    _LP["Y"] = _np.load(os.path.join(OOF, "oof_targets.npy")).astype(_np.float64)
    _LP["folds"] = _pd.read_csv(os.path.join(OOF, "oof_meta.csv"))["fold"].values


def _lp_pair(pair):
    import numpy as _np
    a, b = pair
    embs = _LP["embs"]; Y = _LP["Y"]; folds = _LP["folds"]
    C = Y.shape[1]
    ufolds = sorted(set(folds))
    S = [x for x in ufolds if x not in (a, b)]
    trm = _np.isin(folds, S)
    Xtr = embs[trm].astype(_np.float32); ytr = Y[trm]
    res = []
    for g in (a, b):                       # exactly mirrors serial lp_nested_sel
        f = b if g == a else a
        gm = folds == g
        pred = M.lp_probe(Xtr, ytr, embs[gm].astype(_np.float32), C)
        res.append((int(f), _np.where(gm)[0], pred))
    print(f"  [lp-par] pair {{ {a},{b} }} done ({time.strftime('%H:%M:%S')})", flush=True)
    return res


def lp_nested_sel_par(embs, Y, folds, N, C):
    import itertools as _it
    ufolds = sorted(set(folds))
    sel = {f: np.full((N, C), np.nan, np.float64) for f in ufolds}
    pairs = list(_it.combinations(ufolds, 2))
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:                          # spawned children inherit -> few threads each
        os.environ[k] = os.environ.get("LP_BLAS_THREADS", "4")
    nw = min(len(pairs), int(os.environ.get("LP_WORKERS", "10")))
    print(f"[lp] parallel nested LP probes: {len(pairs)} pairs over {nw} procs "
          f"(BLAS={os.environ['OMP_NUM_THREADS']}/proc)", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=nw, initializer=_lp_init) as pool:
        for res in pool.imap_unordered(_lp_pair, pairs):
            for f, rows, pred in res:
                sel[f][rows] = pred
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return sel


# ---------- BIT-IDENTICAL fast label permutation (speeds the serial null) ----------
# Original permute_labels_within_author does np.where(authors==a) INSIDE a loop over
# every unique author -> O(authors*N) per call, dominating the 50-rep null. We cache
# the per-author row groups once (built in the SAME np.unique sorted order) so the
# rng.permutation() calls receive IDENTICAL index arrays in IDENTICAL sequence ->
# the RNG is consumed identically -> output is BIT-IDENTICAL. null_mean is therefore
# unchanged, so the validation match stays exact (NOT just within noise).
from b2_corrected import permute_labels_within_author as _permute_orig
_PGROUPS = {}


def permute_fast(y, authors, rng):
    key = (id(authors), len(authors))
    g = _PGROUPS.get(key)
    if g is None:
        g = [np.where(authors == a)[0] for a in np.unique(authors)]
        _PGROUPS[key] = g
    out = np.empty_like(y)
    for idx in g:
        out[idx] = y[rng.permutation(idx)]
    return out


# ---------- FAITHFUL parallel label-perm null ----------
# The 50 null replicates are: permute y within author (consumes rng), then evaluate a
# DETERMINISTIC oracle delta. We generate all 50 permuted-label matrices SERIALLY in
# the parent (identical rng stream -> identical yps as the serial code), then farm out
# the deterministic per-yp oracle evaluation to a fork pool (oracle uses only numpy
# argsort, NO BLAS -> fork-safe). Same yps + deterministic oracle => null_mean is
# BIT-IDENTICAL to the serial run (only the order of evaluation changes).
_NULL = {}


def _null_worker(i):
    yp = _NULL["yps"][i]
    folds = _NULL["folds"]; names = _NULL["names"]
    best = _NULL["apply_mats"][_NULL["best_name"]]
    if _NULL["mode"] == "global":
        models = {nm: _NULL["apply_mats"][nm] for nm in names}
        orc, _, _ = M.heldout_oracle_nested(models, yp, folds)
    else:
        orc, _ = M.nested_oracle(_NULL["apply_mats"], _NULL["sel_mats"], names, yp, folds)
    (a_best, a_orc), nc = vec_macro_auc_common(yp, [best, orc])
    if nc and np.isfinite(a_orc) and np.isfinite(a_best):
        return a_orc - a_best
    return None


def _parallel_null(mode, apply_mats, sel_mats, names, y, folds, authors,
                   best_name, n_rep, seed):
    rng = np.random.default_rng(seed)
    yps = [permute_fast(y, authors, rng) for _ in range(n_rep)]   # SERIAL rng = faithful
    _NULL.clear()
    _NULL.update(mode=mode, apply_mats=apply_mats, sel_mats=sel_mats, names=names,
                 folds=folds, best_name=best_name, yps=yps)
    nw = min(n_rep, int(os.environ.get("NULL_WORKERS", "50")))
    print(f"    [null-{mode}] {n_rep} reps over {nw} procs ({time.strftime('%H:%M:%S')})",
          flush=True)
    vals = []
    with mp.get_context("fork").Pool(processes=nw) as pool:
        for v in pool.imap(_null_worker, range(n_rep)):   # imap preserves order
            if v is not None:
                vals.append(v)
    _NULL.clear()
    return (float(np.mean(vals)) if vals else 0.0), len(vals)


def global_null_par(apply_mats, names, y, folds, authors, best_name,
                    n_rep=M.N_NULL, seed=0):
    return _parallel_null("global", apply_mats, None, names, y, folds, authors,
                          best_name, n_rep, seed)


def nested_null_par(apply_mats, sel_mats, names, y, folds, authors, best_name,
                    n_rep=M.N_NULL, seed=0):
    return _parallel_null("nested", apply_mats, sel_mats, names, y, folds, authors,
                          best_name, n_rep, seed)


# monkeypatch the names used inside supp_nested_oracle
M.permute_labels_within_author = permute_fast
M.lp_nested_sel = lp_nested_sel_par
M.clustered_bootstrap_pair = clustered_bootstrap_pair_par
M.global_null_labelperm = global_null_par
M.nested_null_labelperm = nested_null_par

if __name__ == "__main__":
    print(f"[par] N_WORKERS={N_WORKERS} start={time.strftime('%H:%M:%S')}", flush=True)
    # faithfulness self-check: fast permute is bit-identical to the original across
    # SEQUENTIAL reps (so the shared rng stream is consumed identically).
    _OOF = os.path.join(ROOT, "data", "oof")
    _a = pd.read_csv(os.path.join(_OOF, "oof_meta.csv"))["author"].values
    _y = np.load(os.path.join(_OOF, "oof_targets.npy")).astype(np.float64)
    _r1 = np.random.default_rng(123); _r2 = np.random.default_rng(123)
    for _k in range(3):
        assert np.array_equal(_permute_orig(_y, _a, _r1), permute_fast(_y, _a, _r2)), \
            f"permute_fast diverged at rep {_k}"
    _PGROUPS.clear()   # reset cache so the run rebuilds it cleanly
    print("[selfcheck] permute_fast == original over 3 sequential reps: OK", flush=True)
    M.N_WORKERS = N_WORKERS
    M.main()
    # stamp n_workers into the output json
    import json
    jp = os.path.join(ROOT, "data", "oof", "supp_nestedcv.json")
    with open(jp) as f:
        d = json.load(f)
    d["n_workers"] = N_WORKERS
    with open(jp, "w") as f:
        json.dump(d, f, indent=2, default=float)
    print(f"[par] stamped n_workers={N_WORKERS} into {jp}", flush=True)
