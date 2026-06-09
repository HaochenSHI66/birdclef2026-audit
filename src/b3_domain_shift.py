"""Task 3 — Domain-shift (focal XC -> soundscape) feasibility + measurement.

Labeled soundscape data EXISTS locally:
  data/train_soundscapes_labels.csv : 1478 labeled 5 s windows over 66 soundscape
  files, multi-label (';'-separated primary_label codes), columns
  (filename, start, end, primary_label). Audio in data/train_soundscapes/.

This makes the HONEST SUPERVISED domain-shift comparison feasible:
  1. Run Perch-V2 CPU-ONNX on each labeled 5 s soundscape window -> label_head[14795]
     + embedding[1536].
  2. Map label_head -> 234 zero-shot probs with the SAME predeclared mapping as the
     focal anchor (perch_zeroshot_map.csv), aggregation = max.
  3. Per-class ROC-AUC on the soundscape eval set (skip zero-positive) vs the focal-CV
     per-class zero-shot AUC (from focal OOF). Report the per-class AUC DROP on the
     common evaluable class set, plus the Perch zero-shot soundscape macro-AUC.
  4. Descriptive embedding-distribution shift: MMD (unbiased, RBF, median-heuristic
     bandwidth) between focal-train Perch embeddings (cached) and soundscape Perch
     embeddings; plus simple per-dim mean/std L2 gap. Unsupervised, descriptive only.

Robustness gap (Perch zero-shot is the only model with soundscape predictions here;
the trained sidecars CNN/PANN/BEATs are NOT run on soundscapes because that would need
their checkpoints, which were intentionally not saved -> stated as infeasible, no
fabrication). We report the focal-vs-soundscape gap for Perch zero-shot only and say so.

OUTPUT: data/oof/domain_shift.json
"""
import argparse
import csv
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

SR = 32000
WIN = 5 * SR
N_LABEL_HEAD = 14795


def hhmmss_to_sec(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def load_perch_labels(perch_dir):
    with open(os.path.join(perch_dir, "perch_v2_ebird_classes.csv")) as f:
        r = csv.reader(f); next(r)
        ebird = [row[0].strip() for row in r]
    with open(os.path.join(perch_dir, "labels.csv")) as f:
        r = csv.reader(f); next(r)
        inat = [row[0].strip() for row in r]
    assert len(ebird) == len(inat) == N_LABEL_HEAD
    return ebird, inat


def build_map(tax, ebird, inat):
    ebird_to_idxs = defaultdict(list)
    for i, c in enumerate(ebird):
        if c != "no_ebird_code":
            ebird_to_idxs[c].append(i)
    sci_to_idxs = defaultdict(list)
    for i, s in enumerate(inat):
        if s:
            sci_to_idxs[s.lower()].append(i)
    mapping = []
    for t in tax.itertuples(index=False):
        pl = str(t.primary_label).strip()
        sci = str(t.scientific_name).strip().lower()
        if pl in ebird_to_idxs:
            idxs = ebird_to_idxs[pl]
        elif sci in sci_to_idxs:
            idxs = sci_to_idxs[sci]
        else:
            idxs = []
        mapping.append(idxs)
    return mapping


def make_session(onnx_path, n_threads=16):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = n_threads
    return ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def load_window(path, start_sec, sr=SR, win=WIN):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True, offset=start_sec, duration=5.0)
    if len(y) == 0:
        return np.zeros(win, dtype=np.float32)
    if len(y) >= win:
        y = y[:win]
    else:
        y = np.pad(y, (0, win - len(y)))
    return y.astype(np.float32)


def macro_auc_per_class(y_true, y_pred):
    """Return dict class_idx -> auc for classes with >=1 pos and finite preds."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            col = y_pred[:, i]
            if np.isnan(col).any():
                continue
            try:
                out[i] = float(roc_auc_score(y_true[:, i], col))
            except Exception:
                pass
    return out


def rbf_mmd2_unbiased(X, Y, gamma):
    """Unbiased squared MMD with RBF kernel. X:[n,d], Y:[m,d]."""
    Xs = (X * X).sum(1)
    Ys = (Y * Y).sum(1)
    Kxx = np.exp(-gamma * (Xs[:, None] + Xs[None, :] - 2 * X @ X.T))
    Kyy = np.exp(-gamma * (Ys[:, None] + Ys[None, :] - 2 * Y @ Y.T))
    Kxy = np.exp(-gamma * (Xs[:, None] + Ys[None, :] - 2 * X @ Y.T))
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_xx = Kxx.sum() / (n * (n - 1))
    term_yy = Kyy.sum() / (m * (m - 1))
    term_xy = Kxy.mean()
    return float(term_xx + term_yy - 2 * term_xy)


def median_heuristic_gamma(Z, max_pairs=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(Z), size=min(len(Z), max_pairs), replace=False)
    S = Z[idx]
    sq = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    med = np.median(sq[np.triu_indices_from(sq, k=1)])
    return 1.0 / (med + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.expanduser("~/SHC/birdclef2026_clef/data"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "oof")
    perch_dir = os.path.join(args.data_dir, "perch_labels")
    onnx = os.path.join(args.data_dir, "perch_onnx", "perch_v2_no_dft.onnx")
    ss_dir = os.path.join(args.data_dir, "train_soundscapes")

    tax = pd.read_csv(os.path.join(args.data_dir, "taxonomy.csv"))
    class_order = tax["primary_label"].astype(str).tolist()
    cls_idx = {c: i for i, c in enumerate(class_order)}
    ebird, inat = load_perch_labels(perch_dir)
    mapping = build_map(tax, ebird, inat)

    # --- read labeled soundscape windows ---
    lab = pd.read_csv(os.path.join(args.data_dir, "train_soundscapes_labels.csv"))
    print(f"[ss] {len(lab)} labeled windows, {lab['filename'].nunique()} files", flush=True)
    # multi-label target matrix [W, 234]
    W = len(lab)
    Yss = np.zeros((W, 234), dtype=np.float32)
    n_in_tax = 0
    code_seen = set()
    for r, row in enumerate(lab.itertuples(index=False)):
        codes = str(row.primary_label).split(";")
        for c in codes:
            c = c.strip()
            code_seen.add(c)
            if c in cls_idx:
                Yss[r, cls_idx[c]] = 1.0
                n_in_tax += 1
    print(f"[ss] distinct codes {len(code_seen)}; "
          f"codes in 234-taxonomy: {len(code_seen & set(class_order))}", flush=True)

    starts = [hhmmss_to_sec(s) for s in lab["start"].tolist()]
    files = lab["filename"].tolist()

    # --- run Perch on each window (cache embeddings + label_head) ---
    cache_emb = os.path.join(args.out_dir, "ss_perch_emb.npy")
    cache_lab = os.path.join(args.out_dir, "ss_perch_labelhead.npy")
    if os.path.exists(cache_emb) and os.path.exists(cache_lab):
        emb = np.load(cache_emb); lh = np.load(cache_lab)
        print(f"[ss] loaded cached emb {emb.shape} labelhead {lh.shape}", flush=True)
    else:
        sess = make_session(onnx, args.threads)
        out_names = [o.name for o in sess.get_outputs()]
        emb = np.zeros((W, 1536), dtype=np.float16)
        lh = np.zeros((W, N_LABEL_HEAD), dtype=np.float16)
        t0 = time.time()
        for j in range(0, W, args.batch_size):
            bs = min(args.batch_size, W - j)
            batch = np.stack([load_window(os.path.join(ss_dir, files[j + k]),
                                          starts[j + k]) for k in range(bs)]).astype(np.float32)
            outs = sess.run(None, {"inputs": batch})
            d = dict(zip(out_names, outs))
            emb[j:j + bs] = d["embedding"].astype(np.float16)
            lh[j:j + bs] = d["label"].astype(np.float16)
            if j % 160 == 0:
                el = time.time() - t0
                print(f"[ss] {j+bs}/{W} | {(j+bs)/max(el,1e-6):.1f} win/s", flush=True)
        np.save(cache_emb, emb); np.save(cache_lab, lh)
        print(f"[ss] DONE perch in {(time.time()-t0)/60:.1f} min", flush=True)

    # --- zero-shot 234 probs on soundscapes ---
    probs = 1.0 / (1.0 + np.exp(-lh.astype(np.float64)))
    zs = np.full((W, 234), np.nan, dtype=np.float32)
    for c, idxs in enumerate(mapping):
        if idxs:
            zs[:, c] = probs[:, idxs].max(axis=1).astype(np.float32)

    ss_auc = macro_auc_per_class(Yss, np.nan_to_num(zs, nan=0.0))
    ss_macro = float(np.mean(list(ss_auc.values()))) if ss_auc else 0.0
    print(f"[ss] soundscape zero-shot macro-AUC = {ss_macro:.4f} over "
          f"{len(ss_auc)} eval classes", flush=True)

    # --- focal-CV per-class zero-shot AUC (from focal OOF) ---
    focal_zs = np.load(os.path.join(args.out_dir, "perch_zeroshot_oof.npy"))
    Yf = np.load(os.path.join(args.out_dir, "oof_targets.npy"))
    focal_auc = macro_auc_per_class(Yf, np.nan_to_num(focal_zs, nan=0.0))
    focal_macro = float(np.mean(list(focal_auc.values())))

    common = sorted(set(ss_auc) & set(focal_auc))
    drops = [focal_auc[i] - ss_auc[i] for i in common]
    per_class = [{"class_idx": int(i), "primary_label": class_order[i],
                  "class_name": str(tax.iloc[i]["class_name"]),
                  "focal_auc": round(focal_auc[i], 4),
                  "soundscape_auc": round(ss_auc[i], 4),
                  "drop": round(focal_auc[i] - ss_auc[i], 4)}
                 for i in common]
    per_class.sort(key=lambda x: -x["drop"])
    mean_drop = float(np.mean(drops)) if drops else None
    # paired bootstrap CI on mean drop over common classes
    rng = np.random.default_rng(0)
    boots = []
    da = np.array(drops)
    for _ in range(2000):
        bs = rng.choice(len(da), len(da), replace=True)
        boots.append(da[bs].mean())
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # --- embedding MMD focal vs soundscape (descriptive) ---
    f_emb = np.load(os.path.join(args.out_dir, "perch_cache", "embeddings.npy"),
                    mmap_mode="r")
    rng2 = np.random.default_rng(1)
    n_samp = min(1500, len(f_emb))
    fi = np.sort(rng2.choice(len(f_emb), n_samp, replace=False))
    Xf = np.asarray(f_emb[fi], dtype=np.float32)
    Xs = emb.astype(np.float32)
    Z = np.vstack([Xf, Xs])
    gamma = median_heuristic_gamma(Z, rng=rng2)
    mmd2 = rbf_mmd2_unbiased(Xf, Xs[:min(len(Xs), n_samp)], gamma)
    # simple stats: L2 of mean diff, ratio of mean std
    mean_l2 = float(np.linalg.norm(Xf.mean(0) - Xs.mean(0)))
    std_ratio = float(Xs.std(0).mean() / (Xf.std(0).mean() + 1e-12))

    result = {
        "task": "domain_shift_focal_to_soundscape",
        "feasible": True,
        "labeled_soundscape": {
            "n_windows": W, "n_files": int(lab["filename"].nunique()),
            "n_distinct_codes": len(code_seen),
            "n_codes_in_234_taxonomy": len(code_seen & set(class_order)),
        },
        "supervised_perch_zeroshot": {
            "soundscape_macro_auc": round(ss_macro, 4),
            "soundscape_eval_classes": len(ss_auc),
            "focal_macro_auc_full": round(focal_macro, 4),
            "n_common_eval_classes": len(common),
            "mean_per_class_auc_drop_focal_minus_soundscape": round(mean_drop, 4)
            if mean_drop is not None else None,
            "drop_ci95_bootstrap_over_classes": [round(ci[0], 4), round(ci[1], 4)],
            "n_classes_with_drop_gt_0": int((da > 0).sum()),
            "per_class_top": per_class[:20],
        },
        "robustness_gap_note": (
            "Only Perch zero-shot has soundscape predictions; trained sidecars "
            "(CNN/PANN/BEATs) were OOF-only with no saved checkpoints, so a Perch-vs-"
            "sidecar soundscape robustness gap is NOT computable here without retraining "
            "+ re-inference. Reported as infeasible to avoid fabrication."),
        "embedding_distribution_shift_descriptive": {
            "n_focal_sampled": int(n_samp), "n_soundscape": int(len(Xs)),
            "rbf_mmd2_unbiased": round(mmd2, 6),
            "rbf_gamma_median_heuristic": float(gamma),
            "mean_embedding_l2_gap": round(mean_l2, 4),
            "soundscape_to_focal_std_ratio": round(std_ratio, 4),
        },
    }
    import json
    out_path = os.path.join(args.out_dir, "domain_shift.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE", out_path, flush=True)
    print(json.dumps(result, indent=2)[:1500], flush=True)


if __name__ == "__main__":
    main()
