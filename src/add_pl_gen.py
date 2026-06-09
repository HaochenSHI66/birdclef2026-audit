"""Soundscape PSEUDO-LABEL generation for the PL-headroom audit.

Generates pseudo-labels on the UNLABELED soundscape pool (10,592 train_soundscape
files NOT in train_soundscapes_labels.csv) using the strong target-domain anchor
Perch-V2 zero-shot (CPU ONNX), with a standard confidence-thresholded multi-label
PL rule. Outputs a pseudo-labeled clip set (precomputed base mel + 234-multi-hot
targets) that the PL CNN training run consumes as extra training data.

REUSES exactly the focal/Perch protocol pieces:
  - wave_to_mel (from b1_cnn_train): identical mel features as focal training.
  - load_window/make_session/load_perch_labels/build_map (from b3_domain_shift):
    identical Perch ONNX + 234 zero-shot mapping (max-agg) as domain_shift.json.

PL rule (standard): per 5 s window run Perch ZS -> 234 probs; keep classes with
prob >= tau as positives; drop windows with no class >= tau. tau is a CLI arg.

OUTPUT (to <data_dir>/oof):
  pl_pseudo_mel.npy      [P,224,224] fp16  base log-mel for kept pseudo windows
  pl_pseudo_targets.npy  [P,234]     fp32  multi-hot pseudo labels
  pl_pseudo_meta.csv     (filename,start_sec,n_pos)
  pl_config.json         source model, tau, files scanned, windows, #pseudo, coverage
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/SHC/birdclef2026_clef")
sys.path.insert(0, os.path.join(ROOT, "src"))

from b1_cnn_train import wave_to_mel, load_taxonomy_order
from b3_domain_shift import (load_perch_labels, build_map, load_window, make_session)

SR = 32000
WIN = 5 * SR
N_LABEL_HEAD = 14795


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_files", type=int, default=3000, help="# unlabeled files to scan")
    ap.add_argument("--win_starts", default="5,25,45", help="window offsets (s) per 60s file")
    ap.add_argument("--topk", type=int, default=1, help="PL rule: top-k classes per window")
    ap.add_argument("--tau", type=float, default=0.5, help="(legacy threshold; unused for topk)")
    ap.add_argument("--max_pos_per_win", type=int, default=5, help="cap positives per window")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--onnx_bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scan_only", action="store_true", help="print threshold stats, no save")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "oof")
    rng = np.random.default_rng(args.seed)

    class_order = load_taxonomy_order(args.data_dir)
    tax = pd.read_csv(os.path.join(args.data_dir, "taxonomy.csv"))
    ebird, inat = load_perch_labels(os.path.join(args.data_dir, "perch_labels"))
    mapping = build_map(tax, ebird, inat)
    onnx = os.path.join(args.data_dir, "perch_onnx", "perch_v2_no_dft.onnx")
    ss_dir = os.path.join(args.data_dir, "train_soundscapes")

    # --- unlabeled pool = all soundscape ogg minus the 66 labeled files ---
    lab = pd.read_csv(os.path.join(args.data_dir, "train_soundscapes_labels.csv"))
    labeled = set(lab["filename"].unique())
    allf = sorted([f for f in os.listdir(ss_dir) if f.endswith(".ogg")])
    pool = [f for f in allf if f not in labeled]
    print(f"[pool] total ogg {len(allf)} | labeled {len(labeled)} | unlabeled pool {len(pool)}",
          flush=True)
    pick = rng.choice(len(pool), size=min(args.n_files, len(pool)), replace=False)
    sel_files = [pool[i] for i in sorted(pick)]
    starts = [int(x) for x in args.win_starts.split(",")]

    # build (file,start) window list
    win_file, win_start = [], []
    for f in sel_files:
        for s in starts:
            win_file.append(f); win_start.append(s)
    Wn = len(win_file)
    print(f"[scan] {len(sel_files)} files x {len(starts)} windows = {Wn} windows", flush=True)

    # --- Perch ONNX zero-shot probs on each window ---
    sess = make_session(onnx, args.threads)
    out_names = [o.name for o in sess.get_outputs()]
    zs = np.full((Wn, 234), np.nan, dtype=np.float32)
    waves = [None] * Wn  # keep decoded waves only for mel of kept windows (recompute cheap)
    t0 = time.time()
    for j in range(0, Wn, args.onnx_bs):
        bs = min(args.onnx_bs, Wn - j)
        batch = np.stack([load_window(os.path.join(ss_dir, win_file[j + k]), win_start[j + k])
                          for k in range(bs)]).astype(np.float32)
        outs = sess.run(None, {"inputs": batch})
        d = dict(zip(out_names, outs))
        lh = d["label"].astype(np.float64)
        probs = 1.0 / (1.0 + np.exp(-lh))
        for c, idxs in enumerate(mapping):
            if idxs:
                zs[j:j + bs, c] = probs[:, idxs].max(axis=1).astype(np.float32)
        if j % (args.onnx_bs * 20) == 0:
            el = time.time() - t0
            print(f"[scan] {j+bs}/{Wn} | {(j+bs)/max(el,1e-6):.1f} win/s", flush=True)
    print(f"[scan] perch done in {(time.time()-t0)/60:.2f} min", flush=True)

    # mapped classes only (others NaN); for ranking treat NaN as -inf
    zr = np.where(np.isnan(zs), -np.inf, zs)
    zmax = zr.max(axis=1)
    print("[stats] max-prob percentiles:",
          {p: round(float(np.percentile(zmax, p)), 4) for p in [50, 75, 90, 95, 99]}, flush=True)
    for kk in [1, 2, 3, 5]:
        top = np.argsort(-zr, axis=1)[:, :kk]
        cov = len(np.unique(top))
        print(f"[stats] top-{kk}: distinct classes covered {cov}/234 | "
              f"positives/win {kk}", flush=True)
    if args.scan_only:
        return

    # --- apply PL rule: top-k classes per window ---
    keep_rows = []
    Ytgt = []
    meta = []
    for i in range(Wn):
        order = np.argsort(-zr[i])[:args.topk]
        cols = [c for c in order if np.isfinite(zr[i, c])]
        if len(cols) == 0:
            continue
        y = np.zeros(234, dtype=np.float32)
        y[cols] = 1.0
        Ytgt.append(y)
        keep_rows.append(i)
        meta.append((win_file[i], win_start[i], len(cols)))
    P = len(keep_rows)
    Ytgt = np.asarray(Ytgt, dtype=np.float32) if P else np.zeros((0, 234), np.float32)
    print(f"[pl] tau={args.tau}: kept {P} pseudo windows", flush=True)

    # --- precompute base mel (fp16) for kept windows (recompute wave -> mel) ---
    mel = np.zeros((P, args.img_size, args.img_size), dtype=np.float16)
    t0 = time.time()
    for r, i in enumerate(keep_rows):
        y = load_window(os.path.join(ss_dir, win_file[i]), win_start[i])
        mel[r] = wave_to_mel(y, args.img_size).astype(np.float16)
        if r % 500 == 0:
            print(f"[mel] {r}/{P} | {r/max(time.time()-t0,1e-6):.1f} win/s", flush=True)
    print(f"[mel] done in {(time.time()-t0)/60:.2f} min", flush=True)

    cov = int((Ytgt.sum(0) > 0).sum())
    np.save(os.path.join(args.out_dir, "pl_pseudo_mel.npy"), mel)
    np.save(os.path.join(args.out_dir, "pl_pseudo_targets.npy"), Ytgt)
    pd.DataFrame(meta, columns=["filename", "start_sec", "n_pos"]).to_csv(
        os.path.join(args.out_dir, "pl_pseudo_meta.csv"), index=False)
    cfg = {
        "source_model": "perch_v2_zeroshot_ebird_head_maxagg",
        "pl_rule": f"per-window top-{args.topk} (sigmoid label-head saturated -> "
                   f"absolute threshold meaningless; ranking-based top-k is standard)",
        "topk": args.topk,
        "win_starts_sec": starts,
        "n_unlabeled_pool": len(pool),
        "n_files_scanned": len(sel_files),
        "n_windows_scanned": Wn,
        "n_pseudo_clips": P,
        "n_classes_covered_by_pseudo": cov,
        "mean_pos_per_clip": round(float(Ytgt.sum(1).mean()), 3) if P else 0.0,
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "pl_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("WROTE pl_pseudo_mel/targets/meta + pl_config.json", flush=True)
    print(json.dumps(cfg, indent=2), flush=True)


if __name__ == "__main__":
    main()
