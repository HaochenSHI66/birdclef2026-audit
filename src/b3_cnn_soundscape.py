"""Robustness-gap experiment (Codex review item 3): run the trained author CNN sidecar
on the 66-file labeled soundscape set and compute its focal->soundscape macro-AUC drop,
then compare to Perch zero-shot's drop (from domain_shift.json).

Closes the hole that was previously "infeasible" because the 5-fold matrix saved no CNN
checkpoints. Here we trained ONE CNN on folds 0-3 (held fold 4) WITH a saved checkpoint
(b1_cnn_train_ckpt.py --save_ckpt), and re-infer it on the SAME labeled soundscape windows
(train_soundscapes_labels.csv) that domain_shift.json used for Perch.

OUTPUT: data/oof/add_robustness_gap.json
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

import torch
try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

# EXACT training preprocessing / model
from b1_cnn_train import wave_to_mel, BirdCLEFModel, load_taxonomy_order
# EXACT Perch soundscape protocol pieces (for matched-class recompute)
from b3_domain_shift import (load_perch_labels, build_map, hhmmss_to_sec,
                             load_window, macro_auc_per_class)

SR = 32000
WIN = 5 * SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--train_dir", default=None,
                    help="dir holding the CNN checkpoint + focal OOF (default: <data_dir>/oof)")
    ap.add_argument("--tag", default="cnn", help="CNN run tag used at train time")
    ap.add_argument("--ckpt", default=None, help="CNN checkpoint .pt override")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--fold", type=int, default=4, help="held-out focal fold for the focal AUC")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.join(args.data_dir, "oof")          # Perch artifacts + final json live here
    train_dir = args.train_dir or out_dir                 # CNN artifacts live here
    if args.ckpt is None:
        args.ckpt = os.path.join(train_dir, f"{args.tag}_fold{args.fold}_seed{args.seed}.pt")

    if HAS_NPU and "npu" in args.device:
        device = torch.device(args.device); torch_npu.npu.set_device(device); dtype_dev = "npu"
    elif torch.cuda.is_available():
        device = torch.device("cuda:0"); dtype_dev = "cuda"
    else:
        device = torch.device("cpu"); dtype_dev = "cpu"

    tax = pd.read_csv(os.path.join(args.data_dir, "taxonomy.csv"))
    class_order = load_taxonomy_order(args.data_dir)
    cls_idx = {c: i for i, c in enumerate(class_order)}

    # ---- load CNN checkpoint ----
    ck = torch.load(args.ckpt, map_location="cpu")
    model = BirdCLEFModel(ck["backbone"], ck["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    img_size = ck["img_size"]
    print(f"[ckpt] {args.ckpt} backbone={ck['backbone']} best_val_auc={ck['best_val_macro_auc']:.4f} "
          f"@ep{ck['best_epoch']} img={img_size}", flush=True)

    # ---- which classes did THIS CNN actually train on (>=1 pos in folds != held)? ----
    folds = pd.read_csv(os.path.join(ROOT, "folds.csv"))  # filename, fold (df order)
    Y_all = np.load(os.path.join(train_dir, "cnn_oof_targets.npy"))  # [N,234] aligned to folds.csv
    assert len(Y_all) == len(folds), f"targets {len(Y_all)} != folds {len(folds)}"
    tr_pos = (Y_all[folds["fold"].values != args.fold].sum(0) > 0)
    print(f"[cnn] trained classes (>=1 pos in train folds) = {int(tr_pos.sum())}/234", flush=True)

    # =========================================================================
    # 1) CNN on the labeled soundscape windows
    # =========================================================================
    lab = pd.read_csv(os.path.join(args.data_dir, "train_soundscapes_labels.csv"))
    ss_dir = os.path.join(args.data_dir, "train_soundscapes")
    W = len(lab)
    Yss = np.zeros((W, 234), dtype=np.float32)
    for r, row in enumerate(lab.itertuples(index=False)):
        for c in str(row.primary_label).split(";"):
            c = c.strip()
            if c in cls_idx:
                Yss[r, cls_idx[c]] = 1.0
    starts = [hhmmss_to_sec(s) for s in lab["start"].tolist()]
    files = lab["filename"].tolist()
    print(f"[ss] {W} windows, {lab['filename'].nunique()} files", flush=True)

    cache_ss = os.path.join(train_dir, f"ss_{args.tag}_preds.npy")
    if os.path.exists(cache_ss):
        cnn_ss = np.load(cache_ss)
        print(f"[ss] loaded cached CNN preds {cnn_ss.shape}", flush=True)
    else:
        cnn_ss = np.zeros((W, 234), dtype=np.float32)
        t0 = time.time()
        with torch.no_grad():
            for j in range(0, W, args.batch_size):
                bs = min(args.batch_size, W - j)
                mels = []
                for k in range(bs):
                    y = load_window(os.path.join(ss_dir, files[j + k]), starts[j + k])
                    mel = wave_to_mel(y, img_size)
                    mels.append(np.stack([mel, mel, mel], axis=0))
                x = torch.from_numpy(np.stack(mels)).float().to(device)
                with torch.autocast(device_type=dtype_dev, dtype=torch.bfloat16):
                    logits = model(x)
                cnn_ss[j:j + bs] = torch.sigmoid(logits).float().cpu().numpy()
                if j % (args.batch_size * 5) == 0:
                    el = time.time() - t0
                    print(f"[ss] {j+bs}/{W} | {(j+bs)/max(el,1e-6):.1f} win/s", flush=True)
        np.save(cache_ss, cnn_ss)
        print(f"[ss] CNN inference done in {(time.time()-t0)/60:.2f} min -> {cache_ss}", flush=True)

    # mask out classes the CNN never trained on (unsupervised head columns -> NaN)
    cnn_ss_masked = cnn_ss.copy()
    cnn_ss_masked[:, ~tr_pos] = np.nan
    cnn_ss_auc = macro_auc_per_class(Yss, np.nan_to_num(cnn_ss_masked, nan=0.0))
    # only keep classes that are actually scorable AND trained
    cnn_ss_auc = {i: v for i, v in cnn_ss_auc.items() if tr_pos[i]}
    cnn_ss_macro = float(np.mean(list(cnn_ss_auc.values()))) if cnn_ss_auc else 0.0
    print(f"[cnn] soundscape macro-AUC = {cnn_ss_macro:.4f} over {len(cnn_ss_auc)} cls", flush=True)

    # =========================================================================
    # 2) CNN focal per-class AUC on the held-out focal fold (fold 4)
    # =========================================================================
    focal_preds = np.load(os.path.join(train_dir, f"{args.tag}_fold{args.fold}_seed{args.seed}.npy"))  # [n_val,234] NaN untrained
    focal_rows = np.load(os.path.join(train_dir, f"{args.tag}_fold{args.fold}_seed{args.seed}_rows.npy"))
    Yf = Y_all[focal_rows]
    cnn_focal_auc = macro_auc_per_class(Yf, np.nan_to_num(focal_preds, nan=0.0))
    cnn_focal_auc = {i: v for i, v in cnn_focal_auc.items() if tr_pos[i]}
    cnn_focal_macro = float(np.mean(list(cnn_focal_auc.values()))) if cnn_focal_auc else 0.0
    print(f"[cnn] focal (fold{args.fold}) macro-AUC = {cnn_focal_macro:.4f} over "
          f"{len(cnn_focal_auc)} cls", flush=True)

    # =========================================================================
    # 3) Perch focal/soundscape per-class AUC (recompute for matched-class gap)
    # =========================================================================
    perch_dir = os.path.join(args.data_dir, "perch_labels")
    ebird, inat = load_perch_labels(perch_dir)
    mapping = build_map(tax, ebird, inat)
    lh = np.load(os.path.join(out_dir, "ss_perch_labelhead.npy"))
    probs = 1.0 / (1.0 + np.exp(-lh.astype(np.float64)))
    perch_zs = np.full((W, 234), np.nan, dtype=np.float32)
    for c, idxs in enumerate(mapping):
        if idxs:
            perch_zs[:, c] = probs[:, idxs].max(axis=1).astype(np.float32)
    perch_ss_auc = macro_auc_per_class(Yss, np.nan_to_num(perch_zs, nan=0.0))
    perch_focal = np.load(os.path.join(out_dir, "perch_zeroshot_oof.npy"))
    Yf_perch = np.load(os.path.join(out_dir, "oof_targets.npy"))
    perch_focal_auc = macro_auc_per_class(Yf_perch, np.nan_to_num(perch_focal, nan=0.0))

    # =========================================================================
    # 4) Robustness gap
    # =========================================================================
    # Perch drop from the published artifact (same protocol)
    with open(os.path.join(out_dir, "domain_shift.json")) as f:
        ds = json.load(f)
    perch_drop_json = ds["supervised_perch_zeroshot"]["mean_per_class_auc_drop_focal_minus_soundscape"]
    perch_common = sorted(set(perch_ss_auc) & set(perch_focal_auc))
    perch_drop_recomp = float(np.mean([perch_focal_auc[i] - perch_ss_auc[i] for i in perch_common]))

    # CNN native-common drop (same definition Perch uses for its own common set)
    cnn_common = sorted(set(cnn_ss_auc) & set(cnn_focal_auc))
    cnn_focal_common = float(np.mean([cnn_focal_auc[i] for i in cnn_common]))
    cnn_ss_common = float(np.mean([cnn_ss_auc[i] for i in cnn_common]))
    cnn_drop = cnn_focal_common - cnn_ss_common

    # MATCHED-class gap (apples-to-apples: identical class set for both models)
    joint = sorted(set(cnn_common) & set(perch_common))
    perch_drop_joint = float(np.mean([perch_focal_auc[i] - perch_ss_auc[i] for i in joint])) if joint else None
    cnn_drop_joint = float(np.mean([cnn_focal_auc[i] - cnn_ss_auc[i] for i in joint])) if joint else None
    cnn_focal_joint = float(np.mean([cnn_focal_auc[i] for i in joint])) if joint else None
    cnn_ss_joint = float(np.mean([cnn_ss_auc[i] for i in joint])) if joint else None
    perch_focal_joint = float(np.mean([perch_focal_auc[i] for i in joint])) if joint else None
    perch_ss_joint = float(np.mean([perch_ss_auc[i] for i in joint])) if joint else None

    gap = cnn_drop - perch_drop_json                      # requested: CNN(native-common) vs Perch(json)
    gap_matched = (cnn_drop_joint - perch_drop_joint) if joint else None

    perch_more_robust = bool(cnn_drop_joint > perch_drop_joint) if joint else None

    result = {
        "task": "robustness_gap_cnn_vs_perch_focal_to_soundscape",
        "experiment": ("retrained author CNN on folds 0-3 (held fold 4) with saved checkpoint, "
                       "re-inferred on the 66-file labeled soundscape set; closes the previously "
                       "infeasible Perch-vs-sidecar robustness gap"),
        "checkpoint": {
            "path": args.ckpt, "backbone": ck["backbone"], "seed": int(args.seed),
            "held_fold": int(args.fold),
            "best_val_macro_auc_native": round(float(ck["best_val_macro_auc"]), 4),
            "best_epoch": int(ck["best_epoch"]),
            "n_trained_classes": int(tr_pos.sum()),
        },
        # ---- requested headline fields ----
        "cnn_focal_auc": round(cnn_focal_common, 4),
        "cnn_soundscape_auc": round(cnn_ss_common, 4),
        "cnn_drop": round(cnn_drop, 4),
        "perch_drop": round(float(perch_drop_json), 4),
        "gap": round(gap, 4),
        "common_class_count": len(cnn_common),
        "perch_more_robust_than_cnn": perch_more_robust,
        # ---- native macro AUCs (each model's full scorable set) ----
        "cnn_focal_macro_native": round(cnn_focal_macro, 4),
        "cnn_soundscape_macro_native": round(cnn_ss_macro, 4),
        "cnn_focal_eval_classes": len(cnn_focal_auc),
        "cnn_soundscape_eval_classes": len(cnn_ss_auc),
        "perch_focal_macro_full": ds["supervised_perch_zeroshot"]["focal_macro_auc_full"],
        "perch_soundscape_macro": ds["supervised_perch_zeroshot"]["soundscape_macro_auc"],
        "perch_drop_recomputed_check": round(perch_drop_recomp, 4),
        "perch_n_common": len(perch_common),
        # ---- MATCHED-class gap (cleanest apples-to-apples) ----
        "matched_class": {
            "n_joint_classes": len(joint),
            "cnn_focal_auc": round(cnn_focal_joint, 4) if joint else None,
            "cnn_soundscape_auc": round(cnn_ss_joint, 4) if joint else None,
            "cnn_drop": round(cnn_drop_joint, 4) if joint else None,
            "perch_focal_auc": round(perch_focal_joint, 4) if joint else None,
            "perch_soundscape_auc": round(perch_ss_joint, 4) if joint else None,
            "perch_drop": round(perch_drop_joint, 4) if joint else None,
            "gap_cnn_minus_perch": round(gap_matched, 4) if joint else None,
            "perch_more_robust_than_cnn": perch_more_robust,
        },
        "interpretation": (
            "drop = focal_AUC - soundscape_AUC (larger drop = LESS robust to focal->soundscape "
            "shift). gap = cnn_drop - perch_drop; gap>0 means Perch (foundation embedding) is "
            "MORE robust than the from-scratch CNN. Matched-class section uses the identical "
            "class set for both models and is the apples-to-apples number."),
    }
    out_path = os.path.join(out_dir, "add_robustness_gap.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE", out_path, flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
