"""Soundscape PSEUDO-LABEL headroom audit (verdict step).

Loads the matched BASELINE (focal only) and PL (focal + pseudo-labeled soundscapes)
CNN checkpoints, re-infers BOTH on the 66-file labeled soundscape set (same windows as
domain_shift.json), and measures whether PL adds real TARGET-DOMAIN headroom:

  - soundscape macro-AUC for base and PL on a COMMON class set (classes the BASELINE
    trained on in folds 0-3 AND scorable in soundscapes) -> paired apples-to-apples;
  - DeltaAUC = PL - base, with FILE-CLUSTERED bootstrap 95% CI (resample the 66
    soundscape files with replacement);
  - focal-CV effect: each model's held-out fold-4 focal macro-AUC -> DeltaAUC_focal
    (should be neutral/aux);
  - secondary: each model's macro-AUC on its OWN trained set (does PL unlock classes?).

OUTPUT: data/oof/add_pseudolabel_audit.json
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

from b1_cnn_train import wave_to_mel, BirdCLEFModel, load_taxonomy_order
from b3_domain_shift import hhmmss_to_sec, load_window, macro_auc_per_class

SR = 32000
WIN = 5 * SR


def infer_soundscape(ckpt_path, files, starts, ss_dir, device, dtype_dev, bs, cache_path):
    if os.path.exists(cache_path):
        print(f"[ss] cached {cache_path}", flush=True)
        return np.load(cache_path)
    ck = torch.load(ckpt_path, map_location="cpu")
    model = BirdCLEFModel(ck["backbone"], ck["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    img = ck["img_size"]
    W = len(files)
    preds = np.zeros((W, ck["num_classes"]), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for j in range(0, W, bs):
            n = min(bs, W - j)
            mels = []
            for k in range(n):
                y = load_window(os.path.join(ss_dir, files[j + k]), starts[j + k])
                mel = wave_to_mel(y, img)
                mels.append(np.stack([mel, mel, mel], axis=0))
            x = torch.from_numpy(np.stack(mels)).float().to(device)
            with torch.autocast(device_type=dtype_dev, dtype=torch.bfloat16):
                logits = model(x)
            preds[j:j + n] = torch.sigmoid(logits).float().cpu().numpy()
    print(f"[ss] {os.path.basename(ckpt_path)} inferred {W} win in {(time.time()-t0)/60:.2f} min",
          flush=True)
    np.save(cache_path, preds)
    return preds


def macro_on_classes(Y, P, classes):
    """mean per-class AUC over the given class indices (skip classes w/o positives)."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in classes:
        if Y[:, c].sum() > 0:
            col = P[:, c]
            if np.isnan(col).any():
                continue
            try:
                aucs.append(roc_auc_score(Y[:, c], col))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--base_tag", default="cnnplbase")
    ap.add_argument("--pl_tag", default="cnnplpl")
    ap.add_argument("--fold", type=int, default=4, help="held-out focal fold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "oof")
    od = args.out_dir

    if HAS_NPU and "npu" in args.device:
        device = torch.device(args.device); torch_npu.npu.set_device(device); dtype_dev = "npu"
    elif torch.cuda.is_available():
        device = torch.device("cuda:0"); dtype_dev = "cuda"
    else:
        device = torch.device("cpu"); dtype_dev = "cpu"

    class_order = load_taxonomy_order(args.data_dir)
    cls_idx = {c: i for i, c in enumerate(class_order)}

    # ---- soundscape labels (66 files / 1478 windows) ----
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
    file_arr = np.array(files)
    uniq_files = sorted(set(files))
    print(f"[ss] {W} windows / {len(uniq_files)} files", flush=True)

    # ---- focal trained-class sets (>=1 pos in folds != held) ----
    folds = pd.read_csv(os.path.join(ROOT, "folds.csv"))
    Yfoc = np.load(os.path.join(od, f"{args.base_tag}_oof_targets.npy"))  # focal targets (both tags identical)
    tr_pos_base = (Yfoc[folds["fold"].values != args.fold].sum(0) > 0)
    # PL trained classes = focal-trained UNION pseudo-covered
    pl_cfg = {}
    pseudo_pos = np.zeros(234, dtype=bool)
    if os.path.exists(os.path.join(od, "pl_pseudo_targets.npy")):
        pl_t = np.load(os.path.join(od, "pl_pseudo_targets.npy"))
        pseudo_pos = (pl_t.sum(0) > 0)
        with open(os.path.join(od, "pl_config.json")) as f:
            pl_cfg = json.load(f)
    tr_pos_pl = tr_pos_base | pseudo_pos

    # ---- infer both on soundscapes ----
    base_ck = os.path.join(od, f"{args.base_tag}_fold{args.fold}_seed{args.seed}.pt")
    pl_ck = os.path.join(od, f"{args.pl_tag}_fold{args.fold}_seed{args.seed}.pt")
    base_ss = infer_soundscape(base_ck, files, starts, ss_dir, device, dtype_dev,
                               args.batch_size, os.path.join(od, f"ss_{args.base_tag}_preds.npy"))
    pl_ss = infer_soundscape(pl_ck, files, starts, ss_dir, device, dtype_dev,
                             args.batch_size, os.path.join(od, f"ss_{args.pl_tag}_preds.npy"))

    # ---- COMMON class set: baseline-trained AND soundscape-scorable ----
    ss_scorable = (Yss.sum(0) > 0)
    common = [c for c in range(234) if tr_pos_base[c] and ss_scorable[c]]
    print(f"[eval] common (base-trained & ss-scorable) classes = {len(common)}", flush=True)

    base_macro = macro_on_classes(Yss, base_ss, common)
    pl_macro = macro_on_classes(Yss, pl_ss, common)
    delta = pl_macro - base_macro
    print(f"[ss] base macro-AUC {base_macro:.4f} | PL macro-AUC {pl_macro:.4f} | "
          f"Delta {delta:+.4f}", flush=True)

    # ---- file-clustered bootstrap on Delta ----
    rng = np.random.default_rng(args.seed)
    file_to_rows = {f: np.where(file_arr == f)[0] for f in uniq_files}
    deltas, b_list, p_list = [], [], []
    for _ in range(args.n_boot):
        samp = rng.choice(uniq_files, size=len(uniq_files), replace=True)
        rows = np.concatenate([file_to_rows[f] for f in samp])
        Yb, Bb, Pb = Yss[rows], base_ss[rows], pl_ss[rows]
        mb = macro_on_classes(Yb, Bb, common)
        mp = macro_on_classes(Yb, Pb, common)
        if not (np.isnan(mb) or np.isnan(mp)):
            deltas.append(mp - mb); b_list.append(mb); p_list.append(mp)
    deltas = np.array(deltas)
    ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
    excludes0 = bool(ci[0] > 0 or ci[1] < 0)
    print(f"[boot] Delta {delta:+.4f}  95% CI [{ci[0]:+.4f},{ci[1]:+.4f}] "
          f"(excl 0: {excludes0}) n_boot={len(deltas)}", flush=True)

    # ---- focal-CV effect (held-out fold 4) ----
    def focal_macro(tag):
        pr = np.load(os.path.join(od, f"{tag}_fold{args.fold}_seed{args.seed}.npy"))
        rows = np.load(os.path.join(od, f"{tag}_fold{args.fold}_seed{args.seed}_rows.npy"))
        Yf = Yfoc[rows]
        foc_classes = [c for c in common]  # same common set on focal where scorable
        return macro_on_classes(Yf, np.nan_to_num(pr, nan=0.0), foc_classes), Yf, pr
    base_focal, _, _ = focal_macro(args.base_tag)
    pl_focal, _, _ = focal_macro(args.pl_tag)
    delta_focal = pl_focal - base_focal
    print(f"[focal] base {base_focal:.4f} | PL {pl_focal:.4f} | Delta {delta_focal:+.4f}", flush=True)

    # ---- secondary: each model on its OWN trained set (does PL unlock classes?) ----
    base_own = [c for c in range(234) if tr_pos_base[c] and ss_scorable[c]]
    pl_own = [c for c in range(234) if tr_pos_pl[c] and ss_scorable[c]]
    base_macro_own = macro_on_classes(Yss, base_ss, base_own)
    pl_macro_own = macro_on_classes(Yss, pl_ss, pl_own)

    # ---- checkpoint metadata ----
    def ck_meta(p):
        c = torch.load(p, map_location="cpu")
        return {"best_val_macro_auc": round(float(c["best_val_macro_auc"]), 4),
                "best_epoch": int(c["best_epoch"])}

    verdict = ("PL ADDS target-domain headroom (CI excludes 0, Delta>0)" if (excludes0 and delta > 0)
               else "PL HURTS target domain (CI excludes 0, Delta<0)" if (excludes0 and delta < 0)
               else "NULL: no detectable PL target-domain headroom (CI includes 0)")

    result = {
        "task": "soundscape_pseudolabel_headroom_audit",
        "question": ("does soundscape pseudo-labeling (the field's dominant lever) add real "
                     "target-domain headroom in a clean de-biased audit, over a matched "
                     "focal-only baseline?"),
        "protocol": ("two CNNs, identical seed/config/fold (train folds 0-3, hold fold 4, "
                     "focal loss, mixup, SpecAugment, mel cache); BASELINE = focal only; "
                     "PL = focal + Perch-ZS pseudo-labeled unlabeled soundscapes; both evaluated "
                     "on the 66-file labeled soundscape set on a common class set."),
        "pl_config": pl_cfg,
        "soundscape_eval": {
            "n_files": len(uniq_files), "n_windows": int(W),
            "common_class_count": len(common),
            "baseline_macro_auc": round(base_macro, 4),
            "pl_macro_auc": round(pl_macro, 4),
            "delta_auc_pl_minus_baseline": round(float(delta), 4),
            "delta_auc_ci95_file_clustered": [round(ci[0], 4), round(ci[1], 4)],
            "ci_excludes_zero": excludes0,
            "n_boot": len(deltas),
        },
        "soundscape_own_trained_set": {
            "baseline_classes": len(base_own), "baseline_macro_auc": round(base_macro_own, 4),
            "pl_classes": len(pl_own), "pl_macro_auc": round(pl_macro_own, 4),
            "pl_extra_classes_from_pseudo": int(len(pl_own) - len(base_own)),
        },
        "focal_cv_effect": {
            "common_class_count": len(common),
            "baseline_focal_macro_auc": round(base_focal, 4),
            "pl_focal_macro_auc": round(pl_focal, 4),
            "delta_auc_focal": round(float(delta_focal), 4),
            "note": "auxiliary; PL should be neutral on clean focal CV",
        },
        "checkpoints": {
            "baseline": {"path": base_ck, **ck_meta(base_ck)},
            "pl": {"path": pl_ck, **ck_meta(pl_ck)},
        },
        "verdict": verdict,
    }
    out_path = os.path.join(od, "add_pseudolabel_audit.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE", out_path, flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
