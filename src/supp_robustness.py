"""SUPP #6 — full robustness gap for PANN and BEATs (mirrors b3_cnn_soundscape.py).

Trains the sidecar on folds 0-3 (holds fold 4) WITH a saved checkpoint, then re-infers
it on the SAME 66-file / 1478-window labeled soundscape set used by domain_shift.json,
and computes its focal->soundscape macro-AUC drop vs Perch zero-shot's drop (0.928->0.737,
drop 0.0393). Verdict per model: does "foundation embedding (Perch) is more robust than the
from-scratch / pretrained challenger" hold? Writes data/oof/supp_rg_{model}.json.

The CNN was already done by b3_cnn_soundscape.py (add_robustness_gap.json); this closes
the same gap for PANN (CNN14, AudioSet-pretrained) and BEATs (frozen SSL encoder + head).
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch

ROOT = os.path.expanduser("~/SHC/birdclef2026_clef")
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    import torch_npu  # noqa
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

import b1_pann_train as MP
import b1_beats_train as MB
from b3_domain_shift import (load_perch_labels, build_map, hhmmss_to_sec,
                             load_window, macro_auc_per_class)

SR = 32000
WIN = 5 * SR


def build_df(data_dir):
    df = pd.read_csv(os.path.join(ROOT, "folds.csv"))
    df["cache_row"] = np.arange(len(df))
    meta = pd.read_csv(os.path.join(data_dir, "train.csv"))[
        ["filename", "primary_label", "secondary_labels", "author"]]
    df = df.merge(meta, on="filename", how="left")
    return df


def train_fold4(model, df, Y, args, device, device_type):
    """Train on folds 0-3, validate on fold 4, return (model, best_preds_fold4,
    val_rows, best_auc, best_epoch)."""
    from torch.utils.data import DataLoader
    data_dir = args.data_dir
    fv = df["fold"].values
    tr = fv != 4
    va = fv == 4
    audio_root = os.path.join(data_dir, "train_audio")
    if model == "pann":
        mod = MP
        cache = np.load(os.path.join(data_dir, "mel_cache", "pann", "mel.npy"), mmap_mode="r")
        tr_ds = MP.ClipDataset(df["filename"].values[tr], Y[tr], audio_root, augment=True,
                               cache_rows=df["cache_row"].values[tr], mel_cache=cache)
        va_ds = MP.ClipDataset(df["filename"].values[va], Y[va], audio_root, augment=False,
                               cache_rows=df["cache_row"].values[va], mel_cache=cache)
        m = MP.Cnn14(num_classes=Y.shape[1])
        MP.load_pretrained_cnn14(m, os.path.join(data_dir, "panns_ckpt", "Cnn14_mAP0.431.pth"))
        lr = 5e-4
    else:
        mod = MB
        cache = np.load(os.path.join(data_dir, "fbank_cache", "beats", "fbank.npy"), mmap_mode="r")
        tr_ds = MB.ClipDataset(df["filename"].values[tr], Y[tr], augment=True,
                               cache_rows=df["cache_row"].values[tr], fbank_cache=cache)
        va_ds = MB.ClipDataset(df["filename"].values[va], Y[va], augment=False,
                               cache_rows=df["cache_row"].values[va], fbank_cache=cache)
        m, *_ = MB.load_pretrained_beats(
            os.path.join(data_dir, "beats_ckpt", "BEATs_iter3_plus_AS2M.pt"), Y.shape[1])
        lr = 1e-3
    m = m.to(device)
    g = torch.Generator(); g.manual_seed(args.seed + 4)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=False, generator=g)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.num_workers)
    crit = mod.FocalLoss()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    Yva = Y[va]
    best_auc, best_preds, best_ep, since, best_state = -1.0, None, 0, 0, None
    for ep in range(args.epochs):
        m.train()
        if model == "beats":
            m.beats.eval()
        t0 = time.time(); tl = 0.0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            if args.mixup_alpha > 0:
                x, y = mod.mixup_data(x, y, args.mixup_alpha)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                loss = crit(m(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); tl += loss.item()
        preds = mod._predict_val(m, va_ld, device, device_type, Y.shape[1])
        vauc = mod.compute_macro_auc(Yva, np.nan_to_num(preds, nan=0.0))
        print(f"  [{model} fold4] ep{ep+1}/{args.epochs} loss={tl/max(len(tr_ld),1):.4f} "
              f"val_auc={vauc:.4f} ({time.time()-t0:.1f}s)", flush=True)
        if vauc > best_auc + 1e-5:
            best_auc, best_preds, best_ep, since = vauc, preds, ep + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            since += 1
            if since >= args.patience:
                print(f"  EARLY-STOP @ep{ep+1} (best {best_auc:.4f}@ep{best_ep})", flush=True)
                break
    # restore best weights for soundscape inference
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    return m, best_preds, df["cache_row"].values[va], best_auc, best_ep, best_state


def soundscape_preds(model, m, df_lab, ss_dir, device, device_type, batch_size=32):
    W = len(df_lab)
    starts = [hhmmss_to_sec(s) for s in df_lab["start"].tolist()]
    files = df_lab["filename"].tolist()
    out = np.zeros((W, 234), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for j in range(0, W, batch_size):
            bs = min(batch_size, W - j)
            xs = []
            for k in range(bs):
                y = load_window(os.path.join(ss_dir, files[j + k]), starts[j + k])
                if model == "pann":
                    mel = MP.wave_to_logmel(y)            # [64, T]
                    xs.append(mel.T[None])                # [1, T, 64]
                else:
                    fb = MB.wave_to_fbank(y)              # [T, 128]
                    xs.append(fb)
            x = torch.from_numpy(np.stack(xs)).float().to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = m(x)
            out[j:j + bs] = torch.sigmoid(logits).float().cpu().numpy()
            if j % (batch_size * 5) == 0:
                print(f"  [ss {model}] {j+bs}/{W} {(j+bs)/max(time.time()-t0,1e-6):.1f} win/s",
                      flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["pann", "beats"])
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--mixup_alpha", type=float, default=0.5)
    args = ap.parse_args()
    args.data_dir = os.path.join(ROOT, "data")
    if args.epochs == 0:
        args.epochs = {"pann": 20, "beats": 15}[args.model]
    data_dir = args.data_dir
    out_dir = os.path.join(data_dir, "oof")
    rg_dir = os.path.join(data_dir, "oof_rg")
    os.makedirs(rg_dir, exist_ok=True)

    MP.seed_everything(args.seed)
    if HAS_NPU and "npu" in args.device:
        device = torch.device(args.device); torch_npu.npu.set_device(device); device_type = "npu"
    elif torch.cuda.is_available():
        device = torch.device("cuda:0"); device_type = "cuda"
    else:
        device = torch.device("cpu"); device_type = "cpu"
    print(f"[cfg] model={args.model} device={device} epochs={args.epochs} "
          f"ASCEND={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}", flush=True)

    tax = pd.read_csv(os.path.join(data_dir, "taxonomy.csv"))
    class_order = MP.load_taxonomy_order(data_dir)
    cls_idx = {c: i for i, c in enumerate(class_order)}
    df = build_df(data_dir)
    Y_all = MP.build_targets(df, class_order)

    # --- train fold4 (folds 0-3) ---
    m, focal_preds, focal_rows, best_auc, best_ep, best_state = train_fold4(
        args.model, df, Y_all, args, device, device_type)
    ckpt_path = os.path.join(rg_dir, f"{args.model}rg_fold4_seed{args.seed}.pt")
    torch.save({"state_dict": best_state, "model": args.model,
                "best_val_macro_auc": float(best_auc), "best_epoch": int(best_ep),
                "num_classes": 234}, ckpt_path)
    print(f"[ckpt] saved {ckpt_path} best_val={best_auc:.4f}@ep{best_ep}", flush=True)

    tr_pos = (Y_all[df["fold"].values != 4].sum(0) > 0)
    n_trained = int(tr_pos.sum())

    # --- focal per-class AUC on fold4 (trained classes only) ---
    Yf = Y_all[focal_rows]
    fp = focal_preds.copy(); fp[:, ~tr_pos] = np.nan
    focal_auc = macro_auc_per_class(Yf, np.nan_to_num(fp, nan=0.0))
    focal_auc = {i: v for i, v in focal_auc.items() if tr_pos[i]}
    focal_macro = float(np.mean(list(focal_auc.values()))) if focal_auc else 0.0
    print(f"[{args.model}] focal fold4 macro={focal_macro:.4f} over {len(focal_auc)} cls", flush=True)

    # --- soundscape inference ---
    lab = pd.read_csv(os.path.join(data_dir, "train_soundscapes_labels.csv"))
    ss_dir = os.path.join(data_dir, "train_soundscapes")
    W = len(lab)
    Yss = np.zeros((W, 234), dtype=np.float32)
    for r, row in enumerate(lab.itertuples(index=False)):
        for c in str(row.primary_label).split(";"):
            c = c.strip()
            if c in cls_idx:
                Yss[r, cls_idx[c]] = 1.0
    ss_cache = os.path.join(rg_dir, f"ss_{args.model}rg_preds.npy")
    ss = soundscape_preds(args.model, m, lab, ss_dir, device, device_type, args.batch_size)
    np.save(ss_cache, ss)
    ss_m = ss.copy(); ss_m[:, ~tr_pos] = np.nan
    ss_auc = macro_auc_per_class(Yss, np.nan_to_num(ss_m, nan=0.0))
    ss_auc = {i: v for i, v in ss_auc.items() if tr_pos[i]}
    ss_macro = float(np.mean(list(ss_auc.values()))) if ss_auc else 0.0
    print(f"[{args.model}] soundscape macro={ss_macro:.4f} over {len(ss_auc)} cls", flush=True)

    # --- Perch focal/soundscape recompute (matched-class) ---
    perch_dir = os.path.join(data_dir, "perch_labels")
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

    with open(os.path.join(out_dir, "domain_shift.json")) as f:
        ds = json.load(f)
    perch_drop_json = ds["supervised_perch_zeroshot"]["mean_per_class_auc_drop_focal_minus_soundscape"]

    # model native-common drop
    common = sorted(set(ss_auc) & set(focal_auc))
    m_focal_c = float(np.mean([focal_auc[i] for i in common]))
    m_ss_c = float(np.mean([ss_auc[i] for i in common]))
    m_drop = m_focal_c - m_ss_c
    perch_common = sorted(set(perch_ss_auc) & set(perch_focal_auc))
    # matched (apples-to-apples) joint class set
    joint = sorted(set(common) & set(perch_common))
    m_focal_j = float(np.mean([focal_auc[i] for i in joint])) if joint else None
    m_ss_j = float(np.mean([ss_auc[i] for i in joint])) if joint else None
    m_drop_j = (m_focal_j - m_ss_j) if joint else None
    p_focal_j = float(np.mean([perch_focal_auc[i] for i in joint])) if joint else None
    p_ss_j = float(np.mean([perch_ss_auc[i] for i in joint])) if joint else None
    p_drop_j = (p_focal_j - p_ss_j) if joint else None
    perch_more_robust = bool(m_drop_j > p_drop_j) if joint else None

    result = {
        "task": f"robustness_gap_{args.model}_vs_perch_focal_to_soundscape",
        "checkpoint": {"path": ckpt_path, "seed": args.seed, "held_fold": 4,
                       "best_val_macro_auc_native": round(float(best_auc), 4),
                       "best_epoch": int(best_ep), "n_trained_classes": n_trained},
        f"{args.model}_focal_auc": round(m_focal_c, 4),
        f"{args.model}_soundscape_auc": round(m_ss_c, 4),
        f"{args.model}_drop": round(m_drop, 4),
        "perch_drop": round(float(perch_drop_json), 4),
        "gap": round(m_drop - perch_drop_json, 4),
        "common_class_count": len(common),
        f"{args.model}_focal_macro_native": round(focal_macro, 4),
        f"{args.model}_soundscape_macro_native": round(ss_macro, 4),
        f"{args.model}_focal_eval_classes": len(focal_auc),
        f"{args.model}_soundscape_eval_classes": len(ss_auc),
        "perch_focal_macro_full": ds["supervised_perch_zeroshot"]["focal_macro_auc_full"],
        "perch_soundscape_macro": ds["supervised_perch_zeroshot"]["soundscape_macro_auc"],
        "matched_class": {
            "n_joint_classes": len(joint),
            f"{args.model}_focal_auc": round(m_focal_j, 4) if joint else None,
            f"{args.model}_soundscape_auc": round(m_ss_j, 4) if joint else None,
            f"{args.model}_drop": round(m_drop_j, 4) if joint else None,
            "perch_focal_auc": round(p_focal_j, 4) if joint else None,
            "perch_soundscape_auc": round(p_ss_j, 4) if joint else None,
            "perch_drop": round(p_drop_j, 4) if joint else None,
            "gap_model_minus_perch": round(m_drop_j - p_drop_j, 4) if joint else None,
            "perch_more_robust_than_model": perch_more_robust,
        },
        "interpretation": ("drop=focal_AUC-soundscape_AUC (larger drop=LESS robust). "
                           "gap=model_drop-perch_drop; gap>0 => Perch more robust. matched_class "
                           "uses the identical class set for both models."),
    }
    out_path = os.path.join(out_dir, f"supp_rg_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE", out_path, flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
