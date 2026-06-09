"""SUPP #1 — nested-CV inner-OOF base trainer (ONE SEED).

For the oracle/selector to never see base predictions trained on the OUTER fold, we
need inner-OOF base preds: for each unordered pair {a,b} of folds, train the base
model on the 3 remaining folds S = {0..4}\{a,b} and predict folds a and b. Then the
prediction on fold g (trained WITHOUT g and WITHOUT f) serves the selection set of
outer fold f (for the pair {f,g}). There are C(5,2)=10 such 3-fold models per base.

Early-stop mirrors the global OOF procedure: validate on the union of the two held
folds (a,b), keep best-epoch preds. This removes ONLY the outer-fold leakage relative
to the global one-seed OOF, holding everything else identical -> a clean nested-vs-global
comparison.

Reuses the locked model / dataset / loss primitives from the b1_* base trainers
(imported, not modified). Writes per-(model,pair,fold) preds + row indices to
data/oof/supp_nested/.  ONE NPU per invocation (pass --device npu:0 with
ASCEND_RT_VISIBLE_DEVICES pinning to a free card 4/5/6).
"""
import argparse, itertools, os, random, sys, time
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

import b1_cnn_train as M_CNN
import b1_pann_train as M_PANN
import b1_beats_train as M_BEATS


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if HAS_NPU:
        torch_npu.npu.manual_seed_all(seed)


def build_df(data_dir):
    folds_csv = os.path.join(ROOT, "folds.csv")
    df = pd.read_csv(folds_csv)
    df["cache_row"] = np.arange(len(df))
    meta = pd.read_csv(os.path.join(data_dir, "train.csv"))[
        ["filename", "primary_label", "secondary_labels", "author"]]
    df = df.merge(meta, on="filename", how="left")
    assert df["author"].notna().all()
    return df


def make_model(model, C, args):
    if model == "cnn":
        return M_CNN.BirdCLEFModel(args.backbone, C, pretrained=True)
    if model == "pann":
        m = M_PANN.Cnn14(num_classes=C)
        ckpt = os.path.join(args.data_dir, "panns_ckpt", "Cnn14_mAP0.431.pth")
        M_PANN.load_pretrained_cnn14(m, ckpt)
        return m
    if model == "beats":
        ckpt = os.path.join(args.data_dir, "beats_ckpt", "BEATs_iter3_plus_AS2M.pt")
        m, *_ = M_BEATS.load_pretrained_beats(ckpt, C)
        return m
    raise ValueError(model)


def make_ds(model, files, labels, rows, augment, audio_root, args, caches):
    if model == "cnn":
        return M_CNN.ClipDataset(files, labels, audio_root, args.img_size, augment=augment,
                                 cache_rows=rows, mel_cache=caches)
    if model == "pann":
        return M_PANN.ClipDataset(files, labels, audio_root, augment=augment,
                                  cache_rows=rows, mel_cache=caches)
    if model == "beats":
        return M_BEATS.ClipDataset(files, labels, augment=augment,
                                   cache_rows=rows, fbank_cache=caches)


def load_cache(model, data_dir):
    if model == "cnn":
        p = os.path.join(data_dir, "mel_cache", "cnn", "mel.npy")
    elif model == "pann":
        p = os.path.join(data_dir, "mel_cache", "pann", "mel.npy")
    else:
        p = os.path.join(data_dir, "fbank_cache", "beats", "fbank.npy")
    assert os.path.exists(p), f"cache missing {p}"
    return np.load(p, mmap_mode="r")


def train_subset(model, df, Y, S, held, args, device, device_type, cache):
    """Train on folds in S, validate/predict on the two held folds. Returns
    (val_rows, val_folds, best_preds[n_val,C], best_auc, best_epoch)."""
    audio_root = os.path.join(args.data_dir, "train_audio")
    fv = df["fold"].values
    tr = np.isin(fv, list(S))
    va = np.isin(fv, list(held))
    from torch.utils.data import DataLoader
    bs_mod = M_CNN if model == "cnn" else (M_PANN if model == "pann" else M_BEATS)
    tr_ds = make_ds(model, df["filename"].values[tr], Y[tr], df["cache_row"].values[tr],
                    True, audio_root, args, cache)
    va_ds = make_ds(model, df["filename"].values[va], Y[va], df["cache_row"].values[va],
                    False, audio_root, args, cache)
    g = torch.Generator(); g.manual_seed(args.seed + int(min(held)))
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=False, generator=g)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.num_workers)
    m = make_model(model, Y.shape[1], args).to(device)
    crit = bs_mod.FocalLoss()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    Yva = Y[va]
    best_auc, best_preds, best_ep, since = -1.0, None, 0, 0
    for ep in range(args.epochs):
        m.train()
        if model == "beats":
            m.beats.eval()
        t0 = time.time(); tl = 0.0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            if args.mixup_alpha > 0:
                x, y = bs_mod.mixup_data(x, y, args.mixup_alpha)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                loss = crit(m(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); tl += loss.item()
        preds = bs_mod._predict_val(m, va_ld, device, device_type, Y.shape[1])
        vauc = bs_mod.compute_macro_auc(Yva, np.nan_to_num(preds, nan=0.0))
        print(f"    [{model} S={sorted(S)} held={sorted(held)}] ep{ep+1}/{args.epochs} "
              f"loss={tl/max(len(tr_ld),1):.4f} val_auc={vauc:.4f} ({time.time()-t0:.1f}s)",
              flush=True)
        if vauc > best_auc + 1e-5:
            best_auc, best_preds, best_ep, since = vauc, preds, ep + 1, 0
        else:
            since += 1
            if since >= args.patience:
                print(f"    EARLY-STOP @ep{ep+1} (best {best_auc:.4f}@ep{best_ep})", flush=True)
                break
    if best_preds is None:
        best_preds = bs_mod._predict_val(m, va_ld, device, device_type, Y.shape[1])
    return df["cache_row"].values[va], fv[va], best_preds, best_auc, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cnn", "pann", "beats"])
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--mixup_alpha", type=float, default=0.5)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--backbone", default="tf_efficientnet_b0_ns")
    args = ap.parse_args()
    args.data_dir = os.path.join(ROOT, "data")
    # per-model defaults matching the base trainers
    if args.epochs == 0:
        args.epochs = {"cnn": 30, "pann": 20, "beats": 15}[args.model]
    if args.lr == 0.0:
        args.lr = {"cnn": 1e-3, "pann": 5e-4, "beats": 1e-3}[args.model]
    out_dir = os.path.join(args.data_dir, "oof", "supp_nested")
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(args.seed)

    if HAS_NPU and "npu" in args.device:
        device = torch.device(args.device); torch_npu.npu.set_device(device); device_type = "npu"
    elif torch.cuda.is_available():
        device = torch.device("cuda:0"); device_type = "cuda"
    else:
        device = torch.device("cpu"); device_type = "cpu"
    print(f"[cfg] model={args.model} device={device} epochs={args.epochs} lr={args.lr} "
          f"ASCEND={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}", flush=True)

    df = build_df(args.data_dir)
    class_order = M_CNN.load_taxonomy_order(args.data_dir)
    Y = M_CNN.build_targets(df, class_order)
    cache = load_cache(args.model, args.data_dir)
    print(f"[cache] {args.model} {cache.shape}", flush=True)

    folds = sorted(df["fold"].unique())
    pairs = list(itertools.combinations(folds, 2))  # 10
    t_all = time.time()
    for pi, (a, b) in enumerate(pairs):
        held = {a, b}; S = set(folds) - held
        tag = f"{args.model}_ex{a}_{b}_seed{args.seed}"
        done_a = os.path.join(out_dir, f"{tag}_fold{a}.npy")
        done_b = os.path.join(out_dir, f"{tag}_fold{b}.npy")
        if os.path.exists(done_a) and os.path.exists(done_b):
            print(f"[skip] pair {pi+1}/10 {{ {a},{b} }} already done", flush=True)
            continue
        print(f"[pair {pi+1}/10] held={{ {a},{b} }} train S={sorted(S)}", flush=True)
        val_rows, val_folds, preds, bauc, bep = train_subset(
            args.model, df, Y, S, held, args, device, device_type, cache)
        tr_pos = (Y[np.isin(df["fold"].values, list(S))].sum(0) > 0)
        block = preds.copy(); block[:, ~tr_pos] = np.nan
        for f in (a, b):
            sel = val_folds == f
            np.save(os.path.join(out_dir, f"{tag}_fold{f}.npy"), block[sel].astype(np.float32))
            np.save(os.path.join(out_dir, f"{tag}_fold{f}_rows.npy"),
                    val_rows[sel].astype(np.int64))
        print(f"  [pair {pi+1}/10] best_auc={bauc:.4f}@ep{bep} saved fold{a}&fold{b}", flush=True)
    print(f"[done] {args.model} nested inner-OOF in {(time.time()-t_all)/60:.1f} min -> {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
