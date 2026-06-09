"""B1' Task-S3 — STRONG orthogonal challenger: BEATs (microsoft, AudioSet/SSL) on Ascend NPU.

exp-plan-v4 S3: ONE strong orthogonal pretrained challenger as a PARALLEL LANE on the same
locked folds, evaluated through the SAME pooled-OOF / nested-calibration / fusion / oracle
pipeline as the PANN / author-CNN sidecars. Preferred = BEATs (general-audio SSL); AVES is
the fallback. Frozen-encoder + trainable head (768 -> 234), as v4 specifies.

WHY BEATs (not fairseq): the inference graph is self-contained pure-PyTorch
(BEATs.py/backbone.py/modules.py/kaldi.py) copied into src/beats_mod/. NO fairseq, NO torchaudio
import is needed for feature extraction (the local kaldi.fbank replaces torchaudio.compliance.kaldi;
the only torchaudio reference is in the MFCC/DCT path, never hit by fbank). Checkpoint
BEATs_iter3_plus_AS2M.pt (lpepino/beats_ckpts) is a plain {cfg, model} dict loaded with torch.load
-> no fairseq runtime. embed_dim=512, encoder_embed_dim=768, finetuned_model=None (encoder-only,
no AudioSet head), deep_norm=True.

DESIGN (mirrors b1_pann_train.py so B2' fuses identically):
  - fbank computed on CPU (local kaldi.fbank, 128 mel bins, 16 kHz, 25/10 ms) and PRECOMPUTED into
    a fp16 cache (data/fbank_cache/beats/fbank.npy [N, T, 128], row-aligned to the wav cache) by
    b1_build_fbank_cache.py -> per-epoch decode/fbank skipped (like the mel caches).
  - The wav cache is 32 kHz [N,160000] fp16; BEATs needs 16 kHz -> resample 32k->16k
    (scipy.resample_poly 1/2) at cache-build time, before fbank.
  - BEATs encoder is FROZEN (eval, no grad) and runs ON NPU each epoch from the cached fbank;
    only the linear head (768 -> 234) trains. SpecAugment is applied ON TOP of the cached base
    fbank at train time (val = no aug), keeping training stochastic + mathematically the same as
    on-the-fly. bf16 autocast on NPU, as in PANN.
  - OOF emitted in the SAME taxonomy-aligned (N,234) format + naming as PANN:
    {tag}_fold{f}_seed{s}.npy  + {tag}_fold{f}_seed{s}_rows.npy   (tag default = "beats").

DRY/SMOKE (--dry_run): 1 fold, 1 epoch, --dry_n clips/fold on ONE NPU; proves the pretrained
encoder loads + a head trains + OOF emits; prints sec/epoch + extrapolation. Use
ASCEND_RT_VISIBLE_DEVICES in {4,5,6} ONLY.
"""
import argparse
import ast
import os
import random
import sys
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

ROOT = os.path.expanduser("~/SHC/birdclef2026_clef")
sys.path.insert(0, os.path.join(ROOT, "src"))
from beats_mod.BEATs import BEATs, BEATsConfig  # noqa: E402

SR_SRC = 32000        # wav cache sample rate
SR_BEATS = 16000      # BEATs fbank sample rate
WIN_SRC = 5 * SR_SRC  # 160000
N_MELS = 128          # BEATs fbank bins
FBANK_MEAN = 15.41663
FBANK_STD = 6.55582


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if HAS_NPU:
        torch_npu.npu.manual_seed_all(seed)


def compute_macro_auc(y_true, y_pred):
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            col = y_pred[:, i]
            if np.isnan(col).any():
                continue
            try:
                aucs.append(roc_auc_score(y_true[:, i], col))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


# ----------------------------- CPU fbank (raw, pre-norm) -----------------------------
def wave_to_fbank(y32k):
    """RAW BEATs fbank [T, 128] (pre global-norm) from a 5 s 32 kHz waveform.
    Resample 32k->16k, scale *2^15, local kaldi.fbank (128 bins, 16k, 25/10 ms).
    The (x-mean)/(2*std) global norm is applied later (in the model fwd), so the cache stays the
    exact BEATs preprocess minus the cheap affine -> byte-equivalent encoder input after norm."""
    from scipy.signal import resample_poly
    from beats_mod.kaldi import fbank as kaldi_fbank
    y16 = resample_poly(y32k.astype(np.float32), 1, 2).astype(np.float32)  # 32k->16k
    w = torch.from_numpy(y16).unsqueeze(0) * (2 ** 15)
    fb = kaldi_fbank(w, num_mel_bins=N_MELS, sample_frequency=SR_BEATS,
                     frame_length=25, frame_shift=10)               # [T, 128]
    return fb.numpy().astype(np.float32)


# ----------------------------- dataset -----------------------------
class ClipDataset(Dataset):
    def __init__(self, filenames, labels, augment=False,
                 wav_cache=None, cache_rows=None, fbank_cache=None):
        self.filenames = filenames; self.labels = labels; self.augment = augment
        self.wav_cache = wav_cache; self.cache_rows = cache_rows
        # fbank_cache: precomputed RAW fbank fp16 memmap [Nall, T, 128]; when present we SKIP
        # resample+fbank. SpecAugment still applied ON TOP at train time.
        self.fbank_cache = fbank_cache

    def __len__(self):
        return len(self.filenames)

    def _spec_augment(self, fb, num_masks=2, freq_mask=16, time_mask=48):
        # fb: [T, 128]
        fb = fb.copy(); t, n_mels = fb.shape
        fill = fb.min()
        for _ in range(num_masks):
            f = random.randint(0, freq_mask); f0 = random.randint(0, max(0, n_mels - f))
            fb[:, f0:f0 + f] = fill
            tt = random.randint(0, time_mask); t0 = random.randint(0, max(0, t - tt))
            fb[t0:t0 + tt, :] = fill
        return fb

    def __getitem__(self, i):
        if self.fbank_cache is not None:
            fb = np.asarray(self.fbank_cache[self.cache_rows[i]]).astype(np.float32)  # [T,128]
        elif self.wav_cache is not None:
            y = np.asarray(self.wav_cache[self.cache_rows[i]]).astype(np.float32)
            fb = wave_to_fbank(y)
        else:
            raise RuntimeError("no fbank/wav cache available")
        if self.augment:
            fb = self._spec_augment(fb)
        x = torch.from_numpy(fb.copy()).float()                 # [T, 128]
        y = torch.from_numpy(self.labels[i].copy()).float()
        return x, y


# ----------------------------- BEATs frozen encoder + head -----------------------------
class BeatsHead(nn.Module):
    """Frozen BEATs encoder (runs on NPU) + trainable linear head 768 -> num_classes.
    Input = RAW fbank [B, T, 128] (pre global-norm); we apply BEATs' (x-mean)/(2*std) here,
    then run the patch_embedding/encoder stack (the body of extract_features, minus preprocess)."""
    def __init__(self, cfg: BEATsConfig, num_classes=234):
        super().__init__()
        self.beats = BEATs(cfg)
        self.head = nn.Linear(cfg.encoder_embed_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight); self.head.bias.data.zero_()

    def _encode(self, fbank_raw):
        # replicate extract_features() body after preprocess, with the global norm applied here
        b = self.beats
        fbank = (fbank_raw - FBANK_MEAN) / (2 * FBANK_STD)
        fbank = fbank.unsqueeze(1)                              # [B,1,T,128]
        features = b.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1).transpose(1, 2)
        features = b.layer_norm(features)
        if b.post_extract_proj is not None:
            features = b.post_extract_proj(features)
        x = b.dropout_input(features)
        x, _ = b.encoder(x, padding_mask=None)                 # [B, T', 768]
        return x.mean(dim=1)                                    # mean-pool over time -> [B,768]

    def forward(self, fbank_raw):
        with torch.no_grad():
            emb = self._encode(fbank_raw)
        return self.head(emb)


def load_pretrained_beats(ckpt_path, num_classes=234):
    """Build BEATs from the checkpoint cfg and load encoder weights; head is fresh.
    Returns (model, n_loaded, n_total, n_missing, n_unexpected)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(ck["cfg"])
    model = BeatsHead(cfg, num_classes=num_classes)
    sd = ck["model"]
    # checkpoint keys are bare BEATs keys; prefix with "beats." to match BeatsHead
    tgt = model.state_dict()
    to_load = {}
    for k, v in sd.items():
        kk = "beats." + k
        if kk in tgt and tgt[kk].shape == v.shape:
            to_load[kk] = v
    res = model.load_state_dict(to_load, strict=False)
    # freeze encoder
    for p in model.beats.parameters():
        p.requires_grad = False
    model.beats.eval()
    return (model, len(to_load), len(sd),
            len([m for m in res.missing_keys if not m.startswith("head")]),
            len(res.unexpected_keys))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        return ((1 - pt) ** self.gamma * bce).mean()


def mixup_data(x, y, alpha=0.5):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


# ----------------------------- labels -----------------------------
def load_taxonomy_order(data_dir):
    tax = pd.read_csv(os.path.join(data_dir, "taxonomy.csv"))
    cl = tax["primary_label"].astype(str).tolist()
    assert len(cl) == 234
    return cl


def build_targets(df, class_order):
    idx = {c: i for i, c in enumerate(class_order)}
    Y = np.zeros((len(df), len(class_order)), dtype=np.float32)
    for r, row in enumerate(df.itertuples(index=False)):
        p = str(getattr(row, "primary_label"))
        if p in idx:
            Y[r, idx[p]] = 1.0
        sec = getattr(row, "secondary_labels", None)
        if isinstance(sec, str) and sec.startswith("["):
            try:
                for s in ast.literal_eval(sec):
                    if str(s) in idx:
                        Y[r, idx[str(s)]] = 1.0
            except Exception:
                pass
    return Y


# ----------------------------- train one fold -----------------------------
def _predict_val(model, va_ld, device, device_type, C):
    model.eval(); preds = []
    with torch.no_grad():
        for x, _ in va_ld:
            x = x.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(x)
            preds.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros((0, C), np.float32)


def train_fold(df, Y, fold, args, device, device_type, wav_cache=None, fbank_cache=None):
    tr = df["fold"].values != fold
    va = df["fold"].values == fold
    use_rows = (wav_cache is not None) or (fbank_cache is not None)
    tr_rows = df["cache_row"].values[tr] if use_rows else None
    va_rows = df["cache_row"].values[va] if use_rows else None
    tr_ds = ClipDataset(df["filename"].values[tr], Y[tr], augment=True,
                        wav_cache=wav_cache, cache_rows=tr_rows, fbank_cache=fbank_cache)
    va_ds = ClipDataset(df["filename"].values[va], Y[va], augment=False,
                        wav_cache=wav_cache, cache_rows=va_rows, fbank_cache=fbank_cache)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.num_workers)

    model, n_loaded, n_total, n_missing, n_unexp = load_pretrained_beats(args.ckpt, Y.shape[1])
    model = model.to(device)
    print(f"  [pretrained] BEATs loaded {n_loaded}/{n_total} encoder tensors from "
          f"{os.path.basename(args.ckpt)}; non-head missing_in_model={n_missing} "
          f"unexpected={n_unexp} (head 768->{Y.shape[1]} fresh, encoder FROZEN)", flush=True)

    crit = FocalLoss() if args.loss == "focal" else nn.BCEWithLogitsLoss()
    # only the head trains
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)

    Yva = Y[va]
    sec_per_epoch = None
    best_auc, best_preds, best_epoch, since_improve = -1.0, None, 0, 0
    for epoch in range(args.epochs):
        model.train(); model.beats.eval()   # keep frozen encoder in eval (BN/dropout off)
        t0 = time.time(); tl = 0.0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            if args.mixup_alpha > 0:
                x, y = mixup_data(x, y, args.mixup_alpha)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                loss = crit(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); tl += loss.item()
        sec_per_epoch = time.time() - t0
        preds = _predict_val(model, va_ld, device, device_type, Y.shape[1])
        vauc = compute_macro_auc(Yva, np.nan_to_num(preds, nan=0.0))
        print(f"  [fold {fold}] epoch {epoch+1}/{args.epochs} "
              f"train_loss={tl/max(len(tr_ld),1):.4f} val_macro_auc={vauc:.4f} "
              f"({sec_per_epoch:.1f}s, {len(tr_ds)} clips)", flush=True)
        if vauc > best_auc + 1e-5:
            best_auc, best_preds, best_epoch, since_improve = vauc, preds, epoch + 1, 0
        else:
            since_improve += 1
            if since_improve >= args.patience:
                print(f"  [fold {fold}] EARLY-STOP @ epoch {epoch+1} "
                      f"(best val_auc={best_auc:.4f} @ ep{best_epoch})", flush=True)
                break
    if best_preds is None:
        best_preds = _predict_val(model, va_ld, device, device_type, Y.shape[1])
    return va, best_preds, sec_per_epoch, int(tr.sum()), best_auc, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--folds_csv", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--ckpt", default=None,
                    help="BEATs .pt (default <data_dir>/beats_ckpt/BEATs_iter3_plus_AS2M.pt)")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="focal", choices=["bce", "focal"])
    ap.add_argument("--mixup_alpha", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--wav_cache", default=None)
    ap.add_argument("--fbank_cache", default=None,
                    help="precomputed RAW fbank fp16 cache .npy [Nall,T,128] "
                         "(default <data_dir>/fbank_cache/beats/fbank.npy if present)")
    ap.add_argument("--tag", default="beats")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dry_n", type=int, default=40)
    args = ap.parse_args()

    if args.folds_csv is None:
        cand = os.path.join(os.path.dirname(args.data_dir.rstrip("/")), "folds.csv")
        args.folds_csv = cand if os.path.exists(cand) else os.path.join(args.data_dir, "folds.csv")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "oof")
    if args.ckpt is None:
        args.ckpt = os.path.join(args.data_dir, "beats_ckpt", "BEATs_iter3_plus_AS2M.pt")
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)

    if HAS_NPU and "npu" in args.device:
        device = torch.device(args.device); torch_npu.npu.set_device(device)
        device_type = "npu"
    elif torch.cuda.is_available():
        device = torch.device("cuda:0"); device_type = "cuda"
    else:
        device = torch.device("cpu"); device_type = "cpu"
    print(f"[cfg] device={device} dry_run={args.dry_run} HAS_NPU={HAS_NPU} "
          f"ASCEND={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} ckpt={args.ckpt}", flush=True)

    df = pd.read_csv(args.folds_csv)
    df["cache_row"] = np.arange(len(df))
    meta = pd.read_csv(os.path.join(args.data_dir, "train.csv"))[
        ["filename", "primary_label", "secondary_labels", "author"]]
    df = df.merge(meta, on="filename", how="left")
    assert df["author"].notna().all(), "author join NaN -> grouping broken"

    if args.fbank_cache is None:
        cand = os.path.join(args.data_dir, "fbank_cache", "beats", "fbank.npy")
        args.fbank_cache = cand if os.path.exists(cand) else None
    fbank_cache = None
    if args.fbank_cache and os.path.exists(args.fbank_cache):
        fbank_cache = np.load(args.fbank_cache, mmap_mode="r")
        print(f"[cache] using fbank_cache {args.fbank_cache} shape {fbank_cache.shape} "
              f"(FAST: base fbank)", flush=True)

    if args.wav_cache is None:
        cand = os.path.join(args.data_dir, "wav_cache", "wav_cache.npy")
        args.wav_cache = cand if os.path.exists(cand) else None
    wav_cache = None
    if args.wav_cache and os.path.exists(args.wav_cache):
        wav_cache = np.load(args.wav_cache, mmap_mode="r")
        print(f"[cache] wav_cache available {args.wav_cache} shape {wav_cache.shape}", flush=True)
    if fbank_cache is None and wav_cache is None:
        raise RuntimeError("need fbank_cache or wav_cache")
    elif fbank_cache is None:
        print("[cache] no fbank_cache -> fbank computed on the fly from wav_cache (SLOW)", flush=True)

    folds_to_run = [int(x) for x in args.folds.split(",")]
    if args.dry_run:
        folds_to_run = folds_to_run[:1]; args.epochs = 1
        df = df.groupby("fold", group_keys=False).head(args.dry_n).reset_index(drop=True)
        print(f"[DRY] {len(df)} clips, fold {folds_to_run}, 1 epoch", flush=True)

    class_order = load_taxonomy_order(args.data_dir)
    Y = build_targets(df, class_order)
    N, C = Y.shape
    oof = np.full((N, C), np.nan, dtype=np.float32)

    suffix = "_dry" if args.dry_run else ""
    spe_list = []
    for f in folds_to_run:
        va_mask, preds, spe, n_tr, best_auc, best_ep = train_fold(
            df, Y, f, args, device, device_type, wav_cache=wav_cache, fbank_cache=fbank_cache)
        tr_pos = (Y[df["fold"].values != f].sum(0) > 0)
        block = preds.copy(); block[:, ~tr_pos] = np.nan
        oof[va_mask] = block
        auc = compute_macro_auc(Y[va_mask], np.nan_to_num(preds, nan=0.0))
        print(f"  [fold {f}] BEST VAL macro-AUC = {best_auc:.4f} @ep{best_ep}  "
              f"(recheck {auc:.4f}) emitted {preds.shape[0]} rows x {C} cols", flush=True)
        run_rows = df["cache_row"].values[va_mask]
        rtag = f"{args.tag}_fold{f}_seed{args.seed}{suffix}"
        np.save(os.path.join(args.out_dir, f"{rtag}.npy"), block.astype(np.float32))
        np.save(os.path.join(args.out_dir, f"{rtag}_rows.npy"), run_rows.astype(np.int64))
        print(f"[run-oof] {rtag}.npy {block.shape} rows->{rtag}_rows.npy", flush=True)
        if spe:
            spe_list.append((spe, n_tr))

    out = os.path.join(args.out_dir, f"beats_sidecar_oof{suffix}.npy")
    np.save(out, oof)
    df[["filename", "fold", "author"]].to_csv(
        os.path.join(args.out_dir, f"beats_oof_meta{suffix}.csv"), index=False)
    np.save(os.path.join(args.out_dir, f"beats_oof_targets{suffix}.npy"), Y)
    print(f"[done] BEATs sidecar OOF -> {out} {oof.shape} "
          f"(non-NaN rows={int((~np.isnan(oof).all(1)).sum())})", flush=True)

    if spe_list:
        spe, n_tr = spe_list[0]
        per_clip = spe / max(n_tr, 1)
        full_train_per_fold = 35549 * 4 / 5
        sec_full_epoch = per_clip * full_train_per_fold
        epochs_real = 15
        h_5fold = sec_full_epoch * epochs_real * 5 / 3600.0
        print(f"[extrap] smoke {spe:.1f}s for {n_tr} train clips -> "
              f"{per_clip*1000:.1f} ms/clip; full-fold epoch ~{sec_full_epoch/60:.1f} min; "
              f"5-fold x {epochs_real}ep ~= {h_5fold:.1f} NPU-h (single card). "
              f"NOTE: extrapolated from ONE tiny epoch; remeasure 1st real epoch.", flush=True)


if __name__ == "__main__":
    main()
