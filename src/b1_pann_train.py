"""B1' Task-2 — Strong PANNs CNN14 sidecar (AudioSet-pretrained) on Ascend NPU.

Codex round-1 verdict (d): the headline "extra models do not complement Perch"
claim is under-earned with only a from-scratch EfficientNet. This adds a SERIOUS
non-Perch pretrained model: PANNs CNN14 (Kong et al., 2020), AudioSet-pretrained
(mAP=0.431), head adapted 527 -> 234 BirdCLEF classes.

DESIGN (per task + Codex (d)):
  - log-mel / fbank computed on CPU (librosa), model trained/inferred on NPU
    (torch_npu). CNN14 is pure conv/BN/pool -> safest torch_npu candidate.
  - We re-implement the canonical qiuqiangkong CNN14 graph and load the pretrained
    state_dict (Cnn14_mAP=0.431.pth) EXCEPT its built-in Spectrogram/LogmelFilterBank
    frontend (we feed CPU log-mel directly) and the final fc_audioset (527) which is
    replaced by a fresh 234-class head. We report exactly how many tensors loaded.
  - OOF emitted in the SAME taxonomy-aligned (N,234) format as b1_perch_extract.py /
    b1_cnn_train.py so B2' fuses anchor + this sidecar on identical rows.

PRETRAINED WEIGHTS:
  data/panns_ckpt/Cnn14_mAP0.431.pth  (from HF thelou1s/panns-inference, 327 MB).
  The checkpoint dict has key "model" holding the state_dict.

CNN14 frontend params (must match pretraining for the conv stack to transfer):
  sr=32000, n_fft=1024, hop=320, n_mels=64, fmin=50, fmax=14000, power log-mel
  (these are the AudioSet PANNs settings; we replicate on CPU with librosa).

DRY/SMOKE (--dry_run): 1 fold, 1 epoch, --dry_n clips/fold on ONE NPU; prints
sec/epoch and 5-fold extrapolation; writes the dry OOF block. Does NOT launch full
training. Use ASCEND_RT_VISIBLE_DEVICES=5 (NPU 5 ONLY).
"""
import argparse
import ast
import os
import random
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

SR = 32000
WIN = 5 * SR
N_FFT = 1024
HOP = 320
N_MELS = 64          # CNN14 AudioSet pretraining uses 64 mel bins
FMIN = 50
FMAX = 14000


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


# ----------------------------- CPU log-mel -----------------------------
def wave_to_logmel(y):
    """CPU log-mel [n_mels, T] from an already-decoded 5 s waveform."""
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    return librosa.power_to_db(m, ref=1.0, top_db=None).astype(np.float32)


def clip_to_logmel(path):
    """CPU log-mel [n_mels, T] with PANNs/AudioSet frontend params."""
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) >= WIN:
        s = (len(y) - WIN) // 2
        y = y[s:s + WIN]
    else:
        y = np.pad(y, (0, WIN - len(y)))
    return wave_to_logmel(y)  # [n_mels, T]


# ----------------------------- CNN14 (canonical qiuqiangkong graph) ---------
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        return x


class Cnn14(nn.Module):
    """Canonical CNN14 trunk (matches qiuqiangkong state_dict keys), but WITHOUT
    the built-in spectrogram/logmel/SpecAug frontend: forward() takes a precomputed
    log-mel [B, 1, T, n_mels]. fc_audioset replaced by a 234-class head."""
    def __init__(self, num_classes=234):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, num_classes, bias=True)  # fresh head
        init_bn(self.bn0); init_layer(self.fc1); init_layer(self.fc_audioset)

    def forward(self, x):
        # x: [B, 1, T, n_mels] (time x freq). bn0 normalizes the freq axis.
        x = x.transpose(1, 3)            # [B, n_mels, T, 1]
        x = self.bn0(x)
        x = x.transpose(1, 3)            # [B, 1, T, n_mels]
        x = self.conv_block1(x, (2, 2), "avg"); x = F.dropout(x, 0.2, self.training)
        x = self.conv_block2(x, (2, 2), "avg"); x = F.dropout(x, 0.2, self.training)
        x = self.conv_block3(x, (2, 2), "avg"); x = F.dropout(x, 0.2, self.training)
        x = self.conv_block4(x, (2, 2), "avg"); x = F.dropout(x, 0.2, self.training)
        x = self.conv_block5(x, (2, 2), "avg"); x = F.dropout(x, 0.2, self.training)
        x = self.conv_block6(x, (1, 1), "avg"); x = F.dropout(x, 0.2, self.training)
        x = torch.mean(x, dim=3)         # mean over freq -> [B, C, T]
        (x1, _) = torch.max(x, dim=2)    # max over time
        x2 = torch.mean(x, dim=2)        # mean over time
        x = x1 + x2
        x = F.dropout(x, 0.5, self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, 0.5, self.training)
        return self.fc_audioset(x)       # logits [B, num_classes]


def load_pretrained_cnn14(model, ckpt_path):
    """Load AudioSet-pretrained CNN14 weights into the trunk, skipping the
    frontend (spectrogram_extractor/logmel_extractor/spec_augmenter) and the old
    527-class fc_audioset. Returns (n_loaded, n_total_model, skipped_keys)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    model_sd = model.state_dict()
    to_load, skipped = {}, []
    for k, v in sd.items():
        if k.startswith(("spectrogram_extractor", "logmel_extractor", "spec_augmenter")):
            skipped.append(k); continue
        if k in model_sd and model_sd[k].shape == v.shape:
            to_load[k] = v
        else:
            skipped.append(k)             # shape-mismatch (fc_audioset 527->234)
    missing = model.load_state_dict(to_load, strict=False)
    n_loaded = len(to_load)
    return n_loaded, len(model_sd), skipped, missing


# ----------------------------- dataset -----------------------------
class ClipDataset(Dataset):
    def __init__(self, filenames, labels, audio_root, augment=False,
                 wav_cache=None, cache_rows=None, mel_cache=None):
        self.filenames = filenames; self.labels = labels
        self.audio_root = audio_root; self.augment = augment
        self.wav_cache = wav_cache; self.cache_rows = cache_rows
        # mel_cache: precomputed BASE log-mel fp16 memmap [Nall, 64, T], row-aligned to
        # cache_rows. When present we SKIP decode+mel; SpecAugment still applied ON TOP
        # at train time so training stays mathematically equivalent.
        self.mel_cache = mel_cache

    def __len__(self):
        return len(self.filenames)

    def _spec_augment(self, mel, num_masks=2, freq_mask=8, time_mask=40):
        mel = mel.copy(); n_mels, t = mel.shape
        for _ in range(num_masks):
            f = random.randint(0, freq_mask); f0 = random.randint(0, max(0, n_mels - f))
            mel[f0:f0 + f, :] = mel.min()
            tt = random.randint(0, time_mask); t0 = random.randint(0, max(0, t - tt))
            mel[:, t0:t0 + tt] = mel.min()
        return mel

    def __getitem__(self, i):
        if self.mel_cache is not None:
            # deterministic BASE log-mel from cache (fp16 -> fp32); identical every epoch
            mel = np.asarray(self.mel_cache[self.cache_rows[i]]).astype(np.float32)
        elif self.wav_cache is not None:
            y = np.asarray(self.wav_cache[self.cache_rows[i]]).astype(np.float32)
            mel = wave_to_logmel(y)
        else:
            mel = clip_to_logmel(os.path.join(self.audio_root, self.filenames[i]))
        if self.augment:
            mel = self._spec_augment(mel)
        # -> [1, T, n_mels] (time x freq), matching Cnn14.forward expectation
        x = torch.from_numpy(mel.T.copy())[None].float()
        y = torch.from_numpy(self.labels[i].copy()).float()
        return x, y


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


def train_fold(df, Y, fold, args, device, device_type, wav_cache=None, mel_cache=None):
    audio_root = os.path.join(args.data_dir, "train_audio")
    tr = df["fold"].values != fold
    va = df["fold"].values == fold
    use_rows = (wav_cache is not None) or (mel_cache is not None)
    tr_rows = df["cache_row"].values[tr] if use_rows else None
    va_rows = df["cache_row"].values[va] if use_rows else None
    tr_ds = ClipDataset(df["filename"].values[tr], Y[tr], audio_root, augment=True,
                        wav_cache=wav_cache, cache_rows=tr_rows, mel_cache=mel_cache)
    va_ds = ClipDataset(df["filename"].values[va], Y[va], audio_root, augment=False,
                        wav_cache=wav_cache, cache_rows=va_rows, mel_cache=mel_cache)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.num_workers)

    model = Cnn14(num_classes=Y.shape[1]).to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        n_loaded, n_total, skipped, missing = load_pretrained_cnn14(model, args.ckpt)
        print(f"  [pretrained] loaded {n_loaded}/{n_total} model tensors from "
              f"{os.path.basename(args.ckpt)}; skipped {len(skipped)} src keys "
              f"(frontend + 527-class fc). missing_in_model={len(missing.missing_keys)} "
              f"unexpected={len(missing.unexpected_keys)}")
    else:
        print(f"  [pretrained] WARNING: ckpt not found ({args.ckpt}) -> random init")

    crit = FocalLoss() if args.loss == "focal" else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    Yva = Y[va]
    sec_per_epoch = None
    best_auc, best_preds, best_epoch, since_improve = -1.0, None, 0, 0
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); tl = 0.0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            if args.mixup_alpha > 0:
                x, y = mixup_data(x, y, args.mixup_alpha)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                loss = crit(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    ap.add_argument("--data_dir", default=os.path.expanduser("~/SHC/birdclef2026_clef/data"))
    ap.add_argument("--folds_csv", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--ckpt", default=None,
                    help="CNN14 pretrained .pth (default <data_dir>/panns_ckpt/Cnn14_mAP0.431.pth)")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--loss", default="focal", choices=["bce", "focal"])
    ap.add_argument("--mixup_alpha", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=3, help="early-stop patience on val macro-AUC")
    ap.add_argument("--wav_cache", default=None,
                    help="float16 [Nall,160000] decoded-waveform cache .npy "
                         "(default <data_dir>/wav_cache/wav_cache.npy if present)")
    ap.add_argument("--mel_cache", default=None,
                    help="precomputed BASE log-mel fp16 cache .npy [Nall,64,T] "
                         "(default <data_dir>/mel_cache/pann/mel.npy if present); when present "
                         "skips decode+mel, SpecAugment still applied on top at train time")
    ap.add_argument("--tag", default="pann", help="model tag for per-run OOF filename")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dry_n", type=int, default=40)
    args = ap.parse_args()

    if args.folds_csv is None:
        cand = os.path.join(os.path.dirname(args.data_dir.rstrip("/")), "folds.csv")
        args.folds_csv = cand if os.path.exists(cand) else os.path.join(args.data_dir, "folds.csv")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "oof")
    if args.ckpt is None:
        args.ckpt = os.path.join(args.data_dir, "panns_ckpt", "Cnn14_mAP0.431.pth")
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
          f"ASCEND={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} ckpt={args.ckpt}")

    df = pd.read_csv(args.folds_csv)
    df["cache_row"] = np.arange(len(df))  # cache row index == folds.csv order
    meta = pd.read_csv(os.path.join(args.data_dir, "train.csv"))[
        ["filename", "primary_label", "secondary_labels", "author"]]
    df = df.merge(meta, on="filename", how="left")
    assert df["author"].notna().all(), "author join NaN -> grouping broken"

    # precomputed BASE mel cache (preferred; replaces per-epoch decode+mel)
    if args.mel_cache is None:
        cand = os.path.join(args.data_dir, "mel_cache", "pann", "mel.npy")
        args.mel_cache = cand if os.path.exists(cand) else None
    mel_cache = None
    if args.mel_cache and os.path.exists(args.mel_cache):
        mel_cache = np.load(args.mel_cache, mmap_mode="r")
        print(f"[cache] using mel_cache {args.mel_cache} shape {mel_cache.shape} (FAST: base mel)")

    if args.wav_cache is None:
        cand = os.path.join(args.data_dir, "wav_cache", "wav_cache.npy")
        args.wav_cache = cand if os.path.exists(cand) else None
    wav_cache = None
    if args.wav_cache and os.path.exists(args.wav_cache):
        wav_cache = np.load(args.wav_cache, mmap_mode="r")
        print(f"[cache] wav_cache available {args.wav_cache} shape {wav_cache.shape}")
    if mel_cache is None and wav_cache is None:
        print("[cache] NO mel/wav cache -> decoding ogg + mel on the fly (SLOW)")
    elif mel_cache is None:
        print("[cache] no mel_cache -> mel computed on the fly from wav_cache")

    folds_to_run = [int(x) for x in args.folds.split(",")]
    if args.dry_run:
        folds_to_run = folds_to_run[:1]; args.epochs = 1
        df = df.groupby("fold", group_keys=False).head(args.dry_n).reset_index(drop=True)
        print(f"[DRY] {len(df)} clips, fold {folds_to_run}, 1 epoch")

    class_order = load_taxonomy_order(args.data_dir)
    Y = build_targets(df, class_order)
    N, C = Y.shape
    oof = np.full((N, C), np.nan, dtype=np.float32)

    suffix = "_dry" if args.dry_run else ""
    spe_list = []
    for f in folds_to_run:
        va_mask, preds, spe, n_tr, best_auc, best_ep = train_fold(
            df, Y, f, args, device, device_type, wav_cache=wav_cache, mel_cache=mel_cache)
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

    out = os.path.join(args.out_dir, f"pann_sidecar_oof{suffix}.npy")
    np.save(out, oof)
    df[["filename", "fold", "author"]].to_csv(
        os.path.join(args.out_dir, f"pann_oof_meta{suffix}.csv"), index=False)
    np.save(os.path.join(args.out_dir, f"pann_oof_targets{suffix}.npy"), Y)
    print(f"[done] PANN sidecar OOF -> {out} {oof.shape} "
          f"(non-NaN rows={int((~np.isnan(oof).all(1)).sum())})")

    # ----- 5-fold cost extrapolation from the smoke epoch -----
    if spe_list:
        spe, n_tr = spe_list[0]
        per_clip = spe / max(n_tr, 1)
        full_train_per_fold = 35549 * 4 / 5  # ~4/5 of data trains each fold
        sec_full_epoch = per_clip * full_train_per_fold
        # default real run = args.epochs (20) per fold x 5 folds
        epochs_real = 20
        h_5fold = sec_full_epoch * epochs_real * 5 / 3600.0
        print(f"[extrap] smoke {spe:.1f}s for {n_tr} train clips -> "
              f"{per_clip*1000:.1f} ms/clip; full-fold epoch ~{sec_full_epoch/60:.1f} min; "
              f"5-fold x {epochs_real}ep ~= {h_5fold:.1f} NPU-h (single NPU-5). "
              f"NOTE: extrapolated from ONE tiny epoch; remeasure 1st real epoch.")


if __name__ == "__main__":
    main()
