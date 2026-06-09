"""B1' sidecar — author's timm-CNN trained per fold on NPU, emits 234-aligned OOF.

Adapted from scripts/train.py (same backbone family, FocalLoss, mixup, SpecAugment,
macro-AUC skip-zero-pos metric) but:
  (a) reads data/folds.csv (cols: filename, fold; group=author via train.csv join),
  (b) trains the CNN per fold on Ascend NPU (torch_npu, ASCEND_RT_VISIBLE_DEVICES=4,5,6),
  (c) emits OOF per-fold predictions in the SAME taxonomy-aligned (N, 234) format as
      b1_perch_extract.py, so B2' can fuse anchor (Perch) + sidecar (CNN) on identical rows.

Mel features are computed on the fly from raw audio (no mel cache exists on the server),
using the bronze-notebook spectrogram params (SR=32000, N_MELS=128, HOP=320, N_FFT=1024).

epochs / folds are CLI args. DRY-RUN: --dry_run trains 1 fold, 1 epoch on a tiny subset
(--dry_n clips/fold) on ONE NPU and writes the dry OOF block, proving train+emit works.
Do NOT pass --dry_run for the real 5-fold run.

OOF alignment with the anchor:
  - We reuse the SAME oof_meta.csv (filename, fold, author) row order produced by
    b1_perch_extract so the (N,234) matrices line up row-for-row.
  - In dry mode, rows = the dry subset only (oof_meta_dry.csv from the Perch dry run,
    OR a freshly built tiny subset if that file is absent).
  - Classes absent from a train fold are left NaN (206/234 gap), matching the anchor.
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

import timm

SR = 32000
WIN = 5 * SR
N_MELS = 128
HOP = 320
N_FFT = 1024
FMIN = 20
FMAX = 16000


def seed_everything(seed=42):
    """Deterministic seeding across python random, numpy, torch (CPU+CUDA),
    and torch_npu, so every reported number traces to (run, seed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if HAS_NPU:
        torch_npu.npu.manual_seed(seed)
        torch_npu.npu.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """Per-worker deterministic seeding for DataLoader (SpecAugment uses python
    random, mixup uses numpy RNG; without this each worker forks identical or
    OS-time-dependent RNG state and the run is not reproducible)."""
    base = torch.initial_seed() % 2 ** 32
    s = (base + worker_id) % 2 ** 32
    np.random.seed(s)
    random.seed(s)


# ----------------------------- shared metric -----------------------------
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


# ----------------------------- audio -> mel -----------------------------
def wave_to_mel(y, img_size):
    """log-mel -> normalized square img from an already-decoded 5 s waveform."""
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    m = librosa.power_to_db(m, ref=np.max)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    from PIL import Image
    img = Image.fromarray((m * 255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR)
    return (np.array(img).astype(np.float32) / 255.0)


def clip_to_mel(path, img_size):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    if len(y) >= WIN:
        s = (len(y) - WIN) // 2
        y = y[s:s + WIN]
    else:
        y = np.pad(y, (0, WIN - len(y)))
    return wave_to_mel(y, img_size)


class ClipDataset(Dataset):
    def __init__(self, filenames, labels, audio_root, img_size=224, augment=False,
                 wav_cache=None, cache_rows=None, mel_cache=None):
        self.filenames = filenames
        self.labels = labels
        self.audio_root = audio_root
        self.img_size = img_size
        self.augment = augment
        # wav_cache: float16 memmap [Nall, WIN]; cache_rows: per-item cache index.
        self.wav_cache = wav_cache
        self.cache_rows = cache_rows
        # mel_cache: precomputed BASE log-mel fp16 memmap [Nall, img, img], row-aligned to
        # cache_rows. When present we SKIP decode+mel and load the deterministic base mel;
        # stochastic augmentation (SpecAugment) is still applied ON TOP at train time.
        self.mel_cache = mel_cache

    def __len__(self):
        return len(self.filenames)

    def _spec_augment(self, mel, num_masks=2, freq_mask=10, time_mask=20):
        h, w = mel.shape
        for _ in range(num_masks):
            f = random.randint(0, freq_mask); f0 = random.randint(0, max(0, h - f))
            mel[f0:f0 + f, :] = 0.0
            t = random.randint(0, time_mask); t0 = random.randint(0, max(0, w - t))
            mel[:, t0:t0 + t] = 0.0
        return mel

    def __getitem__(self, i):
        if self.mel_cache is not None:
            # deterministic BASE mel from cache (fp16 -> fp32); identical every epoch
            mel = np.asarray(self.mel_cache[self.cache_rows[i]]).astype(np.float32)
        elif self.wav_cache is not None:
            y = np.asarray(self.wav_cache[self.cache_rows[i]]).astype(np.float32)
            mel = wave_to_mel(y, self.img_size)
        else:
            mel = clip_to_mel(os.path.join(self.audio_root, self.filenames[i]), self.img_size)
        if self.augment:
            mel = self._spec_augment(mel)
        mel = np.stack([mel, mel, mel], axis=0)
        x = torch.from_numpy(mel.copy()).float()
        y = torch.from_numpy(self.labels[i].copy()).float()
        return x, y


# ----------------------------- model / loss -----------------------------
class BirdCLEFModel(nn.Module):
    def __init__(self, backbone="tf_efficientnet_b0_ns", num_classes=234, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
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
    tr = (df["fold"].values != fold)
    va = (df["fold"].values == fold)
    tr_files = df["filename"].values[tr]
    va_files = df["filename"].values[va]
    use_rows = (wav_cache is not None) or (mel_cache is not None)
    tr_rows = df["cache_row"].values[tr] if use_rows else None
    va_rows = df["cache_row"].values[va] if use_rows else None

    tr_ds = ClipDataset(tr_files, Y[tr], audio_root, args.img_size, augment=True,
                        wav_cache=wav_cache, cache_rows=tr_rows, mel_cache=mel_cache)
    va_ds = ClipDataset(va_files, Y[va], audio_root, args.img_size, augment=False,
                        wav_cache=wav_cache, cache_rows=va_rows, mel_cache=mel_cache)
    g = torch.Generator()
    g.manual_seed(args.seed + fold)  # deterministic shuffle order, distinct per fold
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=False,
                       worker_init_fn=worker_init_fn, generator=g)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    model = BirdCLEFModel(args.backbone, Y.shape[1], pretrained=args.pretrained).to(device)
    crit = FocalLoss() if args.loss == "focal" else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    Yva = Y[va]
    best_auc, best_preds, best_epoch, since_improve = -1.0, None, 0, 0
    best_state = None
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
        ep_s = time.time() - t0
        preds = _predict_val(model, va_ld, device, device_type, Y.shape[1])
        vauc = compute_macro_auc(Yva, np.nan_to_num(preds, nan=0.0))
        print(f"  [fold {fold}] epoch {epoch+1}/{args.epochs} "
              f"train_loss={tl/max(len(tr_ld),1):.4f} val_macro_auc={vauc:.4f} "
              f"({ep_s:.1f}s)", flush=True)
        if vauc > best_auc + 1e-5:
            best_auc, best_preds, best_epoch, since_improve = vauc, preds, epoch + 1, 0
            if args.save_ckpt:
                # snapshot best-epoch weights to CPU so the early-stop model is the one saved
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_improve += 1
            if since_improve >= args.patience:
                print(f"  [fold {fold}] EARLY-STOP @ epoch {epoch+1} "
                      f"(best val_auc={best_auc:.4f} @ ep{best_epoch})", flush=True)
                break
    if best_preds is None:
        best_preds = _predict_val(model, va_ld, device, device_type, Y.shape[1])
        if args.save_ckpt and best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return va, best_preds, best_auc, best_epoch, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.expanduser("~/SHC/birdclef2026_clef/data"))
    ap.add_argument("--folds_csv", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--backbone", default="tf_efficientnet_b0_ns")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4", help="comma list of folds to run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="focal", choices=["bce", "focal"])
    ap.add_argument("--mixup_alpha", type=float, default=0.5)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=3, help="early-stop patience on val macro-AUC")
    ap.add_argument("--wav_cache", default=None,
                    help="float16 [Nall,160000] decoded-waveform cache .npy "
                         "(default <data_dir>/wav_cache/wav_cache.npy if present)")
    ap.add_argument("--mel_cache", default=None,
                    help="precomputed BASE log-mel fp16 cache .npy [Nall,img,img] "
                         "(default <data_dir>/mel_cache/cnn/mel.npy if present); when present "
                         "skips decode+mel, aug still applied on top at train time")
    ap.add_argument("--tag", default="cnn", help="model tag for per-run OOF filename")
    ap.add_argument("--save_ckpt", action="store_true",
                    help="save best-epoch (early-stop) model weights per fold to "
                         "<out_dir>/<tag>_fold<f>_seed<s>.pt for downstream soundscape inference")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dry_n", type=int, default=40, help="clips/fold in dry-run")
    args = ap.parse_args()

    if args.folds_csv is None:
        cand = os.path.join(os.path.dirname(args.data_dir.rstrip("/")), "folds.csv")
        args.folds_csv = cand if os.path.exists(cand) else os.path.join(args.data_dir, "folds.csv")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "oof")
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)

    if HAS_NPU and "npu" in args.device:
        device = torch.device(args.device); torch_npu.npu.set_device(device)
        device_type = "npu"
    elif torch.cuda.is_available():
        device = torch.device("cuda:0"); device_type = "cuda"
    else:
        device = torch.device("cpu"); device_type = "cpu"
    print(f"[cfg] device={device} backbone={args.backbone} dry_run={args.dry_run} "
          f"HAS_NPU={HAS_NPU} ASCEND={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}")

    df = pd.read_csv(args.folds_csv)  # filename, fold (== cache row order)
    df["cache_row"] = np.arange(len(df))  # cache row index == folds.csv order
    meta = pd.read_csv(os.path.join(args.data_dir, "train.csv"))[
        ["filename", "primary_label", "secondary_labels", "author"]]
    df = df.merge(meta, on="filename", how="left")
    assert df["author"].notna().all(), "author join NaN -> grouping broken"

    # optional precomputed BASE mel cache (replaces per-epoch decode+mel). Preferred.
    if args.mel_cache is None:
        cand = os.path.join(args.data_dir, "mel_cache", "cnn", "mel.npy")
        args.mel_cache = cand if os.path.exists(cand) else None
    mel_cache = None
    if args.mel_cache and os.path.exists(args.mel_cache):
        mel_cache = np.load(args.mel_cache, mmap_mode="r")
        print(f"[cache] using mel_cache {args.mel_cache} shape {mel_cache.shape} (FAST: base mel)")

    # optional decoded-waveform cache (used only if no mel cache; replaces ogg decode)
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
        folds_to_run = folds_to_run[:1]
        args.epochs = 1
        df = df.groupby("fold", group_keys=False).head(args.dry_n).reset_index(drop=True)
        print(f"[DRY] {len(df)} clips, fold {folds_to_run}, 1 epoch")

    class_order = load_taxonomy_order(args.data_dir)
    Y = build_targets(df, class_order)
    N, C = Y.shape
    oof = np.full((N, C), np.nan, dtype=np.float32)

    suffix = "_dry" if args.dry_run else ""
    for f in folds_to_run:
        va_mask, preds, best_auc, best_ep, best_state = train_fold(
            df, Y, f, args, device, device_type, wav_cache=wav_cache, mel_cache=mel_cache)
        if args.save_ckpt and best_state is not None:
            ckpt_path = os.path.join(args.out_dir, f"{args.tag}_fold{f}_seed{args.seed}{suffix}.pt")
            torch.save({
                "state_dict": best_state, "backbone": args.backbone,
                "num_classes": int(C), "img_size": int(args.img_size),
                "fold": int(f), "seed": int(args.seed),
                "best_val_macro_auc": float(best_auc), "best_epoch": int(best_ep),
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path} (best_val_auc={best_auc:.4f} @ep{best_ep})", flush=True)
        # only fill columns that had >=1 positive in TRAIN (else leave NaN); the
        # CNN predicts all 234 anyway, but for absent-in-train classes its head was
        # never supervised -> mark NaN to mirror the 206/234 anchor handling.
        tr_pos = (Y[df["fold"].values != f].sum(0) > 0)
        block = preds.copy()
        block[:, ~tr_pos] = np.nan
        oof[va_mask] = block
        auc = compute_macro_auc(Y[va_mask], np.nan_to_num(preds, nan=0.0))
        print(f"  [fold {f}] BEST VAL macro-AUC = {best_auc:.4f} @ep{best_ep}  "
              f"(recheck {auc:.4f}) emitted {preds.shape[0]} rows x {C} cols", flush=True)
        # per-run OOF: VAL rows only, tagged {model}_fold{f}_seed{s}, + row index
        # into the global folds.csv order so MERGE can scatter it back.
        run_oof = block  # [n_val, 234], NaN for untrained classes
        run_rows = df["cache_row"].values[va_mask]
        rtag = f"{args.tag}_fold{f}_seed{args.seed}{suffix}"
        np.save(os.path.join(args.out_dir, f"{rtag}.npy"), run_oof.astype(np.float32))
        np.save(os.path.join(args.out_dir, f"{rtag}_rows.npy"), run_rows.astype(np.int64))
        print(f"[run-oof] {rtag}.npy {run_oof.shape} rows->{rtag}_rows.npy", flush=True)

    out = os.path.join(args.out_dir, f"cnn_sidecar_oof{suffix}.npy")
    np.save(out, oof)
    df[["filename", "fold", "author"]].to_csv(
        os.path.join(args.out_dir, f"cnn_oof_meta{suffix}.csv"), index=False)
    np.save(os.path.join(args.out_dir, f"cnn_oof_targets{suffix}.npy"), Y)
    print(f"[done] sidecar OOF -> {out} shape {oof.shape} "
          f"(non-NaN rows = {int((~np.isnan(oof).all(1)).sum())})")


if __name__ == "__main__":
    main()
