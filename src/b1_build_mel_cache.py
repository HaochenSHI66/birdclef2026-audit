"""Precompute BASE log-mel caches from the decoded wav cache (mmap), sharded + resumable.

CORRECTNESS: caches ONLY the deterministic base mel, using the EXACT mel functions from
the training scripts (imported, not re-implemented), so cached == on-the-fly base mel.
  - CNN  : wave_to_mel(y, 224)          -> [224, 224] float32 in [0,1]   (b1_cnn_train)
  - PANN : wave_to_logmel(y)            -> [64, 501]  float32 (power_to_db ref=1.0)  (b1_pann_train)
All STOCHASTIC augmentation (SpecAugment / mixup) stays in the loader at train time, applied
ON TOP of the cached base mel. val = no aug. So training is mathematically equivalent, faster.

Output (per config):
  data/mel_cache/{tag}/mel.npy            fp16 memmap, shape [N, *mel_shape], row i == wav_cache row i
  data/mel_cache/{tag}/shards/done_*.flag
Sharded (SHARD clips/shard), resumable (skip shards whose flag exists), logged every shard,
disk_guard aborts a shard launch if df > MAX_DISK_PCT. Validates cached==fresh on a sample.

Usage:
  python src/b1_build_mel_cache.py --tag cnn  --shard 2000
  python src/b1_build_mel_cache.py --tag pann --shard 2000
"""
import argparse
import os
import sys
import time
import numpy as np
from multiprocessing import Pool

ROOT = os.path.expanduser("~/SHC/birdclef2026_clef")
sys.path.insert(0, os.path.join(ROOT, "src"))

# import the EXACT base-mel functions used on the fly (guarantees identical params)
from b1_cnn_train import wave_to_mel as cnn_wave_to_mel          # -> [img,img] float32 [0,1]
from b1_pann_train import wave_to_logmel as pann_wave_to_logmel  # -> [64,T]  float32

WIN = 5 * 32000

# ---- multiprocessing worker: each worker opens the wav cache once (mmap) ----
_W = {}


def _winit(wav_path, tag, img_size):
    _W["wav"] = np.load(wav_path, mmap_mode="r")
    _W["tag"] = tag
    _W["img"] = img_size


def _wmel(r):
    y = np.asarray(_W["wav"][r]).astype(np.float32)
    if _W["tag"] == "cnn":
        return r, cnn_wave_to_mel(y, _W["img"]).astype(np.float16)
    return r, pann_wave_to_logmel(y).astype(np.float16)


def disk_pct(path):
    out = os.popen(f"df -P {path}").read().splitlines()[-1]
    return int(out.split()[4].rstrip("%"))


def mel_shape_for(tag, img_size):
    if tag == "cnn":
        return (img_size, img_size)
    elif tag == "pann":
        return pann_wave_to_logmel(np.zeros(WIN, np.float32)).shape  # [64, T]
    raise ValueError(tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["cnn", "pann"])
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--wav_cache", default=os.path.join(ROOT, "data", "wav_cache", "wav_cache.npy"))
    ap.add_argument("--img_size", type=int, default=224)  # CNN only
    ap.add_argument("--shard", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max_disk_pct", type=int, default=92)
    ap.add_argument("--log_every", type=int, default=1)  # shards
    args = ap.parse_args()

    wav = np.load(args.wav_cache, mmap_mode="r")          # [N, 160000] fp16
    N = wav.shape[0]
    mel_shape = mel_shape_for(args.tag, args.img_size)
    print(f"[mel {args.tag}] wav_cache N={N} win={wav.shape[1]} -> mel_shape={mel_shape} fp16",
          flush=True)

    out_dir = os.path.join(args.data_dir, "mel_cache", args.tag)
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    mel_path = os.path.join(out_dir, "mel.npy")

    # open/create the full fp16 memmap [N, *mel_shape]
    full_shape = (N,) + tuple(mel_shape)
    if os.path.exists(mel_path):
        mel = np.lib.format.open_memmap(mel_path, mode="r+")
        assert mel.shape == full_shape, f"existing mel.npy shape {mel.shape} != {full_shape}"
        print(f"[mel {args.tag}] reopened existing memmap {mel.shape}", flush=True)
    else:
        if disk_pct(args.data_dir) > args.max_disk_pct:
            print(f"[mel {args.tag}] DISK GUARD {disk_pct(args.data_dir)}% before alloc -> abort")
            sys.exit(3)
        mel = np.lib.format.open_memmap(mel_path, mode="w+", dtype=np.float16, shape=full_shape)
        print(f"[mel {args.tag}] allocated memmap {mel_path} {full_shape} "
              f"({np.prod(full_shape)*2/1e9:.2f} GB)", flush=True)

    n_shards = (N + args.shard - 1) // args.shard
    t_start = time.time()
    done_rows = 0
    pool = Pool(args.workers, initializer=_winit,
                initargs=(args.wav_cache, args.tag, args.img_size))
    try:
        for sidx in range(n_shards):
            flag = os.path.join(shard_dir, f"done_{sidx:04d}.flag")
            if os.path.exists(flag):
                continue
            if disk_pct(args.data_dir) > args.max_disk_pct:
                print(f"[mel {args.tag}] DISK GUARD {disk_pct(args.data_dir)}% > "
                      f"{args.max_disk_pct}% -> abort at shard {sidx}", flush=True)
                pool.terminate()
                sys.exit(3)
            lo = sidx * args.shard
            hi = min(N, lo + args.shard)
            t0 = time.time()
            for r, m in pool.imap_unordered(_wmel, range(lo, hi), chunksize=16):
                mel[r] = m
            mel.flush()
            open(flag, "w").write("ok\n")
            done_rows += (hi - lo)
            if sidx % args.log_every == 0 or hi == N:
                el = time.time() - t_start
                rate = done_rows / max(el, 1e-6)
                remain_rows = N - hi
                eta = remain_rows / max(rate, 1e-6) / 60.0
                print(f"[mel {args.tag}] shard {sidx+1}/{n_shards} rows[{lo}:{hi}] "
                      f"{(time.time()-t0):.1f}s {rate:.0f} clip/s eta {eta:.1f} min "
                      f"disk {disk_pct(args.data_dir)}%", flush=True)
    finally:
        pool.close()
        pool.join()

    # ---- validate cached == fresh on a sample ----
    # report (a) cached_fp16 vs fp16-cast-of-fresh  -> must be EXACTLY 0 (cache integrity), and
    #        (b) cached_fp16 vs float32-fresh        -> only fp16 rounding (~1e-3), the new base.
    rng = np.random.RandomState(0)
    idxs = rng.choice(N, size=5, replace=False)
    if args.tag == "cnn":
        base32 = lambda yy: cnn_wave_to_mel(yy, args.img_size)  # float32 on-the-fly base
    else:
        base32 = lambda yy: pann_wave_to_logmel(yy)
    max_fp16 = 0.0    # cached vs fp16(fresh)  -> exactly 0 = cache integrity
    max_f32 = 0.0     # cached vs float32 fresh -> fp16 rounding only
    for r in idxs:
        y = np.asarray(wav[r]).astype(np.float32)
        f32 = base32(y).astype(np.float32)           # on-the-fly float32 base
        fresh_fp16 = f32.astype(np.float16)          # the value the cache should hold
        cached = np.asarray(mel[r])                  # fp16 from disk
        max_fp16 = max(max_fp16, float(np.max(np.abs(
            fresh_fp16.astype(np.float32) - cached.astype(np.float32)))))
        max_f32 = max(max_f32, float(np.max(np.abs(f32 - cached.astype(np.float32)))))
    print(f"[mel {args.tag}] VALIDATE on {len(idxs)} rows: "
          f"max|cached_fp16 - fp16(fresh)|={max_fp16:.3e} (cache integrity, want 0) ; "
          f"max|cached_fp16 - float32(fresh)|={max_f32:.3e} (fp16 rounding only) -> "
          f"{'OK' if max_fp16==0.0 else 'CHECK'}", flush=True)
    print(f"[mel {args.tag}] DONE {mel_path} {full_shape} in {(time.time()-t_start)/60:.1f} min "
          f"disk {disk_pct(args.data_dir)}%", flush=True)


if __name__ == "__main__":
    main()
