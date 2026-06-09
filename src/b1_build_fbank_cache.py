"""Precompute the BASE BEATs fbank cache from the decoded wav cache (mmap), sharded + resumable.

Mirrors b1_build_mel_cache.py. Caches ONLY the deterministic RAW fbank (pre global-norm) so
cached == on-the-fly base. SpecAugment stays in the loader at train time, applied ON TOP.
val = no aug. Mathematically equivalent, faster.

  data/fbank_cache/beats/fbank.npy   fp16 memmap [N, T, 128], row i == wav_cache row i
  data/fbank_cache/beats/shards/done_*.flag
Sharded, resumable, logged per shard, disk_guard aborts if df > MAX_DISK_PCT. Validates cached==fresh.

ROBUSTNESS: uses a SPAWN multiprocessing context with torch_npu auto-load DISABLED in workers
(TORCH_DEVICE_BACKEND_AUTOLOAD=0) + single-thread BLAS. The earlier fork-based pool deadlocked
because forking after torch_npu init inherits locked runtime state (workers sat at 0% CPU).
Workers do CPU-only fbank, so no NPU runtime is needed.

Usage:  python src/b1_build_fbank_cache.py --shard 1000 --workers 8
"""
import argparse
import os
import sys
import time
import numpy as np
import multiprocessing as mp

# keep the PARENT torch-free too (we only need numpy here); workers import a torch-free fbank
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = os.path.expanduser("~/SHC/birdclef2026_clef")
WIN = 5 * 32000


# ---- torch-free worker fbank: scipy resample + local kaldi.fbank (kaldi uses torch but NOT npu) ----
def _wave_to_fbank(y32k):
    from scipy.signal import resample_poly
    import torch
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from beats_mod.kaldi import fbank as kaldi_fbank
    y16 = resample_poly(y32k.astype(np.float32), 1, 2).astype(np.float32)
    w = torch.from_numpy(y16).unsqueeze(0) * (2 ** 15)
    fb = kaldi_fbank(w, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
    return fb.numpy().astype(np.float32)


_W = {}


def _winit(wav_path):
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    _W["wav"] = np.load(wav_path, mmap_mode="r")


def _wfb(r):
    y = np.asarray(_W["wav"][r]).astype(np.float32)
    return r, _wave_to_fbank(y).astype(np.float16)


def disk_pct(path):
    out = os.popen(f"df -P {path}").read().splitlines()[-1]
    return int(out.split()[4].rstrip("%"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--wav_cache", default=os.path.join(ROOT, "data", "wav_cache", "wav_cache.npy"))
    ap.add_argument("--shard", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_disk_pct", type=int, default=92)
    ap.add_argument("--log_every", type=int, default=1)
    args = ap.parse_args()

    wav = np.load(args.wav_cache, mmap_mode="r")
    N = wav.shape[0]
    fb_shape = _wave_to_fbank(np.zeros(WIN, np.float32)).shape  # [T, 128]
    print(f"[fbank] wav_cache N={N} win={wav.shape[1]} -> fbank_shape={fb_shape} fp16 "
          f"workers={args.workers} shard={args.shard} (spawn, npu-autoload off)", flush=True)

    out_dir = os.path.join(args.data_dir, "fbank_cache", "beats")
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    fb_path = os.path.join(out_dir, "fbank.npy")

    full_shape = (N,) + tuple(fb_shape)
    if os.path.exists(fb_path):
        fb = np.lib.format.open_memmap(fb_path, mode="r+")
        assert fb.shape == full_shape, f"existing fbank.npy shape {fb.shape} != {full_shape}"
        print(f"[fbank] reopened existing memmap {fb.shape}", flush=True)
    else:
        if disk_pct(args.data_dir) > args.max_disk_pct:
            print(f"[fbank] DISK GUARD {disk_pct(args.data_dir)}% before alloc -> abort"); sys.exit(3)
        fb = np.lib.format.open_memmap(fb_path, mode="w+", dtype=np.float16, shape=full_shape)
        print(f"[fbank] allocated memmap {fb_path} {full_shape} "
              f"({np.prod(full_shape)*2/1e9:.2f} GB)", flush=True)

    n_shards = (N + args.shard - 1) // args.shard
    t_start = time.time(); done_rows = 0
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_winit, initargs=(args.wav_cache,))
    try:
        for sidx in range(n_shards):
            flag = os.path.join(shard_dir, f"done_{sidx:04d}.flag")
            if os.path.exists(flag):
                continue
            if disk_pct(args.data_dir) > args.max_disk_pct:
                print(f"[fbank] DISK GUARD {disk_pct(args.data_dir)}% > {args.max_disk_pct}% "
                      f"-> abort at shard {sidx}", flush=True)
                pool.terminate(); sys.exit(3)
            lo = sidx * args.shard; hi = min(N, lo + args.shard); t0 = time.time()
            for r, m in pool.imap_unordered(_wfb, range(lo, hi), chunksize=8):
                fb[r] = m
            fb.flush(); open(flag, "w").write("ok\n"); done_rows += (hi - lo)
            if sidx % args.log_every == 0 or hi == N:
                el = time.time() - t_start; rate = done_rows / max(el, 1e-6)
                eta = (N - hi) / max(rate, 1e-6) / 60.0
                print(f"[fbank] shard {sidx+1}/{n_shards} rows[{lo}:{hi}] {(time.time()-t0):.1f}s "
                      f"{rate:.0f} clip/s eta {eta:.1f} min disk {disk_pct(args.data_dir)}%", flush=True)
    finally:
        pool.close(); pool.join()

    rng = np.random.RandomState(0)
    idxs = rng.choice(N, size=5, replace=False)
    max_fp16 = 0.0; max_f32 = 0.0
    for r in idxs:
        y = np.asarray(wav[r]).astype(np.float32)
        f32 = _wave_to_fbank(y).astype(np.float32)
        fresh_fp16 = f32.astype(np.float16)
        cached = np.asarray(fb[r])
        max_fp16 = max(max_fp16, float(np.max(np.abs(
            fresh_fp16.astype(np.float32) - cached.astype(np.float32)))))
        max_f32 = max(max_f32, float(np.max(np.abs(f32 - cached.astype(np.float32)))))
    print(f"[fbank] VALIDATE on {len(idxs)} rows: "
          f"max|cached_fp16 - fp16(fresh)|={max_fp16:.3e} (want 0) ; "
          f"max|cached_fp16 - float32(fresh)|={max_f32:.3e} (fp16 rounding) -> "
          f"{'OK' if max_fp16==0.0 else 'CHECK'}", flush=True)
    print(f"[fbank] DONE {fb_path} {full_shape} in {(time.time()-t_start)/60:.1f} min "
          f"disk {disk_pct(args.data_dir)}%", flush=True)


if __name__ == "__main__":
    main()
