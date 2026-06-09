"""Build a decoded-waveform cache (one pass) for the quality-max sidecar matrix.

Decodes EVERY train clip (folds.csv order) to a 32 kHz, 5 s center window
(160000 samples, identical to b1_perch_extract.load_center_window) and stores it
as float16 in a single contiguous .npy:  wav_cache.npy  shape (N, 160000).

This serves BOTH sidecars (author-CNN n_mels=128 and PANNs CNN14 n_mels=64): each
loader recomputes its own log-mel on the fly from the cached waveform, so the ONLY
thing removed is the expensive librosa ogg decode (~176 ms/clip), which was the
audio-I/O bottleneck (NPU idle, 667 s/epoch). Mel compute from an in-RAM/ mmap'd
float16 waveform is cheap.

Layout (under <out_dir>, default data/wav_cache/):
  wav_cache.npy        float16 [N, 160000]   (~11.4 GB for N=35549)
  wav_cache_index.csv  row, filename, fold   (row order == folds.csv == OOF rows)
  wav_cache_meta.json  {n, win, sr, dtype, built_utc}

Parallel CPU decode (ProcessPool). Writes into a pre-allocated memmap so peak RAM
stays low. Validates a random clip vs a fresh decode at the end.

Usage:
  python src/b1_build_wav_cache.py --workers 16
  python src/b1_build_wav_cache.py --validate_only   # re-check existing cache
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

SR = 32000
WIN = 5 * SR  # 160000


def load_center_window(path, sr=SR, win=WIN):
    """Center 5 s window, pad/truncate to `win`. IDENTICAL to b1_perch_extract."""
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) == 0:
        return np.zeros(win, dtype=np.float32)
    if len(y) >= win:
        start = (len(y) - win) // 2
        y = y[start:start + win]
    else:
        y = np.pad(y, (0, win - len(y)))
    return y.astype(np.float32)


def _decode_one(args):
    i, path = args
    try:
        y = load_center_window(path)
        return i, y.astype(np.float16), None
    except Exception as e:  # never let one bad clip kill the pass
        return i, np.zeros(WIN, dtype=np.float16), f"{path}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.expanduser("~/SHC/birdclef2026_clef/data"))
    ap.add_argument("--folds_csv", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()

    if args.folds_csv is None:
        cand = os.path.join(os.path.dirname(args.data_dir.rstrip("/")), "folds.csv")
        args.folds_csv = cand if os.path.exists(cand) else os.path.join(args.data_dir, "folds.csv")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "wav_cache")
    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, "wav_cache.npy")
    index_path = os.path.join(args.out_dir, "wav_cache_index.csv")
    meta_path = os.path.join(args.out_dir, "wav_cache_meta.json")
    audio_root = os.path.join(args.data_dir, "train_audio")

    df = pd.read_csv(args.folds_csv)  # filename, fold  (defines row order)
    N = len(df)
    paths = [os.path.join(audio_root, fn) for fn in df["filename"]]

    if args.validate_only:
        return validate(cache_path, df, paths)

    # disk guard: abort if cache would push /home over ~90%
    import shutil
    total, used, free = shutil.disk_usage(args.out_dir)
    need = N * WIN * 2  # float16
    proj_pct = 100.0 * (used + need) / total
    print(f"[disk] free={free/1e9:.1f}GB need={need/1e9:.2f}GB "
          f"projected_used={proj_pct:.1f}%")
    if proj_pct > 90.0:
        raise SystemExit(f"[ABORT] cache would push /home to {proj_pct:.1f}% (>90%).")

    print(f"[build] N={N} clips -> {cache_path} (float16 [{N},{WIN}], "
          f"{need/1e9:.2f}GB), workers={args.workers}")
    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16,
                                      shape=(N, WIN))
    t0 = time.time()
    n_done = 0
    n_err = 0
    errs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_decode_one, (i, p)) for i, p in enumerate(paths)]
        for fut in as_completed(futs):
            i, y, err = fut.result()
            cache[i] = y
            if err:
                n_err += 1
                if len(errs) < 20:
                    errs.append(err)
            n_done += 1
            if n_done % 2000 == 0:
                el = time.time() - t0
                rate = n_done / el
                eta = (N - n_done) / max(rate, 1e-6)
                print(f"[build] {n_done}/{N} ({rate:.1f} clip/s, "
                      f"eta {eta/60:.1f} min, errs={n_err})", flush=True)
    cache.flush()
    del cache
    dt = time.time() - t0
    print(f"[build] DONE {n_done} clips in {dt/60:.1f} min "
          f"({n_done/dt:.1f} clip/s); errors={n_err}")
    if errs:
        print(f"[build] first errors: {errs}")

    df_idx = df.copy()
    df_idx.insert(0, "row", range(N))
    df_idx[["row", "filename", "fold"]].to_csv(index_path, index=False)
    json.dump({"n": N, "win": WIN, "sr": SR, "dtype": "float16",
               "n_err": n_err, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(meta_path, "w"), indent=2)
    print(f"[build] wrote index -> {index_path}, meta -> {meta_path}")

    validate(cache_path, df, paths)


def validate(cache_path, df, paths, n_check=3):
    """Confirm cached clips match a FRESH decode (float16 round-trip tolerance)."""
    cache = np.load(cache_path, mmap_mode="r")
    print(f"[validate] cache shape {cache.shape} dtype {cache.dtype}")
    rng = np.random.RandomState(0)
    rows = rng.choice(len(df), size=min(n_check, len(df)), replace=False)
    ok = True
    for r in rows:
        fresh = load_center_window(paths[r]).astype(np.float16).astype(np.float32)
        cached = np.asarray(cache[r]).astype(np.float32)
        max_abs = float(np.max(np.abs(fresh - cached)))
        same = max_abs < 1e-3
        ok = ok and same
        print(f"[validate] row {r} {df['filename'].iloc[r]}: "
              f"max|fresh-cached|={max_abs:.2e} -> {'OK' if same else 'MISMATCH'}")
    print(f"[validate] {'ALL OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    main()
