"""Driver: wait for the resumable Perch extraction to finish assembling the caches,
then produce BOTH anchors. Resumable/idempotent — safe to re-run.

Waits (polling, with a hard cap) for:
    data/oof/perch_cache/embeddings.npy   (1536-d, for linear probe)
    data/oof/perch_cache/label_head.npy   (14795, for zero-shot)
    data/oof/oof_targets.npy + oof_meta.csv
to all exist, then runs:
    1) linear-probe anchor  -> perch_anchor_oof.npy        (per-fold logistic on 1536d)
    2) zero-shot anchor     -> perch_zeroshot_oof.npy       (14795->234 via map csv)
Both are no-ops if their output already exists and --force is not set.
"""
import argparse
import os
import subprocess
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))


def exists_all(out_dir):
    need = [
        os.path.join(out_dir, "perch_cache", "embeddings.npy"),
        os.path.join(out_dir, "perch_cache", "label_head.npy"),
        os.path.join(out_dir, "oof_targets.npy"),
        os.path.join(out_dir, "oof_meta.csv"),
    ]
    return all(os.path.exists(p) for p in need), need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.expanduser("~/SHC/birdclef2026_clef/data/oof"))
    ap.add_argument("--data_dir", default=os.path.expanduser("~/SHC/birdclef2026_clef/data"))
    ap.add_argument("--poll_sec", type=int, default=120)
    ap.add_argument("--max_wait_min", type=int, default=180)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    deadline = time.time() + args.max_wait_min * 60
    while True:
        ok, need = exists_all(args.out_dir)
        if ok:
            print("[anchors] caches present; proceeding.", flush=True)
            break
        if time.time() > deadline:
            missing = [p for p in need if not os.path.exists(p)]
            print(f"[anchors] TIMEOUT waiting for caches; missing: {missing}", flush=True)
            sys.exit(2)
        missing = [os.path.basename(p) for p in need if not os.path.exists(p)]
        print(f"[anchors] waiting for {missing} ... (poll {args.poll_sec}s)", flush=True)
        time.sleep(args.poll_sec)

    py = sys.executable
    # 1) linear-probe anchor
    lp_out = os.path.join(args.out_dir, "perch_anchor_oof.npy")
    if args.force or not os.path.exists(lp_out):
        print("[anchors] running linear-probe anchor ...", flush=True)
        subprocess.run([py, os.path.join(SRC, "b1_perch_linear_probe.py"),
                        "--out_dir", args.out_dir, "--max_iter", "1000"], check=True)
    else:
        print(f"[anchors] linear-probe already present: {lp_out}", flush=True)

    # 2) zero-shot anchor
    zs_out = os.path.join(args.out_dir, "perch_zeroshot_oof.npy")
    if args.force or not os.path.exists(zs_out):
        print("[anchors] running zero-shot anchor ...", flush=True)
        subprocess.run([py, os.path.join(SRC, "b1_perch_zeroshot.py"),
                        "--data_dir", args.data_dir, "--out_dir", args.out_dir,
                        "--agg", "max"], check=True)
    else:
        print(f"[anchors] zero-shot already present: {zs_out}", flush=True)

    print("[anchors] DONE: perch_anchor_oof.npy + perch_zeroshot_oof.npy", flush=True)


if __name__ == "__main__":
    main()
