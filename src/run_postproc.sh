#!/bin/bash
# Atomic-locked launcher for the post-proc ablation. mkdir is atomic, so even if this
# script is delivered/started multiple times over a flaky tunnel, only ONE run proceeds.
set -e
cd ~/SHC/birdclef2026_clef
export PATH=~/SHC/miniconda3/envs/sft/bin:$PATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

LOCK=logs/postproc.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "LOCK_HELD: another run owns $LOCK; not starting a duplicate."
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

rm -f data/oof/postproc_ablation.json
echo "RUN_START $(date)"
python -u src/b2_postproc_v2.py --n_boot 200
echo "RUN_DONE $(date)"
