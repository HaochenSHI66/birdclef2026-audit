#!/usr/bin/env bash
# Parallel B2 FINAL orchestrator: fire 8 single-cell processes (one per anchor x combo),
# each writing its own per-cell json; when ALL 8 jsons exist, merge into b2_final.json
# in b2_corrected's schema. Detached; CPU-only; never touches NPU.
cd ~/SHC/birdclef2026_clef || exit 1
source ~/SHC/Ascend/ascend-toolkit/set_env.sh
source ~/SHC/Ascend/nnal/atb/set_env.sh
export PATH=~/SHC/miniconda3/envs/sft/bin:$PATH
export ASCEND_RT_VISIBLE_DEVICES=""          # CPU only, never touch NPU
# Keep each cell single-threaded-ish so 8 cells share cores cleanly (cores are plentiful).
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

CELLDIR=data/oof/b2_par_cells
mkdir -p "$CELLDIR" logs

echo "[orch] start $(date) — launching 8 cells"

PIDS=()
for c in 0 1 2 3 4 5 6 7; do
  out="$CELLDIR/b2cell_$(printf '%d' $c).json"
  log="logs/b2cell_${c}.log"
  python src/b2_par_cell.py --oof_dir data/oof --cell $c \
      --out_json "$out" --n_boot 1000 --n_null 50 --seed 0 > "$log" 2>&1 &
  PIDS+=($!)
  echo "[orch] launched cell $c pid=$! -> $out (log $log)"
done

echo "[orch] waiting on PIDs: ${PIDS[*]}"
FAIL=0
for p in "${PIDS[@]}"; do
  wait "$p" || { echo "[orch] cell pid $p FAILED rc=$?"; FAIL=1; }
done

# verify all 8 jsons exist
N=$(ls "$CELLDIR"/b2cell_*.json 2>/dev/null | wc -l)
echo "[orch] per-cell jsons present: $N/8 (FAIL=$FAIL)"
if [ "$N" -eq 8 ]; then
  python src/b2_par_merge.py --cell_dir "$CELLDIR" --out_json data/oof/b2_final.json \
    && echo "[orch] MERGE OK -> data/oof/b2_final.json"
else
  echo "[orch] NOT merging: only $N/8 cells"
fi
echo "[orch] done $(date)"
