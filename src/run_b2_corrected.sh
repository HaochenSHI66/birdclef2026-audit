#!/bin/bash
cd ~/SHC/birdclef2026_clef
source ~/SHC/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source ~/SHC/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
export PATH=~/SHC/miniconda3/envs/sft/bin:$PATH
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
python src/b2_corrected.py --oof_dir data/oof --out_json data/oof/b2_corrected_preview.json --n_boot 1000 --n_null 50 --seed 0
echo "EXIT_CODE=$?"
