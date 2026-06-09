#!/bin/bash
# Detached 4-stage soundscape pseudo-label headroom audit. NPU4 (free).
source ~/SHC/Ascend/ascend-toolkit/set_env.sh
source ~/SHC/Ascend/nnal/atb/set_env.sh
export PATH=~/SHC/miniconda3/envs/sft/bin:$PATH
export ASCEND_RT_VISIBLE_DEVICES=4
cd ~/SHC/birdclef2026_clef || exit 1

echo "=== STAGE1 PL-gen (Perch-ZS pseudo-labels on unlabeled soundscapes) $(date) ==="
python src/add_pl_gen.py --n_files 3000 --win_starts 5,25,45 --topk 1 --seed 42 \
  || { echo "STAGE1 FAILED rc=$?"; exit 1; }

echo "=== STAGE2 BASELINE CNN (focal only, train folds 0-3, hold 4) $(date) ==="
python src/b1_cnn_train_pl.py --folds 4 --seed 42 --epochs 30 --patience 3 \
  --save_ckpt --tag cnnplbase --device npu:0 \
  || { echo "STAGE2 FAILED rc=$?"; exit 1; }

echo "=== STAGE3 PL CNN (focal + pseudo soundscapes) $(date) ==="
python src/b1_cnn_train_pl.py --folds 4 --seed 42 --epochs 30 --patience 3 \
  --save_ckpt --tag cnnplpl --device npu:0 \
  --extra_mel data/oof/pl_pseudo_mel.npy --extra_targets data/oof/pl_pseudo_targets.npy \
  || { echo "STAGE3 FAILED rc=$?"; exit 1; }

echo "=== STAGE4 AUDIT (soundscape DeltaAUC + file-clustered bootstrap) $(date) ==="
python src/add_pseudolabel_audit.py --base_tag cnnplbase --pl_tag cnnplpl \
  --fold 4 --seed 42 --device npu:0 --n_boot 1000 \
  || { echo "STAGE4 FAILED rc=$?"; exit 1; }

echo "=== PL_AUDIT_DONE $(date) ==="
