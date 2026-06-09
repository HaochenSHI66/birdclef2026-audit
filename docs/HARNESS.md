# BirdCLEF++ 2026 — B1'/B2' Experiment Harness

Reproducible harness for the "Measuring the Ceiling" oracle-blend headroom audit.
All paths are on the Ascend server `~/SHC/birdclef2026_clef/` unless noted.

## Environment

```bash
# NPU / torch jobs (CNN training, Perch ONNX via sft env):
source ~/SHC/Ascend/ascend-toolkit/set_env.sh
source ~/SHC/Ascend/nnal/atb/set_env.sh
export PATH=~/SHC/miniconda3/envs/sft/bin:$PATH      # torch 2.10.0, torch_npu 2.10.0,
                                                     # onnxruntime 1.23.2, timm 0.9.16,
                                                     # librosa 0.11.0, sklearn 1.7.2
# kaggle 2.2.1 lives in the BASE env only (data/asset downloads).
```

NPU policy: shared 8x910B2. Use ONLY `ASCEND_RT_VISIBLE_DEVICES=4,5,6`. Never touch 0-3.

## Data flow

```
folds.csv (filename,fold)  +  data/train.csv (author,primary_label,secondary_labels)
        │  join on filename                taxonomy.csv (234 primary_label order)
        ▼
  ┌─────────────────────┐     ┌──────────────────────┐
  │ b1_perch_extract.py │     │  b1_cnn_train.py     │
  │ Perch CPU-ONNX      │     │  timm CNN on NPU     │
  │ emb[N,1536]→probe   │     │  mel→backbone→head   │
  └─────────┬───────────┘     └──────────┬───────────┘
            ▼                            ▼
  perch_anchor_oof.npy [N,234]   cnn_sidecar_oof.npy [N,234]
  oof_targets.npy / oof_meta.csv (filename,fold,author)  ← shared row order
            └────────────┬───────────────┘
                         ▼
              b2_oracle_headroom.py  (CPU)
   anchor-only / sidecar-only / calibrated-unclamped fusion / per-class ORACLE
   dAUC=oracle-anchor ± 95% CI (bootstrap clustered by author) + MDE
   complementarity: Spearman/Pearson/Kuncheva-Q/double-fault
```

## The Perch ONNX model (verified)

`data/perch_onnx/perch_v2_no_dft.onnx` (413 MB, from kaggle `tuckerarrants/perch-v2-no-dft-onnx`).
- input  `inputs` `[batch, 160000]` = 5 s @ 32 kHz raw mono audio.
- outputs: `embedding[batch,1536]`, `spatial_embedding[batch,16,4,1536]`,
  `spectrogram[batch,500,128]`, `label[batch,14795]`.
- `get_available_providers()` = `['AzureExecutionProvider','CPUExecutionProvider']` (CPU used).

DECISION: `label` is Perch's **14795-class GLOBAL taxonomy**, not the 234 scored
classes, and NO label-name map ships with the dataset. The bronze pipeline uses the
**embedding** (ProtoSSM prototype matching), not the raw head. So the reproducible
anchor head = **per-fold logistic-regression linear probe on the 1536-d embedding ->
234 classes**. The 14795 head is cached only for the parity sanity report.

## 206 vs 234 class handling

- Column order = `taxonomy.csv` primary_label order (== sample_submission columns).
- Only 206 of 234 scored classes have focal training clips. The 28 absent classes
  have no positives in any train fold -> their OOF columns are left **NaN** in both
  anchor and sidecar matrices.
- Macro ROC-AUC is **skip-zero-positive** (scripts/optimize_ensemble.py:19): classes
  with no positives in the eval set are never scored, and any NaN column is skipped.
  So the 28-class gap never inflates/deflates the metric.

## Commands

### 0. Assets (base env, one-time)
```bash
conda activate base
cd ~/SHC/birdclef2026_clef/data/perch_onnx
kaggle datasets download -d tuckerarrants/perch-v2-no-dft-onnx --unzip
```

### 1. B1' Perch extract + OOF anchor (CPU)
```bash
# DRY (proven): 40/fold, ~101 ms/clip
python src/b1_perch_extract.py --mode all --dry_run --dry_n 40 --batch_size 8
# FULL: extract once (cache emb) then build OOF
python src/b1_perch_extract.py --mode extract --batch_size 16 --threads 8 --cache_label_head
python src/b1_perch_extract.py --mode oof
# parallelise extraction across CPU workers by sharding folds_csv if wall-time matters
```

### 2. B1' CNN sidecar (NPU 4,5,6 — one fold per NPU)
```bash
export ASCEND_RT_VISIBLE_DEVICES=4   # then 5, then 6 in parallel shells
# DRY (proven): 1 fold, 1 epoch, 40/fold, NPU
python src/b1_cnn_train.py --dry_run --dry_n 40 --device npu:0 --batch_size 8
# FULL per fold (set --folds to the one this NPU owns):
python src/b1_cnn_train.py --folds 0 --epochs 30 --backbone tf_efficientnet_b0_ns \
    --device npu:0 --batch_size 32
# NOTE: each NPU is masked to a single visible device, so always --device npu:0.
# Merge per-fold OOF blocks (each run fills only its own VAL rows; combine by
# overwriting non-NaN rows) before B2'. Easiest: run all 5 folds writing to the
# same cnn_sidecar_oof.npy via a small merge, or run folds sequentially per NPU.
```

### 3. B2' oracle headroom (CPU)
```bash
python src/b2_oracle_headroom.py --selftest        # synthetic asserts (proven)
python src/b2_oracle_headroom.py \
    --anchor data/oof/perch_anchor_oof.npy \
    --sidecar data/oof/cnn_sidecar_oof.npy \
    --targets data/oof/oof_targets.npy \
    --meta data/oof/oof_meta.csv \
    --n_boot 1000 --out_json data/oof/b2_headroom.json
```

## Projected full-run cost

- **B1' Perch extract (CPU-ONNX):** ~101 ms/clip incl audio I/O (measured, 1 batch
  stream) -> **~1.0 h** for 35,549 clips single-stream; embarrassingly parallel across
  CPU workers (shard folds_csv) -> <20 min on 4+ workers. OOF probe fit: minutes.
- **B1' CNN (NPU):** dry epoch = 6.5 s for 160 train clips @ bs8 on 1 NPU
  (tf_efficientnet_b0_ns). Full fold ~28k train clips x 30 epochs. Scaling linearly
  from the dry rate (~24 clips/s incl mel compute at bs8; higher at bs32) gives a
  rough **~10-15 NPU-h per fold**; 5 folds across 3 NPUs (4,5,6) -> **~20-30 NPU-h
  wall**. Re-measure the first real epoch before committing; mel-on-the-fly I/O is
  the main variable (precomputing mels would cut this substantially).
- **B2' analysis (CPU):** seconds to a couple minutes (1000-boot clustered).

Total under the plan's ~50 NPU-h budget. If CNN proves I/O-bound, add a one-time
mel-precompute pass (mirror scripts/precompute_mels.py).

## Outputs (data/oof/)
`perch_anchor_oof.npy`, `cnn_sidecar_oof.npy`, `oof_targets.npy`, `oof_meta.csv`,
`perch_cache/embeddings.npy` (+ `label_head.npy` if `--cache_label_head`),
`b2_headroom.json`. Dry-run variants carry a `_dry` suffix.
