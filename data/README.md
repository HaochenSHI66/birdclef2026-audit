# Data

Small, fully-shareable reproducibility inputs. **No competition audio and no third-party model weights are
included here** (see [../README.md](../README.md#data-provenance--component-attribution)).

| file | description |
|---|---|
| `folds.csv` | The locked recordist-grouped 5-fold split: `filename,fold`. Groups = `author` (Xeno-Canto/iNat recordist); StratifiedGroupKFold on `primary_label`. Leakage gate: 0 authors span >1 fold. Fold sizes {0:5099, 1:6847, 2:7963, 3:8718, 4:6922}. |
| `oof_meta.csv` | Shared OOF row order: `filename,fold,author` (35,549 rows). The `author` column is the bootstrap clustering key. These recordist names are already public in the competition `train.csv`. |
| `perch_zeroshot_map.csv` | Verified 234→Perch-eBird mapping: `primary_label,class_name,stage,perch_idxs`. `stage` ∈ {exact_ebird (158), sciname_fallback (45), unmapped (31)}; `perch_idxs` indexes into Perch's 14,795-class eBird output head. This is the artifact behind the coverage wall. |
| `sample_oof/` | A **2000-row real sample** of the per-model OOF matrices (`perch_anchor`, `perch_zeroshot`, `cnn_ensemble`, `pann_ensemble`, `beats_ensemble`) + `oof_targets.npy` + matching `oof_meta.csv`, so the analysis scripts run out-of-the-box. |

## Regenerating the full OOF arrays

The full per-model OOF matrices are `[35549, 234]` float32 (~33 MB each) and are **not committed** to keep the
repo lean. Regenerate them with the `b1_*` scripts in [`../src/`](../src/) (see
[../REPRODUCE.md](../REPRODUCE.md) Level B). The released `results/*.json` already contain every derived number,
so full regeneration is only needed to reproduce the analysis end-to-end from raw audio.
