# Reproduce

Two levels of reproduction are supported.

- **Level A — re-derive the headline from released artifacts (minutes, CPU).** Everything in
  [`results/`](results/) is the actual analysis output; you can re-read or re-run the analysis layer on the
  bundled 2000-row sample without any model or competition data.
- **Level B — full regeneration (extraction + training + analysis).** Requires the BirdCLEF++ 2026
  competition audio and the third-party models listed in [README.md](README.md#data-provenance--component-attribution).

The fixed split is [`data/folds.csv`](data/folds.csv) (recordist-grouped 5-fold; group = `author`, 0 authors
span >1 fold). The shared OOF row order is [`data/oof_meta.csv`](data/oof_meta.csv). The verified
234→Perch-eBird mapping is [`data/perch_zeroshot_map.csv`](data/perch_zeroshot_map.csv).

---

## Level A — re-derive the headline (no competition data needed)

```bash
pip install -r requirements.txt

# 1. Estimator self-test (synthetic asserts)
python src/b2_oracle_headroom.py --selftest

# 2. Run the core estimator on the bundled real 2000-row OOF sample
python src/b2_oracle_headroom.py \
    --anchor  data/sample_oof/perch_anchor_oof.npy \
    --sidecar data/sample_oof/cnn_ensemble_oof.npy \
    --targets data/sample_oof/oof_targets.npy \
    --meta    data/sample_oof/oof_meta.csv \
    --n_boot  200 --out_json /tmp/sample_headroom.json

# 3. Inspect the released full-data headline (8-cell de-biased oracle headroom)
python - <<'PY'
import json
d = json.load(open("results/b2_final.json"))
for k, c in d["cells"].items():
    h = c["headroom_heldout_debiased"]; t = c["tost_delta_0.01"]; f = c["fusion_vs_best"]
    print(f"{k:30s} debiased ΔAUC={h['debiased_mean']:+.4f} "
          f"CI=[{h['debiased_ci95'][0]:+.4f},{h['debiased_ci95'][1]:+.4f}] "
          f"TOST.01_equiv={t['equivalent']} fusion={f['delta_mean']:+.4f}")
PY
```

`b2_oracle_headroom.py` is the single-cell estimator; `b2_headline_driver.py` / `b2_corrected.py` orchestrate
the full 8-cell matrix with the recentered label-permutation null and TOST. `b2_format_results.py` renders
the table.

## Level B — full regeneration

Environment notes (Ascend NPU; substitute a matching torch build on CUDA/CPU) are in
[`docs/HARNESS.md`](docs/HARNESS.md), which also documents the Perch ONNX I/O contract and the 206-vs-234
class handling.

```bash
# 0. Obtain (NOT included): BirdCLEF++ 2026 audio + train.csv + taxonomy;
#    Perch-V2 ONNX; PANNs CNN14 weights; BEATs code (src/beats_mod/) + checkpoint.

# 1. Perch-V2 anchors (CPU-ONNX): embeddings + zero-shot eBird head + linear probe
python src/b1_perch_extract.py --mode extract --batch_size 16 --threads 8 --cache_label_head
python src/b1_perch_extract.py --mode oof
python src/b1_perch_zeroshot.py        # -> perch_zeroshot_oof.npy (+ perch_zeroshot_map.csv)
python src/b1_perch_linear_probe.py    # -> perch_anchor_oof.npy

# 2. Author challengers (5 folds x 5 seeds each; NPU/GPU)
python src/b1_build_mel_cache.py       # mel cache (≈17-20x speedup over on-the-fly)
python src/b1_cnn_train.py  --folds 0 --epochs 30 --backbone tf_efficientnet_b0_ns
python src/b1_pann_train.py --folds 0 --epochs 30
python src/b1_beats_train.py --folds 0 --epochs 30   # needs src/beats_mod/ + BEATs ckpt
python src/b1_matrix_merge.py          # merge per-fold/seed OOF -> *_ensemble_oof.npy
python src/b1_merge_beats.py
python src/b1_pooled_auc_compare.py    # -> results/pooled_auc_compare.json

# 3. Headline + corroborating audits (CPU)
python src/b2_corrected.py             # -> results/b2_final.json (8-cell de-biased headline)
python src/b2_postproc_ablation.py     # -> results/postproc_ablation.json
python src/b3_domain_shift.py          # -> results/domain_shift.json
python src/supp_robustness.py && python src/supp_robustness_merge.py   # robustness gap
python src/supp_stats_hardening.py     # repeated-splits / leave-one-author / FDR
python src/supp_nested_oracle.py       # nested-CV corroboration
python src/add_coverage.py             # -> results/coverage_audit.json
python src/cv_lb_concordance.py        # -> results/cv_lb_concordance.json
```

Wall-time guidance and the exact NPU scheduling/caching strategy are in `docs/HARNESS.md`. The analysis layer
runs in seconds-to-minutes; full sidecar training is the only heavy cost.
