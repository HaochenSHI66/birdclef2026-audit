# Measuring the Ceiling — A Reproducible Oracle-Headroom & Coverage Audit of Perch-V2 for BirdCLEF++ 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Status: research artifact](https://img.shields.io/badge/status-research%20artifact-informational.svg)](#)
[![Reproducible: OOF + scripts](https://img.shields.io/badge/reproducible-OOF%20%2B%20scripts-success.svg)](REPRODUCE.md)

> **TL;DR** — We could not find any *deployable* per-class hard-selection headroom over the single best
> model when a Perch-V2 bioacoustic foundation head is one of the candidates (de-biased oracle ΔAUC ≤ 0
> in **all 8** anchor×challenger cells; formal equivalence at δ=0.01 in 2). Model complementarity is real
> but is exploitable **only by calibrated soft fusion** (up to +0.009 AUC). Separately, **31 of 234** scored
> classes are structurally outside Perch's fixed eBird output vocabulary (incl. 25 undescribed insect
> morphospecies), and Perch degrades much less than from-scratch CNN/PANN/BEATs challengers under
> focal→soundscape shift. This repo releases the **code, fixed splits, result artifacts, and label-coverage
> maps** so every number in the paper is independently re-derivable.

This is the companion reproducibility package for our **CLEF 2026 / LifeCLEF (BirdCLEF++) working note**.
It is an **audit harness**, not a competition solution: it does not ship a new SOTA model and makes no
ranking claim. Its contribution is a *falsifiable measurement* with confidence intervals, equivalence
tests, and a transparent provenance trail.

---

## What this is

A reproducible **oracle-headroom + eBird label-coverage audit** of Google's Perch-V2 bioacoustic
foundation model on BirdCLEF++ 2026 (LifeCLEF Task 2; Pantanal multi-taxa passive acoustic monitoring).
On a single locked, recordist-grouped 5-fold cross-validation reconstruction we ask three questions a
practitioner actually faces:

1. **Is there per-class oracle-selection headroom** over the single best model when a strong foundation
   anchor is in the candidate set — and is any of it *deployable* (i.e. survives leak-free out-of-fold
   selection)?
2. **Do the ecological post-processing knobs** that nudge the community Kaggle leaderboard stack help a
   clean foundation-embedding baseline?
3. **How badly does the foundation model degrade** from focal (Xeno-Canto) recordings to in-situ
   soundscape conditions, and how does that compare to trained challengers?

The estimator is a **leak-free held-out (nested) per-class hard-selection oracle**, de-biased against a
label-permutation null, with **author-clustered bootstrap** confidence intervals, a **minimum detectable
effect** (MDE), and a **pre-registered TOST** equivalence test. We add a coverage audit of Perch's eBird
output vocabulary, a domain-shift / robustness-gap diagnosis, a post-processing ablation, and a CV↔LB
concordance sanity check.

---

## Key findings

All numbers below are re-derivable from the released artifacts in [`results/`](results/) and the scripts in
[`src/`](src/). The source artifact for each result is named in brackets.

- **No detectable deployable hard-selection headroom.** Across **8/8** anchor×challenger cells the
  de-biased held-out oracle ΔAUC is **≤ 0** (range −0.0019 to −0.0242). Formal **TOST equivalence at
  δ=0.01** passes in 2 cells (LP+PANN, ZS+BEATs); MDE ≈ 0.005–0.007. This is a *power-bounded null on one
  operator* (per-class hard selection), **not** a claim that the models are redundant. [`results/b2_final.json`]
- **Soft fusion is the only payoff, and it is small.** Calibrated, unclamped soft fusion beats the best
  single model by **+0.0011…+0.0092 AUC**, significant (95% CI excludes 0) in **5/8** cells. Complementarity
  exists; hard per-class routing simply cannot capture it out-of-fold. [`results/b2_final.json`]
- **A foundation coverage wall.** Of 234 scored classes, **203 map** into Perch's fixed eBird output space
  (158 exact eBird-code + 45 scientific-name fallback); **31 do not** — including **25 undescribed insect
  "son01–25" morphospecies** (no binomial, zero focal training positives) and 6 absent named taxa (3
  Amphibia, 2 Mammalia, 1 Reptilia). A fixed foundation output vocabulary structurally cannot express
  undescribed taxa. [`results/coverage_audit.json`]
- **Perch is markedly more robust under domain shift.** Focal→soundscape macro-AUC drop: **Perch 0.039**
  vs **CNN 0.198 / PANN 0.185 / BEATs 0.182** (matched-class gaps +0.16/+0.15/+0.15). Perch beats all three
  challengers. [`results/supp_robustness_full.json`, `results/domain_shift.json`]
- **Post-processing knobs do not help the clean baseline.** On the reproducible Perch zero-shot baseline,
  taxonomy smoothing is null (+0.0008, CI⊇0), cross-taxa gating hurts (−0.0032), and a naive global spatial
  prior strongly hurts (−0.0306). None help. [`results/postproc_ablation.json`]
- **Honest provenance.** Our Kaggle Bronze (private LB 0.94221, rank **262/4085, top 6.4%**) came from an
  **attributed community foundation-embedding stack**; our own from-scratch acoustic sidecars added **zero**
  private-LB movement. The medal is *not* the contribution — the reproducible audit is.

---

## Results at a glance

### Per-model pooled macro-AUC (common evaluable-class mask)
Source: [`results/pooled_auc_compare.json`] (common 197-class mask) and [`results/b2_final.json`] (BEATs).

| model | macro-AUC | note |
|---|---|---|
| CNN ensemble (5-seed) | **0.9398** | author timm EfficientNet-B0, ImageNet-pretrained + linear head |
| PANN ensemble (5-seed) | **0.9379** | author PANNs CNN14, AudioSet-pretrained |
| Perch-V2 zero-shot (anchor A1) | **0.9356** | Perch-V2 eBird head, max-aggregated, no fine-tune |
| Perch-V2 linear probe (anchor A2) | **0.9201** | per-fold logistic regression on the 1536-d embedding |
| BEATs ensemble (5-seed) | **0.8876** | BEATs frozen encoder + head (weakest family) |

Seed spread (training noise only): CNN mean 0.9212 ± 0.0017, PANN mean 0.9269 ± 0.0010
[`results/pooled_auc_compare.json`]. Note: zero-shot Perch (0.9356) beats its own supervised linear probe
(0.9201).

### Headline: 8-cell de-biased oracle-headroom table
Source: [`results/b2_final.json`] — n_boot = 1000, n_null = 50, author-clustered bootstrap, common
evaluable-class mask (196 for LP cells, 194 for ZS cells), pre-registered TOST at δ=0.01.

| cell | best single | apparent oracle | held-out oracle | **de-biased ΔAUC** [95% CI] | TOST δ=0.01 | soft-fusion ΔAUC [95% CI] |
|---|---|---|---|---|---|---|
| LP + CNN            | 0.9396 (cnn)     | 0.9482 | 0.9396 | **−0.0064** [−0.0132, −0.0005] | not equiv | +0.0011 [−0.0014, +0.0038] |
| LP + PANN           | 0.9365 (pann)    | 0.9461 | 0.9394 | **−0.0019** [−0.0075, +0.0035] | **EQUIV**  | +0.0043 [+0.0013, +0.0078] \* |
| LP + BEATs          | 0.9202 (perch_lp)| 0.9312 | 0.9170 | **−0.0178** [−0.0223, −0.0136] | not equiv | +0.0030 [−0.0018, +0.0083] |
| LP + CNN+PANN+BEATs | 0.9396 (cnn)     | 0.9508 | 0.9402 | **−0.0084** [−0.0139, −0.0034] | not equiv | +0.0045 [+0.0017, +0.0076] \* |
| ZS + CNN            | 0.9412 (cnn)     | 0.9514 | 0.9050 | **−0.0242** [−0.0305, −0.0170] | not equiv | +0.0069 [+0.0042, +0.0095] \* |
| ZS + PANN           | 0.9379 (pann)    | 0.9515 | 0.9111 | **−0.0138** [−0.0199, −0.0065] | not equiv | +0.0092 [+0.0050, +0.0139] \* |
| ZS + BEATs          | 0.9355 (perch_zs)| 0.9405 | 0.9198 | **−0.0033** [−0.0072, +0.0007] | **EQUIV**  | −0.0000 [−0.0032, +0.0033] |
| ZS + CNN+PANN+BEATs | 0.9412 (cnn)     | 0.9545 | 0.9140 | **−0.0140** [−0.0202, −0.0071] | not equiv | +0.0090 [+0.0067, +0.0114] \* |

LP = Perch linear-probe anchor; ZS = Perch zero-shot anchor. `\*` = soft-fusion 95% CI excludes 0
(significant). The large gap between *apparent* oracle (0.94–0.95) and *held-out* oracle (0.905–0.940) on
the ZS-anchor cells is **selection optimism**: apparent class-specialist diversity does not survive
leak-free out-of-fold selection.

### Domain shift, post-processing, coverage, CV↔LB

| diagnosis | result | source |
|---|---|---|
| Perch focal→soundscape macro-AUC | 0.9283 → 0.7367 (mean per-class drop +0.0393, CI [0.0010, 0.0835]) | `results/domain_shift.json` |
| Robustness gap (drop, lower = more robust) | Perch 0.039 vs CNN 0.198 / PANN 0.185 / BEATs 0.182 | `results/supp_robustness_full.json` |
| Post-proc on clean baseline (Δ macro-AUC) | taxonomy +0.0008 (null) · cross-taxa −0.0032 · spatial −0.0306 — none help | `results/postproc_ablation.json` |
| eBird coverage | 203/234 mapped (158 exact + 45 sci-name); 31 unmapped (25 insect morphospecies + 6 absent taxa) | `results/coverage_audit.json` |
| CV↔LB concordance (n=3 reproducible) | Spearman ρ = 1.0 (sanity check only; Kendall p = 0.333 is the n=3 floor) | `results/cv_lb_concordance.json` |

---

## Repository structure

```
birdclef2026-audit/
├── README.md
├── LICENSE                  # MIT (our code only; third-party components NOT included — see below)
├── requirements.txt
├── REPRODUCE.md             # exact commands, in 3 steps
├── src/                     # our analysis / eval / training code (46 files)
│   ├── b1_perch_extract*.py        # Perch-V2 CPU-ONNX embedding/label extraction
│   ├── b1_perch_zeroshot.py        # zero-shot eBird-head anchor (A1)
│   ├── b1_perch_linear_probe.py    # per-fold linear-probe anchor (A2)
│   ├── b1_cnn_train.py / b1_pann_train.py / b1_beats_train.py   # author challengers
│   ├── b1_build_*_cache.py, b1_matrix_*.py, run_*.py            # caching + scheduling
│   ├── b2_oracle_headroom.py       # core oracle-headroom estimator (+ self-test)
│   ├── b2_headline_driver.py       # the 8-cell headline driver
│   ├── b2_corrected.py             # recentered-null / TOST corrected analysis
│   ├── b2_postproc_ablation.py     # post-processing ablation
│   ├── b3_domain_shift.py, b3_cnn_soundscape.py   # focal→soundscape shift
│   ├── supp_stats_hardening.py     # repeated-splits, leave-one-author, FDR hardening
│   ├── supp_nested_oracle*.py      # nested-CV corroboration of the headline
│   ├── supp_robustness*.py         # Perch-vs-challenger robustness gap
│   ├── supp_frozen_transfer.py     # frozen selector/fusion transfer under shift
│   ├── add_coverage.py             # eBird coverage-wall audit
│   └── cv_lb_concordance.py, add_*.py             # corroborating analyses
├── results/                 # released result JSON artifacts (every paper number traces here)
│   ├── b2_final.json                # 8-cell headline (THE main result)
│   ├── pooled_auc_compare.json      # per-model macro-AUC table
│   ├── coverage_audit.json          # 31/234 coverage wall
│   ├── domain_shift.json, supp_robustness_full.json   # shift + robustness
│   ├── postproc_ablation.json       # post-proc null
│   ├── cv_lb_concordance.json       # CV↔LB sanity
│   └── supp_*.json, add_*.json       # hardening + corroborating audits
├── data/
│   ├── folds.csv                    # recordist-grouped 5-fold split (filename,fold)
│   ├── oof_meta.csv                 # OOF row order (filename,fold,author) — clustering key
│   ├── perch_zeroshot_map.csv       # verified 234→Perch eBird mapping (primary_label,stage,perch_idxs)
│   └── sample_oof/                  # 2000-row real OOF sample so analysis runs out-of-the-box
└── docs/
    └── HARNESS.md                   # environment, data flow, model/I-O contract, full-run cost
```

---

## Reproduce in 3 steps

```bash
# 1. Environment (CPU is enough for the analysis layer; NPU/GPU only for training)
pip install -r requirements.txt

# 2. Smoke-test the core estimator on the bundled 2000-row OOF sample
python src/b2_oracle_headroom.py --selftest
python src/b2_oracle_headroom.py \
    --anchor data/sample_oof/perch_anchor_oof.npy \
    --sidecar data/sample_oof/cnn_ensemble_oof.npy \
    --targets data/sample_oof/oof_targets.npy \
    --meta data/sample_oof/oof_meta.csv \
    --n_boot 200 --out_json /tmp/sample_headroom.json

# 3. Re-derive the headline from the released JSON (no recompute needed),
#    or regenerate full OOF + the 8-cell table (see REPRODUCE.md for the full pipeline)
python -c "import json;d=json.load(open('results/b2_final.json'));\
print({k:round(v['headroom_heldout_debiased']['debiased_mean'],4) for k,v in d['cells'].items()})"
```

See [REPRODUCE.md](REPRODUCE.md) for the full extraction → training → analysis pipeline and exact commands.
Full per-model OOF arrays (~33 MB each) are **not** committed to keep the repo lean; they are regenerated
by the `b1_*` scripts, and a 2000-row real sample is included for smoke testing.

---

## Data provenance & component attribution

**This repository contains ONLY our own reproducible audit work.** Third-party code and models are
**cited and linked, not redistributed** — please obtain them from their original sources under their own
licenses.

| Component | Status | Where to obtain |
|---|---|---|
| **Perch-V2** bioacoustic foundation model (Google) | third-party, **not included** | Perch-V2 ([van Merriënboer et al., 2025](https://github.com/google-research/perch)); ONNX export via Kaggle `tuckerarrants/perch-v2-no-dft-onnx` |
| **PANNs CNN14** AudioSet-pretrained weights | third-party, **not included** | [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) (Kong et al., 2020) |
| **BEATs** encoder code + `BEATs_iter3+AS2M` checkpoint | third-party, **not included** | [microsoft/unilm/beats](https://github.com/microsoft/unilm/tree/master/beats) (Chen et al., 2023); ckpt via `lpepino/beats_ckpts`. `src/b1_beats_train.py` expects a `src/beats_mod/` you supply. |
| Community Kaggle Bronze stack (ProtoSSM/ResidualSSM head, PowerOpt branch, BirdNET gating, ONNX wrapper) | third-party, **not included** | Original Kaggle authors (yukiZ ProtoSSM; Karnakbayev PowerOpt; BirdNET [Kahl et al., 2021]). Used only for the LB *anecdote* — **not** part of the reproducible CV audit. |
| BirdCLEF++ 2026 audio + `train.csv` + taxonomy | competition data, **not included** | [Kaggle: birdclef-2026](https://www.kaggle.com/competitions/birdclef-2026) |

The bundled `data/oof_meta.csv` contains recordist (author) IDs/names that are already public in the
competition's `train.csv`; they are used purely as the bootstrap **clustering key** (no leakage across folds).

**Honest scope.** The reproducible audit is built on the **Perch-V2 CPU-ONNX head + our own trained
challengers**. The full community TF/JAX Bronze pipeline is *not* reproducible per-fold and is reported only
as a clearly separated leaderboard anecdote (`results/cv_lb_concordance.json`). Our author sidecars added
zero private-LB gain; we do not claim model novelty.

---

## Limitations (read before citing)

- **Single-level global OOF base features.** Each base model saw the outer fold's rows during its own
  training; the held-out *selection/fusion* is leak-free, but the base *features* are not. This **inflates**
  the best single model and therefore **shrinks** apparent headroom — i.e. the "no headroom" finding is
  **conservative**. A per-seed sub-ensemble proxy bounds the bias (headroom spread ≈ 0.0015–0.0051).
- **One CV split.** The bootstrap resamples authors, not folds/seeds/splits; split-design uncertainty is not
  in the CIs. Five seeds estimate training noise only.
- **Soundscape eval is small** (66 files / 47 common scorable classes, multi-label); the domain-shift
  macro-AUC is Perch-only, the robustness gap uses re-trained checkpoints.
- **CV↔LB is a sanity check only** (n = 3 strictly-reproducible configs; late Kaggle submission is closed).
- **BEATs** is a frozen encoder + head (not fine-tuned) and is the weakest single family.

---

## Citation

If you use this harness, please cite the working note (citation to be finalized on acceptance):

```bibtex
@inproceedings{shi2026measuringceiling,
  title     = {Measuring the Ceiling: A Reproducible Oracle-Blend Audit of
               Perch-V2 Ensemble Headroom in BirdCLEF++ 2026},
  author    = {Shi, Haochen},
  booktitle = {Working Notes of CLEF 2026 -- Conference and Labs of the
               Evaluation Forum (LifeCLEF/BirdCLEF++)},
  year      = {2026},
  series    = {CEUR Workshop Proceedings},
  note      = {Reproducibility package: https://github.com/HaochenSHI66/birdclef2026-audit}
}
```

## License

Our code and analysis artifacts in this repository are released under the **MIT License** (see
[LICENSE](LICENSE)). Third-party models, checkpoints, and community competition code are **not** included and
remain under their respective licenses — see the attribution table above.
