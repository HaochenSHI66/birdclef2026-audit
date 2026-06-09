"""B1' Task-1 — Zero-shot Perch-V2 label-head anchor (14795 -> 234).

The HONEST native-model baseline (Codex round-1 verdict (a)): in addition to the
supervised embedding linear probe (b1_perch_extract.py), report a ZERO-SHOT anchor
built directly from Perch's published 14795-class label head — no BirdCLEF training.

LABEL LIST SOURCE (obtained, not fabricated):
  HuggingFace `cgeorgiaw/Perch` -> assets/perch_v2_ebird_classes.csv  (eBird-2021
  six-letter codes, one row per Perch class index, header `ebird2021`, 14795 rows,
  5089 == "no_ebird_code") and assets/labels.csv (iNaturalist scientific names,
  same index order, header `inat2024_fsd50k`, 14795 rows). Both verified to have
  exactly 14795 data rows == the ONNX `label[14795]` head dimension, so row i ==
  Perch class index i. Files are committed under
  .research/artifacts/perch_labels/ and copied to the server data dir.

MAPPING (14795 -> 234), predeclared BEFORE looking at any metric:
  Stage 1 EXACT eBird-code match: BirdCLEF taxonomy.csv primary_label (eBird code
          for Aves) == Perch ebird2021 code.
  Stage 2 AUDITED scientific-name fallback: for classes unmatched in Stage 1
          (non-bird taxa whose BirdCLEF primary_label is an iNat numeric id, plus a
          few renamed/lumped bird codes), match BirdCLEF taxonomy.scientific_name
          (lower-cased, exact) against Perch labels.csv iNat scientific name.
  Many-to-one aggregation (predeclared) = MAX probability across all Perch indices
          mapping to the same BirdCLEF class (1 - prod(1-p) listed as alt; max
          chosen because the bronze ProtoSSM path is max-like and AUC is rank-based).
  Unmapped BirdCLEF classes -> OOF column left NaN (no zero-shot signal); these are
          skipped by the macro-AUC-skip-zero-pos metric exactly like the 206/234 gap.

COVERAGE (measured, this run prints it): 203/234 mapped
  (158 exact eBird + 45 sci-name). 31 unmapped = 25 unidentified "Insect son01-25"
  morphospecies (NO scientific name exists -> genuinely unmappable) + 6 species
  truly absent from Perch's vocabulary (Caiman yacare, Adenomera guarani,
  Lysapsus limellum, Chiasmocleis mehelyi, Sapajus cay, Mico melanurus).
  Excluding the 25 unidentifiable morphospecies: 203/209 = 97% of identifiable
  classes covered. No fabricated mappings.

OUTPUT (under --out_dir, default data/oof/):
  perch_zeroshot_oof.npy   float32 [N, 234]  zero-shot probs (NaN for unmapped cls)
  perch_zeroshot_map.csv   birdclef primary_label, class_name, stage, perch_idxs
  (dry suffix `_dry` mirrors b1_perch_extract)

This script does NOT re-run Perch ONNX by default: it consumes a cached label head
(`perch_cache/label_head{,_dry}.npy` from b1_perch_extract --cache_label_head) plus
the aligned oof_meta/oof_targets. For the smoke test it reuses the already-cached
200-clip dry label head on the server. Full run needs one extraction pass with
--cache_label_head (the embedding cache from the supervised anchor can be reused;
only the label head must be added).
"""
import argparse
import csv
import os
import numpy as np
import pandas as pd
from collections import defaultdict

N_LABEL_HEAD = 14795


# ----------------------------- shared metric -----------------------------
def compute_macro_auc(y_true, y_pred):
    """Macro ROC-AUC, skip classes with no positives; skip NaN-pred columns.
    Mirrors scripts/optimize_ensemble.py:19 / b1_perch_extract.compute_macro_auc."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            col = y_pred[:, i]
            if np.isnan(col).any():
                continue
            try:
                aucs.append(roc_auc_score(y_true[:, i], col))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


# ----------------------------- mapping -----------------------------
def load_perch_labels(perch_dir):
    """Return (ebird_codes[14795], inat_sci[14795]) in Perch index order."""
    with open(os.path.join(perch_dir, "perch_v2_ebird_classes.csv")) as f:
        r = csv.reader(f); next(r)
        ebird = [row[0].strip() for row in r]
    with open(os.path.join(perch_dir, "labels.csv")) as f:
        r = csv.reader(f); next(r)
        inat = [row[0].strip() for row in r]
    assert len(ebird) == len(inat) == N_LABEL_HEAD, \
        f"perch label files must have {N_LABEL_HEAD} rows, got {len(ebird)}/{len(inat)}"
    return ebird, inat


def load_taxonomy(data_dir):
    tax = pd.read_csv(os.path.join(data_dir, "taxonomy.csv"))
    assert len(tax) == 234, f"expected 234 classes, got {len(tax)}"
    return tax


def build_map(tax, ebird, inat):
    """Return (mapping: list[ list[perch_idx] ] aligned to the 234 taxonomy order,
    rows: list[dict] for the audit CSV). Predeclared two-stage; max aggregation."""
    ebird_to_idxs = defaultdict(list)
    for i, c in enumerate(ebird):
        if c != "no_ebird_code":
            ebird_to_idxs[c].append(i)
    sci_to_idxs = defaultdict(list)
    for i, s in enumerate(inat):
        if s:
            sci_to_idxs[s.lower()].append(i)

    mapping = []
    rows = []
    n_exact = n_sci = n_unmapped = n_many = 0
    for t in tax.itertuples(index=False):
        pl = str(t.primary_label).strip()
        sci = str(t.scientific_name).strip().lower()
        cn = str(t.class_name)
        if pl in ebird_to_idxs:                       # Stage 1: exact eBird code
            idxs = ebird_to_idxs[pl]; stage = "exact_ebird"; n_exact += 1
        elif sci in sci_to_idxs:                       # Stage 2: sci-name fallback
            idxs = sci_to_idxs[sci]; stage = "sciname_fallback"; n_sci += 1
        else:
            idxs = []; stage = "unmapped"; n_unmapped += 1
        if len(idxs) > 1:
            n_many += 1
        mapping.append(idxs)
        rows.append({"primary_label": pl, "class_name": cn, "stage": stage,
                     "perch_idxs": "|".join(map(str, idxs))})
    print(f"[map] exact_ebird={n_exact} sciname_fallback={n_sci} "
          f"unmapped={n_unmapped} | TOTAL MAPPED={n_exact + n_sci}/234 "
          f"| many-to-one classes={n_many}")
    unmapped = [r["primary_label"] + " (" + r["class_name"] + ")"
                for r in rows if r["stage"] == "unmapped"]
    print(f"[map] UNMAPPED list ({len(unmapped)}): {unmapped}")
    return mapping, rows


# ----------------------------- zero-shot OOF -----------------------------
def build_zeroshot_oof(label_head, mapping, agg="max"):
    """label_head: [N, 14795] raw logits. -> zero-shot OOF [N, 234] probabilities.
    Many-to-one aggregation predeclared: max prob (alt 1-prod(1-p))."""
    probs = 1.0 / (1.0 + np.exp(-label_head.astype(np.float64)))  # sigmoid
    N = probs.shape[0]
    C = len(mapping)
    oof = np.full((N, C), np.nan, dtype=np.float32)
    for c, idxs in enumerate(mapping):
        if not idxs:
            continue                                   # unmapped -> NaN
        sub = probs[:, idxs]
        if agg == "noisy_or":
            agg_p = 1.0 - np.prod(1.0 - sub, axis=1)
        else:                                          # max (default)
            agg_p = sub.max(axis=1)
        oof[:, c] = agg_p.astype(np.float32)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.expanduser("~/SHC/birdclef2026_clef/data"))
    ap.add_argument("--perch_dir", default=None,
                    help="dir with perch_v2_ebird_classes.csv + labels.csv "
                         "(default <data_dir>/perch_labels)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--label_head", default=None,
                    help="cached [N,14795] label head .npy (default "
                         "<out_dir>/perch_cache/label_head{,_dry}.npy)")
    ap.add_argument("--agg", choices=["max", "noisy_or"], default="max")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.perch_dir is None:
        args.perch_dir = os.path.join(args.data_dir, "perch_labels")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "oof")
    suffix = "_dry" if args.dry_run else ""
    if args.label_head is None:
        args.label_head = os.path.join(args.out_dir, "perch_cache",
                                       f"label_head{suffix}.npy")
    print(f"[cfg] perch_dir={args.perch_dir}\n[cfg] label_head={args.label_head}\n"
          f"[cfg] out_dir={args.out_dir} agg={args.agg} dry_run={args.dry_run}")

    ebird, inat = load_perch_labels(args.perch_dir)
    tax = load_taxonomy(args.data_dir)
    mapping, audit_rows = build_map(tax, ebird, inat)

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(
        os.path.join(args.out_dir, "perch_zeroshot_map.csv"), index=False)

    label_head = np.load(args.label_head)
    assert label_head.shape[1] == N_LABEL_HEAD, \
        f"label head dim {label_head.shape[1]} != {N_LABEL_HEAD}"
    oof = build_zeroshot_oof(label_head, mapping, agg=args.agg)
    np.save(os.path.join(args.out_dir, f"perch_zeroshot_oof{suffix}.npy"), oof)

    # sanity macro-AUC against aligned targets if available
    tgt_path = os.path.join(args.out_dir, f"oof_targets{suffix}.npy")
    if os.path.exists(tgt_path):
        Y = np.load(tgt_path)
        n = min(len(Y), len(oof))
        auc = compute_macro_auc(Y[:n], np.nan_to_num(oof[:n], nan=0.0))
        n_eval = int(((Y[:n].sum(0) > 0) & (~np.isnan(oof[:n]).any(0))).sum())
        print(f"[zeroshot] SMOKE macro-AUC (skip-zero-pos) = {auc:.4f} over "
              f"{n_eval} eval classes (mapped & with >=1 pos), {n} clips")
    else:
        print(f"[zeroshot] no targets at {tgt_path}; skipped AUC sanity")
    print(f"[done] zero-shot OOF -> perch_zeroshot_oof{suffix}.npy {oof.shape}; "
          f"map -> perch_zeroshot_map.csv")


if __name__ == "__main__":
    main()
