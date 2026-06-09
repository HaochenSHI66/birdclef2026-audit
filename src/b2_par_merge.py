"""Merge the 8 per-cell jsons (b2_par_cell.py output) into b2_final.json using the
EXACT schema b2_corrected.main() writes: a top-level dict with run metadata + a
"cells" dict keyed by f"{anchor}{combo}", each value being the run_cell() result.
"""
import argparse, json, glob, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_dir", required=True)   # dir holding b2cell_*.json
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    cell_files = sorted(glob.glob(os.path.join(args.cell_dir, "b2cell_*.json")))
    cells = {}
    meta = None
    for cf in cell_files:
        with open(cf) as fh:
            obj = json.load(fh)
        if meta is None:
            meta = obj
        cells[obj["key"]] = obj["result"]

    assert len(cells) == 8, f"expected 8 cells, got {len(cells)}: {sorted(cells)}"

    # Reconstruct b2_corrected.main()'s top-level schema verbatim.
    results = {
        "rows": meta["rows"],
        "n_authors": meta["n_authors"],
        "fold_sizes": meta["fold_sizes"],
        "n_boot": meta["n_boot"],
        "n_null": meta["n_null"],
        "models_present": "perch_lp,perch_zs,cnn,pann,beats (ALL 4 model families)",
        "global_oof_caveat": (
            "Base OOF columns are single-level GLOBAL OOF (each base model saw the "
            "outer fold's rows during its own training). The held-out SELECTION/fusion "
            "here is leak-free; the base FEATURES are not. Global-OOF inflates the best "
            "single model -> SHRINKS apparent residual headroom -> the 'no headroom' "
            "finding is CONSERVATIVE (biased toward our own conclusion). The per-seed "
            "sub-ensemble proxy bounds the bias empirically."),
        "cells": {},
    }
    # Preserve the canonical cell order LP{+CNN,+PANN,+BEATs,+all3}, ZS{...}.
    order = [
        "perch_linear_probe+CNN", "perch_linear_probe+PANN",
        "perch_linear_probe+BEATs", "perch_linear_probe+CNN+PANN+BEATs",
        "perch_zeroshot+CNN", "perch_zeroshot+PANN",
        "perch_zeroshot+BEATs", "perch_zeroshot+CNN+PANN+BEATs",
    ]
    for k in order:
        assert k in cells, f"missing cell {k}"
        results["cells"][k] = cells[k]

    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"[merge] wrote {args.out_json} with {len(results['cells'])} cells")


if __name__ == "__main__":
    main()
