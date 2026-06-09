"""Parallel SINGLE-CELL runner for B2 FINAL.

Runs ONE (anchor x combo) cell of b2_corrected.py in its own process and writes a
per-cell json. The estimator is run_cell() imported VERBATIM from b2_corrected, with
the SAME inputs and the SAME per-cell seed the sequential driver uses (args.seed,
default 0, identical for every cell). Therefore each cell is bit-for-bit identical to
what the sequential `main()` computes for that cell.

Cell order MUST match b2_corrected.main():
  anchors = {perch_linear_probe, perch_zeroshot}  (in that order)
  combos  = [+CNN, +PANN, +BEATs, +CNN+PANN+BEATs] (in that order)
  -> cells 0..7 = LP+CNN, LP+PANN, LP+BEATs, LP+all3, ZS+CNN, ZS+PANN, ZS+BEATs, ZS+all3
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
# Import the EXACT estimator + cell function from the corrected driver.
from b2_corrected import run_cell


def load_all(d):
    def L(name):
        return np.load(os.path.join(d, name)).astype(np.float64)
    perch_lp = L("perch_anchor_oof.npy")
    perch_zs = L("perch_zeroshot_oof.npy")
    cnn = L("cnn_ensemble_oof.npy")
    pann = L("pann_ensemble_oof.npy")
    beats = L("beats_ensemble_oof.npy")
    y = L("oof_targets.npy")
    meta = pd.read_csv(os.path.join(d, "oof_meta.csv"))
    authors = meta["author"].values
    folds = meta["fold"].values

    def load_seeds(prefix):
        arrs = []
        for s in range(41, 46):
            p = os.path.join(d, f"{prefix}_seed{s}_oof.npy")
            if os.path.exists(p):
                arrs.append(np.load(p).astype(np.float64))
        return arrs
    cnn_seeds = load_seeds("cnn")
    pann_seeds = load_seeds("pann")
    beats_seeds = load_seeds("beats")

    # IDENTICAL row-keep to b2_corrected.main()
    keep = ~(np.isnan(perch_lp).all(1) | np.isnan(perch_zs).all(1)
             | np.isnan(cnn).all(1) | np.isnan(pann).all(1) | np.isnan(beats).all(1))
    if not keep.all():
        perch_lp, perch_zs = perch_lp[keep], perch_zs[keep]
        cnn, pann, beats, y = cnn[keep], pann[keep], beats[keep], y[keep]
        authors, folds = authors[keep], folds[keep]
        cnn_seeds = [a[keep] for a in cnn_seeds]
        pann_seeds = [a[keep] for a in pann_seeds]
        beats_seeds = [a[keep] for a in beats_seeds]

    seed_arrs = {}
    if cnn_seeds:
        seed_arrs["cnn"] = cnn_seeds
    if pann_seeds:
        seed_arrs["pann"] = pann_seeds
    if beats_seeds:
        seed_arrs["beats"] = beats_seeds

    anchors = {"perch_linear_probe": perch_lp, "perch_zeroshot": perch_zs}
    combos = [
        ("+CNN", [("cnn", cnn)]),
        ("+PANN", [("pann", pann)]),
        ("+BEATs", [("beats", beats)]),
        ("+CNN+PANN+BEATs", [("cnn", cnn), ("pann", pann), ("beats", beats)]),
    ]
    return anchors, combos, seed_arrs, y, folds, authors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_dir", required=True)
    ap.add_argument("--cell", type=int, required=True)  # 0..7
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_null", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    anchors, combos, seed_arrs, y, folds, authors = load_all(args.oof_dir)
    anchor_items = list(anchors.items())  # ordered: LP, ZS

    # Flatten to the SAME cell order as b2_corrected.main(): anchor-outer, combo-inner
    flat = []
    for an, anchor in anchor_items:
        for cname, specs in combos:
            flat.append((an, anchor, cname, specs))
    assert 0 <= args.cell < len(flat), f"cell must be 0..{len(flat)-1}"
    an, anchor, cname, specs = flat[args.cell]
    key = f"{an}{cname}"
    print(f"[par-cell {args.cell}] {key} (seed={args.seed}) ...", flush=True)

    res = run_cell(an, anchor, specs, seed_arrs, y, folds, authors,
                   args.n_boot, args.n_null, args.seed)

    out = {
        "cell_index": args.cell,
        "key": key,
        "rows": int(len(y)),
        "n_authors": int(len(set(authors))),
        "fold_sizes": {int(f): int((folds == f).sum()) for f in sorted(set(folds))},
        "n_boot": args.n_boot,
        "n_null": args.n_null,
        "seed": args.seed,
        "result": res,
    }
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    db = res["headroom_heldout_debiased"]
    t = res["tost_delta_0.01"]
    print(f"[par-cell {args.cell}] DONE {key} best={res['best_single_model']} "
          f"{res['best_single_macro_auc']:.4f} | d_raw={res['heldout_minus_best_single_point']:+.4f} "
          f"| null_lp={res['null_labelperm']['null_mean']:+.4f} "
          f"| debiased={db['debiased_mean']:+.4f} CI{[round(x,4) for x in db['debiased_ci95']]} "
          f"| TOST.01 {'EQUIV' if t['equivalent'] else 'no'} "
          f"| fusion d={res['fusion_minus_best_single_point']:+.4f} -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
