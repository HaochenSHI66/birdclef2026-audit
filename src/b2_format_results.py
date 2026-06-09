"""Format b2_results.json into a markdown log block + console summary."""
import json, sys

d = json.load(open(sys.argv[1]))
cells = d["cells"]

def f(x, n=4):
    return "nan" if x is None else f"{x:.{n}f}"

lines = []
lines.append(f"rows={d['rows']}  authors={d['n_authors']}  fold_sizes={d['fold_sizes']}")
for key, r in cells.items():
    h = r["headroom_heldout_vs_best"]; fv = r["fusion_vs_best"]; nh = r["null_headroom"]
    lines.append("")
    lines.append(f"### {key}  (best_single={r['best_single_model']}, common eval cls={r['n_common_eval_classes']})")
    pm = "  ".join(f"{k}={f(v)}" for k, v in r["per_model_macro_auc"].items())
    lines.append(f"- per-model macro-AUC: {pm}")
    lines.append(f"- ORACLE TRIPLET: apparent(in-sample)={f(r['apparent_oracle_macro_auc'])}  "
                 f"held-out(deployable)={f(r['heldout_oracle_macro_auc'])}  "
                 f"optimism_gap={f(r['optimism_gap'])}")
    lines.append(f"- held-out oracle - best_single (point)={f(r['heldout_minus_best_single_point'])}")
    lines.append(f"- PRIMARY headroom dAUC (held-out oracle vs best, author-clustered): "
                 f"mean={f(h['delta_mean'])}  CI95=[{f(h['ci95'][0])},{f(h['ci95'][1])}]  "
                 f"MDE={f(h['MDE_approx'])}  (n_boot={h['n_boot']}, n_authors={h['n_authors']})")
    lines.append(f"- calibrated UNCLAMPED fusion macro={f(r['calibrated_unclamped_fusion_macro_auc'])}  "
                 f"fusion - best (point)={f(r['fusion_minus_best_single_point'])}  "
                 f"fusion vs best dAUC mean={f(fv['delta_mean'])} CI95=[{f(fv['ci95'][0])},{f(fv['ci95'][1])}]")
    lines.append(f"- NULL-headroom (within-author permuted sidecars): mean={f(nh.get('null_mean'))}  "
                 f"p95={f(nh.get('null_p95'))}  max={f(nh.get('null_max'))}  (n_rep={nh.get('n_rep')})")
    lines.append(f"- dup-anchor control headroom={f(r['dup_anchor_control_headroom'])}")
    cc = r["heldout_choice_counts"]
    lines.append(f"- held-out per-fold choice counts: {cc}")
    for nm, comp in r["complementarity"].items():
        lines.append(f"- complementarity anchor vs {nm}: spearman={f(comp['spearman'],3)} "
                     f"pearson={f(comp['pearson'],3)} Kuncheva_Q={f(comp['kuncheva_Q'],3)} "
                     f"double_fault={f(comp['double_fault'],3)}")

print("\n".join(lines))
