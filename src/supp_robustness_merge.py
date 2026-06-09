"""SUPP #6 merge — combine CNN (add_robustness_gap.json), PANN, BEATs robustness into
data/oof/supp_robustness_full.json with the headline verdict: is Perch (foundation
embedding) more robust to focal->soundscape shift than ALL THREE challengers?"""
import json, os

OOF = os.path.expanduser("~/SHC/birdclef2026_clef/data/oof")


def load(p):
    fp = os.path.join(OOF, p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def main():
    cnn = load("add_robustness_gap.json")
    pann = load("supp_rg_pann.json")
    beats = load("supp_rg_beats.json")

    rows = {}
    if cnn:
        rows["cnn"] = {
            "focal": cnn["cnn_focal_auc"], "soundscape": cnn["cnn_soundscape_auc"],
            "drop": cnn["cnn_drop"], "matched_drop": cnn["matched_class"]["cnn_drop"],
            "perch_matched_drop": cnn["matched_class"]["perch_drop"],
            "matched_gap": cnn["matched_class"]["gap_cnn_minus_perch"],
            "n_joint": cnn["matched_class"]["n_joint_classes"],
            "perch_more_robust": cnn["matched_class"]["perch_more_robust_than_cnn"]}
    for nm, j in [("pann", pann), ("beats", beats)]:
        if j:
            mc = j["matched_class"]
            rows[nm] = {
                "focal": j[f"{nm}_focal_auc"], "soundscape": j[f"{nm}_soundscape_auc"],
                "drop": j[f"{nm}_drop"], "matched_drop": mc[f"{nm}_drop"],
                "perch_matched_drop": mc["perch_drop"],
                "matched_gap": mc["gap_model_minus_perch"], "n_joint": mc["n_joint_classes"],
                "perch_more_robust": mc["perch_more_robust_than_model"]}

    perch_drop = cnn["perch_drop"] if cnn else (pann or beats or {}).get("perch_drop")
    all_more = all(v["perch_more_robust"] for v in rows.values()) if rows else None
    out = {
        "task": "full_robustness_gap_perch_vs_cnn_pann_beats",
        "perch_focal_to_soundscape_drop": perch_drop,
        "perch_focal_macro": (cnn or {}).get("perch_focal_macro_full"),
        "perch_soundscape_macro": (cnn or {}).get("perch_soundscape_macro"),
        "models_present": list(rows.keys()),
        "per_model": rows,
        "verdict_perch_more_robust_than_all": all_more,
        "interpretation": ("Each challenger trained on folds 0-3, held fold 4, checkpointed, "
                           "re-inferred on the 66-file/1478-window labeled soundscape set. "
                           "matched_gap = challenger_drop - perch_drop on the joint class set; "
                           ">0 means Perch is more robust. Verdict True iff Perch beats ALL three."),
    }
    out_path = os.path.join(OOF, "supp_robustness_full.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print("WROTE", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
