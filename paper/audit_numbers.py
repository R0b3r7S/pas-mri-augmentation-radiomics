#!/usr/bin/env python3
"""
audit_numbers.py — re-derive every numeric claim in the 3D-only paper from
the project's CSVs, with full provenance.

Input CSVs:
    shareable/all_test_metrics.csv         (Dice etc., 274 cell rows)
    shareable/per_patient_dice.csv         (per-patient Dice, 5,206 rows;
                                            produced by extract_per_patient_metrics.py)
    radiomics/comparisons/2d/per_corruption_summary.csv  (mean/median rel-err
        per (model, base_domain, corruption, corruption_mode), 274 rows)
    radiomics/comparisons/2d/per_patient_errors.csv      (per-patient × per-feature
        relative error, ~530k rows; produced by compare_radiomics.py)
    radiomics/comparisons/2d/summary_by_feature.csv      (102-feature summary)

Filtering for the 3D-only paper:
    * keep only the 12 augmentation models (6 augs * BTFE/TSE training)
    * keep only corruption_mode in {3d, clean}
    * drop COMBINED + Transfer models (not analysed in this paper)

Run from project root:
    conda activate monai_placenta
    python paper/audit_numbers.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent

METRICS_CSV    = ROOT / "shareable" / "all_test_metrics.csv"
PERCORR_CSV    = ROOT / "radiomics" / "comparisons" / "2d" / "per_corruption_summary.csv"
FEATURES_CSV   = ROOT / "radiomics" / "comparisons" / "2d" / "summary_by_feature.csv"
PERPATIENT_DICE_CSV   = ROOT / "shareable" / "per_patient_dice.csv"
PERPATIENT_ERRORS_CSV = ROOT / "radiomics" / "comparisons" / "2d" / "per_patient_errors.csv"

# 12 augmentation models analysed in the paper
AUG_MODELS = {
    "Baseline":   ["BTFE_unetplusplus_1Fold",      "TSE_unetplusplus_1Fold"],
    "MixUp":      ["BTFE_unetplusplus_mixup",      "TSE_unetplusplus_mixup"],
    "CutMix":     ["BTFE_unetplusplus_cutmix",     "TSE_unetplusplus_cutmix"],
    "AFA":        ["BTFE_unetplusplus_afa",        "TSE_unetplusplus_afa"],
    "CutMix+AFA": ["BTFE_unetplusplus_cutmix_afa", "TSE_unetplusplus_cutmix_afa"],
    "MixUp+AFA":  ["BTFE_unetplusplus_mixup_afa",  "TSE_unetplusplus_mixup_afa"],
}
AUG_MODELS_FLAT = sum(AUG_MODELS.values(), [])

# Each augmentation maps to its "matched baseline" for cell-by-cell improvement
BASELINE_OF = {m: ("BTFE_unetplusplus_1Fold" if m.startswith("BTFE_") else "TSE_unetplusplus_1Fold")
               for ms in AUG_MODELS.values() for m in ms}


def hr(title=""):
    bar = "=" * 78
    print(f"\n{bar}\n {title}\n{bar}" if title else bar)


def main():
    if not METRICS_CSV.is_file() or not PERCORR_CSV.is_file():
        sys.exit(f"missing inputs: {METRICS_CSV} or {PERCORR_CSV}")

    metrics = pd.read_csv(METRICS_CSV)
    percorr = pd.read_csv(PERCORR_CSV)
    feats   = pd.read_csv(FEATURES_CSV)

    hr(" 0. Provenance")
    print(f"  metrics  {METRICS_CSV.relative_to(ROOT)}        rows={len(metrics)}")
    print(f"  percorr  {PERCORR_CSV.relative_to(ROOT)}  rows={len(percorr)}")
    print(f"  features {FEATURES_CSV.relative_to(ROOT)}      rows={len(feats)}")
    assert len(feats) == 102, f"expected 102 features, got {len(feats)}"

    # --- inner-join Dice and rel-err on the 4-key composite key ---
    keys = ["model", "base_domain", "corruption", "corruption_mode"]
    joined_full = pd.merge(
        metrics[keys + ["dice", "train_domain"]],
        percorr[keys + ["feature_rel_error", "feature_rel_error_median"]],
        on=keys, how="inner",
    ).dropna()
    print(f"  joined (all models, all modes)  rows={len(joined_full)}")

    # --- restrict to 12 augmentation models ---
    j12 = joined_full[joined_full["model"].isin(AUG_MODELS_FLAT)].copy()
    print(f"  joined (12 aug models, all modes)  rows={len(j12)}    (was '264 cells')")

    # --- restrict to 3D corruption mode + clean reference ---
    j3d = j12[j12["corruption_mode"].isin(["3d", "clean"])].copy()
    print(f"  joined (12 aug models, 3D + clean)  rows={len(j3d)}")

    n_clean    = (j3d["corruption_mode"] == "clean").sum()
    n_corrupt  = (j3d["corruption_mode"] == "3d").sum()
    print(f"     clean cells:     {n_clean}    (= 12 models * 2 base-domains)")
    print(f"     corrupted cells: {n_corrupt}  (= 12 models * 5 corruptions * 2 base-domains)")

    # =========================================================================
    # CLAIM 1 — number of analysed cells in the 3D paper
    # =========================================================================
    hr(" 1. Total-cells claim (3D-only)")
    print(f"  joined cells = {len(j3d)}")
    claim_total = len(j3d)

    # =========================================================================
    # CLAIM 2 — Pearson + Spearman correlations Dice ↔ feature error
    # =========================================================================
    hr(" 2. Pearson / Spearman: Dice ↔ feature relative error  (3D + clean)")
    r_med,  p_med  = pearsonr (j3d["dice"], j3d["feature_rel_error_median"])
    r_mean, p_mean = pearsonr (j3d["dice"], j3d["feature_rel_error"])
    rho_med, prho_med = spearmanr(j3d["dice"], j3d["feature_rel_error_median"])
    rho_mean, prho_mean = spearmanr(j3d["dice"], j3d["feature_rel_error"])
    print(f"  Pearson  Dice vs MEDIAN  r = {r_med:+.4f}   p = {p_med:.2e}")
    print(f"  Pearson  Dice vs MEAN    r = {r_mean:+.4f}   p = {p_mean:.2e}")
    print(f"  Spearman Dice vs MEDIAN  rho = {rho_med:+.4f}   p = {prho_med:.2e}")
    print(f"  Spearman Dice vs MEAN    rho = {rho_mean:+.4f}   p = {prho_mean:.2e}")

    # also report on corrupted-only subset (drop clean), as that is the more
    # honest test of whether Dice tracks stability under perturbation
    j3d_corr = j3d[j3d["corruption_mode"] == "3d"]
    rc_med, pc_med   = pearsonr (j3d_corr["dice"], j3d_corr["feature_rel_error_median"])
    rc_mean, pc_mean = pearsonr (j3d_corr["dice"], j3d_corr["feature_rel_error"])
    rhoc_med,  prhoc_med  = spearmanr(j3d_corr["dice"], j3d_corr["feature_rel_error_median"])
    rhoc_mean, prhoc_mean = spearmanr(j3d_corr["dice"], j3d_corr["feature_rel_error"])
    print(f"\n  Restricted to 120 corrupted cells (no clean):")
    print(f"  Pearson  Dice vs MEDIAN  r = {rc_med:+.4f}   p = {pc_med:.2e}")
    print(f"  Pearson  Dice vs MEAN    r = {rc_mean:+.4f}   p = {pc_mean:.2e}")
    print(f"  Spearman Dice vs MEDIAN  rho = {rhoc_med:+.4f}   p = {prhoc_med:.2e}")
    print(f"  Spearman Dice vs MEAN    rho = {rhoc_mean:+.4f}   p = {prhoc_mean:.2e}")

    # =========================================================================
    # CLAIM 3 — Catastrophic-failure rate (mean rel-err > 1) per augmentation
    # =========================================================================
    hr(" 3. Catastrophic-failure rate: mean rel-err > 1, corrupted cells only")
    catastrophic = {}
    print(f"  (denominator = 5 corr * 2 base-domain * 2 train-domain = 20 per aug)")
    print(f"  {'Aug':<11s} {'cat':>3s} / {'tot':>3s}    {'rate':>6s}")
    for name, models in AUG_MODELS.items():
        sub = j3d_corr[j3d_corr["model"].isin(models)]
        cat = int((sub["feature_rel_error"] > 1).sum())
        tot = len(sub)
        pct = 100.0 * cat / tot if tot else 0
        catastrophic[name] = {"cat": cat, "tot": tot, "rate_pct": pct}
        print(f"  {name:<11s} {cat:>3d} / {tot:>3d}    {pct:>5.1f}%")

    # which corruption types contribute most to the catastrophic cells?
    cat_rows = j3d_corr[j3d_corr["feature_rel_error"] > 1]
    print("\n  Catastrophic cells broken down by corruption (3D only):")
    print(cat_rows.groupby(["corruption"]).size().to_string())

    # =========================================================================
    # CLAIM 4 — Augmentation summary table (Dice, mean, median, catastrophic)
    # =========================================================================
    hr(" 4. Combined table — same-domain vs OOD, corrupted cells only")
    print(f"  {'Aug':<11s}  {'scope':<5s} {'Dice':>7s} {'med':>7s} {'mean':>7s}   "
          f"{'cat':>3s}/{'tot':>3s}")
    table_rows = {}
    for name, models in AUG_MODELS.items():
        for scope, scope_pred in [("Same", lambda td, bd: td == bd),
                                  ("OOD",  lambda td, bd: td != bd)]:
            cells = []
            for m in models:
                td = "BTFE" if m.startswith("BTFE_") else "TSE"
                sub = j3d_corr[j3d_corr["model"] == m]
                if scope == "Same":
                    sub = sub[sub["base_domain"] == td]
                else:
                    sub = sub[sub["base_domain"] != td]
                cells.append(sub)
            s = pd.concat(cells, ignore_index=True)
            cat = int((s["feature_rel_error"] > 1).sum())
            tot = len(s)
            row = {
                "dice":     float(s["dice"].mean()),
                "med":      float(s["feature_rel_error_median"].mean()),
                "mean":     float(s["feature_rel_error"].mean()),
                "cat": cat, "tot": tot,
                "cat_pct":  100.0 * cat / tot if tot else 0,
            }
            table_rows[f"{name}_{scope}"] = row
            print(f"  {name:<11s}  {scope:<5s} {row['dice']:>7.3f} {row['med']:>7.3f} "
                  f"{row['mean']:>7.3f}   {cat:>3d}/{tot:>3d}")

    # =========================================================================
    # CLAIM 5 — MixUp+AFA improvement rate vs same-modality baseline
    # =========================================================================
    hr(" 5. MixUp+AFA improves median rel-err vs baseline")
    aug = ["BTFE_unetplusplus_mixup_afa", "TSE_unetplusplus_mixup_afa"]

    def pct_improve(aug_models, scenario_filter, df):
        win = tot = 0
        for am in aug_models:
            bl = BASELINE_OF[am]
            td = "BTFE" if am.startswith("BTFE_") else "TSE"
            for _, r in df[df["model"] == am].iterrows():
                if r["corruption"] == "clean":
                    continue
                if not scenario_filter(td, r["base_domain"]):
                    continue
                ref = df[(df["model"] == bl) &
                        (df["base_domain"] == r["base_domain"]) &
                        (df["corruption"] == r["corruption"]) &
                        (df["corruption_mode"] == r["corruption_mode"])]
                if len(ref) == 0:
                    continue
                if r["feature_rel_error_median"] < ref.iloc[0]["feature_rel_error_median"]:
                    win += 1
                tot += 1
        return win, tot, (100 * win / tot if tot else 0)

    win_in, tot_in, p_in = pct_improve(aug, lambda td, bd: td == bd, j3d_corr)
    win_x,  tot_x,  p_x  = pct_improve(aug, lambda td, bd: td != bd, j3d_corr)
    print(f"  in-domain   : {win_in}/{tot_in} = {p_in:.1f}%")
    print(f"  cross-domain: {win_x}/{tot_x}  = {p_x:.1f}%")

    # also report for every augmentation, just in case
    print("\n  Same-domain win-rate per augmentation (vs same-modality baseline):")
    for name, models in AUG_MODELS.items():
        if name == "Baseline":
            continue
        w, t, p = pct_improve(models, lambda td, bd: td == bd, j3d_corr)
        print(f"     {name:<11s} {w}/{t} = {p:.1f}%")
    print("  Cross-domain win-rate:")
    for name, models in AUG_MODELS.items():
        if name == "Baseline":
            continue
        w, t, p = pct_improve(models, lambda td, bd: td != bd, j3d_corr)
        print(f"     {name:<11s} {w}/{t} = {p:.1f}%")

    # =========================================================================
    # CLAIM 6 — per-corruption ranking
    # =========================================================================
    hr(" 6. Per-corruption ranking (3D only, averaged across 12 models * 2 base-doms)")
    agg = (j3d_corr.groupby("corruption")
           .agg(median_re=("feature_rel_error_median", "mean"),
                mean_re  =("feature_rel_error",        "mean"),
                dice     =("dice", "mean"),
                n        =("dice", "count"))
           .reset_index()
           .sort_values("median_re", ascending=False))
    print(agg.round(4).to_string(index=False))

    # =========================================================================
    # CLAIM 7 — BTFE spike-noise improvement spot-check
    # =========================================================================
    hr(" 7. Spot-check: BTFE spike-noise 3D — baseline vs MixUp+AFA")
    sub = j3d_corr[(j3d_corr["model"].isin(["BTFE_unetplusplus_1Fold",
                                            "BTFE_unetplusplus_mixup_afa"])) &
                   (j3d_corr["base_domain"] == "BTFE") &
                   (j3d_corr["corruption"] == "spike_noise")]
    if len(sub) == 2:
        bl_v  = sub[sub["model"].str.endswith("1Fold")]["feature_rel_error_median"].iloc[0]
        aug_v = sub[sub["model"].str.endswith("mixup_afa")]["feature_rel_error_median"].iloc[0]
        impr_med  = 100 * (bl_v - aug_v) / bl_v
        bl_d  = sub[sub["model"].str.endswith("1Fold")]["dice"].iloc[0]
        aug_d = sub[sub["model"].str.endswith("mixup_afa")]["dice"].iloc[0]
        print(f"  baseline median rel-err  = {bl_v:.4f}   dice = {bl_d:.4f}")
        print(f"  MixUp+AFA median rel-err = {aug_v:.4f}   dice = {aug_d:.4f}")
        print(f"  improvement (median)     = {impr_med:+.1f}%")

    # =========================================================================
    # CLAIM 8 — feature-family stability (3D corrupted only)
    # =========================================================================
    hr(" 8. Feature-family stability — uses summary_by_feature.csv as-is")
    print(f"  ({FEATURES_CSV.relative_to(ROOT)} is computed across all 274 cells")
    print(f"   in compare_radiomics.py; we report it for context only.)")
    if "family" in feats.columns:
        fam = feats.groupby("family")["median_rel_error"].median().sort_values(ascending=False)
        print(fam.round(4).to_string())

    # =========================================================================
    # 9. Per-patient four-quadrant table with IQR (paper Table I source)
    # =========================================================================
    hr(" 9. Four-quadrant table with IQR (per-patient aggregation)")

    # Initialised here so the JSON dump at the end of main() always has these
    # names defined, even when the per-patient inputs are missing and we skip
    # the whole computation.
    four_quadrant = None
    group_means = None

    if not PERPATIENT_DICE_CSV.is_file():
        print(f"  WARNING: missing {PERPATIENT_DICE_CSV.relative_to(ROOT)}")
        print(f"           run extract_per_patient_metrics.py first to generate it.")
    elif not PERPATIENT_ERRORS_CSV.is_file():
        print(f"  WARNING: missing {PERPATIENT_ERRORS_CSV.relative_to(ROOT)}")
    else:
        ppd = pd.read_csv(PERPATIENT_DICE_CSV)
        # per_patient_errors.csv is large (~530k rows × 13 cols); keep only what we need
        ppe_cols = ["model", "base_domain", "corruption", "corruption_mode",
                    "patient", "rel_error"]
        ppe = pd.read_csv(PERPATIENT_ERRORS_CSV, usecols=ppe_cols)

        # collapse the 102 features into per-patient median e_f and mean e_f
        ppe_agg = (ppe.groupby(["model", "base_domain", "corruption",
                                "corruption_mode", "patient"])["rel_error"]
                      .agg(median_ef="median", mean_ef="mean")
                      .reset_index())

        # join per-patient Dice with per-patient feature errors
        ppj = pd.merge(
            ppd[["model", "base_domain", "corruption", "corruption_mode",
                 "patient", "dice"]],
            ppe_agg,
            on=["model", "base_domain", "corruption", "corruption_mode",
                "patient"],
            how="inner",
        )

        # restrict to the 12 augmentation models, drop 2D-mode rows
        ppj = ppj[ppj["model"].isin(AUG_MODELS_FLAT)
                  & ppj["corruption_mode"].isin(["3d", "clean"])].copy()

        # tag each row with augmentation, training domain, scope, condition
        suffix_to_aug = {}
        for aug_name, models in AUG_MODELS.items():
            for m in models:
                suffix_to_aug[m] = aug_name
        ppj["aug"]        = ppj["model"].map(suffix_to_aug)
        ppj["train_dom"]  = np.where(ppj["model"].str.startswith("BTFE_"),
                                     "BTFE", "TSE")
        ppj["scope"]      = np.where(ppj["train_dom"] == ppj["base_domain"],
                                     "Same", "OOD")
        ppj["condition"]  = np.where(ppj["corruption_mode"] == "clean",
                                     "Clean", "Corrupted")

        print(f"  joined per-patient rows: {len(ppj)}")
        print(f"  bucket sizes (#patient values per (aug, scope, condition)):")
        bucket_sizes = (ppj.groupby(["aug", "scope", "condition"])
                          .size().unstack(fill_value=0))
        print(bucket_sizes.to_string())

        # cell-level catastrophic counts (kept from the cell-level analysis,
        # so the table stays consistent with previous audits). j3d already
        # contains the (model, base_domain, corruption, corruption_mode,
        # feature_rel_error) columns we need.
        cell_cat = (j3d.groupby(["model", "base_domain", "corruption",
                                 "corruption_mode"])["feature_rel_error"]
                          .first()
                          .reset_index())
        cell_cat["aug"]        = cell_cat["model"].map(suffix_to_aug)
        cell_cat["train_dom"]  = np.where(cell_cat["model"].str.startswith("BTFE_"),
                                          "BTFE", "TSE")
        cell_cat["scope"]      = np.where(cell_cat["train_dom"] == cell_cat["base_domain"],
                                          "Same", "OOD")
        cell_cat["condition"]  = np.where(cell_cat["corruption_mode"] == "clean",
                                          "Clean", "Corrupted")
        cell_cat["catastrophic"] = (cell_cat["feature_rel_error"] > 1).astype(int)
        cell_cat = cell_cat[cell_cat["corruption_mode"].isin(["3d", "clean"])]
        cell_cat = cell_cat[cell_cat["aug"].notna()]

        print(f"\n  Four-quadrant table (mean Dice [IQR]; median e_f [IQR]; mean e_f; cat/n_cells):")
        print(f"  {'aug':<11s} {'scope':<5s} {'cond':<10s} "
              f"{'Dice':>16s}  {'med_ef':>16s}  {'mean_ef':>9s}  {'catastr':>9s}")
        four_quadrant = {}
        for aug in AUG_MODELS:
            for scope in ("Same", "OOD"):
                for cond in ("Clean", "Corrupted"):
                    sub = ppj[(ppj["aug"] == aug)
                              & (ppj["scope"] == scope)
                              & (ppj["condition"] == cond)]
                    if len(sub) == 0:
                        continue
                    dice_q = sub["dice"].quantile([0.25, 0.50, 0.75]).tolist()
                    med_q  = sub["median_ef"].quantile([0.25, 0.50, 0.75]).tolist()
                    cell_sub = cell_cat[(cell_cat["aug"] == aug)
                                        & (cell_cat["scope"] == scope)
                                        & (cell_cat["condition"] == cond)]
                    n_cells     = len(cell_sub)
                    n_catastr   = int(cell_sub["catastrophic"].sum())
                    row = {
                        "n_patient_values":  int(len(sub)),
                        "dice_mean":         float(sub["dice"].mean()),
                        "dice_q1":           float(dice_q[0]),
                        "dice_median":       float(dice_q[1]),
                        "dice_q3":           float(dice_q[2]),
                        "median_ef_median":  float(med_q[1]),
                        "median_ef_q1":      float(med_q[0]),
                        "median_ef_q3":      float(med_q[2]),
                        "mean_ef_mean":      float(sub["mean_ef"].mean()),
                        "n_cells":           n_cells,
                        "catastrophic":      n_catastr,
                    }
                    four_quadrant[f"{aug}|{scope}|{cond}"] = row
                    print(f"  {aug:<11s} {scope:<5s} {cond:<10s} "
                          f"{row['dice_mean']:>5.3f} [{row['dice_q1']:.3f}, {row['dice_q3']:.3f}]  "
                          f"{row['median_ef_median']:>5.3f} [{row['median_ef_q1']:.3f}, {row['median_ef_q3']:.3f}]  "
                          f"{row['mean_ef_mean']:>9.3g}  "
                          f"{n_catastr:>3d}/{n_cells:<3d}")

        # group-mean footer rows (mean across the 6 augmentations in each
        # (scope, condition) group; the discussion section anchors its
        # OOD-vs-corruption magnitudes to these group means).
        print(f"\n  Group means (across 6 augmentations per row):")
        group_means = {}
        for scope in ("Same", "OOD"):
            for cond in ("Clean", "Corrupted"):
                rows = [four_quadrant[f"{a}|{scope}|{cond}"]
                        for a in AUG_MODELS if f"{a}|{scope}|{cond}" in four_quadrant]
                if not rows:
                    continue
                group_means[f"{scope}|{cond}"] = {
                    "dice_mean":        float(np.mean([r["dice_mean"]        for r in rows])),
                    "median_ef_median": float(np.mean([r["median_ef_median"] for r in rows])),
                    "mean_ef_mean":     float(np.mean([r["mean_ef_mean"]     for r in rows])),
                    "catastrophic":     int(sum(r["catastrophic"]            for r in rows)),
                    "n_cells_total":    int(sum(r["n_cells"]                 for r in rows)),
                }
                gm = group_means[f"{scope}|{cond}"]
                print(f"  {scope:<5s} {cond:<10s} (group mean)  "
                      f"Dice={gm['dice_mean']:.3f}  "
                      f"med_ef={gm['median_ef_median']:.3f}  "
                      f"mean_ef={gm['mean_ef_mean']:.3g}  "
                      f"catastr={gm['catastrophic']}/{gm['n_cells_total']}")

    # =========================================================================
    # Dump everything to JSON for the paper-audit map
    # =========================================================================
    out = {
        "scope": {
            "n_aug_models": len(AUG_MODELS_FLAT),
            "n_total_cells_3d": len(j3d),
            "n_corrupted_cells_3d": int(n_corrupt),
            "n_clean_cells_3d": int(n_clean),
            "filter": "model in 12 aug models AND corruption_mode in {3d, clean}",
        },
        "correlations_all_3d_plus_clean": {
            "pearson_median":  {"r": float(r_med),  "p": float(p_med)},
            "pearson_mean":    {"r": float(r_mean), "p": float(p_mean)},
            "spearman_median": {"rho": float(rho_med),  "p": float(prho_med)},
            "spearman_mean":   {"rho": float(rho_mean), "p": float(prho_mean)},
        },
        "correlations_corrupted_only": {
            "pearson_median":  {"r": float(rc_med),  "p": float(pc_med)},
            "pearson_mean":    {"r": float(rc_mean), "p": float(pc_mean)},
            "spearman_median": {"rho": float(rhoc_med),  "p": float(prhoc_med)},
            "spearman_mean":   {"rho": float(rhoc_mean), "p": float(prhoc_mean)},
        },
        "catastrophic": catastrophic,
        "table_rows": table_rows,
        "mixup_afa_winrate": {
            "in_domain":   {"win": win_in, "tot": tot_in, "pct": p_in},
            "cross_domain":{"win": win_x,  "tot": tot_x,  "pct": p_x},
        },
        "per_corruption": agg.round(6).to_dict(orient="records"),
        "four_quadrant_table_with_iqr": four_quadrant,
        "four_quadrant_group_means": group_means,
    }
    out_path = ROOT / "paper" / "audit_3d_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
