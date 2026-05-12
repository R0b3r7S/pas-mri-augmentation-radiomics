#!/usr/bin/env python3
"""
==============================================================================
 Radiomics Feature Error Dashboard
==============================================================================
 Loads the per-patient JSONs produced by compute_radiomics.py and measures
 how much each segmentation model distorts radiomics features compared to
 ground-truth masks.

 Inputs (under radiomics/features/):
   gt/<mode>/<DOMAIN>/<sub>.json
   pred/<MODEL>/test_on_<DOMAIN>/<mode>/<sub>.json

 Outputs (under radiomics/comparisons/<mode>/):
   per_patient_errors.csv        — every (model, domain, patient, feature)
   summary_by_model.csv          — mean relative error per (model, domain, feature)
   summary_by_feature.csv        — ranks features by mean error across models
   augmentation_impact.csv       — baseline vs augmented on the same domain
   heatmap_rel_error.png         — models × top-K features colored by rel-error
   top_worst_features.csv        — top-K features with highest mean error

 Error metrics per feature:
   abs_err = |pred − gt|
   rel_err = |pred − gt| / (|gt| + eps)
==============================================================================
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_ROOT = PROJECT_ROOT / "radiomics" / "features"
OUT_ROOT = PROJECT_ROOT / "radiomics" / "comparisons"

EPS = 1e-8

# Copied from compare_test_metrics.py so the dashboards stay visually consistent.
DISPLAY_NAMES = {
    "BTFE_unetplusplus_1Fold":        "BTFE Baseline",
    "BTFE_unetplusplus_mixup":        "BTFE + MixUp",
    "BTFE_unetplusplus_cutmix":       "BTFE + CutMix",
    "BTFE_unetplusplus_afa":          "BTFE + AFA",
    "BTFE_unetplusplus_mixup_afa":    "BTFE + MixUp+AFA",
    "BTFE_unetplusplus_cutmix_afa":   "BTFE + CutMix+AFA",
    "TSE_unetplusplus_1Fold":         "TSE Baseline",
    "TSE_unetplusplus_mixup":         "TSE + MixUp",
    "TSE_unetplusplus_cutmix":        "TSE + CutMix",
    "TSE_unetplusplus_afa":           "TSE + AFA",
    "TSE_unetplusplus_mixup_afa":     "TSE + MixUp+AFA",
    "TSE_unetplusplus_cutmix_afa":    "TSE + CutMix+AFA",
    "COMBINED_unetplusplus_cutmix_afa": "COMBINED + CutMix+AFA",
    "COMBINED_unetplusplus_mixup_afa":  "COMBINED + MixUp+AFA",
    "COMBINED_unetplusplus_1Fold":    "Combined Baseline",
    "Transfer_BTFE_to_TSE_1Fold":     "Transfer BTFE→TSE",
    "Transfer_TSE_to_BTFE_1Fold":     "Transfer TSE→BTFE",
}

# Baseline lookup (for augmentation_impact.csv)
BASELINE_OF = {
    "BTFE_unetplusplus_mixup":       "BTFE_unetplusplus_1Fold",
    "BTFE_unetplusplus_cutmix":      "BTFE_unetplusplus_1Fold",
    "BTFE_unetplusplus_afa":         "BTFE_unetplusplus_1Fold",
    "BTFE_unetplusplus_mixup_afa":   "BTFE_unetplusplus_1Fold",
    "BTFE_unetplusplus_cutmix_afa":  "BTFE_unetplusplus_1Fold",
    "TSE_unetplusplus_mixup":        "TSE_unetplusplus_1Fold",
    "TSE_unetplusplus_cutmix":       "TSE_unetplusplus_1Fold",
    "TSE_unetplusplus_afa":          "TSE_unetplusplus_1Fold",
    "TSE_unetplusplus_mixup_afa":    "TSE_unetplusplus_1Fold",
    "TSE_unetplusplus_cutmix_afa":   "TSE_unetplusplus_1Fold",
    "COMBINED_unetplusplus_mixup_afa":  "COMBINED_unetplusplus_1Fold",
    "COMBINED_unetplusplus_cutmix_afa": "COMBINED_unetplusplus_1Fold",
}


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------
def load_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


def features_of(payload: dict) -> dict:
    """Return the canonical feature dict regardless of mode."""
    if "features" in payload:
        return payload["features"]
    if "features_mean" in payload:
        return payload["features_mean"]
    return {}


def load_gt(mode: str):
    """Return dict[(domain, patient)] -> feature dict."""
    gt_root = FEATURES_ROOT / "gt" / mode
    out = {}
    if not gt_root.is_dir():
        return out
    for domain_dir in sorted(gt_root.iterdir()):
        if not domain_dir.is_dir():
            continue
        for jp in sorted(domain_dir.glob("*.json")):
            payload = load_json(jp)
            patient = payload.get("patient") or jp.stem
            feats = features_of(payload)
            if feats:
                out[(domain_dir.name, patient)] = feats
    return out


def load_pred(mode: str):
    """Return dict[(model, test_domain)] -> dict[patient] -> feature dict."""
    pred_root = FEATURES_ROOT / "pred"
    out = defaultdict(dict)
    if not pred_root.is_dir():
        return out
    for model_dir in sorted(pred_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for test_dir in sorted(model_dir.iterdir()):
            if not test_dir.name.startswith("test_on_"):
                continue
            mode_dir = test_dir / mode
            if not mode_dir.is_dir():
                continue
            domain_key = test_dir.name[len("test_on_"):]
            for jp in sorted(mode_dir.glob("*.json")):
                payload = load_json(jp)
                patient = payload.get("patient") or jp.stem
                feats = features_of(payload)
                if feats:
                    out[(model_dir.name, domain_key)][patient] = feats
    return out


# ---------------------------------------------------------------------------
# Domain key mapping between runs/ naming and dataset naming
# ---------------------------------------------------------------------------
# runs use "test_on_BTFE" / "test_on_TSE"; GT cache uses "BTFE" / "SSH_TSE"
DOMAIN_ALIAS = {
    "BTFE": "BTFE",
    "TSE": "SSH_TSE",
    "SSH_TSE": "SSH_TSE",
}

# Corrupted variants: "BTFE_bias_field_s3_2d" → base "BTFE"
_VARIANT_RE = re.compile(
    r"^(?P<base>BTFE|TSE|SSH_TSE|COMBINED)"
    r"(?:_(?P<corr>[a-z_]+)_s(?P<sev>\d+)_(?P<cmode>2d|3d))?$"
)


def parse_variant(variant: str):
    m = _VARIANT_RE.match(variant)
    if not m:
        return {"base": variant, "corruption": None, "severity": None, "cmode": None}
    return {
        "base": m.group("base"),
        "corruption": m.group("corr"),
        "severity": int(m.group("sev")) if m.group("sev") else None,
        "cmode": m.group("cmode"),
    }


# ---------------------------------------------------------------------------
# Error table construction
# ---------------------------------------------------------------------------
def build_error_rows(gt_map, pred_map):
    """Yield per-patient/feature error rows."""
    for (model, test_domain), patient_feats in pred_map.items():
        parsed = parse_variant(test_domain)
        gt_domain = DOMAIN_ALIAS.get(parsed["base"], parsed["base"])
        for patient, pred_feats in patient_feats.items():
            gt_feats = gt_map.get((gt_domain, patient))
            if not gt_feats:
                continue
            common = set(pred_feats).intersection(gt_feats)
            for feature in common:
                gt_v = gt_feats[feature]
                pr_v = pred_feats[feature]
                if gt_v is None or pr_v is None:
                    continue
                try:
                    gt_f = float(gt_v)
                    pr_f = float(pr_v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(gt_f) or math.isnan(pr_f):
                    continue
                abs_err = abs(pr_f - gt_f)
                rel_err = abs_err / (abs(gt_f) + EPS)
                yield {
                    "model": model,
                    "model_display": DISPLAY_NAMES.get(model, model),
                    "test_domain": test_domain,
                    "base_domain": parsed["base"],
                    "corruption": parsed["corruption"] or "clean",
                    "severity": parsed["severity"],
                    "corruption_mode": parsed["cmode"] or "clean",
                    "patient": patient,
                    "feature": feature,
                    "gt_value": gt_f,
                    "pred_value": pr_f,
                    "abs_error": abs_err,
                    "rel_error": rel_err,
                }


# ---------------------------------------------------------------------------
# Summaries and plots
# ---------------------------------------------------------------------------
def summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["model", "model_display", "test_domain", "feature"])
             .agg(rel_mean=("rel_error", "mean"),
                  rel_median=("rel_error", "median"),
                  rel_std=("rel_error", "std"),
                  abs_mean=("abs_error", "mean"),
                  abs_median=("abs_error", "median"),
                  abs_std=("abs_error", "std"),
                  n=("rel_error", "size"))
             .reset_index())
    return agg


def summary_by_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Rank features by mean rel error averaged across models+domains."""
    agg = (df.groupby("feature")
             .agg(rel_mean=("rel_error", "mean"),
                  rel_median=("rel_error", "median"),
                  rel_std=("rel_error", "std"),
                  rel_max=("rel_error", "max"),
                  n=("rel_error", "size"))
             .reset_index()
             .sort_values("rel_mean", ascending=False))
    return agg


def augmentation_impact(summary_model: pd.DataFrame) -> pd.DataFrame:
    """For each (augmented-model, test_domain, feature), compute the change
    in *both* mean and median relative error vs. its baseline (same train
    domain, no augmentation). Negative delta = augmentation reduced the
    feature-level error.

    Median is the more robust readout when a few features have GT values
    near zero (firstorder_Minimum, ngtdm_Coarseness, etc.) and inflate the
    mean.
    """
    indexed = summary_model.set_index(["model", "test_domain", "feature"])
    mean_s = indexed["rel_mean"]
    median_s = indexed["rel_median"]

    rows = []
    for (model, domain, feature), rel_mean in mean_s.items():
        base = BASELINE_OF.get(model)
        if base is None:
            continue
        key = (base, domain, feature)
        if key not in mean_s.index:
            continue
        base_mean = mean_s.loc[key]
        rel_median = median_s.loc[(model, domain, feature)]
        base_median = median_s.loc[key]
        rows.append({
            "model": model,
            "model_display": DISPLAY_NAMES.get(model, model),
            "baseline": base,
            "baseline_display": DISPLAY_NAMES.get(base, base),
            "test_domain": domain,
            "feature": feature,
            # Mean-based (sensitive to outlier features)
            "rel_error_model": rel_mean,
            "rel_error_baseline": base_mean,
            "delta_rel_error": rel_mean - base_mean,
            "pct_improvement": (base_mean - rel_mean) / (base_mean + EPS) * 100.0,
            # Median-based (robust to near-zero-GT features)
            "rel_error_model_median": rel_median,
            "rel_error_baseline_median": base_median,
            "delta_rel_error_median": rel_median - base_median,
            "pct_improvement_median": (base_median - rel_median) / (base_median + EPS) * 100.0,
        })
    return pd.DataFrame(rows)


def heatmap_top_features(df: pd.DataFrame, out_path: Path,
                         top_k: int = 25, title: str = "Radiomics rel-error",
                         agg: str = "mean") -> None:
    """Heatmap: rows = top-K features by ``agg`` rel-error, cols = model×domain.

    ``agg`` is either ``"mean"`` or ``"median"`` and selects both the
    feature ranking and the cell aggregation.
    """
    if df.empty:
        return
    if agg not in ("mean", "median"):
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")

    top_feats = (df.groupby("feature")["rel_error"].agg(agg)
                   .nlargest(top_k).index.tolist())
    sub = df[df["feature"].isin(top_feats)].copy()
    sub["model_domain"] = sub["model_display"] + " | " + sub["test_domain"]
    pivot = (sub.groupby(["feature", "model_domain"])["rel_error"]
                 .agg(agg).unstack("model_domain"))
    pivot = pivot.loc[top_feats]

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * pivot.shape[1] + 4),
                                    max(6, 0.35 * pivot.shape[0] + 2)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f.replace("original_", "") for f in pivot.index], fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{agg.capitalize()} relative error |pred − gt| / |gt|")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def dice_like_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean relative error across ALL features per (model, domain) — a single
    scalar per cell so it sits next to the existing Dice comparison table."""
    agg = (df.groupby(["model", "model_display", "test_domain"])
             .agg(feature_rel_error=("rel_error", "mean"),
                  feature_rel_error_median=("rel_error", "median"),
                  n=("rel_error", "size"))
             .reset_index()
             .sort_values(["model_display", "test_domain"]))
    return agg


def summary_by_corruption(df: pd.DataFrame) -> pd.DataFrame:
    """Mean rel-error per (model × base_domain × corruption × corruption_mode),
    averaged across all features and patients. One row per cell."""
    agg = (df.groupby(["model", "model_display", "base_domain",
                       "corruption", "corruption_mode"])
             .agg(feature_rel_error=("rel_error", "mean"),
                  feature_rel_error_median=("rel_error", "median"),
                  feature_rel_error_std=("rel_error", "std"),
                  n=("rel_error", "size"))
             .reset_index()
             .sort_values(["base_domain", "corruption",
                           "corruption_mode", "model_display"]))
    return agg


def corruption_robustness(corr_summary: pd.DataFrame) -> pd.DataFrame:
    """For every (model × base_domain × corruption × cmode), compare its
    feature rel-error (both mean and median) against the same model's CLEAN
    rel-error. Positive ``degradation`` = the corruption made feature
    extraction worse for that model. Useful to see how much feature
    fidelity collapses under Figure-1 corruptions for each augmentation
    variant.
    """
    clean_rows = corr_summary[corr_summary["corruption"] == "clean"]
    clean_mean = (clean_rows.set_index(["model", "base_domain"])
                  ["feature_rel_error"].to_dict())
    clean_median = (clean_rows.set_index(["model", "base_domain"])
                    ["feature_rel_error_median"].to_dict())
    rows = []
    for _, r in corr_summary.iterrows():
        if r["corruption"] == "clean":
            continue
        key = (r["model"], r["base_domain"])
        cm = clean_mean.get(key)
        cmed = clean_median.get(key)
        if cm is None:
            continue
        rows.append({
            "model": r["model"],
            "model_display": r["model_display"],
            "base_domain": r["base_domain"],
            "corruption": r["corruption"],
            "corruption_mode": r["corruption_mode"],
            # Mean-based (sensitive to outliers)
            "rel_error_corrupted": r["feature_rel_error"],
            "rel_error_clean": cm,
            "degradation": r["feature_rel_error"] - cm,
            "degradation_pct": (r["feature_rel_error"] - cm) / (cm + EPS) * 100.0,
            # Median-based (robust)
            "rel_error_corrupted_median": r["feature_rel_error_median"],
            "rel_error_clean_median": cmed,
            "degradation_median": r["feature_rel_error_median"] - (cmed or 0.0),
            "degradation_pct_median": ((r["feature_rel_error_median"] - (cmed or 0.0))
                                        / ((cmed or 0.0) + EPS) * 100.0),
        })
    return pd.DataFrame(rows)


def augmentation_vs_corruption(corr_summary: pd.DataFrame) -> pd.DataFrame:
    """For each (augmented model × corruption × cmode), compare its feature
    rel-error to the baseline model's under the *same* corruption. Answers:
    does this augmentation reduce feature-level damage caused by this
    specific corruption?

    Both mean and median variants are emitted; median is more robust when
    a few features have GT values near zero.
    """
    indexed = corr_summary.set_index(
        ["model", "base_domain", "corruption", "corruption_mode"]
    )
    mean_s = indexed["feature_rel_error"]
    median_s = indexed["feature_rel_error_median"]

    rows = []
    for (model, base, corr, cmode), val in mean_s.items():
        bl = BASELINE_OF.get(model)
        if bl is None or corr == "clean":
            continue
        bl_key = (bl, base, corr, cmode)
        if bl_key not in mean_s.index:
            continue
        bl_val = mean_s.loc[bl_key]
        val_med = median_s.loc[(model, base, corr, cmode)]
        bl_val_med = median_s.loc[bl_key]
        rows.append({
            "model": model,
            "model_display": DISPLAY_NAMES.get(model, model),
            "baseline": bl,
            "baseline_display": DISPLAY_NAMES.get(bl, bl),
            "base_domain": base,
            "corruption": corr,
            "corruption_mode": cmode,
            # Mean-based
            "rel_error_model": val,
            "rel_error_baseline": bl_val,
            "delta_rel_error": val - bl_val,
            "pct_improvement": (bl_val - val) / (bl_val + EPS) * 100.0,
            # Median-based
            "rel_error_model_median": val_med,
            "rel_error_baseline_median": bl_val_med,
            "delta_rel_error_median": val_med - bl_val_med,
            "pct_improvement_median": (bl_val_med - val_med) / (bl_val_med + EPS) * 100.0,
        })
    return (pd.DataFrame(rows)
              .sort_values(["base_domain", "corruption", "corruption_mode", "model"])
              if rows else pd.DataFrame())


def heatmap_corruption(corr_summary: pd.DataFrame, out_path: Path,
                       title: str = "Feature rel-error per corruption",
                       agg: str = "mean") -> None:
    """Heatmap: rows = model, cols = corruption × cmode × base_domain.

    ``agg`` is ``"mean"`` or ``"median"``; selects the cell-aggregation
    column from ``corr_summary``.
    """
    if agg not in ("mean", "median"):
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")
    value_col = "feature_rel_error" if agg == "mean" else "feature_rel_error_median"

    sub = corr_summary[corr_summary["corruption"] != "clean"]
    if sub.empty:
        return
    sub = sub.copy()
    sub["col"] = (sub["base_domain"] + " | " + sub["corruption"]
                  + " | " + sub["corruption_mode"])
    pivot = sub.pivot_table(index="model_display", columns="col",
                            values=value_col, aggfunc="mean")
    pivot = pivot.sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * pivot.shape[1] + 4),
                                    max(4, 0.35 * pivot.shape[0] + 2)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{agg.capitalize()} relative error |pred − gt| / |gt|")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_for_mode(mode: str, top_k: int):
    out_dir = OUT_ROOT / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{mode}] loading features ...")
    gt_map = load_gt(mode)
    pred_map = load_pred(mode)
    print(f"[{mode}] GT patients   : {len(gt_map)}")
    print(f"[{mode}] pred groups   : {len(pred_map)} "
          f"(sum patients: {sum(len(v) for v in pred_map.values())})")

    if not gt_map or not pred_map:
        print(f"[{mode}] nothing to compare; skipping.")
        return

    rows = list(build_error_rows(gt_map, pred_map))
    if not rows:
        print(f"[{mode}] no overlap between GT and pred features; skipping.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_patient_errors.csv", index=False)
    print(f"[{mode}] wrote per_patient_errors.csv ({len(df)} rows)")

    summ_model = summary_by_model(df)
    summ_model.to_csv(out_dir / "summary_by_model.csv", index=False)

    summ_feat = summary_by_feature(df)
    summ_feat.to_csv(out_dir / "summary_by_feature.csv", index=False)
    summ_feat.head(top_k).to_csv(out_dir / "top_worst_features.csv", index=False)

    aug = augmentation_impact(summ_model)
    if not aug.empty:
        aug.sort_values(["test_domain", "feature", "model"]).to_csv(
            out_dir / "augmentation_impact.csv", index=False)

    overall = dice_like_summary(df)
    overall.to_csv(out_dir / "overall_mean_rel_error.csv", index=False)

    # --- corruption analyses (only populate if corrupted variants are present) ---
    corr_summary = summary_by_corruption(df)
    corr_summary.to_csv(out_dir / "per_corruption_summary.csv", index=False)
    have_corruptions = (corr_summary["corruption"] != "clean").any()
    if have_corruptions:
        robust = corruption_robustness(corr_summary)
        if not robust.empty:
            robust.to_csv(out_dir / "corruption_robustness.csv", index=False)
        av_corr = augmentation_vs_corruption(corr_summary)
        if not av_corr.empty:
            av_corr.to_csv(out_dir / "augmentation_vs_corruption.csv", index=False)
        heatmap_corruption(
            corr_summary, out_dir / "heatmap_corruption.png",
            title=f"Feature rel-error by corruption — mean ({mode})",
            agg="mean",
        )
        heatmap_corruption(
            corr_summary, out_dir / "heatmap_corruption_median.png",
            title=f"Feature rel-error by corruption — median ({mode})",
            agg="median",
        )

    heatmap_top_features(
        df, out_dir / "heatmap_rel_error.png",
        top_k=top_k,
        title=f"Top-{top_k} features by mean rel-error ({mode})",
        agg="mean",
    )
    heatmap_top_features(
        df, out_dir / "heatmap_rel_error_median.png",
        top_k=top_k,
        title=f"Top-{top_k} features by median rel-error ({mode})",
        agg="median",
    )

    # Compact console report — both mean and median pivots
    print(f"\n[{mode}] ========== Mean feature rel-error per model/domain ==========")
    pivot_mean = overall.pivot(index="model_display", columns="test_domain",
                               values="feature_rel_error")
    print(pivot_mean.round(4).to_string())

    print(f"\n[{mode}] ========== Median feature rel-error per model/domain "
          f"(robust readout) ==========")
    pivot_med = overall.pivot(index="model_display", columns="test_domain",
                              values="feature_rel_error_median")
    print(pivot_med.round(4).to_string())

    if have_corruptions:
        print(f"\n[{mode}] ========== Mean feature rel-error per corruption "
              f"(all models) ==========")
        cpivot_mean = (corr_summary[corr_summary["corruption"] != "clean"]
                       .pivot_table(index="model_display",
                                    columns=["corruption", "corruption_mode"],
                                    values="feature_rel_error", aggfunc="mean"))
        print(cpivot_mean.round(4).to_string())

        print(f"\n[{mode}] ========== Median feature rel-error per corruption "
              f"(all models) ==========")
        cpivot_med = (corr_summary[corr_summary["corruption"] != "clean"]
                      .pivot_table(index="model_display",
                                   columns=["corruption", "corruption_mode"],
                                   values="feature_rel_error_median", aggfunc="mean"))
        print(cpivot_med.round(4).to_string())

    print(f"\n[{mode}] top {min(top_k, len(summ_feat))} worst features "
          f"(ranked by mean rel-error; median shown for context):")
    for _, r in summ_feat.head(top_k).iterrows():
        print(f"  mean={r['rel_mean']:10.4f}   median={r['rel_median']:8.4f}   {r['feature']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["2d", "3d", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=25,
                        help="Number of worst features to highlight in heatmap/report.")
    args = parser.parse_args()

    modes = ("2d", "3d") if args.mode == "both" else (args.mode,)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        if not (FEATURES_ROOT / "gt" / mode).is_dir():
            print(f"[{mode}] no features cache found at "
                  f"{FEATURES_ROOT / 'gt' / mode}; run compute_radiomics.py first.")
            continue
        run_for_mode(mode, args.top_k)


if __name__ == "__main__":
    main()
