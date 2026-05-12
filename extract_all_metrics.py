#!/usr/bin/env python3
"""
==============================================================================
 Consolidate test_metrics.json from every (model × test_variant) cell into
 a single set of CSVs that's easy to share and analyse.
==============================================================================
 Walks runs/<MODEL>/test_on_<VARIANT>/fold_*/test_metrics.json and writes:

   shareable/all_test_metrics.csv                — flat: one row per cell
                                                   × every aggregate metric
                                                   (dice, iou, sens, prec,
                                                    spec, hd95, msd) + parsed
                                                   variant fields
   shareable/pivot_<metric>.csv                  — one wide table per metric
                                                   (rows = model, cols =
                                                   test_variant)
   shareable/dice_augmentation_vs_corruption.csv — for each augmented model
                                                   vs. its same-train-domain
                                                   baseline, Dice delta and
                                                   pct improvement under the
                                                   *same* corruption variant
   shareable/dice_corruption_robustness.csv      — per-model Dice change vs.
                                                   that model's clean test
                                                   (positive degradation =
                                                   the corruption hurt that
                                                   model)

 Usage:
   python extract_all_metrics.py
   python extract_all_metrics.py --runs-filter BTFE_unetplusplus_mixup_afa
==============================================================================
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs"
OUT_ROOT = PROJECT_ROOT / "shareable"

VARIANT_RE = re.compile(
    r"^(?P<base>BTFE|TSE|SSH_TSE|COMBINED)"
    r"(?:_(?P<corr>[a-z_]+)_s(?P<sev>\d+)_(?P<cmode>2d|3d))?$"
)

# Mirror compare_radiomics.py for naming consistency in shared outputs.
DISPLAY_NAMES = {
    "BTFE_unetplusplus_1Fold":         "BTFE Baseline",
    "BTFE_unetplusplus_mixup":         "BTFE + MixUp",
    "BTFE_unetplusplus_cutmix":        "BTFE + CutMix",
    "BTFE_unetplusplus_afa":           "BTFE + AFA",
    "BTFE_unetplusplus_mixup_afa":     "BTFE + MixUp+AFA",
    "BTFE_unetplusplus_cutmix_afa":    "BTFE + CutMix+AFA",
    "TSE_unetplusplus_1Fold":          "TSE Baseline",
    "TSE_unetplusplus_mixup":          "TSE + MixUp",
    "TSE_unetplusplus_cutmix":         "TSE + CutMix",
    "TSE_unetplusplus_afa":            "TSE + AFA",
    "TSE_unetplusplus_mixup_afa":      "TSE + MixUp+AFA",
    "TSE_unetplusplus_cutmix_afa":     "TSE + CutMix+AFA",
    "COMBINED_unetplusplus_cutmix_afa": "COMBINED + CutMix+AFA",
    "COMBINED_unetplusplus_mixup_afa":  "COMBINED + MixUp+AFA",
    "COMBINED_unetplusplus_1Fold":     "Combined Baseline",
    "Transfer_BTFE_to_TSE_1Fold":      "Transfer BTFE→TSE",
    "Transfer_TSE_to_BTFE_1Fold":      "Transfer TSE→BTFE",
}

# Mapping each augmented model to its same-train-domain baseline (no augmentation).
BASELINE_OF = {
    "BTFE_unetplusplus_mixup":         "BTFE_unetplusplus_1Fold",
    "BTFE_unetplusplus_cutmix":        "BTFE_unetplusplus_1Fold",
    "BTFE_unetplusplus_afa":           "BTFE_unetplusplus_1Fold",
    "BTFE_unetplusplus_mixup_afa":     "BTFE_unetplusplus_1Fold",
    "BTFE_unetplusplus_cutmix_afa":    "BTFE_unetplusplus_1Fold",
    "TSE_unetplusplus_mixup":          "TSE_unetplusplus_1Fold",
    "TSE_unetplusplus_cutmix":         "TSE_unetplusplus_1Fold",
    "TSE_unetplusplus_afa":            "TSE_unetplusplus_1Fold",
    "TSE_unetplusplus_mixup_afa":      "TSE_unetplusplus_1Fold",
    "TSE_unetplusplus_cutmix_afa":     "TSE_unetplusplus_1Fold",
    "COMBINED_unetplusplus_mixup_afa":  "COMBINED_unetplusplus_1Fold",
    "COMBINED_unetplusplus_cutmix_afa": "COMBINED_unetplusplus_1Fold",
}

METRICS = ["dice", "iou", "sens", "prec", "spec", "hd95", "msd"]
EPS = 1e-8


def parse_variant(variant: str) -> dict:
    m = VARIANT_RE.match(variant)
    if not m:
        return {"base": variant, "corruption": "clean",
                "severity": None, "cmode": "clean"}
    return {
        "base": m.group("base"),
        "corruption": m.group("corr") or "clean",
        "severity": int(m.group("sev")) if m.group("sev") else None,
        "cmode": m.group("cmode") or "clean",
    }


def parse_train_domain(model_name: str) -> str:
    if model_name.startswith("Transfer_"):
        return "Transfer"
    for d in ("BTFE", "TSE", "COMBINED"):
        if model_name.startswith(f"{d}_"):
            return d
    return ""


def discover_metrics(runs_filter, variants_filter):
    rows = []
    if not RUNS_ROOT.is_dir():
        print(f"[error] runs root not found: {RUNS_ROOT}", file=sys.stderr)
        return rows
    for model_dir in sorted(RUNS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        if runs_filter and not any(s in model_dir.name for s in runs_filter):
            continue
        for test_dir in sorted(model_dir.iterdir()):
            if not test_dir.is_dir() or not test_dir.name.startswith("test_on_"):
                continue
            variant = test_dir.name[len("test_on_"):]
            if variants_filter and not any(s in variant for s in variants_filter):
                continue
            for fold_dir in sorted(test_dir.iterdir()):
                if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
                    continue
                mp = fold_dir / "test_metrics.json"
                if not mp.is_file():
                    continue
                try:
                    payload = json.loads(mp.read_text())
                except Exception as e:
                    print(f"  [warn] couldn't read {mp}: {e}", file=sys.stderr)
                    continue
                parsed = parse_variant(variant)
                td = parse_train_domain(model_dir.name)
                row = {
                    "model": model_dir.name,
                    "model_display": DISPLAY_NAMES.get(model_dir.name, model_dir.name),
                    "train_domain": td,
                    "test_variant": variant,
                    "base_domain": parsed["base"],
                    "corruption": parsed["corruption"],
                    "severity": parsed["severity"],
                    "corruption_mode": parsed["cmode"],
                    "same_domain": (td == parsed["base"]) or td in ("COMBINED", "Transfer"),
                    "fold": fold_dir.name,
                }
                # SSH_TSE in dataset = "TSE" suffix in run-folder names; fold them
                # together for the "same-domain" flag.
                if td == "TSE" and parsed["base"] == "SSH_TSE":
                    row["same_domain"] = True
                if td == "BTFE" and parsed["base"] == "BTFE":
                    row["same_domain"] = True
                for m in METRICS:
                    val = payload.get(m)
                    row[m] = float(val) if val is not None else None
                rows.append(row)
    return rows


def write_pivots(df: pd.DataFrame, out_root: Path) -> None:
    """One wide CSV per metric: rows = model_display, cols = test_variant."""
    for m in METRICS:
        pivot = df.pivot_table(index="model_display",
                               columns="test_variant",
                               values=m, aggfunc="mean")
        # Stable column order: clean variants first, then alphabetical
        clean_first = [c for c in pivot.columns if "_s3_" not in str(c)]
        corrupted = sorted(c for c in pivot.columns if "_s3_" in str(c))
        pivot = pivot.reindex(columns=clean_first + corrupted)
        path = out_root / f"pivot_{m}.csv"
        pivot.to_csv(path)
        print(f"  wrote {path.relative_to(PROJECT_ROOT)}  "
              f"({pivot.shape[0]}×{pivot.shape[1]})")


def dice_augmentation_vs_corruption(df: pd.DataFrame) -> pd.DataFrame:
    """For each (augmented model × variant), compare Dice to the same-train
    baseline's Dice on the *same* variant. Mirror of the radiomics version.
    """
    if df.empty:
        return pd.DataFrame()
    keyed = df.set_index(["model", "test_variant"])["dice"]
    rows = []
    for (model, variant), dice in keyed.items():
        bl = BASELINE_OF.get(model)
        if bl is None:
            continue
        bl_key = (bl, variant)
        if bl_key not in keyed.index:
            continue
        bl_dice = keyed.loc[bl_key]
        if dice is None or bl_dice is None:
            continue
        parsed = parse_variant(variant)
        rows.append({
            "model": model,
            "model_display": DISPLAY_NAMES.get(model, model),
            "baseline": bl,
            "baseline_display": DISPLAY_NAMES.get(bl, bl),
            "test_variant": variant,
            "base_domain": parsed["base"],
            "corruption": parsed["corruption"],
            "corruption_mode": parsed["cmode"],
            "dice_model": dice,
            "dice_baseline": bl_dice,
            "delta_dice": dice - bl_dice,           # positive = augmentation HELPED
            "pct_improvement": (dice - bl_dice) / (bl_dice + EPS) * 100.0,
        })
    return (pd.DataFrame(rows)
            .sort_values(["base_domain", "corruption", "corruption_mode", "model"]))


def dice_corruption_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model × test_variant) where test_variant is corrupted: how much
    Dice dropped vs. that *same* model's Dice on its clean (same base domain)
    test set. Positive degradation = the corruption hurt that model.
    """
    if df.empty:
        return pd.DataFrame()
    clean = (df[df["corruption"] == "clean"]
             .set_index(["model", "base_domain"])["dice"]
             .to_dict())
    rows = []
    for _, r in df.iterrows():
        if r["corruption"] == "clean":
            continue
        cdice = clean.get((r["model"], r["base_domain"]))
        if cdice is None or r["dice"] is None:
            continue
        rows.append({
            "model": r["model"],
            "model_display": r["model_display"],
            "base_domain": r["base_domain"],
            "corruption": r["corruption"],
            "corruption_mode": r["corruption_mode"],
            "dice_corrupted": r["dice"],
            "dice_clean": cdice,
            "degradation": cdice - r["dice"],          # positive = corruption hurt
            "degradation_pct": (cdice - r["dice"]) / (cdice + EPS) * 100.0,
        })
    return (pd.DataFrame(rows)
            .sort_values(["base_domain", "corruption", "corruption_mode", "model"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-filter", nargs="*", default=None,
                    help="Only include model dirs whose name contains any of these substrings.")
    ap.add_argument("--variants-filter", nargs="*", default=None,
                    help="Only include test_on_<VARIANT> names containing any of these substrings.")
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = discover_metrics(args.runs_filter, args.variants_filter)
    if not rows:
        print("no test_metrics.json files matched")
        return
    df = pd.DataFrame(rows)

    # Stable row order: train_domain group → augmentation → variant
    df = df.sort_values(["train_domain", "model_display", "test_variant"]).reset_index(drop=True)

    flat_path = out_root / "all_test_metrics.csv"
    df.to_csv(flat_path, index=False)
    print(f"wrote {flat_path.relative_to(PROJECT_ROOT)}  ({len(df)} rows)")

    print("\nper-metric pivots:")
    write_pivots(df, out_root)

    # The two comparison tables
    av = dice_augmentation_vs_corruption(df)
    if not av.empty:
        path = out_root / "dice_augmentation_vs_corruption.csv"
        av.to_csv(path, index=False)
        print(f"\nwrote {path.relative_to(PROJECT_ROOT)}  ({len(av)} rows)")

    rob = dice_corruption_robustness(df)
    if not rob.empty:
        path = out_root / "dice_corruption_robustness.csv"
        rob.to_csv(path, index=False)
        print(f"wrote {path.relative_to(PROJECT_ROOT)}  ({len(rob)} rows)")

    # Compact console summary — Dice pivot only
    print("\n========== Dice per (model × test_variant) ==========")
    pivot = df.pivot_table(index="model_display", columns="test_variant",
                           values="dice", aggfunc="mean")
    clean_first = [c for c in pivot.columns if "_s3_" not in str(c)]
    corrupted = sorted(c for c in pivot.columns if "_s3_" in str(c))
    print(pivot.reindex(columns=clean_first + corrupted).round(4).to_string())


if __name__ == "__main__":
    main()
