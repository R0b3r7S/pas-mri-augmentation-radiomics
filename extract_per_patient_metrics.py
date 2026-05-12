#!/usr/bin/env python3
"""
==============================================================================
 Extract per-patient Dice for every (model, test variant) cell.
==============================================================================
 The aggregated `runs/<MODEL>/test_on_<VARIANT>/fold_0/test_metrics.json`
 only stores the mean Dice over all 19 test patients of that variant.
 To compute IQR (and any other per-patient summary statistic) we need the
 raw per-patient values. This script reuses the same volumetric Dice
 computation as extract_best_worst_overlays.py, but instead of keeping
 only the best and worst patient it writes **all** per-patient Dice
 values to one CSV.

 Output:
   shareable/per_patient_dice.csv
     columns:
       model, test_variant, base_domain, corruption, corruption_mode,
       patient, dice

 Usage:
   conda activate monai_placenta
   python extract_per_patient_metrics.py --workers 8

 Run-time on the paper grid (274 cells x 19 patients): ~1-2 min on
 a 12-core CPU. Idempotent — safe to re-run.
==============================================================================
"""

import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "resized"
RUNS_ROOT    = PROJECT_ROOT / "runs"
OUT_CSV      = PROJECT_ROOT / "shareable" / "per_patient_dice.csv"

VARIANT_RE = re.compile(
    r"^(?P<base>BTFE|TSE|SSH_TSE|COMBINED)"
    r"(?:_(?P<corr>[a-z_]+)_s(?P<sev>\d+)_(?P<cmode>2d|3d))?$"
)
DOMAIN_TO_DATASET = {
    "BTFE": "DATASET_BTFE",
    "TSE": "DATASET_SSH_TSE",
    "SSH_TSE": "DATASET_SSH_TSE",
    "COMBINED": "DATASET_COMBINED",
}


def parse_variant(variant: str):
    m = VARIANT_RE.match(variant)
    if not m:
        return None
    return {
        "base":            m.group("base"),
        "corruption":      m.group("corr") or "clean",
        "corruption_mode": m.group("cmode") or "clean",
    }


def gt_mask_dir(variant: str) -> Path | None:
    parsed = parse_variant(variant)
    if parsed is None:
        return None
    ds = DOMAIN_TO_DATASET.get(parsed["base"])
    if ds is None:
        return None
    return DATASET_ROOT / ds / "masks"


def patient_dice(pred_dir: Path, gt_dir: Path) -> float | None:
    """Volumetric Dice across every slice of one patient.

    pred filenames: <sub>_<idx>_pred.png   in pred_dir/<sub>/
    gt   filenames: <sub>_<idx>.png        in gt_dir/<sub>/
    """
    inter = 0
    union = 0
    matched = 0
    for pred_path in sorted(pred_dir.glob(f"{pred_dir.name}_*_pred.png")):
        idx = pred_path.stem.replace("_pred", "").rsplit("_", 1)[-1]
        gt_path = gt_dir / f"{pred_dir.name}_{idx}.png"
        if not gt_path.is_file():
            continue
        pr = (np.asarray(Image.open(pred_path), dtype=np.uint8) > 0)
        gt = (np.asarray(Image.open(gt_path),   dtype=np.uint8) > 0)
        inter += int(np.logical_and(pr, gt).sum())
        union += int(pr.sum() + gt.sum())
        matched += 1
    if matched == 0 or union == 0:
        return None
    return 2.0 * inter / union


def discover_cells():
    """Yield dicts describing every (model, variant) cell to process."""
    if not RUNS_ROOT.is_dir():
        return
    for model_dir in sorted(RUNS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        for test_dir in sorted(model_dir.iterdir()):
            if not test_dir.name.startswith("test_on_"):
                continue
            variant = test_dir.name[len("test_on_"):]
            for fold_dir in sorted(test_dir.iterdir()):
                if not (fold_dir.name.startswith("fold_") and fold_dir.is_dir()):
                    continue
                yield {
                    "model_name": model_dir.name,
                    "variant":    variant,
                    "fold_dir":   str(fold_dir),
                }


def _process_cell(task: dict):
    model = task["model_name"]
    variant = task["variant"]
    fold_dir = Path(task["fold_dir"])

    parsed = parse_variant(variant)
    if parsed is None:
        return {"status": "skip", "model": model, "variant": variant,
                "reason": "could not parse variant"}

    gt_root = gt_mask_dir(variant)
    if gt_root is None:
        return {"status": "skip", "model": model, "variant": variant,
                "reason": "no GT mask root"}

    pred_root = fold_dir / "inference_raw_masks"
    if not pred_root.is_dir():
        return {"status": "skip", "model": model, "variant": variant,
                "reason": "no inference_raw_masks/"}

    rows = []
    for sub_dir in sorted(pred_root.iterdir()):
        if not sub_dir.is_dir():
            continue
        d = patient_dice(sub_dir, gt_root / sub_dir.name)
        if d is None:
            continue
        rows.append({
            "model":            model,
            "test_variant":     variant,
            "base_domain":      parsed["base"],
            "corruption":       parsed["corruption"],
            "corruption_mode":  parsed["corruption_mode"],
            "patient":          sub_dir.name,
            "dice":             round(d, 6),
        })
    return {"status": "ok", "model": model, "variant": variant,
            "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    cells = list(discover_cells())
    if not cells:
        sys.exit(f"no cells found under {RUNS_ROOT}")
    print(f"discovered {len(cells)} cells across {RUNS_ROOT}")

    all_rows = []
    skipped = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_cell, c): c for c in cells}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res["status"] == "ok":
                all_rows.extend(res["rows"])
            else:
                skipped.append(res)
            if i % 25 == 0 or i == len(cells):
                print(f"  processed {i}/{len(cells)} cells "
                      f"(rows so far: {len(all_rows)})")

    if not all_rows:
        sys.exit("no per-patient Dice rows produced")

    # Sort for stable diffs.
    all_rows.sort(key=lambda r: (r["model"], r["test_variant"], r["patient"]))

    fieldnames = ["model", "test_variant", "base_domain", "corruption",
                  "corruption_mode", "patient", "dice"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"\nwrote {OUT_CSV.relative_to(PROJECT_ROOT)} "
          f"({len(all_rows)} rows from {len(cells) - len(skipped)} cells)")
    if skipped:
        print(f"\nskipped {len(skipped)} cells:")
        for s in skipped[:10]:
            print(f"  {s['model']}/{s['variant']}: {s['reason']}")


if __name__ == "__main__":
    main()
