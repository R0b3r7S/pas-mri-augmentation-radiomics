#!/usr/bin/env python3
"""
==============================================================================
 Extract best- and worst-patient inference overlays per (model × test variant)
==============================================================================
 Walks runs/<MODEL>/test_on_<VARIANT>/fold_0/, computes a per-patient Dice
 score by re-loading the predicted masks and the corresponding GT masks,
 then copies the inference_overlays/<sub>/*.png of the *best* and *worst*
 patient (by Dice) into a single sharable directory tree.

 Why we re-compute Dice: test_metrics.json only stores aggregate (mean)
 metrics, not per-patient values. Per-patient Dice is straightforward to
 reconstruct from the binary masks already on disk.

 Output layout:
   shareable/best_worst_overlays/
     <MODEL>/
       test_on_<VARIANT>/
         best_<sub>_dice0.95/   (full overlay PNGs for that patient)
         worst_<sub>_dice0.42/
     summary.csv                (model, variant, best_sub, best_dice,
                                 worst_sub, worst_dice)

 Usage:
   python extract_best_worst_overlays.py
   python extract_best_worst_overlays.py --runs-filter BTFE_unetplusplus_mixup_afa
   python extract_best_worst_overlays.py --variants-filter BTFE BTFE_spike_noise_s3_2d
==============================================================================
"""

import argparse
import csv
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "resized"
RUNS_ROOT = PROJECT_ROOT / "runs"
OUT_ROOT = PROJECT_ROOT / "shareable" / "best_worst_overlays"
SUMMARY_CSV = OUT_ROOT.parent / "best_worst_summary.csv"

# Map test variant name to clean dataset (for GT mask lookup).
# Same convention as compute_radiomics.py / compare_radiomics.py.
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
    return m.group("base"), m.group("corr"), m.group("cmode")


def gt_mask_dir(variant: str) -> Path | None:
    parsed = parse_variant(variant)
    if parsed is None:
        return None
    base = parsed[0]
    ds = DOMAIN_TO_DATASET.get(base)
    if ds is None:
        return None
    return DATASET_ROOT / ds / "masks"


def patient_dice(pred_dir: Path, gt_dir: Path) -> float | None:
    """Aggregated 2D Dice across every slice of one patient.

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


def discover_variants():
    """Yield (model_dir, variant_name, fold_dir)."""
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
                if fold_dir.name.startswith("fold_") and fold_dir.is_dir():
                    yield model_dir.name, variant, fold_dir


def copy_patient_overlays(overlays_dir: Path, dst: Path):
    if not overlays_dir.is_dir():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in overlays_dir.glob("*.png"):
        shutil.copy2(src, dst / src.name)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Worker (one task = one (model, variant) cell)
# ---------------------------------------------------------------------------
def _process_cell(task: dict):
    """Compute per-patient Dice for one cell, optionally copy best/worst overlays.

    `task` must be picklable (passed across processes).
    Returns a dict the parent collects into the summary CSV.
    """
    try:
        model_name = task["model_name"]
        variant = task["variant"]
        fold_dir = Path(task["fold_dir"])
        no_copy = task["no_copy"]
        out_root = Path(task["out_root"])

        gt_root = gt_mask_dir(variant)
        if gt_root is None:
            return {"status": "skip", "reason": f"unknown variant '{variant}'",
                    "model": model_name, "variant": variant}

        pred_root = fold_dir / "inference_raw_masks"
        overlay_root = fold_dir / "inference_overlays"
        if not pred_root.is_dir():
            return {"status": "skip", "reason": "no inference_raw_masks",
                    "model": model_name, "variant": variant}

        scores = []
        for sub_dir in sorted(pred_root.iterdir()):
            if not sub_dir.is_dir():
                continue
            d = patient_dice(sub_dir, gt_root / sub_dir.name)
            if d is not None:
                scores.append((sub_dir.name, d))

        if len(scores) < 2:
            return {"status": "skip", "reason": "<2 patients with computable Dice",
                    "model": model_name, "variant": variant}

        scores.sort(key=lambda x: x[1])
        worst_sub, worst_d = scores[0]
        best_sub,  best_d  = scores[-1]

        if not no_copy:
            base = out_root / model_name / f"test_on_{variant}"
            for tag, sub, d in [("best", best_sub, best_d), ("worst", worst_sub, worst_d)]:
                copy_patient_overlays(overlay_root / sub,
                                      base / f"{tag}_{sub}_dice{d:.3f}")

        return {
            "status": "ok",
            "row": {
                "model": model_name,
                "test_variant": variant,
                "best_patient": best_sub,
                "best_dice": round(best_d, 4),
                "worst_patient": worst_sub,
                "worst_dice": round(worst_d, 4),
                "n_patients": len(scores),
                "median_dice": round(float(np.median([s for _, s in scores])), 4),
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}",
                "model": task.get("model_name"), "variant": task.get("variant")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-filter", nargs="*", default=None,
                    help="Only process model dirs whose name contains any of these substrings.")
    ap.add_argument("--variants-filter", nargs="*", default=None,
                    help="Only process these test_on_<VARIANT> names (substring match).")
    ap.add_argument("--out-dir", default=str(OUT_ROOT),
                    help=f"Override output root (default {OUT_ROOT}).")
    ap.add_argument("--no-copy", action="store_true",
                    help="Compute Dice and write summary.csv but skip copying PNGs (dry-run).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Process pool size; one (model × variant) cell per worker. "
                         "Default = cpu_count() - 1.")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root.parent / "best_worst_summary.csv"

    # Build the task list (one task = one (model, variant) cell)
    tasks = []
    for model_name, variant, fold_dir in discover_variants():
        if args.runs_filter and not any(s in model_name for s in args.runs_filter):
            continue
        if args.variants_filter and not any(s in variant for s in args.variants_filter):
            continue
        tasks.append({
            "model_name": model_name,
            "variant": variant,
            "fold_dir": str(fold_dir),
            "no_copy": args.no_copy,
            "out_root": str(out_root),
        })

    print(f"[overlays] cells queued : {len(tasks)}")
    print(f"[overlays] workers      : {args.workers}")
    print(f"[overlays] copying PNGs : {not args.no_copy}")
    if not tasks:
        print("[overlays] nothing to do")
        return

    rows = []
    n_ok = n_skipped = n_err = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process_cell, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            status = res.get("status")
            if status == "ok":
                rows.append(res["row"])
                n_ok += 1
                r = res["row"]
                print(f"  {r['model']:50s} | {r['test_variant']:35s} | "
                      f"best {r['best_patient']} ({r['best_dice']:.3f})  "
                      f"worst {r['worst_patient']} ({r['worst_dice']:.3f})  "
                      f"n={r['n_patients']}")
            elif status == "skip":
                n_skipped += 1
                print(f"  [skip] {res.get('model')}/{res.get('variant')}: {res.get('reason')}")
            else:
                n_err += 1
                print(f"  [ERR] {res.get('model')}/{res.get('variant')}: {res.get('error')}",
                      file=sys.stderr)

    rows.sort(key=lambda r: (r["model"], r["test_variant"]))
    if rows:
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote summary: {summary_path}")
    else:
        print("\nno (model, variant) pairs produced output")

    elapsed = time.time() - t0
    print(f"\nprocessed={n_ok}  skipped={n_skipped}  err={n_err}  "
          f"elapsed={elapsed:.1f}s")
    if not args.no_copy:
        print(f"overlays under: {out_root}")


if __name__ == "__main__":
    main()
