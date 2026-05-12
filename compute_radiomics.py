#!/usr/bin/env python3
"""
==============================================================================
 Radiomics Feature Extraction for Placenta PAS Segmentation
==============================================================================
 Walks dataset/ (GT masks) and runs/<MODEL>/test_on_<DOMAIN>/fold_0/
 inference_raw_masks/ (predicted masks) and extracts PyRadiomics features
 for every (patient, mask-source) pair. Features are cached as JSON.

 Two modes:
   - 2D: extract features per slice, aggregate per patient (mean/std).
   - 3D: stack slices into a pseudo-3D volume, extract once per patient.

 Physical spacing (from s41598-023-50900-z paper + 512x512 resize):
   in-plane   : 36 cm FOV / 512 px = 0.703125 mm
   slice (3D) : 5 mm thickness + 0.6 mm gap = 5.6 mm

 Output layout:
   radiomics/
     features/
       gt/<mode>/<DOMAIN>/<sub>.json         (computed once per domain)
       pred/<MODEL>/test_on_<DOMAIN>/<mode>/<sub>.json
==============================================================================
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

try:
    from radiomics import featureextractor
    import radiomics
    radiomics.logger.setLevel("ERROR")
except ImportError as e:
    sys.stderr.write(
        "pyradiomics is not installed. Install with:\n"
        "  pip install pyradiomics \"numpy<2\"\n"
    )
    raise

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "resized"
RUNS_ROOT = PROJECT_ROOT / "runs"
OUT_ROOT = PROJECT_ROOT / "radiomics" / "features"

# test_on_<X> folder name -> dataset folder name
DOMAIN_TO_DATASET = {
    "BTFE": "DATASET_BTFE",
    "TSE": "DATASET_SSH_TSE",
    "SSH_TSE": "DATASET_SSH_TSE",
    "COMBINED": "DATASET_COMBINED",
}

# Corrupted-variant naming: e.g. "BTFE_bias_field_s3_2d" → base "BTFE".
# Keeps GT lookups pointed at the CLEAN images (Option 1 semantics: measure
# the pure segmentation-boundary impact on radiomics, isolated from the
# image-intensity corruption the model was fed).
VARIANT_RE = re.compile(
    r"^(?P<base>BTFE|TSE|SSH_TSE|COMBINED)"
    r"(?:_(?P<corr>[a-z_]+)_s(?P<sev>\d+)_(?P<cmode>2d|3d))?$"
)


def parse_variant(variant: str):
    """Split a ``test_on_<variant>`` suffix into its parts.

    Returns a dict with keys ``base``, ``corruption``, ``severity``, ``cmode``
    (corruption-mode). For a clean test domain, corruption is ``None``.
    """
    m = VARIANT_RE.match(variant)
    if not m:
        return None
    return {
        "base": m.group("base"),
        "corruption": m.group("corr"),
        "severity": int(m.group("sev")) if m.group("sev") else None,
        "cmode": m.group("cmode"),
    }


IN_PLANE_SPACING_MM = 360.0 / 512.0   # 0.703125
SLICE_SPACING_MM = 5.6                # 5 mm thickness + 0.6 mm gap

PRED_PATTERN = re.compile(r"^(?P<sub>[\w\-]+)_(?P<idx>\d+)_pred\.png$")
GT_PATTERN = re.compile(r"^(?P<sub>[\w\-]+)_(?P<idx>\d+)\.png$")


# ---------------------------------------------------------------------------
# PyRadiomics extractor builders
# ---------------------------------------------------------------------------
def build_extractor_2d():
    settings = {
        "binWidth": 25,
        "force2D": True,
        "force2Ddimension": 0,          # axial slice
        "interpolator": None,
        "resampledPixelSpacing": None,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllFeatures()
    # shape in 2D -> use shape2D (disable shape which is 3D-only)
    extractor.disableAllFeatures()
    for cls in ("shape2D", "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"):
        extractor.enableFeatureClassByName(cls)
    return extractor


def build_extractor_3d():
    settings = {
        "binWidth": 25,
        "interpolator": None,
        "resampledPixelSpacing": None,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    for cls in ("shape", "firstorder", "glcm", "glrlm", "glszm", "ngtdm", "gldm"):
        extractor.enableFeatureClassByName(cls)
    return extractor


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_gray_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint8)


def to_binary_mask(arr: np.ndarray) -> np.ndarray:
    """Normalise {0,1} or {0,255} encodings to uint8 {0,1}."""
    return (arr > 0).astype(np.uint8)


def list_slices(sub_dir: Path, pattern: re.Pattern):
    """Return sorted list of (idx:int, filename:str)."""
    items = []
    if not sub_dir.is_dir():
        return items
    for f in os.listdir(sub_dir):
        m = pattern.match(f)
        if m:
            items.append((int(m.group("idx")), f))
    items.sort(key=lambda x: x[0])
    return items


def sitk_2d(arr: np.ndarray) -> sitk.Image:
    """Wrap a 2D numpy array as a SimpleITK image with in-plane spacing."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing((IN_PLANE_SPACING_MM, IN_PLANE_SPACING_MM))
    return img


def sitk_3d(vol: np.ndarray) -> sitk.Image:
    """Wrap a [Z, Y, X] numpy array as 3D sitk image with isotropic-in-plane spacing."""
    img = sitk.GetImageFromArray(vol.astype(np.float32))
    # SimpleITK SetSpacing order is (X, Y, Z)
    img.SetSpacing((IN_PLANE_SPACING_MM, IN_PLANE_SPACING_MM, SLICE_SPACING_MM))
    return img


# ---------------------------------------------------------------------------
# Extraction primitives
# ---------------------------------------------------------------------------
def _filter_features(result: dict) -> dict:
    """Drop pyradiomics diagnostics/keys, keep numeric feature values only."""
    out = {}
    for k, v in result.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def extract_2d_patient(
    image_slices: list,  # list of (idx, np.ndarray)
    mask_slices: list,   # list of (idx, np.ndarray) aligned with image_slices
    extractor,
) -> dict:
    """Extract per-slice 2D features, return aggregated dict."""
    per_slice = {}
    errors = []
    for (img_idx, img_arr), (mask_idx, mask_arr) in zip(image_slices, mask_slices):
        assert img_idx == mask_idx
        mask_bin = to_binary_mask(mask_arr)
        if mask_bin.sum() < 3:   # pyradiomics needs a non-trivial ROI
            continue
        try:
            img_sitk = sitk_2d(img_arr)
            mask_sitk = sitk_2d(mask_bin)
            mask_sitk.CopyInformation(img_sitk)
            feats = _filter_features(extractor.execute(img_sitk, mask_sitk, label=1))
            per_slice[str(img_idx)] = feats
        except Exception as e:
            errors.append((img_idx, str(e)))
            continue

    if not per_slice:
        return {
            "n_slices": len(image_slices),
            "n_valid_slices": 0,
            "features_mean": {},
            "features_std": {},
            "per_slice_features": {},
            "errors": errors,
        }

    # Collect keys (union) then aggregate
    keys = sorted({k for sl in per_slice.values() for k in sl})
    mean = {}
    std = {}
    for k in keys:
        vals = [sl[k] for sl in per_slice.values() if k in sl]
        arr = np.asarray(vals, dtype=np.float64)
        mean[k] = float(np.nanmean(arr))
        std[k] = float(np.nanstd(arr))

    return {
        "n_slices": len(image_slices),
        "n_valid_slices": len(per_slice),
        "features_mean": mean,
        "features_std": std,
        "per_slice_features": per_slice,
        "errors": errors,
    }


def extract_3d_patient(
    image_slices: list,   # list of (idx, np.ndarray)
    mask_slices: list,
    extractor,
) -> dict:
    if not image_slices:
        return {"n_slices": 0, "features": {}, "errors": ["no slices"]}

    img_vol = np.stack([a for _, a in image_slices], axis=0)
    mask_vol = np.stack([to_binary_mask(a) for _, a in mask_slices], axis=0)

    if mask_vol.sum() < 10:
        return {
            "n_slices": len(image_slices),
            "features": {},
            "errors": [f"empty mask (voxels={int(mask_vol.sum())})"],
        }

    try:
        img_sitk = sitk_3d(img_vol)
        mask_sitk = sitk_3d(mask_vol)
        mask_sitk.CopyInformation(img_sitk)
        feats = _filter_features(extractor.execute(img_sitk, mask_sitk, label=1))
        return {"n_slices": len(image_slices), "features": feats, "errors": []}
    except Exception as e:
        return {"n_slices": len(image_slices), "features": {}, "errors": [str(e)]}


# ---------------------------------------------------------------------------
# Task construction (what to extract)
# ---------------------------------------------------------------------------
def gt_patients_for_domain(domain_key: str):
    """Return list of (patient_id, image_dir, mask_dir)."""
    ds_name = DOMAIN_TO_DATASET.get(domain_key)
    if ds_name is None:
        return []
    img_root = DATASET_ROOT / ds_name / "images"
    mask_root = DATASET_ROOT / ds_name / "masks"
    if not img_root.is_dir() or not mask_root.is_dir():
        return []
    patients = sorted(p.name for p in mask_root.iterdir() if p.is_dir())
    return [(p, img_root / p, mask_root / p) for p in patients]


def discover_pred_runs():
    """Yield tuples (model_name, domain_key, pred_root_dir, fold_name)."""
    if not RUNS_ROOT.is_dir():
        return
    for model_dir in sorted(RUNS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        for test_dir in sorted(model_dir.iterdir()):
            if not test_dir.is_dir() or not test_dir.name.startswith("test_on_"):
                continue
            domain_key = test_dir.name[len("test_on_"):]
            for fold_dir in sorted(test_dir.iterdir()):
                if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
                    continue
                pred_root = fold_dir / "inference_raw_masks"
                if pred_root.is_dir():
                    yield (model_dir.name, domain_key, pred_root, fold_dir.name)


# ---------------------------------------------------------------------------
# Worker helpers (run in subprocesses)
# ---------------------------------------------------------------------------
_EXTR = {"2d": None, "3d": None}


def _get_extractor(mode: str):
    ex = _EXTR.get(mode)
    if ex is None:
        ex = build_extractor_2d() if mode == "2d" else build_extractor_3d()
        _EXTR[mode] = ex
    return ex


def _load_image_mask_pair(image_dir: Path, mask_dir: Path, mask_pattern: re.Pattern):
    """Return (image_slices, mask_slices) as aligned lists."""
    mask_items = list_slices(mask_dir, mask_pattern)
    if not mask_items:
        return [], []

    image_slices = []
    mask_slices = []
    for idx, mask_fname in mask_items:
        img_fname = f"{mask_dir.name}_{idx}.png"
        img_path = image_dir / img_fname
        mask_path = mask_dir / mask_fname
        if not img_path.is_file() or not mask_path.is_file():
            continue
        image_slices.append((idx, read_gray_png(img_path)))
        mask_slices.append((idx, read_gray_png(mask_path)))
    return image_slices, mask_slices


def _compute_and_save(task):
    """Run one extraction task. `task` is a dict; must be picklable."""
    try:
        kind = task["kind"]       # "gt" or "pred"
        mode = task["mode"]       # "2d" or "3d"
        out_path = Path(task["out_path"])
        image_dir = Path(task["image_dir"])
        mask_dir = Path(task["mask_dir"])
        mask_pattern = PRED_PATTERN if kind == "pred" else GT_PATTERN

        if out_path.is_file() and not task.get("overwrite", False):
            return {"status": "skip", "out": str(out_path)}

        image_slices, mask_slices = _load_image_mask_pair(image_dir, mask_dir, mask_pattern)
        if not image_slices:
            return {"status": "empty", "out": str(out_path)}

        extractor = _get_extractor(mode)
        if mode == "2d":
            payload = extract_2d_patient(image_slices, mask_slices, extractor)
        else:
            payload = extract_3d_patient(image_slices, mask_slices, extractor)

        payload.update({
            "kind": kind,
            "mode": mode,
            "patient": task["patient"],
            "domain": task["domain"],
            "base_domain": task.get("base_domain", task["domain"]),
            "corruption": task.get("corruption"),
            "severity": task.get("severity"),
            "corruption_mode": task.get("cmode"),
            "model": task.get("model"),
            "spacing_mm": [IN_PLANE_SPACING_MM, IN_PLANE_SPACING_MM] + (
                [SLICE_SPACING_MM] if mode == "3d" else []),
        })
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(payload, fh)
        return {"status": "ok", "out": str(out_path)}
    except Exception as e:
        return {
            "status": "error",
            "out": task.get("out_path"),
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_tasks(modes, runs_filter, domain_filter, overwrite):
    tasks = []

    # --- GT tasks (shared across all models) ---
    gt_domains = [d for d in ("BTFE", "SSH_TSE") if not domain_filter or d in domain_filter]
    for domain in gt_domains:
        for patient, img_dir, mask_dir in gt_patients_for_domain(domain):
            for mode in modes:
                out_path = OUT_ROOT / "gt" / mode / domain / f"{patient}.json"
                tasks.append({
                    "kind": "gt",
                    "mode": mode,
                    "domain": domain,
                    "patient": patient,
                    "model": None,
                    "image_dir": str(img_dir),
                    "mask_dir": str(mask_dir),
                    "out_path": str(out_path),
                    "overwrite": overwrite,
                })

    # --- PRED tasks ---
    for model_name, domain_key, pred_root, fold_name in discover_pred_runs():
        if runs_filter and not any(flt in model_name for flt in runs_filter):
            continue

        parsed = parse_variant(domain_key)
        if parsed is None:
            continue
        base_domain = parsed["base"]
        ds_name = DOMAIN_TO_DATASET.get(base_domain)
        if ds_name is None:
            continue
        if domain_filter and not any(
            flt in domain_key or flt == base_domain for flt in domain_filter
        ):
            continue

        image_root = DATASET_ROOT / ds_name / "images"
        for sub_dir in sorted(pred_root.iterdir()):
            if not sub_dir.is_dir():
                continue
            patient = sub_dir.name
            img_dir = image_root / patient
            if not img_dir.is_dir():
                continue
            for mode in modes:
                out_path = (OUT_ROOT / "pred" / model_name
                            / f"test_on_{domain_key}" / mode / f"{patient}.json")
                tasks.append({
                    "kind": "pred",
                    "mode": mode,
                    "domain": domain_key,       # full variant name, e.g. BTFE_bias_field_s3_2d
                    "base_domain": base_domain,
                    "corruption": parsed["corruption"],
                    "severity": parsed["severity"],
                    "cmode": parsed["cmode"],
                    "patient": patient,
                    "model": model_name,
                    "image_dir": str(img_dir),
                    "mask_dir": str(sub_dir),
                    "out_path": str(out_path),
                    "overwrite": overwrite,
                })

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["2d", "3d", "both"], default="both")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if output JSON already exists.")
    parser.add_argument("--runs-filter", nargs="*", default=None,
                        help="Only process model dirs whose name contains any of these substrings.")
    parser.add_argument("--domain-filter", nargs="*", default=None,
                        help="Only process these test domains (BTFE, TSE, SSH_TSE).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print task count and exit.")
    args = parser.parse_args()

    modes = ("2d", "3d") if args.mode == "both" else (args.mode,)

    tasks = build_tasks(modes, args.runs_filter, args.domain_filter, args.overwrite)
    print(f"[radiomics] project root     : {PROJECT_ROOT}")
    print(f"[radiomics] dataset root     : {DATASET_ROOT}")
    print(f"[radiomics] runs root        : {RUNS_ROOT}")
    print(f"[radiomics] output root      : {OUT_ROOT}")
    print(f"[radiomics] in-plane spacing : {IN_PLANE_SPACING_MM} mm")
    print(f"[radiomics] slice spacing 3D : {SLICE_SPACING_MM} mm")
    print(f"[radiomics] modes            : {modes}")
    print(f"[radiomics] tasks queued     : {len(tasks)}")
    if args.dry_run or not tasks:
        return

    ok = skip = empty = err = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_compute_and_save, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            status = res.get("status")
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            elif status == "empty":
                empty += 1
            else:
                err += 1
                print(f"[ERR] {res.get('out')}: {res.get('error')}", file=sys.stderr)
            if i % 25 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                remaining = (len(tasks) - i) / max(rate, 1e-6)
                print(f"  [{i}/{len(tasks)}] ok={ok} skip={skip} empty={empty} err={err} "
                      f"elapsed={elapsed:.0f}s eta={remaining:.0f}s", flush=True)

    print(f"[radiomics] done. ok={ok} skip={skip} empty={empty} err={err} "
          f"elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
