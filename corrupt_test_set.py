#!/usr/bin/env python3
"""
==============================================================================
 Generate corrupted test-set variants (Figure 1 of paper 2505.10223)
==============================================================================
 Applies the same five image corruptions shown in the paper's figure 1 —
 Bias Field, Ghosting, k-Space Subsampling, Rician Noise, Spike Noise —
 at severity level 3, and writes each corrupted dataset alongside the
 clean one so the existing inference/radiomics pipeline can consume them
 unchanged.

 Severity-level-3 parameters are taken directly from the upstream
 MIAGroupUT/augmentations-for-the-unknown repository
 (medg/transforms/defaults.py, index 2 of every severity_controller).
 That folder is not vendored in this repo; see the upstream source if
 you need to verify the values.

 Two corruption modes run in parallel:
   2d: each slice corrupted independently
       (seed = hash(patient, slice_idx, corruption))
   3d: slices of one patient stacked [H, W, N] → corrupted once → split back
       (seed = hash(patient, corruption))

 Output layout (for each corruption × mode × domain combination):
   dataset/resized/DATASET_<BASE>_<corr>_s3_<mode>/
     images/<sub>/<sub>_<idx>.png
     masks/<sub>/<sub>_<idx>.png            (copied unchanged from clean)
     splits.json                            (test-only)

 Usage:
   python corrupt_test_set.py                    # all defaults (both modes, sev 3, BTFE+TSE)
   python corrupt_test_set.py --mode 2d          # only 2D
   python corrupt_test_set.py --corruptions bias_field spike_noise
   python corrupt_test_set.py --domains BTFE
==============================================================================
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torchio as tio
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "resized"

DOMAIN_TO_DIR = {
    "BTFE": "DATASET_BTFE",
    "TSE": "DATASET_SSH_TSE",
    "SSH_TSE": "DATASET_SSH_TSE",
}

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def stable_seed(*parts) -> int:
    s = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "big")


# ---------------------------------------------------------------------------
# Corruption operators
# ---------------------------------------------------------------------------
# Every operator takes a float32 volume of shape [H, W, N] in [0, 1] and a
# seed, and returns a corrupted volume of the same shape in [0, 1].
#
# Severity-level-3 values are documented next to each operator and sourced
# from the upstream MIAGroupUT medg/transforms/defaults.py (index 2 of
# every 5-step severity_controller). The upstream tree is not vendored
# in this repo.
# ---------------------------------------------------------------------------

def apply_bias_field(vol: np.ndarray, seed: int) -> np.ndarray:
    """torchio.RandomBiasField — coefficients=(-0.9, 0.9), order=3 (severity 3)."""
    torch.manual_seed(seed)
    tensor = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0)   # [1, H, W, N]
    tf = tio.RandomBiasField(coefficients=0.9, order=3)
    out = tf(tensor).squeeze(0).numpy()
    return np.clip(out, 0.0, 1.0)


def apply_ghosting(vol: np.ndarray, seed: int) -> np.ndarray:
    """torchio.RandomGhosting — num_ghosts=7, intensity=1.5, axes=(0, 1) (severity 3)."""
    torch.manual_seed(seed)
    tensor = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0)
    tf = tio.RandomGhosting(num_ghosts=(7, 7), intensity=(1.5, 1.5), axes=(0, 1))
    out = tf(tensor).squeeze(0).numpy()
    return np.clip(out, 0.0, 1.0)


def apply_kspace_subsampling(vol: np.ndarray, seed: int,
                             center_fraction: float = 0.08,
                             acceleration: int = 4) -> np.ndarray:
    """fastMRI-style random Cartesian undersampling per slice.

    Severity-3 (from medg): center_fraction=0.08, acceleration=4.
    We undersample along the width axis (phase-encoding direction).
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(vol, dtype=np.float32)
    H, W, N = vol.shape
    center_w = max(1, int(round(W * center_fraction)))
    target_keep = max(center_w, W // acceleration)
    for z in range(N):
        slc = vol[..., z].astype(np.float32)
        k = np.fft.fftshift(np.fft.fft2(slc))
        mask = np.zeros(W, dtype=bool)
        cstart = (W - center_w) // 2
        mask[cstart:cstart + center_w] = True
        remaining = target_keep - center_w
        if remaining > 0:
            candidates = np.where(~mask)[0]
            selected = rng.choice(candidates, size=remaining, replace=False)
            mask[selected] = True
        k *= mask[np.newaxis, :]
        recon = np.abs(np.fft.ifft2(np.fft.ifftshift(k)))
        out[..., z] = recon.astype(np.float32)
    # Renormalise to [0, 1] so PNG round-trip is stable
    mx = out.max()
    if mx > 0:
        out /= mx
    return out


def apply_rician_noise(vol: np.ndarray, seed: int, std: float = 0.48) -> np.ndarray:
    """Rician noise with *relative* std (monai.RandRicianNoise convention).

    Severity-3 (from medg): std = linspace(0, 0.8, 6)[1:][2] = 0.48, relative=True.
    Rician(x) = sqrt((x + n1)^2 + n2^2) with n1, n2 ~ N(0, std * max(x)).
    """
    rng = np.random.default_rng(seed)
    v = vol.astype(np.float32)
    scale = float(v.max() * std)
    if scale <= 0.0:
        return v
    n1 = rng.normal(0.0, scale, size=v.shape).astype(np.float32)
    n2 = rng.normal(0.0, scale, size=v.shape).astype(np.float32)
    noisy = np.sqrt((v + n1) ** 2 + n2 ** 2)
    mx = noisy.max()
    if mx > 0:
        noisy /= mx
    return noisy.astype(np.float32)


def apply_spike_noise(vol: np.ndarray, seed: int,
                      num_spikes: int = 2,
                      intensity_range=(1.0, 1.5)) -> np.ndarray:
    """Random high-amplitude impulses in 2D k-space per slice.

    Severity-3 (from medg RandomSpikeN): num_spikes=2, intensity=(1.0, 1.5).
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(vol, dtype=np.float32)
    H, W, N = vol.shape
    for z in range(N):
        slc = vol[..., z].astype(np.float32)
        k = np.fft.fftshift(np.fft.fft2(slc))
        max_mag = float(np.abs(k).max())
        for _ in range(num_spikes):
            y = int(rng.integers(0, H))
            x = int(rng.integers(0, W))
            amp = float(rng.uniform(*intensity_range)) * max_mag
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            k[y, x] = amp * np.exp(1j * phase)
        recon = np.abs(np.fft.ifft2(np.fft.ifftshift(k)))
        out[..., z] = recon.astype(np.float32)
    mx = out.max()
    if mx > 0:
        out /= mx
    return out


CORRUPTIONS = {
    "bias_field":   apply_bias_field,
    "ghosting":     apply_ghosting,
    "kspace_sub":   apply_kspace_subsampling,
    "rician_noise": apply_rician_noise,
    "spike_noise":  apply_spike_noise,
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_patient_volume(image_dir: Path):
    """Return ([H, W, N] float volume in [0,1], [idx list])."""
    files = sorted(image_dir.glob(f"{image_dir.name}_*.png"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        return None, []
    arrays = []
    indices = []
    for f in files:
        arr = np.asarray(Image.open(f), dtype=np.uint8)
        arrays.append(arr)
        indices.append(int(f.stem.split("_")[-1]))
    vol = np.stack(arrays, axis=-1).astype(np.float32) / 255.0   # [H, W, N]
    return vol, indices


def write_slice(path: Path, arr01: np.ndarray):
    """Save a 2D float array in [0, 1] as uint8 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(arr01 * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(a, mode="L").save(path)


def copy_masks(src_mask_dir: Path, dst_mask_dir: Path):
    dst_mask_dir.mkdir(parents=True, exist_ok=True)
    for f in src_mask_dir.glob("*.png"):
        shutil.copy2(f, dst_mask_dir / f.name)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def process_patient(src_root: Path, dst_root: Path, sub: str,
                    corruption_name: str, mode: str, seed_base: int):
    src_img_dir = src_root / "images" / sub
    src_msk_dir = src_root / "masks" / sub
    dst_img_dir = dst_root / "images" / sub
    dst_msk_dir = dst_root / "masks" / sub

    vol, indices = read_patient_volume(src_img_dir)
    if vol is None:
        return 0

    fn = CORRUPTIONS[corruption_name]

    if mode == "3d":
        seed = stable_seed(seed_base, sub, corruption_name, "3d")
        corrupted = fn(vol, seed)
    else:  # 2d — per-slice
        corrupted = np.empty_like(vol)
        for i, idx in enumerate(indices):
            seed = stable_seed(seed_base, sub, idx, corruption_name, "2d")
            single = vol[..., i:i + 1]          # [H, W, 1]
            corrupted[..., i:i + 1] = fn(single, seed)

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(indices):
        write_slice(dst_img_dir / f"{sub}_{idx}.png", corrupted[..., i])
    copy_masks(src_msk_dir, dst_msk_dir)
    return len(indices)


def write_test_only_splits(src_splits: Path, dst_splits: Path):
    """Write a splits.json that keeps only the test patients, so the existing
    training script in --mode test reads the right list."""
    data = json.loads(src_splits.read_text())
    out = {
        "train": [],
        "val": [],
        "test": data.get("test", []),
        "seed": data.get("seed", 42),
        "mode": "corrupted_test_only",
        "fractions": data.get("fractions", {}),
        "source_splits": str(src_splits),
    }
    dst_splits.parent.mkdir(parents=True, exist_ok=True)
    dst_splits.write_text(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="*", default=["BTFE", "SSH_TSE"],
                        help="Which base domains to corrupt (BTFE, SSH_TSE, TSE).")
    parser.add_argument("--mode", choices=["2d", "3d", "both"], default="both")
    parser.add_argument("--corruptions", nargs="*", default=list(CORRUPTIONS),
                        help="Which corruption types to apply.")
    parser.add_argument("--severity", type=int, default=3,
                        help="Only severity level 3 is hard-coded (the values used by the "
                             "paper); kept for folder naming and future extension.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed mixed with (patient, slice, corruption) for reproducibility.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.severity != 3:
        sys.exit("Only severity=3 is wired up right now (matches the upstream medg "
                 "defaults at index 2; that folder is not vendored here, see the upstream "
                 "MIAGroupUT/augmentations-for-the-unknown repo). "
                 "Extend the severity tables in the apply_* functions to enable other levels.")

    modes = ("2d", "3d") if args.mode == "both" else (args.mode,)
    unknown = [c for c in args.corruptions if c not in CORRUPTIONS]
    if unknown:
        sys.exit(f"Unknown corruption(s): {unknown}. Known: {list(CORRUPTIONS)}")

    print(f"[corrupt] project root : {PROJECT_ROOT}")
    print(f"[corrupt] domains      : {args.domains}")
    print(f"[corrupt] corruptions  : {args.corruptions}")
    print(f"[corrupt] modes        : {modes}")
    print(f"[corrupt] severity     : {args.severity}")
    print(f"[corrupt] base seed    : {args.seed}")

    total_slices = 0
    for domain in args.domains:
        src_dir_name = DOMAIN_TO_DIR.get(domain)
        if src_dir_name is None:
            print(f"  [skip] unknown domain {domain}")
            continue
        src_root = DATASET_ROOT / src_dir_name
        if not src_root.is_dir():
            print(f"  [skip] missing source dataset {src_root}")
            continue

        splits_path = src_root / "splits.json"
        test_subs = json.loads(splits_path.read_text()).get("test", [])
        if not test_subs:
            print(f"  [skip] no test patients in {splits_path}")
            continue

        print(f"\n[{domain}] {len(test_subs)} test patients: {test_subs}")

        for corr in args.corruptions:
            for mode in modes:
                tag = f"{corr}_s{args.severity}_{mode}"
                dst_root = DATASET_ROOT / f"{src_dir_name}_{tag}"
                print(f"  → {dst_root.relative_to(PROJECT_ROOT)}")
                if dst_root.exists() and not args.overwrite:
                    print(f"    already exists, skipping (use --overwrite to force)")
                    continue
                if args.dry_run:
                    continue

                for sub in test_subs:
                    n = process_patient(src_root, dst_root, sub, corr, mode, args.seed)
                    total_slices += n
                write_test_only_splits(splits_path, dst_root / "splits.json")

    print(f"\n[corrupt] done. total corrupted slices written: {total_slices}")


if __name__ == "__main__":
    main()
