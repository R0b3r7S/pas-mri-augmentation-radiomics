# PAS MRI Augmentation × Radiomics

Radiomic-stability evaluation of data-agnostic augmentations (MixUp, CutMix, AFA, and their combinations) for U-Net++ segmentation of the placenta in MRI. The repository trains the segmentation models, evaluates them in-domain and across modalities, applies five clinically-motivated MRI corruptions at severity level 3, extracts 102 PyRadiomics features per (patient, mask), and quantifies how segmentation errors propagate to the radiomic feature signature. Numbers and figures of the paper are regenerated from CSVs by two short scripts in `paper/`.

This repository builds directly on
[MIAGroupUT/augmentations-for-the-unknown](https://github.com/MIAGroupUT/augmentations-for-the-unknown)
(MIDL 2025). The `augmentations/` AFA and Dual Instance-Batch
Normalisation modules are derived from that codebase; the MixUp and
CutMix modules are derived from [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
(DKFZ). Both upstreams are Apache-2.0 — see [NOTICE](NOTICE) for the
required attributions. The
upstream `medg/`, `ml/`, and `nnunet/` trees and the helper scripts
`gen_med_g.py` / `model_surgery.py` are **not** carried in this repo —
they are not used by the paper pipeline, and the upstream copy is
authoritative. The severity-level-3 corruption parameters in
[corrupt_test_set.py](corrupt_test_set.py) were copied from the
upstream `medg/transforms/defaults.py`; see that file in the linked
repository if you want to verify the values.

---

## Acknowledgments and citation

This work uses code from and credit is due to:

- **P. Vaish, F. Meister, T. Heimann, C. Brune, and J. M. Wolterink.**
  *Data-Agnostic Augmentations for Unknown Variations: Out-of-Distribution Generalisation in MRI Segmentation.*
  In Proceedings of Machine Learning Research, vol. 88, pp. 1–29, 2025.
  Medical Imaging with Deep Learning (MIDL) 2025; [arXiv:2505.10223](https://arxiv.org/abs/2505.10223).
  Repository: <https://github.com/MIAGroupUT/augmentations-for-the-unknown>.
  Pre-trained checkpoints and evaluation data for that work are
  hosted on [Zenodo](https://zenodo.org/records/15517159) — note that
  Zenodo entry belongs to the upstream project, **not** to this fork.

- **Huang, Lyu, Lai, Li.**
  *Nomogram model for predicting invasive placenta in patients with placenta previa: integrating MRI findings and clinical characteristics.*
  Scientific Reports 14:200, 2024. [doi:10.1038/s41598-023-50900-z](https://doi.org/10.1038/s41598-023-50900-z).
  Cohort source paper.

- **Mendeley Data.** The placental MRI dataset used in this work is hosted at
  <https://data.mendeley.com/datasets/284gwmf5bh/1>.

If you use this code, please cite this repository's paper (Šojo et al., ELMAR 2026 — see [CITATION.cff](CITATION.cff)), along with the upstream MIDL 2025 paper (Vaish et al. 2025) and the cohort source paper (Huang et al. 2024). The ELMAR 2026 paper is accepted and awaiting publication in IEEE Xplore; the DOI will be added to the citation file once available.

---

## Setup

### Hardware used to produce the paper numbers

- NVIDIA RTX 5080 (16 GB)
- AMD Ryzen 9 7900X (12 cores)
- 128 GB DDR5 memory
- Ubuntu 24.04 LTS

A 16 GB GPU is sufficient for `--batch_size 8` with `--amp --compile`. Smaller GPUs may need `--batch_size 4` or `--batch_size 2`.

### Conda environment

Single conda environment for everything. Create it once:

```bash
conda create -n pas_radiomics python=3.10 -y
conda activate pas_radiomics
pip install -r requirements.txt
```

The `torch` and `torchvision` lines in `requirements.txt` are the CUDA-12 builds used to produce the paper numbers; if your hardware needs a different CUDA version, install PyTorch from the matching index at <https://pytorch.org/get-started/locally/> and then re-run `pip install -r requirements.txt` for the rest.

PyRadiomics 3.1+ supports NumPy 2.x. If `pip install pyradiomics` fails on your platform, install from the upstream repository:

```bash
pip install git+https://github.com/AIM-Harvard/pyradiomics.git
```

### Smoke test (optional but recommended)

After install, run:

```bash
python -m unittest tests.test_smoke
```

This imports every augmentation module and runs the paper-audit pipeline on a synthetic CSV pair. Two tests should pass in a few seconds.

---

## Dataset preparation

### Original cohort

The placental MRI cohort is described in Huang et al. 2024 and is hosted on Mendeley Data:

<https://data.mendeley.com/datasets/284gwmf5bh/1>

The Mendeley archive contains JPG slice images of patients with placenta accreta spectrum (PAS) — one JPG per 2D slice, with a separate matching mask file per slice. Imaging inclusion criterion: MRI performed after 28 weeks of gestation. Acquisitions are balanced turbo field-echo (BTFE) and single-shot turbo spin-echo (ssh\_TSE). Once resized for this repository the images are stored as PNGs.

### Layout this repository expects

After preparing the dataset, your `dataset/` folder should look like this:

```
dataset/
└── resized/
    ├── DATASET_BTFE/
    │   ├── images/
    │   │   ├── <patient_id_1>/<patient_id_1>_<slice_idx>.png
    │   │   └── ...
    │   ├── masks/
    │   │   ├── <patient_id_1>/<patient_id_1>_<slice_idx>.png
    │   │   └── ...
    │   └── splits.json
    └── DATASET_SSH_TSE/
        ├── images/<patient_id>/...
        ├── masks/<patient_id>/...
        └── splits.json
```

- All images: 512 × 512 uint8 PNGs.
- Implied in-plane spacing: 0.703 mm (36 cm field of view ÷ 512).
- Slice spacing: 5.6 mm (5 mm thickness + 0.6 mm inter-slice gap).
- `splits.json` defines `train`/`val`/`test` patient lists per modality.

### Steps to prepare the dataset from the Mendeley archive

The interactive script `pas_preprocessing_toolkit.py` performs every step in this section. Launch it from the repository root:

```bash
python pas_preprocessing_toolkit.py
```

It opens a four-option menu:

1. Manage/Preprocess Dataset (scan, pad/crop, resize, binarise masks)
2. Create Patient Splits (train/val/test with optional slice balancing)
3. Generate Overlays (red mask over grayscale image)
4. Merge Datasets (combine BTFE and ssh\_TSE into a COMBINED dataset)

The walkthrough below uses modules 1, 2, and 3 to reproduce the paper layout. Module 4 is not used — the paper analyses single-modality models only. The toolkit prompts for every parameter interactively; the answers below are exactly what we used.

#### Step A — Download the cohort

Download the JPG image and mask archive from [Mendeley Data](https://data.mendeley.com/datasets/284gwmf5bh/1). You should obtain one folder of slice JPGs per modality (BTFE and ssh\_TSE). Each modality folder contains one JPG per slice; mask files are RGB JPGs in which the placenta region is drawn in red.

#### Step B — Preprocess each modality (toolkit option 1)

Run module 1 **once with the BTFE archive and once with the ssh\_TSE archive**. The settings we used (defaults unless noted otherwise):

| Prompt | Answer |
|---|---|
| Path to IMAGES root folder | `<modality images folder>` |
| Path to MASKS root folder | `<modality masks folder>` |
| What do you want to do? | `Preprocess and write output` |
| Filename regex on STEM | `^(?P<pid>.+)_(?P<sid>\d+)$` (default) |
| Allow fallback patient-id extraction? | `n` |
| Preferred mask extension | `.png` |
| Pairs to open for mode/size stats | `0` (open all pairs) |
| Estimate empty-mask rate & foreground stats? | `y` (optional sanity report) |
| Mask binarisation method (sanity) | `red` |
| R threshold (R ≥ ?) | `1` |
| G max (G ≤ ?) | `20` |
| B max (B ≤ ?) | `20` |
| Output folder | `dataset/resized/DATASET_BTFE` (or `dataset/resized/DATASET_SSH_TSE`) |
| Organisation | `patient` |
| Overwrite existing? | `n` |
| Dry-run? | `n` |
| Convert images to grayscale? | `y` |
| Convert masks via | `red` |
| Mask red-channel threshold (R ≥ ?) | `1` (default) |
| Mask max green (G ≤ ?) | `20` (default) |
| Mask max blue (B ≤ ?) | `20` (default) |
| Write masks as | `01` |
| Padding/cropping | `none` |
| Also RESIZE after pad/crop? | `y` |
| Resize height | `512` |
| Resize width | `512` |
| Upscaling | `bicubic` |
| Downscaling | `bicubic` (not exercised — every slice is upscaled to 512 × 512, never downscaled) |
| Output format for images | `png` |
| Use parallel processing? | `y` |

Each run writes the per-patient folder layout the training script expects:

```
dataset/resized/DATASET_BTFE/
    images/<patient_id>/<patient_id>_<slice_idx>.png
    masks/<patient_id>/<patient_id>_<slice_idx>.png
```

It also writes `scan_*.json` and `scan_*.csv` reports alongside the output folder, including the empty-mask rate and the foreground-ratio statistics.

#### Step C — (Optional) QA overlays before training (toolkit option 3)

We ran module 3 immediately after preprocessing each modality, to visually confirm that the binarised masks line up with the resized images:

| Prompt | Answer |
|---|---|
| Choose mode | `Batch (folder)` |
| Red boost intensity | `120` (default) |
| Images root folder | `dataset/resized/DATASET_BTFE/images` |
| Masks root folder | `dataset/resized/DATASET_BTFE/masks` |
| Output folder | `dataset/resized/DATASET_BTFE/overlays` (default) |
| Use parallel processing? | `y` |
| Preserve subfolder structure? | `y` |

The output is a per-patient folder of red-on-grayscale PNG overlays, purely diagnostic. It is not read by any other step.

#### Step D — Drop unpaired patients (manual)

Keep only patients that appear in **both** modality folders under the same patient identifier. The toolkit operates on one modality at a time and does not enforce this. For our cohort the result is **130 patients per modality**.

A one-liner to compare the two `images/` folders:

```bash
diff <(ls dataset/resized/DATASET_BTFE/images) \
     <(ls dataset/resized/DATASET_SSH_TSE/images)
```

Remove (or move out) any patient subfolder that is missing in either modality, in both `images/` and `masks/`, before generating the splits.

#### Step E — Generate `splits.json` per modality (toolkit option 2)

Run module 2 once for each modality. The settings we used:

| Prompt | Answer |
|---|---|
| Dataset root folder | `dataset/resized/DATASET_BTFE` (or `…SSH_TSE`) |
| Images folder | `images` (default) |
| Train fraction | `0.70` |
| Val fraction | `0.15` |
| Split mode | `patient (each patient counts equally)` |
| Random seed | `42` |
| Output folder | `<dataset root>` (default) |

The script writes `splits.json` next to `images/` with `train` / `val` / `test` patient lists plus a `diagnostics` block. For our cohort the result is a **91 / 20 / 19** patient split per modality. The on-disk format is:

```json
{
  "seed": 42,
  "mode": "patient",
  "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
  "train": ["<patient_id>", ...],
  "val":   ["<patient_id>", ...],
  "test":  ["<patient_id>", ...],
  "diagnostics": { ... }
}
```

How the splits are used at training time — the bash scripts pass `--cv_folds 1`, which is the *single random split* mode. In this mode the training script combines `train` + `val` from `splits.json`, shuffles the combined list with the seed, then partitions it into five equal pieces and uses four pieces for training and one piece for validation (a single 80 / 20 random fold over the 70 + 15 = 85 % patients). The `test` list from `splits.json` is held out and used unchanged for every inference run. So `splits.json` deterministically fixes both the held-out test set and (via the seed) the train / validation re-split.

#### Step F — (Optional, not used in the paper) Build `DATASET_COMBINED/`

Module 4 of the toolkit (Merge Datasets) concatenates BTFE and ssh\_TSE into a single `DATASET_COMBINED/` by prefixing every patient identifier (e.g. `btfe_`, `tse_`) and merging the per-modality `splits.json` files. The paper does *not* analyse combined-cohort models; the shipped `run_augmentation_training.sh` keeps that variant as an opt-in only because the upstream MIDL 2025 work uses it.

`dataset/` is gitignored and never committed.

---

## Repository layout

| Path | Role |
|---|---|
| `pas_preprocessing_toolkit.py` | Interactive dataset-preparation toolkit (preprocess, splits, overlays, merge). Used to build `dataset/resized/DATASET_BTFE` and `…SSH_TSE` from the raw Mendeley archive — see "Dataset preparation" above. |
| `train_placenta_2d_monai_v8.py` | Main training and inference entry point (MONAI). |
| `augmentations/` | MixUp, CutMix, AFA, and Dual Instance-Batch Normalisation modules (AFA + Dual-Norm derived from MIAGroupUT; MixUp/CutMix from nnU-Net — see [NOTICE](NOTICE)). |
| `corrupt_test_set.py` | Generates corrupted test sets at severity level 3 (5 corruptions × 2D / 3D modes). |
| `compute_radiomics.py` | Extracts 102 PyRadiomics features per (patient, mask). |
| `compare_radiomics.py` | Aggregates per-feature errors into the comparison CSVs. |
| `compare_test_metrics.py` | Dice / IoU / HD95 dashboard across all runs. |
| `extract_all_metrics.py` | Consolidates per-cell `test_metrics.json` files into `shareable/all_test_metrics.csv`. |
| `extract_per_patient_metrics.py` | Recomputes volumetric Dice per patient from saved prediction PNGs and writes `shareable/per_patient_dice.csv`. Required for the paper's four-quadrant Table I and the Figure 1 box plot. |
| `extract_best_worst_overlays.py` | Per-patient best/worst overlay extraction. |
| `run_augmentation_training.sh` | Trains 12 paper models + optional COMBINED variants. |
| `run_augmentation_inference.sh` | Same-domain + cross-domain clean inference for all 12 models. |
| `run_corrupted_inference.sh` | Every model × every corruption × {2D, 3D} test variant. |
| `paper/audit_numbers.py` | Re-derives every numeric claim of the paper from the CSVs (4-quadrant Table I, correlations, catastrophic-failure rates) and writes `paper/audit_3d_results.json`. |
| `paper/make_figures_3d.py` | Regenerates the paper's two figures: Figure 1 (per-patient median feature-error box plot) and Figure 2 (Dice-vs-error scatter). |
| `tests/test_smoke.py` | Imports + audit-pipeline smoke test. |
| `requirements.txt` | Pinned package versions. |
| `CITATION.cff` | This paper's citation (ELMAR 2026, DOI pending) + upstream MIDL 2025 and cohort references. |
| `LICENSE` | Apache 2.0 (same as upstream). |
| `NOTICE` | Attribution for the Apache-2.0 code derived from MIAGroupUT (AFA + Dual-Norm) and nnU-Net (MixUp/CutMix). |

Generated at runtime and gitignored: `dataset/`, `runs/`, `log/`, `radiomics/`, `shareable/`, `comparison_results/`, `paper/figures/`, `paper/audit_3d_results.json`.

---

## End-to-end reproduction

After the dataset is in place, run the steps below in order. Each step states what it consumes, what it produces, and the exact command. All paths are relative to the repository root.

### Step 1 — Train every UNet++ variant

**Consumes:** `dataset/resized/DATASET_BTFE/`, `dataset/resized/DATASET_SSH_TSE/` (plus `splits.json` per modality).
**Produces:** `runs/<MODEL>/fold_0/best_model.pth` and per-epoch logs.

```bash
bash run_augmentation_training.sh
```

This trains 12 models — six augmentations (Baseline, MixUp, CutMix, AFA, MixUp+AFA, CutMix+AFA) × two training modalities (BTFE, ssh\_TSE). Each run uses one fold, the train/val/test split from `splits.json`, AdamW with `lr=1e-3` and `weight_decay=1e-5`, a `ReduceLROnPlateau` schedule (factor 0.5, patience 10), early stopping (patience 25), batch size 8, mixed precision, and `torch.compile`. The standard MONAI augmentation pipeline (random horizontal/vertical flips, affine, 2D elastic deformation, zoom, contrast adjustment, Gaussian noise, Gaussian smoothing) is applied to every run. AFA-enabled runs additionally apply random planar sinusoidal-wave injection, with channels split between instance normalisation and a dual batch normalisation that maintains independent running statistics for the AFA-perturbed and the clean batches.

### Step 2 — Same-domain and cross-domain inference

**Consumes:** trained checkpoints from step 1.
**Produces:** `runs/<MODEL>/test_on_<DOMAIN>/fold_0/{test_metrics.json, inference_raw_masks/, inference_overlays/}`.

```bash
bash run_augmentation_inference.sh
```

For each of the 12 models, runs inference on both clean BTFE and clean ssh\_TSE test sets. Models trained with AFA pass `--use_afa` at test time so the dual-norm routing remains active.

### Step 3 — Generate corrupted test sets at severity level 3

**Consumes:** `dataset/resized/DATASET_BTFE/`, `dataset/resized/DATASET_SSH_TSE/`.
**Produces:** `dataset/resized/DATASET_BTFE_<corruption>_s3_<mode>/` and `DATASET_SSH_TSE_<corruption>_s3_<mode>/` for each of five corruptions and both `2d` and `3d` modes (20 corrupted datasets total).

```bash
python corrupt_test_set.py
```

Severity-3 parameters (taken from the upstream MIAGroupUT corruption library):

- **Bias field**: `RandomBiasField(coefficients=0.9, order=3)` (torchio).
- **Ghosting**: 7 ghosts, intensity 1.5, axes (0, 1).
- **k-space subsampling**: centre fraction 0.08, acceleration factor 4 (Cartesian).
- **Rician noise**: relative standard deviation 0.48.
- **Spike noise**: 2 spikes in k-space, intensity range (1.0, 1.5).

`2d` mode applies the corruption per-slice with an independent random seed; `3d` mode stacks the patient's slices into a pseudo-volume and corrupts once with a single seed. Masks are copied unchanged so the ground-truth geometry is identical across all conditions. **The paper analyses use the `3d` mode only**, because volume-level corruption is more clinically realistic; the `2d` mode is generated alongside it and kept on disk as a reference for future work.

### Step 4 — Inference on every corrupted test set

**Consumes:** trained checkpoints + corrupted datasets.
**Produces:** `runs/<MODEL>/test_on_<BASE_DOMAIN>_<corruption>_s3_<mode>/fold_0/...` for every (model, base modality, corruption, mode) cell.

```bash
bash run_corrupted_inference.sh
```

The script iterates over a 240-cell grid. Each factor is:

- **12 trained models** = 6 augmentations (Baseline, MixUp, CutMix, AFA, MixUp+AFA, CutMix+AFA) × 2 *training* modalities (BTFE, ssh\_TSE).
- **× 2 base modalities** = each of the 12 models is evaluated on the corrupted BTFE test set *and* on the corrupted ssh\_TSE test set. For a BTFE-trained model, BTFE is same-domain (ID) and ssh\_TSE is cross-domain (OOD); the reverse for a ssh\_TSE-trained model.
- **× 5 corruptions** = bias field, ghosting, k-space subsampling, Rician noise, spike noise.
- **× 2 modes** = `2d` and `3d` corruption modes.
- **= 240 inference runs.**

The paper analyses only the 120 cells from the `3d` mode; the `2d` mode cells are written to disk and kept as a reference for future work. Idempotent: re-running just rewrites the same `out_dir`.

### Step 5 — Consolidate cell-level and per-patient metrics

This step produces the two CSVs that the audit (Step 9) and the figures (Step 10) read.

**5a — Cell-level metrics.**

**Consumes:** every `runs/<MODEL>/test_on_<VARIANT>/fold_0/test_metrics.json`.
**Produces:** `shareable/all_test_metrics.csv` (one row per (model, variant) test condition).

```bash
python extract_all_metrics.py
```

**5b — Per-patient Dice (required for the 4-quadrant Table I and the Figure 1 box plot).**

**Consumes:** the saved prediction PNGs under `runs/<MODEL>/test_on_<VARIANT>/fold_0/inference_raw_masks/<patient>/...png` and the ground-truth masks under `dataset/resized/DATASET_<modality>/masks/<patient>/...png`.
**Produces:** `shareable/per_patient_dice.csv` (5,206 rows = 274 (model × variant) cells × 19 test patients) by recomputing volumetric Dice per patient.

```bash
python extract_per_patient_metrics.py --workers 12
```

This script must be run before Steps 9 and 10. Skipping it leaves the audit unable to compute the per-patient [Q1, Q3] interquartile ranges in the paper's Table I.

### Step 6 — Extract PyRadiomics features (both 2D and 3D)

**Consumes:** ground-truth masks + every model's predicted masks (clean and corrupted).
**Produces:** `radiomics/features/{gt,pred}/2d/<VARIANT>/<patient>.json` and `radiomics/features/{gt,pred}/3d/<VARIANT>/<patient>.json` — 102 features per patient in each mode. Ground-truth features are cached and reused across model variants.

```bash
python compute_radiomics.py --mode both --workers 8
```

The `2d` extraction computes per-slice radiomic features with 2D shape descriptors (sphericity in 2D, perimeter, area, etc.) and aggregates per patient. The `3d` extraction stacks the patient's slices into a pseudo-volume and computes 3D radiomic features (3D shape, 3D texture matrices) once per patient. The paper analyses use the **2D extraction** (the segmentation model is itself 2D UNet++, operating on individual slices), and only filter the inference results to the **3D corruption mode** because volume-level corruption is more clinically realistic than per-slice corruption. The 3D radiomic extraction is computed for completeness and as a reference for future work, but is not analysed in the paper.

### Step 7 — Aggregate radiomic feature errors (both 2D and 3D)

**Consumes:** the JSONs from step 6.
**Produces:** `radiomics/comparisons/2d/...` and `radiomics/comparisons/3d/...`. Each folder contains `per_corruption_summary.csv` (mean and median relative feature error per (model, base modality, corruption, mode) cell), `summary_by_feature.csv` (one row per feature), and a heat-map dashboard.

```bash
python compare_radiomics.py --mode both
```

The per-feature relative error is

```
e_f = |f_pred - f_gt| / (|f_gt| + 1e-8)
```

Five features (in the 2D extraction) have ground-truth values close to zero (first-order minimum and skewness, GLCM cluster shade and cluster prominence, NGTDM coarseness) and inflate the mean across features by orders of magnitude even when the underlying segmentation is reasonable. The pipeline reports both the median (typical-robustness readout) and the mean (catastrophic-failure indicator).

The paper's audit script (Step 9 below) reads from `radiomics/comparisons/2d/per_corruption_summary.csv`.

### Step 8 — (Optional) Best- and worst-patient overlays

**Consumes:** prediction PNGs from steps 2 and 4.
**Produces:** `shareable/best_worst_summary.csv` and per-patient overlay PNGs.

```bash
python extract_best_worst_overlays.py
```

### Step 9 — Re-derive every numeric claim of the paper

**Consumes:** `shareable/all_test_metrics.csv`, `shareable/per_patient_dice.csv`, `radiomics/comparisons/2d/per_corruption_summary.csv`, and `radiomics/comparisons/2d/per_patient_errors.csv`.
**Produces:** `paper/audit_3d_results.json` plus a printed audit on stdout.

```bash
python paper/audit_numbers.py
```

Filters to the 12 augmentation models analysed in the paper and to `corruption_mode in {3d, clean}`. The output JSON contains the scope (144 test conditions: 24 clean + 120 corrupted), Pearson and Spearman correlations between Dice and the median / mean relative feature error, the per-augmentation catastrophic-failure rate, the per-corruption ranking, and — under the key `four_quadrant_table_with_iqr` — the per-patient values that back the paper's four-quadrant Table I (mean Dice, median e_f and their [Q1, Q3] IQR over 38 per-patient values per clean row and 190 per-patient values per corrupted row, plus group-mean rows). Every numeric value cited in the paper is one entry in this JSON.

### Step 10 — Regenerate the paper figures

**Consumes:** the same four CSVs as step 9.
**Produces:** `paper/figures/figure1.png` (two-panel grouped box plot of per-patient median feature error: same-domain on top, OOD on bottom; 5 corruption columns × 6 augmentation boxes per panel, each box summarising 38 per-patient values) and `paper/figures/figure2.png` (Dice-vs-error two-panel scatter).

```bash
python paper/make_figures_3d.py
```

---

## Key findings (paper Table I and Figure 2)

**Table I — augmentation summary across four (regime, condition) row groups.** Mean Dice and median feature error $e_f$ are followed by the [Q1, Q3] interquartile range computed over the per-patient values that contribute to that row: 38 values for clean rows (19 test patients × 2 trained models per augmentation, BTFE-trained and ssh\_TSE-trained pooled) and 190 for corrupted rows (19 × 5 corruptions × 2). Mean $e_f$ is the average over the same population. Catastr. counts the (model, base modality, corruption) test conditions in the row whose mean $e_f$ exceeds 1. Best value in each column of each row group is in **bold**.

Each quadrant ends with an italic *Group mean* row: the mean across the 6 augmentation rows of that quadrant. The Catastr. denominator on the *Group mean* row is the per-quadrant total (12 for clean rows = 6 × 2; 60 for corrupted rows = 6 × 10).

**(A) Clean / Same-domain (ID).**

| Augmentation | Mean Dice [Q1, Q3] | Median e_f [Q1, Q3] | Mean e_f | Catastr. |
|---|---|---|---|---|
| Baseline    | 0.884 [0.858, 0.915] | 0.025 [0.021, 0.035] | 0.100 | 0 / 2 |
| MixUp       | 0.887 [0.862, 0.919] | 0.025 [0.020, 0.037] | 0.114 | 0 / 2 |
| CutMix      | 0.887 [0.870, 0.921] | 0.027 [0.019, 0.030] | **0.089** | 0 / 2 |
| AFA         | 0.885 [0.865, 0.915] | 0.028 [0.021, 0.035] | 0.099 | 0 / 2 |
| CutMix+AFA  | **0.888** [0.870, 0.923] | 0.027 [0.017, 0.035] | 0.095 | 0 / 2 |
| MixUp+AFA   | **0.888** [0.869, 0.919] | **0.024** [0.019, 0.032] | 0.117 | 0 / 2 |
| *Group mean* | *0.887* | *0.026* | *0.102* | *0 / 12* |

**(B) Clean / Out-of-distribution (OOD).**

| Augmentation | Mean Dice [Q1, Q3] | Median e_f [Q1, Q3] | Mean e_f | Catastr. |
|---|---|---|---|---|
| Baseline    | 0.774 [0.728, 0.850] | **0.055** [0.035, 0.086] | **0.158** | 0 / 2 |
| MixUp       | 0.771 [0.733, 0.840] | 0.063 [0.046, 0.083] | 0.162 | 0 / 2 |
| CutMix      | 0.740 [0.685, 0.823] | 0.072 [0.047, 0.096] | 0.263 | 0 / 2 |
| AFA         | 0.773 [0.747, 0.845] | 0.061 [0.041, 0.082] | 0.201 | 0 / 2 |
| CutMix+AFA  | 0.738 [0.694, 0.811] | 0.082 [0.059, 0.110] | 0.275 | 0 / 2 |
| MixUp+AFA   | **0.783** [0.732, 0.846] | **0.055** [0.044, 0.071] | 0.177 | 0 / 2 |
| *Group mean* | *0.763* | *0.065* | *0.206* | *0 / 12* |

**(C) Corrupted / Same-domain (ID).**

| Augmentation | Mean Dice [Q1, Q3] | Median e_f [Q1, Q3] | Mean e_f | Catastr. |
|---|---|---|---|---|
| Baseline    | 0.742 [0.682, 0.857] | 0.044 [0.029, 0.079] | 2.8 × 10³ | 1 / 10 |
| MixUp       | 0.748 [0.715, 0.871] | 0.041 [0.027, 0.078] | 0.220 | **0 / 10** |
| CutMix      | 0.717 [0.663, 0.868] | 0.044 [0.027, 0.090] | 1.4 × 10³ | 1 / 10 |
| AFA         | 0.757 [0.714, 0.874] | 0.041 [0.027, 0.072] | 1.4 × 10³ | 1 / 10 |
| CutMix+AFA  | 0.740 [0.723, 0.879] | **0.035** [0.024, 0.071] | 0.191 | **0 / 10** |
| MixUp+AFA   | **0.774** [0.740, 0.877] | **0.035** [0.024, 0.076] | **0.183** | **0 / 10** |
| *Group mean* | *0.746* | *0.040* | *9.3 × 10²* | *3 / 60* |

**(D) Corrupted / Out-of-distribution (OOD).**

| Augmentation | Mean Dice [Q1, Q3] | Median e_f [Q1, Q3] | Mean e_f | Catastr. |
|---|---|---|---|---|
| Baseline    | 0.635 [0.534, 0.763] | 0.083 [0.049, 0.130] | 1.6 × 10³ | 2 / 10 |
| MixUp       | 0.639 [0.519, 0.770] | 0.072 [0.050, 0.130] | 0.291 | **0 / 10** |
| CutMix      | 0.569 [0.461, 0.716] | 0.110 [0.069, 0.169] | 4.4 × 10³ | 3 / 10 |
| AFA         | 0.644 [0.535, 0.776] | 0.071 [0.052, 0.118] | **0.266** | **0 / 10** |
| CutMix+AFA  | 0.593 [0.505, 0.742] | 0.095 [0.061, 0.150] | 1.5 × 10³ | 1 / 10 |
| MixUp+AFA   | **0.671** [0.569, 0.791] | **0.068** [0.042, 0.110] | 0.284 | **0 / 10** |
| *Group mean* | *0.625* | *0.083* | *1.3 × 10³* | *6 / 60* |

**Dice as a stability proxy across the 120 corrupted test conditions:**

- Pearson `r = -0.92` (median feature error) — strong linear association.
- Pearson `r = +0.05` (mean feature error) — no linear association.
- Spearman `ρ = -0.91` (median) and `ρ = -0.69` (mean) — Spearman recovers the monotonic trend in the mean that Pearson hides because of catastrophic outliers.

All numbers above are reproduced exactly by `python paper/audit_numbers.py` and stored in `paper/audit_3d_results.json` (key `four_quadrant_table_with_iqr`) after running the upstream pipeline (steps 1 through 7).

---

## License

**Apache License 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The `augmentations/` modules are derived and modified from Apache-2.0 upstream code (AFA + Dual Instance-Batch Normalisation — MIA Group-UT; MixUp / CutMix — nnU-Net / DKFZ), so this project is distributed under the same permissive license; the NOTICE file records the required attributions.
