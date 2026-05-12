import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevents GUI errors on headless servers
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm 

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import CacheDataset, Dataset
from monai.losses import DiceCELoss
import gc # Add this at the top with your other imports

# --- Data-Agnostic Augmentations (Paper: arXiv 2505.10223) ---
# Ensure the project root is in the path so 'augmentations' package is found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augmentations import (
    RandomMixUpBinary, RandomCutMixBinary, AFA,
    convert_to_dual_norm, set_dual_norm_route
)

# Change your MONAI metrics import to look exactly like this:
from monai.metrics import DiceMetric, compute_hausdorff_distance, compute_average_surface_distance
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, 
    Lambdad, EnsureTyped, RandFlipd, RandAffined, RandZoomd,
    RandAdjustContrastd, RandGaussianNoised, RandGaussianSmoothd
)
from monai.utils import set_determinism

# ---------------------------------------------------------------------------
# 1. Dataset Parsing
# ---------------------------------------------------------------------------
def read_splits(splits_path):
    with open(splits_path, "r") as f:
        return json.load(f)

def build_items(images_dir, masks_dir, patient_ids):
    items = []
    pid_set = set(patient_ids)
    
    for img_path in Path(images_dir).rglob("*.png"):
        patient_id = img_path.parent.name 
        if patient_id in pid_set:
            rel_path = img_path.relative_to(images_dir)
            mask_path = Path(masks_dir) / rel_path
            if mask_path.exists():
                items.append({
                    "image": str(img_path),
                    "label": str(mask_path),
                    "patient_id": patient_id,
                    "slice_id": img_path.stem 
                })
    return items

# ---------------------------------------------------------------------------
# 2. Network Switcher
# ---------------------------------------------------------------------------
def get_network(name):
    # ==========================================
    # 1. THE U-NET FAMILY (CNNs)
    # ==========================================
    if name == "unet":
        from monai.networks.nets import UNet
        return UNet(spatial_dims=2, in_channels=1, out_channels=1, 
                    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2)
                    
    elif name == "attentionunet":
        from monai.networks.nets import AttentionUnet
        return AttentionUnet(spatial_dims=2, in_channels=1, out_channels=1, 
                             channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2))
                             
    elif name == "basicunet":
        from monai.networks.nets import BasicUNet
        # A lightweight, ultra-fast baseline U-Net
        return BasicUNet(spatial_dims=2, in_channels=1, out_channels=1)
        
    elif name == "unetplusplus":
        from monai.networks.nets import BasicUNetPlusPlus
        # Dense, nested skip connections for complex boundaries
        return BasicUNetPlusPlus(spatial_dims=2, in_channels=1, out_channels=1, deep_supervision=False)

    elif name == "flexunet":
        from monai.networks.nets import FlexibleUNet
        # Uses an EfficientNet backbone (highly memory efficient and accurate)
        return FlexibleUNet(in_channels=1, out_channels=1, backbone="efficientnet-b2", spatial_dims=2)
    
    elif name == "dynunet":
        from monai.networks.nets import DynUNet
        
        # 1. THE MATH FOR 512x512 IMAGES:
        # We need 7 levels. The first level stays at 512.
        # The next 6 levels halve the size: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        
        # Kernel size is always 3x3 for every block
        kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        
        # Stride of 1 keeps the size the same. Stride of 2 cuts it in half.
        strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
        
        # Upsampling kernels must perfectly match the downsampling strides
        upsample_kernels = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
        
        # The number of channels learned at each depth level
        filters = [32, 64, 128, 256, 512, 1024, 1024]

        return DynUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=upsample_kernels,
            filters=filters,
            dropout=0.1,             # Adds a 10% dropout to prevent memorization
            res_block=True,          # Turns on Residual connections (nnU-Net style)
            deep_supervision=False   # Keep False to ensure it only outputs 1 final mask
        )

    # ==========================================
    # 2. THE RESIDUAL & HIGH-RES FAMILY
    # ==========================================
    elif name == "segresnet":
        from monai.networks.nets import SegResNet
        return SegResNet(spatial_dims=2, in_channels=1, out_channels=1, 
                         init_filters=32, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))

    elif name == "vnet":
        from monai.networks.nets import VNet
        # Replaces standard convolutions with continuous residual blocks
        return VNet(spatial_dims=2, in_channels=1, out_channels=1)

    elif name == "highresnet":
        from monai.networks.nets import HighResNet
        # Does not downsample the image heavily; maintains high resolution throughout
        return HighResNet(spatial_dims=2, in_channels=1, out_channels=1)

    # ==========================================
    # 3. THE VISION TRANSFORMERS (ViTs)
    # ==========================================
    elif name == "unetr":
        from monai.networks.nets import UNETR
        # Standard Vision Transformer with a CNN decoder
        return UNETR(in_channels=1, out_channels=1, img_size=(512, 512), spatial_dims=2)

    elif name == "swinunetr":
        from monai.networks.nets import SwinUNETR
        # Hierarchical 'Shifted Window' Transformer
        return SwinUNETR(img_size=(512, 512), in_channels=1, out_channels=1, feature_size=24, spatial_dims=2)

    else:
        raise ValueError(f"Unknown network: {name}")

# ---------------------------------------------------------------------------
# 3. Patient-Total Comprehensive Metrics Tracker
# ---------------------------------------------------------------------------
class PatientMetricsTracker:
    def __init__(self):
        self.tp, self.fp, self.fn, self.tn = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
        self.hd95_vals = defaultdict(list)
        self.msd_vals = defaultdict(list)
        # We completely deleted the MONAI metric objects here!

    def reset(self):
        self.tp.clear(); self.fp.clear(); self.fn.clear(); self.tn.clear()
        self.hd95_vals.clear(); self.msd_vals.clear()

    @torch.no_grad()
    def update(self, preds, labels, patient_ids):
        # 1. Stateless functional boundary distances (No memory buffers!)
        hd = compute_hausdorff_distance(y_pred=preds, y=labels, include_background=True, percentile=95)
        msd = compute_average_surface_distance(y_pred=preds, y=labels, include_background=True)

        # 2. Binarize for pixel-level counting
        preds_bin = (preds > 0.5).int()
        labels_bin = (labels > 0.5).int()

        for i, pid in enumerate(patient_ids):
            p = preds_bin[i].flatten()
            l = labels_bin[i].flatten()

            self.tp[pid] += torch.sum((p == 1) & (l == 1)).item()
            self.fp[pid] += torch.sum((p == 1) & (l == 0)).item()
            self.fn[pid] += torch.sum((p == 0) & (l == 1)).item()
            self.tn[pid] += torch.sum((p == 0) & (l == 0)).item()

            if not torch.isnan(hd[i][0]) and not torch.isinf(hd[i][0]):
                self.hd95_vals[pid].append(hd[i][0].item())
            if not torch.isnan(msd[i][0]) and not torch.isinf(msd[i][0]):
                self.msd_vals[pid].append(msd[i][0].item())

    def compute(self):
        if not self.tp: return {}
        res = {}
        for pid in self.tp.keys():
            tp, fp, fn, tn = self.tp[pid], self.fp[pid], self.fn[pid], self.tn[pid]
            
            # Prevent division by zero
            denom_dice = (2.0 * tp + fp + fn)
            denom_iou = (tp + fp + fn)
            denom_sens = (tp + fn)
            denom_prec = (tp + fp)
            denom_spec = (tn + fp)

            dice = (2.0 * tp / denom_dice) if denom_dice > 0 else 1.0
            iou = (tp / denom_iou) if denom_iou > 0 else 1.0
            sens = (tp / denom_sens) if denom_sens > 0 else 1.0
            prec = (tp / denom_prec) if denom_prec > 0 else 1.0
            spec = (tn / denom_spec) if denom_spec > 0 else 1.0

            hd95_mean = np.mean(self.hd95_vals[pid]) if self.hd95_vals[pid] else 0.0
            msd_mean = np.mean(self.msd_vals[pid]) if self.msd_vals[pid] else 0.0

            res[pid] = {"dice": dice, "iou": iou, "sens": sens, "prec": prec, "spec": spec, "hd95": hd95_mean, "msd": msd_mean}

        # Average all patient scores into final dataset-wide numbers
        metrics = ["dice", "iou", "sens", "prec", "spec", "hd95", "msd"]
        avg_res = {k: float(np.mean([res[pid][k] for pid in res.keys()])) for k in metrics}
        return avg_res

# ---------------------------------------------------------------------------
# 4. Inference Savers (Visual Overlay + Raw Mask)
# ---------------------------------------------------------------------------
def save_inference_outputs(image, label, pred, patient_id, slice_id, out_dir):
    """Saves 4-panel visual overlays, error maps, and raw binary masks."""
    out_dir = Path(out_dir)
    
    img_np = image[0, 0].cpu().numpy().T
    lbl_np = label[0, 0].cpu().numpy().T
    pred_np = pred[0, 0].cpu().numpy().T
    
    # --- 1. Save Raw Binary Mask ---
    raw_dir = out_dir / "inference_raw_masks" / patient_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_img = (pred_np * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(raw_dir / f"{slice_id}_pred.png")

    # --- 2. Generate Color-Coded Error Map ---
    H, W = pred_np.shape
    error_map = np.zeros((H, W, 3), dtype=np.uint8)
    error_map[(pred_np == 1) & (lbl_np == 1)] = [0, 255, 0]   # True Positive -> Green
    error_map[(pred_np == 1) & (lbl_np == 0)] = [255, 0, 0]   # False Positive -> Red (Hallucination)
    error_map[(pred_np == 0) & (lbl_np == 1)] = [0, 0, 255]   # False Negative -> Blue (Missed)
    
    # --- 3. Slice-Level Metric Stamp ---
    tp = np.sum((pred_np == 1) & (lbl_np == 1))
    fp = np.sum((pred_np == 1) & (lbl_np == 0))
    fn = np.sum((pred_np == 0) & (lbl_np == 1))
    s_dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    s_sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    s_prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    title_stamp = f"Patient: {patient_id} | Slice: {slice_id}\nDice: {s_dice:.2f} | Sens: {s_sens:.2f} | Prec: {s_prec:.2f}"

    # --- 4. Plot 4-Panel Figure ---
    overlay_dir = out_dir / "inference_overlays" / patient_id
    overlay_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title_stamp, fontsize=14, fontweight='bold')
    
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("1. Input MRI")
    axes[0].axis("off")
    
    axes[1].imshow(img_np, cmap="gray")
    axes[1].imshow(lbl_np, cmap="Greens", alpha=0.4)
    axes[1].set_title("2. Ground Truth Mask")
    axes[1].axis("off")
    
    axes[2].imshow(img_np, cmap="gray")
    axes[2].imshow(pred_np, cmap="Reds", alpha=0.4)
    axes[2].set_title("3. Predicted Mask")
    axes[2].axis("off")

    axes[3].imshow(error_map)
    axes[3].set_title("4. Error Map (G=TP, R=FP, B=FN)")
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(overlay_dir / f"{slice_id}_overlay.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Main Script
# ---------------------------------------------------------------------------
def main(args):
    set_determinism(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable cuDNN benchmark for extra speed (especially good for static 512x512 images)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    splits = read_splits(args.dataset_root + "/splits.json")
    
    pre_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader="PILReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]), 
        Lambdad(keys=["label"], func=lambda x: (x > 0).astype(np.float32)), 
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])

    # aug_transforms = Compose([
    #     RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    #     RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    #     RandAffined(keys=["image", "label"], prob=0.5, rotate_range=0.15, translate_range=10, scale_range=0.1),
    #     RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.95, max_zoom=1.05),
    #     RandAdjustContrastd(keys=["image"], prob=0.3),
    #     RandGaussianNoised(keys=["image"], prob=0.15),
    #     RandGaussianSmoothd(keys=["image"], prob=0.1),
    # ]) if args.use_augmentation else Compose([])
    
    aug_transforms = Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        # Increased rotation to ~17 degrees (0.3 rad) for better angle robustness
        RandAffined(keys=["image", "label"], prob=0.5, rotate_range=0.3, translate_range=15, scale_range=0.15),
        # ADDED: Elastic deformation (The gold standard for soft tissue like placentas!)
        monai.transforms.Rand2DElasticd(keys=["image", "label"], prob=0.3, spacing=(20, 20), magnitude_range=(1, 2)),
        RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.90, max_zoom=1.10),
        RandAdjustContrastd(keys=["image"], prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.15),
        RandGaussianSmoothd(keys=["image"], prob=0.1),
    ]) if args.use_augmentation else Compose([])

    # # --- CROSS-VALIDATION LOGIC ---
    # if args.cv_folds > 1:
    #     # Combine train and val into one giant pool, sort to ensure consistency
    #     all_patients = sorted(splits["train"] + splits["val"])
    #     rng = np.random.default_rng(args.seed)
    #     rng.shuffle(all_patients)
    #     # Split into pieces (folds)
    #     patient_folds = np.array_split(all_patients, args.cv_folds)
    #     num_runs = args.cv_folds
    # else:
    #     num_runs = 1 # Runs only once for classic train/val split
    #     patient_folds = None

    # --- CROSS-VALIDATION LOGIC ---
    if args.cv_folds > 0:
        # Combine train and val into one giant pool, sort to ensure consistency
        all_patients = sorted(splits["train"] + splits["val"])
        rng = np.random.default_rng(args.seed)
        rng.shuffle(all_patients)
        
        # 🧠 The Trick: If they want 1 fold, we still must slice the pie into 5 pieces to get a 20% validation exam!
        split_pieces = 5 if args.cv_folds == 1 else args.cv_folds
        patient_folds = np.array_split(all_patients, split_pieces)
        
        num_runs = args.cv_folds # If 1, it runs once. If 5, it runs 5 times.
    else:
        num_runs = 1 # Runs only once for classic train/val split
        patient_folds = None

    # # ==========================================
    # # MODE: TRAIN
    # # ==========================================
    # if args.mode == "train":
    #     for fold in range(num_runs):
    #         if args.cv_folds > 1:
    #             print(f"\n{'='*50}\n🚀 STARTING 5-FOLD CV: FOLD {fold}/{args.cv_folds - 1}\n{'='*50}")
    #             current_out_dir = base_out_dir / f"fold_{fold}"
    #             val_ids = patient_folds[fold].tolist()
    #             train_ids = [p for i, f in enumerate(patient_folds) if i != fold for p in f.tolist()]
    #         else:
    #             print(f"\n{'='*50}\n📚 STARTING CLASSIC TRAIN/VAL SPLIT\n{'='*50}")
    #             current_out_dir = base_out_dir
    #             train_ids = splits["train"]
    #             val_ids = splits["val"]

    # ==========================================
    # MODE: TRAIN
    # ==========================================
    if args.mode == "train":
        for fold in range(num_runs):
            if args.cv_folds > 0:
                print(f"\n{'='*50}\n🚀 STARTING RANDOM SPLIT: FOLD {fold}/{max(0, args.cv_folds - 1)}\n{'='*50}")
                current_out_dir = base_out_dir / f"fold_{fold}"
                val_ids = patient_folds[fold].tolist()
                train_ids = [p for i, f in enumerate(patient_folds) if i != fold for p in f.tolist()]
            else:
                print(f"\n{'='*50}\n📚 STARTING CLASSIC TRAIN/VAL SPLIT\n{'='*50}")
                current_out_dir = base_out_dir
                train_ids = splits["train"]
                val_ids = splits["val"]
                
            current_out_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(current_out_dir / "logs"))
            
            # # --- 1. INITIALIZE NETWORK & COMPILER ---
            # net = get_network(args.network).to(device)
            # if args.compile:
            #     print("⚡ Compiling network with torch.compile() (this may take 1-3 minutes)...")
            #     net = torch.compile(net)

            # --- 1. INITIALIZE NETWORK & COMPILER ---
            net = get_network(args.network).to(device)
            
            # ---> ADD THIS BLOCK FOR TRANSFER LEARNING <---
            if args.checkpoint and Path(args.checkpoint).exists():
                print(f"🔄 TRANSFER LEARNING: Loading pre-trained weights from {args.checkpoint}")
                state_dict = torch.load(args.checkpoint, map_location=device)
                clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                net.load_state_dict(clean_state_dict)
            # ----------------------------------------------

            # --- 2. DUAL NORMALIZATION FOR AFA (Paper: 2505.10223) ---
            if args.use_afa:
                print("🔀 Converting normalization layers to dual-path for AFA...")
                net = convert_to_dual_norm(net)
                net = net.to(device)
            # -------------------------------------------------------

            if args.compile:
                print("⚡ Compiling network with torch.compile() (this may take 1-3 minutes)...")
                net = torch.compile(net)

            loss_function = DiceCELoss(sigmoid=True)
            optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

            # --- 3. INITIALIZE DATA-AGNOSTIC AUGMENTATIONS ---
            mixup_aug = None
            cutmix_aug = None
            afa_aug = None
            if args.use_mixup:
                mixup_aug = RandomMixUpBinary(p=1.0, alpha=args.mixup_alpha)
                print(f"📊 MixUp enabled (alpha={args.mixup_alpha})")
            if args.use_cutmix:
                cutmix_aug = RandomCutMixBinary(p=1.0, alpha=args.cutmix_alpha)
                print(f"✂️  CutMix enabled (alpha={args.cutmix_alpha})")
            if args.use_afa:
                afa_aug = AFA(min_str=args.afa_min_str, mean_str=args.afa_mean_str)
                print(f"🌊 AFA enabled (min_str={args.afa_min_str}, mean_str={args.afa_mean_str})")
            # -------------------------------------------------

            if args.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            elif args.scheduler == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
            else:
                scheduler = None

            metrics_tracker = PatientMetricsTracker()

            train_items = build_items(args.dataset_root + "/images", args.dataset_root + "/masks", train_ids)
            val_items = build_items(args.dataset_root + "/images", args.dataset_root + "/masks", val_ids)
            
            print("Caching datasets (this makes training super fast!)...")
            train_cached_ds = CacheDataset(data=train_items, transform=pre_transforms, cache_rate=1.0)
            val_cached_ds = CacheDataset(data=val_items, transform=pre_transforms, cache_rate=1.0)
            train_ds = Dataset(data=train_cached_ds, transform=aug_transforms)
            
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_cached_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            best_val_patient_dice = 0.0
            epochs_no_improve = 0

            for epoch in range(1, args.epochs + 1):
                print(f"\n--- Fold {fold} | Epoch {epoch}/{args.epochs} ---")
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current Learning Rate: {current_lr}")
                
                # --- TRAIN ---
                net.train()
                if args.use_afa:
                    set_dual_norm_route(net, 'M')  # Start each epoch on main route
                epoch_loss = 0
                for batch in tqdm(train_loader, desc="Training", leave=False):
                    images, labels = batch["image"].to(device), batch["label"].to(device)
                    
                    # --- Apply batch-level augmentations (MixUp or CutMix) ---
                    if mixup_aug is not None:
                        images, labels = mixup_aug(images, labels)
                    elif cutmix_aug is not None:
                        images, labels = cutmix_aug(images, labels)
                    
                    optimizer.zero_grad()
                    
                    if afa_aug is not None:
                        # === AFA DUAL-PATH (memory-efficient sequential backward) ===
                        # Instead of holding BOTH computation graphs in memory,
                        # we backward each path immediately to free its activations.
                        
                        # Path 1: Clean/mixed images → Main BN statistics
                        with torch.amp.autocast('cuda', enabled=args.amp):
                            set_dual_norm_route(net, 'M')
                            outputs = net(images)
                            if isinstance(outputs, (list, tuple)):
                                outputs = outputs[0]
                            loss_clean = loss_function(outputs, labels)
                        
                        # Backward pass 1 — frees the computation graph, saves VRAM
                        scaler.scale(loss_clean * 0.5).backward()
                        
                        # Path 2: AFA-perturbed images → Auxiliary BN statistics
                        with torch.amp.autocast('cuda', enabled=args.amp):
                            set_dual_norm_route(net, 'A')
                            outputs_afa = net(afa_aug(images))
                            if isinstance(outputs_afa, (list, tuple)):
                                outputs_afa = outputs_afa[0]
                            loss_afa = loss_function(outputs_afa, labels)
                        
                        # Backward pass 2 — gradients accumulate with pass 1
                        scaler.scale(loss_afa * 0.5).backward()
                        
                        set_dual_norm_route(net, 'M')
                        loss = 0.5 * (loss_clean.detach() + loss_afa.detach())  # For logging
                        # =============================================================
                    else:
                        # --- Original forward pass (unchanged) ---
                        with torch.amp.autocast('cuda', enabled=args.amp):
                            outputs = net(images)
                            if isinstance(outputs, (list, tuple)):  # Safety net for UNet++
                                outputs = outputs[0]                
                            loss = loss_function(outputs, labels)
                        scaler.scale(loss).backward()
                    
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += loss.item()

                epoch_loss /= len(train_loader)
                writer.add_scalar("Train/Loss", epoch_loss, epoch)

                # --- VALIDATION ---
                net.eval()
                if args.use_afa:
                    set_dual_norm_route(net, 'M')  # Always use main route for validation
                metrics_tracker.reset()
                val_loss = 0

                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validating", leave=False):
                        images, labels, pids = batch["image"].to(device), batch["label"].to(device), batch["patient_id"]
                        
                        with torch.amp.autocast('cuda', enabled=args.amp):
                            outputs = net(images)
                            if isinstance(outputs, (list, tuple)):  # Safety net for UNet++
                                outputs = outputs[0]                
                            loss = loss_function(outputs, labels)
                        val_loss += loss.item()

                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        metrics_tracker.update(preds, labels, pids)
                        
                        # Force Python to destroy these tensors instantly
                        del images, labels, outputs, loss, preds

                val_loss /= len(val_loader)
                val_metrics = metrics_tracker.compute()
                
                val_patient_dice = val_metrics.get("dice", 0.0)

                print(f"Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"Metrics -> Dice: {val_patient_dice:.4f} | IoU: {val_metrics.get('iou', 0.0):.4f} | Sens: {val_metrics.get('sens', 0.0):.4f} | Prec: {val_metrics.get('prec', 0.0):.4f}")
                print(f"Boundary-> HD95: {val_metrics.get('hd95', 0.0):.2f} px | MSD: {val_metrics.get('msd', 0.0):.2f} px")

                writer.add_scalar("Val_Loss/Loss", val_loss, epoch)
                writer.add_scalar("Val_Metrics/Patient_Dice", val_patient_dice, epoch)
                writer.add_scalar("Val_Metrics/Patient_IoU", val_metrics.get("iou", 0.0), epoch)
                writer.add_scalar("Val_Metrics/Patient_Sensitivity", val_metrics.get("sens", 0.0), epoch)
                writer.add_scalar("Val_Metrics/Patient_Precision", val_metrics.get("prec", 0.0), epoch)
                writer.add_scalar("Val_Metrics/Patient_Specificity", val_metrics.get("spec", 0.0), epoch)
                writer.add_scalar("Val_Distances/Patient_HD95_px", val_metrics.get("hd95", 0.0), epoch)
                writer.add_scalar("Val_Distances/Patient_MSD_px", val_metrics.get("msd", 0.0), epoch)

                # --- CHECKPOINTING & EARLY STOPPING ---
                if val_patient_dice > best_val_patient_dice:
                    best_val_patient_dice = val_patient_dice
                    epochs_no_improve = 0
                    torch.save(net.state_dict(), current_out_dir / "best_model.pth")
                    print("🌟 New best model saved!")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epochs.")

                if args.scheduler == "cosine":
                    scheduler.step()
                elif args.scheduler == "plateau":
                    scheduler.step(val_patient_dice)

                if args.early_stopping and epochs_no_improve >= args.patience:
                    print(f"🛑 Early stopping triggered after {epoch} epochs.")
                    break
                    
                # Nuke the PyTorch memory cache!
                gc.collect()
                torch.cuda.empty_cache()

            print(f"\n✅ Fold {fold} training complete!")
            
            # --- VRAM WIPE BETWEEN FOLDS ---
            del net, optimizer, scheduler, train_loader, val_loader, train_cached_ds, val_cached_ds
            gc.collect()
            torch.cuda.empty_cache()

    # ==========================================
    # MODE: TEST / INFERENCE
    # ==========================================
    elif args.mode == "test":
        print("\n--- Inference / Testing Phase ---")
        
        # # We loop through the folds again. If CV was 1, it runs once for base_out_dir
        # for fold in range(num_runs):
        #     if args.cv_folds > 1:
        #         print(f"\n{'='*40}\n🧪 TESTING FOLD {fold}\n{'='*40}")
        #         current_out_dir = base_out_dir / f"fold_{fold}"
        #     else:
        #         current_out_dir = base_out_dir

        # We loop through the folds again. If CV was 0, it runs once for base_out_dir
        for fold in range(num_runs):
            if args.cv_folds > 0:
                print(f"\n{'='*40}\n🧪 TESTING FOLD {fold}\n{'='*40}")
                current_out_dir = base_out_dir / f"fold_{fold}"
            else:
                current_out_dir = base_out_dir

            net = get_network(args.network).to(device)
            # (We don't need to compile the network for testing, pure PyTorch is safer for saving PNGs)
            
            # If model was trained with AFA, we must convert to dual norm BEFORE loading weights
            if args.use_afa:
                print("🔀 Converting normalization layers to dual-path for AFA-trained model...")
                net = convert_to_dual_norm(net)
                net = net.to(device)
            
            ckpt_path = args.checkpoint if args.checkpoint else current_out_dir / "best_model.pth"
            if not Path(ckpt_path).exists():
                print(f"⚠️ Cannot find checkpoint at {ckpt_path}. Skipping Fold {fold}.")
                continue
            
            # --- PYTORCH 2.0 COMPILE FIX ---
            state_dict = torch.load(ckpt_path, map_location=device)
            # Automatically strip the '_orig_mod.' prefix if it exists!
            clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            net.load_state_dict(clean_state_dict)
            net.eval()
            if args.use_afa:
                set_dual_norm_route(net, 'M')  # Inference always uses main route
            # -------------------------------

            test_items = build_items(args.dataset_root + "/images", args.dataset_root + "/masks", splits["test"])
            test_cached_ds = CacheDataset(data=test_items, transform=pre_transforms, cache_rate=1.0)
            test_loader = DataLoader(test_cached_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

            metrics_tracker = PatientMetricsTracker()

            print(f"Generating 4-panel overlays for {len(test_items)} slices...")
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Testing & Saving"):
                    images, labels = batch["image"].to(device), batch["label"].to(device)
                    pids, slice_ids = batch["patient_id"], batch["slice_id"]
                    
                    with torch.amp.autocast('cuda', enabled=args.amp):
                        outputs = net(images)
                        if isinstance(outputs, (list, tuple)):  # Safety net for UNet++
                            outputs = outputs[0]                
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    
                    metrics_tracker.update(preds, labels, pids)
                    save_inference_outputs(images, labels, preds, pids[0], slice_ids[0], current_out_dir)

            final_metrics = metrics_tracker.compute()

            print(f"\n✅ Final Test Metrics for Fold {fold} (Patient-Averaged):")
            for k, v in final_metrics.items():
                print(f" - {k.upper()}: {v:.4f}")

            with open(current_out_dir / "test_metrics.json", "w") as f:
                json.dump(final_metrics, f, indent=4)
            print(f"\nAll metrics saved to {current_out_dir / 'test_metrics.json'}")
            print(f"Raw binary masks saved in {current_out_dir / 'inference_raw_masks'}")
            print(f"Visual overlays saved in {current_out_dir / 'inference_overlays'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to your dataset folder")
    parser.add_argument("--out_dir", type=str, default="./runs_placenta", help="Where to save logs/models")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Run mode")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model weights (for test mode)")
    parser.add_argument("--network", type=str, default="unet", choices=[
        "unet", "attentionunet", "basicunet", "unetplusplus", "flexunet", 
        "segresnet", "vnet", "highresnet", "unetr", "swinunetr", "dynunet"
    ])
    
    # --- NEW ADVANCED FLAGS ---
    # parser.add_argument("--cv_folds", type=int, default=1, help="Number of CV folds (e.g. 5). Set to 1 for classic split.")
    parser.add_argument("--cv_folds", type=int, default=0, help="0=Classic Split, 1=Single Random 80/20 Split, >1=Full CV")
    parser.add_argument("--compile", action="store_true", help="Enable PyTorch 2.0 torch.compile() for ~20% faster training")
    # --------------------------
    
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use_augmentation", action="store_true")
    
    # --- DATA-AGNOSTIC AUGMENTATIONS (Paper: arXiv 2505.10223) ---
    # MixUp and CutMix are mutually exclusive (use one or the other)
    aug_group = parser.add_mutually_exclusive_group()
    aug_group.add_argument("--use_mixup", action="store_true",
        help="Enable MixUp augmentation (blends pairs of images and masks)")
    aug_group.add_argument("--use_cutmix", action="store_true",
        help="Enable CutMix augmentation (pastes rectangular patches between samples)")
    parser.add_argument("--use_afa", action="store_true",
        help="Enable Auxiliary Fourier Augmentation with dual batch norm. "
             "Can be combined with --use_mixup or --use_cutmix. "
             "IMPORTANT: If you trained with --use_afa, you MUST also pass it during --mode test")
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
        help="MixUp Beta distribution parameter. Smaller=less mixing. Paper default: 0.2")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0,
        help="CutMix Beta distribution parameter. Controls cut region size. Default: 1.0")
    parser.add_argument("--afa_min_str", type=float, default=10.0,
        help="AFA minimum perturbation strength. Paper default: 10")
    parser.add_argument("--afa_mean_str", type=float, default=20.0,
        help="AFA mean perturbation strength (exponential distribution). Paper default: 20")
    # -------------------------------------------------------------
    
    args = parser.parse_args()
    main(args)

