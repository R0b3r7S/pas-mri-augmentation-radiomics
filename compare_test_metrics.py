#!/usr/bin/env python3
"""
==============================================================================
 CONFERENCE PAPER — Model Comparison Dashboard
 Automatically discovers all test_metrics.json in ./runs/ and generates:
   1. Console comparison tables
   2. CSV export files
   3. LaTeX tables (copy-paste into your paper)
   4. Bar chart visualizations (PDF + PNG)
   5. Radar/spider chart for multi-metric comparison
   6. Heatmap of cross-domain generalization
==============================================================================
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import csv

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
RUNS_DIR = Path(__file__).parent / "runs"
OUTPUT_DIR = Path(__file__).parent / "comparison_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Display-friendly names for the run folders
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

# Group models for analysis
BTFE_TRAINED = [
    "BTFE_unetplusplus_1Fold",
    "BTFE_unetplusplus_mixup",
    "BTFE_unetplusplus_cutmix",
    "BTFE_unetplusplus_afa",
    "BTFE_unetplusplus_mixup_afa",
    "BTFE_unetplusplus_cutmix_afa",
]
TSE_TRAINED = [
    "TSE_unetplusplus_1Fold",
    "TSE_unetplusplus_mixup",
    "TSE_unetplusplus_cutmix",
    "TSE_unetplusplus_afa",
    "TSE_unetplusplus_mixup_afa",
    "TSE_unetplusplus_cutmix_afa",
]
OTHER_MODELS = [
    "COMBINED_unetplusplus_cutmix_afa",
    "COMBINED_unetplusplus_mixup_afa",
    "COMBINED_unetplusplus_1Fold",
    "Transfer_BTFE_to_TSE_1Fold",
    "Transfer_TSE_to_BTFE_1Fold",
]

METRICS_ORDER = ["dice", "iou", "sens", "prec", "spec", "hd95", "msd"]
METRICS_LABELS = {
    "dice": "Dice ↑",
    "iou": "IoU ↑",
    "sens": "Sensitivity ↑",
    "prec": "Precision ↑",
    "spec": "Specificity ↑",
    "hd95": "HD95 (px) ↓",
    "msd": "MSD (px) ↓",
}
# Higher is better for these, lower is better for hd95/msd
HIGHER_IS_BETTER = {"dice", "iou", "sens", "prec", "spec"}

# ============================================================================
# 2. DATA LOADING
# ============================================================================
def discover_metrics(runs_dir):
    """Automatically find all test_metrics.json files."""
    results = {}  # {model_name: {test_domain: {metric: value}}}
    
    for metrics_file in sorted(runs_dir.rglob("test_metrics.json")):
        parts = metrics_file.relative_to(runs_dir).parts
        # Expected: MODEL_NAME / test_on_DOMAIN / fold_X / test_metrics.json
        # or:       MODEL_NAME / test_on_DOMAIN / test_metrics.json (no fold)
        
        model_name = parts[0]
        test_domain = None
        for p in parts:
            if p.startswith("test_on_"):
                test_domain = p.replace("test_on_", "")
                break
        
        if test_domain is None:
            continue
            
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        if model_name not in results:
            results[model_name] = {}
        results[model_name][test_domain] = metrics
    
    return results


# ============================================================================
# 3. CONSOLE OUTPUT
# ============================================================================
def print_table(results, test_domain, model_list=None, title=""):
    """Print a formatted comparison table for a specific test domain."""
    if model_list is None:
        model_list = sorted(results.keys())
    
    # Filter to models that have this test domain
    available = [m for m in model_list if m in results and test_domain in results[m]]
    
    if not available:
        print(f"  No results for test domain: {test_domain}")
        return
    
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    
    # Header
    header = f"{'Model':<28}"
    for metric in METRICS_ORDER:
        header += f" {METRICS_LABELS[metric]:>14}"
    print(header)
    print("-" * 110)
    
    # Find best values
    best_vals = {}
    for metric in METRICS_ORDER:
        vals = [results[m][test_domain].get(metric, float('nan')) for m in available]
        if metric in HIGHER_IS_BETTER:
            best_vals[metric] = max(vals)
        else:
            best_vals[metric] = min(vals)
    
    # Rows
    for model in available:
        name = DISPLAY_NAMES.get(model, model)[:27]
        row = f"{name:<28}"
        for metric in METRICS_ORDER:
            val = results[model][test_domain].get(metric, float('nan'))
            is_best = abs(val - best_vals[metric]) < 1e-6
            marker = " ★" if is_best else "  "
            if metric in ("hd95", "msd"):
                row += f" {val:>10.2f}{marker}"
            else:
                row += f" {val:>10.4f}{marker}"
        print(row)
    
    print("-" * 110)
    print("  ★ = Best value in column")


def print_cross_domain_analysis(results, model_list, train_domain, title=""):
    """Print cross-domain gap analysis."""
    other_domain = "TSE" if train_domain == "BTFE" else "BTFE"
    
    available = [m for m in model_list 
                 if m in results 
                 and train_domain in results[m] 
                 and other_domain in results[m]]
    
    if not available:
        return
    
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"{'Model':<28} {'Same-Domain':>12} {'Cross-Domain':>12} {'Δ Gap':>10} {'Δ vs Baseline':>14}")
    print("-" * 90)
    
    # Get baseline gap
    baseline_key = f"{train_domain}_unetplusplus_1Fold"
    baseline_same = results.get(baseline_key, {}).get(train_domain, {}).get("dice", 0)
    baseline_cross = results.get(baseline_key, {}).get(other_domain, {}).get("dice", 0)
    baseline_gap = baseline_same - baseline_cross
    
    for model in available:
        name = DISPLAY_NAMES.get(model, model)[:27]
        same_dice = results[model][train_domain]["dice"]
        cross_dice = results[model][other_domain]["dice"]
        gap = same_dice - cross_dice
        delta_vs_baseline = cross_dice - baseline_cross
        
        sign = "+" if delta_vs_baseline >= 0 else ""
        print(f"{name:<28} {same_dice:>12.4f} {cross_dice:>12.4f} {gap:>10.4f} {sign}{delta_vs_baseline:>13.4f}")
    
    print("-" * 90)


# ============================================================================
# 4. CSV EXPORT
# ============================================================================
def export_csv(results, output_dir):
    """Export results to CSV files."""
    for test_domain in ["BTFE", "TSE"]:
        csv_path = output_dir / f"comparison_test_on_{test_domain}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"] + [METRICS_LABELS[m] for m in METRICS_ORDER])
            
            for model in sorted(results.keys()):
                if test_domain in results[model]:
                    name = DISPLAY_NAMES.get(model, model)
                    row = [name]
                    for metric in METRICS_ORDER:
                        row.append(f"{results[model][test_domain].get(metric, 0):.4f}")
                    writer.writerow(row)
        print(f"  📄 Saved: {csv_path}")


# ============================================================================
# 5. LATEX TABLE EXPORT
# ============================================================================
def export_latex(results, output_dir):
    """Generate publication-ready LaTeX tables."""
    
    for test_domain in ["BTFE", "TSE"]:
        latex_path = output_dir / f"latex_table_test_on_{test_domain}.tex"
        
        available = sorted([m for m in results if test_domain in results[m]])
        
        # Find best values
        best_vals = {}
        for metric in METRICS_ORDER:
            vals = [results[m][test_domain].get(metric, float('nan')) for m in available]
            if metric in HIGHER_IS_BETTER:
                best_vals[metric] = max(vals)
            else:
                best_vals[metric] = min(vals)
        
        with open(latex_path, "w") as f:
            # Compact metrics for paper: Dice, IoU, Sensitivity, Precision, HD95
            paper_metrics = ["dice", "iou", "sens", "prec", "hd95", "msd"]
            col_spec = "l" + "c" * len(paper_metrics)
            
            f.write(f"% Auto-generated LaTeX table — Test on {test_domain}\n")
            f.write(f"\\begin{{table}}[htbp]\n")
            f.write(f"\\centering\n")
            f.write(f"\\caption{{Segmentation performance on {test_domain} test set (patient-averaged).}}\n")
            f.write(f"\\label{{tab:test_{test_domain.lower()}}}\n")
            f.write(f"\\resizebox{{\\textwidth}}{{!}}{{\n")
            f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
            f.write(f"\\toprule\n")
            
            # Header row
            header_parts = ["Model"]
            for m in paper_metrics:
                arrow = "$\\uparrow$" if m in HIGHER_IS_BETTER else "$\\downarrow$"
                label = m.upper() if m not in ("sens", "prec", "hd95", "msd") else {
                    "sens": "Sens", "prec": "Prec", "hd95": "HD95", "msd": "MSD"
                }[m]
                header_parts.append(f"{label} {arrow}")
            f.write(" & ".join(header_parts) + " \\\\\n")
            f.write("\\midrule\n")
            
            # Data rows
            for model in available:
                name = DISPLAY_NAMES.get(model, model)
                # Escape special LaTeX characters
                name = name.replace("→", "$\\rightarrow$").replace("+", "+")
                
                row_parts = [name]
                for metric in paper_metrics:
                    val = results[model][test_domain].get(metric, 0)
                    is_best = abs(val - best_vals[metric]) < 1e-6
                    
                    if metric in ("hd95", "msd"):
                        formatted = f"{val:.2f}"
                    else:
                        formatted = f"{val:.4f}"
                    
                    if is_best:
                        formatted = f"\\textbf{{{formatted}}}"
                    
                    row_parts.append(formatted)
                
                f.write(" & ".join(row_parts) + " \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write(f"\\end{{tabular}}\n")
            f.write(f"}}\n")
            f.write(f"\\end{{table}}\n")
        
        print(f"  📄 Saved: {latex_path}")
    
    # Combined cross-domain table
    latex_path = output_dir / "latex_table_cross_domain.tex"
    with open(latex_path, "w") as f:
        f.write("% Auto-generated — Cross-Domain Dice Comparison\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Cross-domain generalization (Dice score). Bold = best in group.}\n")
        f.write("\\label{tab:cross_domain}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Model & BTFE $\\rightarrow$ BTFE & BTFE $\\rightarrow$ TSE & $\\Delta$ Gap \\\\\n")
        f.write("\\midrule\n")
        
        for group, train_domain in [(BTFE_TRAINED, "BTFE"), (TSE_TRAINED, "TSE")]:
            if train_domain == "TSE":
                f.write("\\midrule\n")
                f.write("Model & TSE $\\rightarrow$ TSE & TSE $\\rightarrow$ BTFE & $\\Delta$ Gap \\\\\n")
                f.write("\\midrule\n")
            
            other_domain = "TSE" if train_domain == "BTFE" else "BTFE"
            
            # Find best cross-domain dice in this group
            cross_vals = []
            for m in group:
                if m in results and other_domain in results[m]:
                    cross_vals.append(results[m][other_domain]["dice"])
            best_cross = max(cross_vals) if cross_vals else 0
            
            for model in group:
                if model not in results:
                    continue
                name = DISPLAY_NAMES.get(model, model)
                name = name.replace("→", "$\\rightarrow$")
                
                same_dice = results[model].get(train_domain, {}).get("dice", 0)
                cross_dice = results[model].get(other_domain, {}).get("dice", 0)
                gap = same_dice - cross_dice
                
                cross_str = f"{cross_dice:.4f}"
                if abs(cross_dice - best_cross) < 1e-6:
                    cross_str = f"\\textbf{{{cross_str}}}"
                
                f.write(f"{name} & {same_dice:.4f} & {cross_str} & {gap:.4f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"  📄 Saved: {latex_path}")


# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
def plot_bar_charts(results, output_dir):
    """Generate grouped bar charts comparing models."""
    
    # # Color palette
    # colors = {
    #     "Baseline": "#4A90D9",
    #     "MixUp": "#50C878",
    #     "CutMix": "#FF6B6B",
    #     "AFA": "#FFB347",
    #     "MixUp+AFA": "#9B59B6",
    # }
    
    # def get_color(model_name):
    #     display = DISPLAY_NAMES.get(model_name, model_name)
    #     if "MixUp+AFA" in display: return colors["MixUp+AFA"]
    #     if "MixUp" in display: return colors["MixUp"]
    #     if "CutMix" in display: return colors["CutMix"]
    #     if "AFA" in display: return colors["AFA"]
    #     return colors["Baseline"]
    
    # def get_aug_label(model_name):
    #     display = DISPLAY_NAMES.get(model_name, model_name)
    #     if "MixUp+AFA" in display: return "MixUp+AFA"
    #     if "MixUp" in display: return "MixUp"
    #     if "CutMix" in display: return "CutMix"
    #     if "AFA" in display: return "AFA"
    #     return "Baseline"

    # Color palette
    colors = {
        "Baseline": "#4A90D9",
        "MixUp": "#50C878",
        "CutMix": "#FF6B6B",
        "AFA": "#FFB347",
        "MixUp+AFA": "#9B59B6",
        "CutMix+AFA": "#E67E22",  # <-- ADDED NEW COLOR (Orange)
    }
    
    def get_color(model_name):
        display = DISPLAY_NAMES.get(model_name, model_name)
        if "MixUp+AFA" in display: return colors["MixUp+AFA"]
        if "CutMix+AFA" in display: return colors["CutMix+AFA"] # <-- ADDED
        if "MixUp" in display: return colors["MixUp"]
        if "CutMix" in display: return colors["CutMix"]
        if "AFA" in display: return colors["AFA"]
        return colors["Baseline"]
    
    def get_aug_label(model_name):
        display = DISPLAY_NAMES.get(model_name, model_name)
        if "MixUp+AFA" in display: return "MixUp+AFA"
        if "CutMix+AFA" in display: return "CutMix+AFA" # <-- ADDED
        if "MixUp" in display: return "MixUp"
        if "CutMix" in display: return "CutMix"
        if "AFA" in display: return "AFA"
        return "Baseline"
    
    for group, group_name, train_domain in [
        (BTFE_TRAINED, "BTFE-Trained Models", "BTFE"),
        (TSE_TRAINED, "TSE-Trained Models", "TSE"),
    ]:
        other_domain = "TSE" if train_domain == "BTFE" else "BTFE"
        available = [m for m in group if m in results]
        
        if not available:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"{group_name} — Dice Score Comparison", fontsize=16, fontweight='bold', y=1.02)
        
        for ax_idx, (test_domain, ax_title) in enumerate([
            (train_domain, f"Same-Domain ({train_domain} → {train_domain})"),
            (other_domain, f"Cross-Domain ({train_domain} → {other_domain})"),
        ]):
            ax = axes[ax_idx]
            
            labels = [get_aug_label(m) for m in available]
            dice_vals = [results[m].get(test_domain, {}).get("dice", 0) for m in available]
            bar_colors = [get_color(m) for m in available]
            
            bars = ax.bar(labels, dice_vals, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)
            
            # Add value labels on bars
            for bar, val in zip(bars, dice_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax.set_title(ax_title, fontsize=13, fontweight='bold')
            ax.set_ylabel("Dice Score", fontsize=11)
            ax.set_ylim(0.65, 0.96)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Baseline reference line
            baseline_key = f"{train_domain}_unetplusplus_1Fold"
            if baseline_key in results and test_domain in results[baseline_key]:
                baseline_val = results[baseline_key][test_domain]["dice"]
                ax.axhline(y=baseline_val, color='gray', linestyle='--', alpha=0.6, label=f'Baseline ({baseline_val:.3f})')
                ax.legend(loc='lower right', fontsize=9)
        
        plt.tight_layout()
        save_path = output_dir / f"bar_chart_{train_domain}_models.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  📊 Saved: {save_path}")
    
    # --- All Models Comparison (Both Domains) ---
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    fig.suptitle("Complete Model Comparison — All Training Strategies", fontsize=16, fontweight='bold')
    
    all_models = BTFE_TRAINED + TSE_TRAINED + OTHER_MODELS
    available = [m for m in all_models if m in results]
    
    for ax_idx, test_domain in enumerate(["BTFE", "TSE"]):
        ax = axes[ax_idx]
        
        models_with_data = [m for m in available if test_domain in results[m]]
        labels = [DISPLAY_NAMES.get(m, m) for m in models_with_data]
        dice_vals = [results[m][test_domain]["dice"] for m in models_with_data]
        
        # Color by training data source
        bar_colors = []
        for m in models_with_data:
            if m.startswith("BTFE"): bar_colors.append("#4A90D9")
            elif m.startswith("TSE"): bar_colors.append("#50C878")
            elif m.startswith("COMBINED"): bar_colors.append("#FFB347")
            else: bar_colors.append("#9B59B6")
        
        bars = ax.barh(range(len(labels)), dice_vals, color=bar_colors, edgecolor='white', linewidth=1, height=0.7)
        
        for i, (bar, val) in enumerate(zip(bars, dice_vals)):
            ax.text(val + 0.003, i, f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Dice Score", fontsize=11)
        ax.set_title(f"Tested on {test_domain}", fontsize=13, fontweight='bold')
        ax.set_xlim(0.7, 0.95)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
    
    plt.tight_layout()
    save_path = output_dir / "bar_chart_all_models.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {save_path}")


def plot_cross_domain_heatmap(results, output_dir):
    """Generate a heatmap showing cross-domain generalization matrix."""
    
    all_models = BTFE_TRAINED + TSE_TRAINED + OTHER_MODELS
    available = [m for m in all_models if m in results]
    
    domains = ["BTFE", "TSE"]
    
    # Build matrix
    n_models = len(available)
    matrix = np.zeros((n_models, 2))
    
    for i, model in enumerate(available):
        for j, domain in enumerate(domains):
            matrix[i, j] = results[model].get(domain, {}).get("dice", 0)
    
    fig, ax = plt.subplots(figsize=(8, 12))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.70, vmax=0.92)
    
    ax.set_xticks(range(2))
    ax.set_xticklabels(["Test on BTFE", "Test on TSE"], fontsize=12, fontweight='bold')
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([DISPLAY_NAMES.get(m, m) for m in available], fontsize=10)
    
    # Add text annotations
    for i in range(n_models):
        for j in range(2):
            val = matrix[i, j]
            text_color = 'white' if val < 0.80 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=text_color)
    
    ax.set_title("Cross-Domain Generalization Heatmap (Dice Score)", fontsize=14, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Dice Score")
    
    plt.tight_layout()
    save_path = output_dir / "heatmap_cross_domain.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {save_path}")


def plot_augmentation_improvement(results, output_dir):
    """Bar chart showing cross-domain Dice improvement over baseline."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cross-Domain Dice Improvement Over Baseline", fontsize=15, fontweight='bold')
    
    for ax_idx, (group, train_domain) in enumerate([
        (BTFE_TRAINED, "BTFE"), (TSE_TRAINED, "TSE")
    ]):
        ax = axes[ax_idx]
        other_domain = "TSE" if train_domain == "BTFE" else "BTFE"
        
        baseline_key = f"{train_domain}_unetplusplus_1Fold"
        if baseline_key not in results or other_domain not in results.get(baseline_key, {}):
            continue
        
        baseline_cross = results[baseline_key][other_domain]["dice"]
        
        aug_models = [m for m in group if m != baseline_key and m in results and other_domain in results[m]]
        labels = []
        improvements = []
        
        for m in aug_models:
            labels.append(DISPLAY_NAMES.get(m, m).replace(f"{train_domain} + ", ""))
            improvements.append(results[m][other_domain]["dice"] - baseline_cross)
        
        bar_colors = ['#50C878' if v >= 0 else '#FF6B6B' for v in improvements]
        bars = ax.bar(labels, [v * 100 for v in improvements], color=bar_colors, edgecolor='white', width=0.6)
        
        for bar, val in zip(bars, improvements):
            y_pos = bar.get_height() + 0.2 if val >= 0 else bar.get_height() - 0.8
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{val*100:+.2f}%', ha='center', va='bottom' if val >= 0 else 'top',
                    fontweight='bold', fontsize=11)
        
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_title(f"{train_domain}-Trained → {other_domain}\n(Baseline Cross: {baseline_cross:.4f})",
                     fontsize=12, fontweight='bold')
        ax.set_ylabel("Dice Improvement (%points)", fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_path = output_dir / "improvement_over_baseline.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {save_path}")


# ============================================================================
# 7. ANALYSIS: Which augmentation combination should you try next?
# ============================================================================
def print_recommendations(results):
    """Analyze results and suggest missing augmentation experiments."""
    
    print(f"\n{'='*90}")
    print(f"  🧬 AUGMENTATION ANALYSIS & RECOMMENDATIONS")
    print(f"{'='*90}")
    
    # BTFE Analysis
    base_btfe = "BTFE_unetplusplus_1Fold"
    if base_btfe in results:
        btfe_same = results[base_btfe].get("BTFE", {}).get("dice", 0)
        btfe_cross = results[base_btfe].get("TSE", {}).get("dice", 0)
        
        print(f"\n  📊 BTFE-Trained Models (Baseline: same={btfe_same:.4f}, cross={btfe_cross:.4f})")
        print(f"  {'Augmentation':<20} {'Same-Domain':>12} {'Δ Same':>10} {'Cross-Domain':>12} {'Δ Cross':>10}")
        print(f"  {'-'*70}")
        
        for model in BTFE_TRAINED:
            if model == base_btfe or model not in results:
                continue
            aug_name = DISPLAY_NAMES[model].replace("BTFE + ", "")
            same = results[model].get("BTFE", {}).get("dice", 0)
            cross = results[model].get("TSE", {}).get("dice", 0)
            delta_same = same - btfe_same
            delta_cross = cross - btfe_cross
            s_sign = "+" if delta_same >= 0 else ""
            c_sign = "+" if delta_cross >= 0 else ""
            print(f"  {aug_name:<20} {same:>12.4f} {s_sign}{delta_same:>9.4f} {cross:>12.4f} {c_sign}{delta_cross:>9.4f}")
    
    # TSE Analysis
    base_tse = "TSE_unetplusplus_1Fold"
    if base_tse in results:
        tse_same = results[base_tse].get("TSE", {}).get("dice", 0)
        tse_cross = results[base_tse].get("BTFE", {}).get("dice", 0)
        
        print(f"\n  📊 TSE-Trained Models (Baseline: same={tse_same:.4f}, cross={tse_cross:.4f})")
        print(f"  {'Augmentation':<20} {'Same-Domain':>12} {'Δ Same':>10} {'Cross-Domain':>12} {'Δ Cross':>10}")
        print(f"  {'-'*70}")
        
        for model in TSE_TRAINED:
            if model == base_tse or model not in results:
                continue
            aug_name = DISPLAY_NAMES[model].replace("TSE + ", "")
            same = results[model].get("TSE", {}).get("dice", 0)
            cross = results[model].get("BTFE", {}).get("dice", 0)
            delta_same = same - tse_same
            delta_cross = cross - tse_cross
            s_sign = "+" if delta_same >= 0 else ""
            c_sign = "+" if delta_cross >= 0 else ""
            print(f"  {aug_name:<20} {same:>12.4f} {s_sign}{delta_same:>9.4f} {cross:>12.4f} {c_sign}{delta_cross:>9.4f}")
    
    # # Missing combinations
    # print(f"\n  🔬 MISSING AUGMENTATION EXPERIMENTS:")
    
    # existing_btfe = {m.replace("BTFE_unetplusplus_", "") for m in BTFE_TRAINED if m in results}
    # existing_tse = {m.replace("TSE_unetplusplus_", "") for m in TSE_TRAINED if m in results}
    
    # all_possible = {"1Fold", "mixup", "cutmix", "afa", "mixup_afa", "cutmix_afa"}
    
    # missing_btfe = all_possible - existing_btfe
    # missing_tse = all_possible - existing_tse
    
    # if missing_btfe:
    #     print(f"    BTFE missing: {', '.join(sorted(missing_btfe))}")
    # if missing_tse:
    #     print(f"    TSE missing:  {', '.join(sorted(missing_tse))}")
    
    # if "cutmix_afa" in missing_btfe or "cutmix_afa" in missing_tse:
    #     print(f"\n  💡 RECOMMENDATION: Train CutMix+AFA on both BTFE and TSE.")
    #     print(f"     Rationale: You have MixUp+AFA but NOT CutMix+AFA.")
    #     print(f"     CutMix creates hard region boundaries (vs MixUp's soft blending).")
    #     print(f"     Combined with AFA's Fourier perturbation, this tests a complementary")
    #     print(f"     augmentation strategy that may outperform on cross-domain.")
    
    # # Also suggest Combined + augmentations
    # print(f"\n  💡 ADDITIONAL SUGGESTION: Combined + Augmentations")
    # print(f"     Your Combined baseline achieves the smallest cross-domain gap.")
    # print(f"     Adding MixUp, CutMix, or AFA to the Combined model could further")
    # print(f"     improve both same-domain and cross-domain performance.")
    # Missing combinations
    print(f"\n  🔬 MISSING AUGMENTATION EXPERIMENTS:")
    
    existing_btfe = {m.replace("BTFE_unetplusplus_", "") for m in BTFE_TRAINED if m in results}
    existing_tse = {m.replace("TSE_unetplusplus_", "") for m in TSE_TRAINED if m in results}
    
    all_possible = {"1Fold", "mixup", "cutmix", "afa", "mixup_afa", "cutmix_afa"}
    
    missing_btfe = all_possible - existing_btfe
    missing_tse = all_possible - existing_tse
    
    if missing_btfe:
        print(f"    BTFE missing: {', '.join(sorted(missing_btfe))}")
    elif missing_tse:
        print(f"    TSE missing:  {', '.join(sorted(missing_tse))}")
    else:
        print(f"    ✅ ALL foundational experiments are complete! Your grid is fully populated.")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  🏆 PLACENTA SEGMENTATION — MODEL COMPARISON DASHBOARD")
    print("=" * 60)
    
    results = discover_metrics(RUNS_DIR)
    
    print(f"\n  Discovered {len(results)} models with test metrics:")
    for model in sorted(results.keys()):
        domains = ", ".join(sorted(results[model].keys()))
        print(f"    - {DISPLAY_NAMES.get(model, model):<28} tested on: {domains}")
    
    # --- Console Tables ---
    print_table(results, "BTFE", BTFE_TRAINED, "BTFE-TRAINED MODELS → Tested on BTFE (Same-Domain)")
    print_table(results, "TSE", BTFE_TRAINED, "BTFE-TRAINED MODELS → Tested on TSE (Cross-Domain)")
    print_table(results, "TSE", TSE_TRAINED, "TSE-TRAINED MODELS → Tested on TSE (Same-Domain)")
    print_table(results, "BTFE", TSE_TRAINED, "TSE-TRAINED MODELS → Tested on BTFE (Cross-Domain)")
    print_table(results, "BTFE", OTHER_MODELS, "COMBINED & TRANSFER MODELS → Tested on BTFE")
    print_table(results, "TSE", OTHER_MODELS, "COMBINED & TRANSFER MODELS → Tested on TSE")
    
    # --- Cross-Domain Analysis ---
    print_cross_domain_analysis(results, BTFE_TRAINED, "BTFE", "BTFE-TRAINED: Cross-Domain Gap Analysis (Dice)")
    print_cross_domain_analysis(results, TSE_TRAINED, "TSE", "TSE-TRAINED: Cross-Domain Gap Analysis (Dice)")
    
    # --- Recommendations ---
    print_recommendations(results)
    
    # --- Exports ---
    print(f"\n{'='*60}")
    print(f"  📁 EXPORTING RESULTS TO: {OUTPUT_DIR}")
    print(f"{'='*60}")
    
    export_csv(results, OUTPUT_DIR)
    export_latex(results, OUTPUT_DIR)
    plot_bar_charts(results, OUTPUT_DIR)
    plot_cross_domain_heatmap(results, OUTPUT_DIR)
    plot_augmentation_improvement(results, OUTPUT_DIR)
    
    print(f"\n🎉 Done! All results saved to: {OUTPUT_DIR}")
