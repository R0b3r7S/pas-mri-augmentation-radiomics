#!/usr/bin/env python3
"""
make_figures_3d.py — generate the two figures for the 3D-only paper.

Figure 1 (box plot).  Per-patient median PyRadiomics relative error,
grouped by augmentation, for each corruption type, in the same-domain
(ID) and out-of-distribution (OOD) regimes. Two stacked panels:
   - top:    ID    (model evaluated on its own training modality)
   - bottom: OOD   (model evaluated on the other modality)
Within each panel, 5 corruption columns x 6 augmentation box plots.

Figure 2.  Two-panel Dice-vs-feature-error scatter.
    Left panel:  Dice vs. median relative error  (correlated, r ~ -0.92)
    Right panel: Dice vs. mean   relative error  (uncorrelated, r ~ 0)

Inputs are the same CSVs as audit_numbers.py, plus the two per-patient
CSVs needed for the box plot (shareable/per_patient_dice.csv and
radiomics/comparisons/2d/per_patient_errors.csv).
Outputs are written to paper/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
METRICS_CSV    = ROOT / "shareable" / "all_test_metrics.csv"
PERCORR_CSV    = ROOT / "radiomics" / "comparisons" / "2d" / "per_corruption_summary.csv"
PERPATIENT_DICE_CSV   = ROOT / "shareable" / "per_patient_dice.csv"
PERPATIENT_ERRORS_CSV = ROOT / "radiomics" / "comparisons" / "2d" / "per_patient_errors.csv"
OUT_DIR        = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUG_LABEL = {
    "1Fold":       "Baseline",
    "mixup":       "MixUp",
    "cutmix":      "CutMix",
    "afa":         "AFA",
    "cutmix_afa":  "CutMix+AFA",
    "mixup_afa":   "MixUp+AFA",
}
AUG_ORDER = ["Baseline", "MixUp", "CutMix", "AFA", "CutMix+AFA", "MixUp+AFA"]
AUG_COLOR = {
    "Baseline":   "#7f7f7f",
    "MixUp":      "#1f77b4",
    "CutMix":     "#d62728",
    "AFA":        "#2ca02c",
    "CutMix+AFA": "#ff7f0e",
    "MixUp+AFA":  "#9467bd",
}
CORR_LABEL = {
    "bias_field":   "Bias field",
    "ghosting":     "Ghosting",
    "kspace_sub":   "k-space sub.",
    "rician_noise": "Rician noise",
    "spike_noise":  "Spike noise",
}
CORR_ORDER = ["bias_field", "ghosting", "kspace_sub", "rician_noise", "spike_noise"]


def aug_name(model: str) -> str:
    suffix = model.split("_unetplusplus_", 1)[1]
    return AUG_LABEL[suffix]


def train_dom(model: str) -> str:
    return "BTFE" if model.startswith("BTFE_") else "TSE"


def load_joined_3d() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS_CSV)
    percorr = pd.read_csv(PERCORR_CSV)
    keys = ["model", "base_domain", "corruption", "corruption_mode"]
    j = pd.merge(
        metrics[keys + ["dice", "train_domain"]],
        percorr[keys + ["feature_rel_error", "feature_rel_error_median"]],
        on=keys, how="inner",
    ).dropna()
    keep = [m for m in j["model"].unique()
            if m.startswith(("BTFE_unetplusplus_", "TSE_unetplusplus_"))
            and not m.startswith(("COMBINED_", "Transfer_"))]
    j = j[j["model"].isin(keep)]
    j = j[j["corruption_mode"].isin(["3d", "clean"])].copy()
    j["aug"]        = j["model"].map(aug_name)
    j["train_dom"]  = j["model"].map(train_dom)
    j["scope"]      = np.where(j["train_dom"] == j["base_domain"], "Same", "OOD")
    return j


def load_per_patient_3d() -> pd.DataFrame:
    """Build a per-patient dataframe restricted to 12 aug models, 3D mode only.

    Columns:
        model, base_domain, corruption, corruption_mode, patient,
        median_ef, mean_ef, aug, train_dom, scope
    """
    ppe = pd.read_csv(PERPATIENT_ERRORS_CSV,
                      usecols=["model", "base_domain", "corruption",
                               "corruption_mode", "patient", "rel_error"])
    # Aggregate the 102 features into per-patient median e_f and mean e_f
    pp = (ppe.groupby(["model", "base_domain", "corruption",
                       "corruption_mode", "patient"])["rel_error"]
              .agg(median_ef="median", mean_ef="mean")
              .reset_index())

    # Restrict to the 12 augmentation models and 3D-corrupted cells only
    keep_models = [f"{td}_unetplusplus_{suf}"
                   for td in ("BTFE", "TSE")
                   for suf in AUG_LABEL.keys()]
    pp = pp[pp["model"].isin(keep_models)
            & (pp["corruption_mode"] == "3d")].copy()
    pp["aug"]        = pp["model"].map(aug_name)
    pp["train_dom"]  = pp["model"].map(train_dom)
    pp["scope"]      = np.where(pp["train_dom"] == pp["base_domain"],
                                "Same", "OOD")
    return pp


def make_boxplot(pp: pd.DataFrame):
    """Two-row two-column grouped box plot of per-patient median e_f.

    Rows:    ID  (top, same-domain), OOD (bottom).
    Columns: the four "low-error" corruptions on the left, Rician noise on
             the right with its OWN y-axis. Rician dominates the magnitude
             of the per-patient errors (~0.18-0.30 vs ~0.02-0.10 for the
             others); sharing one y-axis crushes the other four columns
             against the bottom, so we split them.
    Inside each subplot, each corruption group has 6 augmentation boxes.
    Each box contains 38 patient values (19 patients x 2 training
    modalities).
    Top and right spines are hidden on every subplot (despined).
    """
    LEFT_CORRS  = ["bias_field", "ghosting", "kspace_sub", "spike_noise"]
    RIGHT_CORRS = ["rician_noise"]

    fig, axes = plt.subplots(
        2, 2, figsize=(7.0, 3.4),
        gridspec_kw={"width_ratios": [len(LEFT_CORRS), len(RIGHT_CORRS)],
                     "wspace": 0.16, "hspace": 0.30},
        sharex="col", sharey="col",
    )

    n_aug = len(AUG_ORDER)
    box_w = 0.7
    group_w = n_aug * box_w + 1.0          # spacing between corruption groups
    pos_in_group = np.arange(n_aug) * box_w  # 6 box positions inside one group

    def _draw_panel(ax, corrs, scope_value, scope_label, show_ylabel):
        for c_idx, corr in enumerate(corrs):
            x_centre = c_idx * group_w
            box_data    = []
            box_colours = []
            for aug in AUG_ORDER:
                vals = pp[(pp["aug"] == aug)
                          & (pp["scope"] == scope_value)
                          & (pp["corruption"] == corr)]["median_ef"].values
                box_data.append(vals)
                box_colours.append(AUG_COLOR[aug])
            positions = x_centre + pos_in_group - (n_aug - 1) * box_w / 2
            bp = ax.boxplot(
                box_data, positions=positions, widths=box_w * 0.85,
                patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=2.5,
                                markerfacecolor="0.4", markeredgecolor="none",
                                alpha=0.5),
                medianprops=dict(color="black", lw=1.0),
                whiskerprops=dict(color="0.3", lw=0.7),
                capprops=dict(color="0.3", lw=0.7),
                boxprops=dict(lw=0.7),
            )
            for patch, col in zip(bp["boxes"], box_colours):
                patch.set_facecolor(col)
                patch.set_alpha(0.65)
                patch.set_edgecolor("0.2")

        ax.set_xticks([c * group_w for c in range(len(corrs))])
        ax.set_xticklabels([CORR_LABEL[c] for c in corrs], fontsize=8)
        if show_ylabel:
            ax.set_ylabel("Median rel. feature error", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        if scope_label:
            ax.set_title(scope_label, fontsize=9, loc="left")
        ax.set_ylim(bottom=0)
        # Despine: hide the top and right borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    rows = [(axes[0, 0], axes[0, 1], "Same-domain (ID)",          "Same"),
            (axes[1, 0], axes[1, 1], "Out-of-distribution (OOD)", "OOD")]
    for ax_left, ax_right, scope_label, scope_value in rows:
        _draw_panel(ax_left,  LEFT_CORRS,  scope_value, scope_label, show_ylabel=True)
        # Right subplot: no scope title (the left panel of the same row
        # carries it) and no y-label (its scale is just "Median rel.
        # feature error" on a different range than the left).
        _draw_panel(ax_right, RIGHT_CORRS, scope_value, "",          show_ylabel=False)

    fig.tight_layout()
    handles = [mpl.patches.Patch(facecolor=AUG_COLOR[a], alpha=0.65,
                                 edgecolor="0.2", label=a)
               for a in AUG_ORDER]
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, 1.0), ncol=6,
                     fontsize=8, frameon=False,
                     handletextpad=0.4, columnspacing=1.2)
    out = OUT_DIR / "figure1.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                bbox_extra_artists=(leg,))
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def make_scatter(j: pd.DataFrame):
    """Two-panel scatter: Dice vs median (left), Dice vs mean (right).
    Colour by augmentation, marker by scope (Same / OOD)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.3))

    for ax, ycol, ylab, log in [
        (axL, "feature_rel_error_median", "Median rel. feature error", False),
        (axR, "feature_rel_error",        "Mean rel. feature error",   True),
    ]:
        for aug in AUG_ORDER:
            for scope, marker in [("Same", "o"), ("OOD", "x")]:
                sub = j[(j["aug"] == aug) & (j["scope"] == scope) &
                        (j["corruption_mode"] == "3d")]
                ax.scatter(sub["dice"], sub[ycol],
                           s=18, marker=marker, alpha=0.75,
                           edgecolors="none" if marker == "o" else None,
                           color=AUG_COLOR[aug],
                           label=f"{aug} ({scope})" if ax is axL else None)
        ax.set_xlabel("Dice")
        ax.set_ylabel(ylab)
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        # Despine: hide the top and right borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        d = j[j["corruption_mode"] == "3d"]
        r, p = pearsonr(d["dice"], d[ycol])
        if p < 1e-100:
            p_str = r"$p\!<\!10^{-100}$"
        else:
            p_str = f"$p={p:.2g}$"
        ax.text(0.04, 0.04, f"$r={r:+.2f}$, {p_str}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.7", lw=0.5))

    fig.tight_layout()
    # Place the legend ABOVE the panels with explicit gap so it never clips.
    handles, labels = axL.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = fig.legend(by_label.values(), by_label.keys(),
                     loc="lower center", bbox_to_anchor=(0.5, 1.02),
                     ncol=6, fontsize=7, frameon=False,
                     handletextpad=0.4, columnspacing=1.0)

    out = OUT_DIR / "figure2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                bbox_extra_artists=(leg,))
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main():
    j = load_joined_3d()
    print(f"joined 3D+clean cells: {len(j)}")
    print(f"corrupted only:        {(j['corruption_mode']=='3d').sum()}")
    pp = load_per_patient_3d()
    print(f"per-patient rows (3D corrupted, 12 aug models): {len(pp)}")
    make_boxplot(pp)
    make_scatter(j)


if __name__ == "__main__":
    main()
