"""Cross-PVT Analysis and Publication Figures for SRAM Fault Diagnosis.

Implements Stage 7 of project roadmap and Section 6.4 of PLAN (3).md:
1. Feature Spread Analysis (Healthy class across 15 PVT corners):
   - Compares distribution spread (Coefficient of Variation) of Raw vs Invariant features.
   - Verifies that invariant physics normalization significantly reduces PVT dispersion.
2. Cross-PVT Degradation Analysis:
   - Compares Leave-One-Corner-Out (LOCO) generalization for Raw vs Invariant across all 4 models.
   - Produces publication-ready figures saved in `reports/figures/`.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports" / "final_results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Professional typography and styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def analyze_feature_spread(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> pd.DataFrame:
    """Analyze and plot feature spread across 15 PVT corners for healthy cells."""
    out_dir.mkdir(parents=True, exist_ok=True)
    healthy_df = df[df["fault_class"] == 0].copy()

    pairs = [
        ("i_ddq_uA", "iddq_norm", "Standby Leakage I_ddq (uA)", "Normalized I_ddq (ratio)"),
        ("t_write_ps", "t_write_norm", "Write Time t_write (ps)", "Normalized Write Time (ratio)"),
        ("snm_hold_v", "snm_hold_norm", "Hold SNM (V)", "Normalized Hold SNM (ratio)"),
        ("dv_bl_strobe_mV", "dv_bl_norm", "Sense Voltage dV_BL (mV)", "Normalized dV_BL (ratio)"),
    ]

    spread_records = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (raw_col, norm_col, raw_lbl, norm_lbl) in enumerate(pairs):
        raw_vals = healthy_df[raw_col].values
        norm_vals = healthy_df[norm_col].values

        raw_mean = float(np.mean(raw_vals))
        raw_std = float(np.std(raw_vals, ddof=1))
        raw_cv = (raw_std / abs(raw_mean)) * 100.0 if raw_mean != 0 else 0.0

        norm_mean = float(np.mean(norm_vals))
        norm_std = float(np.std(norm_vals, ddof=1))
        norm_cv = (norm_std / abs(norm_mean)) * 100.0 if norm_mean != 0 else 0.0

        reduction = (1.0 - norm_cv / max(raw_cv, 1e-6)) * 100.0

        spread_records.append({
            "feature": raw_col.split("_")[0],
            "raw_mean": round(raw_mean, 3),
            "raw_std": round(raw_std, 3),
            "raw_cv_percent": round(raw_cv, 2),
            "norm_mean": round(norm_mean, 3),
            "norm_std": round(norm_std, 3),
            "norm_cv_percent": round(norm_cv, 2),
            "cv_reduction_percent": round(reduction, 2),
        })

        ax = axes[idx]
        plot_data = pd.DataFrame({
            "Corner": healthy_df["corner_id"],
            "Raw": (raw_vals - raw_mean) / max(raw_std, 1e-6),  # z-scored for visual comparison
            "Invariant": (norm_vals - norm_mean) / max(norm_std, 1e-6),
        }).melt(id_vars=["Corner"], value_vars=["Raw", "Invariant"], var_name="Feature Type", value_name="Z-Score")

        sns.boxplot(data=plot_data, x="Feature Type", y="Z-Score", ax=ax, palette=["#d95f02", "#1b9e77"])
        ax.set_title(f"{raw_lbl.split('(')[0].strip()}: Spread CV {raw_cv:.1f}% -> {norm_cv:.1f}%")
        ax.set_ylabel("Standardized Deviation (Z-score)")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = out_dir / "fig1_feature_dispersion_comparison.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved feature dispersion plot: {fig_path}")

    spread_df = pd.DataFrame(spread_records)
    spread_df_path = REPORTS_DIR / "feature_spread_comparison.csv"
    spread_df.to_csv(spread_df_path, index=False)
    print(f"Saved feature spread metrics: {spread_df_path}")
    return spread_df


def plot_loco_comparison(
    comparison_csv: Path = REPORTS_DIR / "comparison_table.csv",
    out_dir: Path = FIGURES_DIR,
):
    """Plot Cross-PVT Leave-One-Corner-Out (LOCO) performance across models."""
    if not comparison_csv.exists():
        print(f"Comparison table not found at {comparison_csv}. Skipping LOCO plot.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(comparison_csv)
    loco_df = df[df["protocol"] == "leave_one_corner_out"].copy()
    if len(loco_df) == 0:
        return

    # Filter out baseline for model-to-model ML chart if desired or keep
    ml_df = loco_df[loco_df["model"] != "NonML_Baseline"].copy()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=ml_df,
        x="model",
        y="accuracy",
        hue="feature_set",
        palette=["#e7298a", "#7570b3"],
        errorbar="sd",
        capsize=0.1,
    )
    plt.title("Cross-PVT Generalization: Raw vs. Physics-Normalized Invariant Features (LOCO)")
    plt.ylabel("Mean Test Accuracy (+/- 1 Std Dev)")
    plt.xlabel("Machine Learning Classifier")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Feature Representation", loc="lower right")

    fig_path = out_dir / "fig2_cross_pvt_loco_accuracy.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved LOCO comparison plot: {fig_path}")

    # Heatmap of per-corner performance for best model
    rf_df = loco_df[loco_df["model"] == "RandomForest"].copy()
    if len(rf_df) > 0:
        pivot = rf_df.pivot(index="corner", columns="feature_set", values="accuracy")
        pivot["Improvement (%)"] = (pivot["invariant"] - pivot["raw"]) * 100.0

        plt.figure(figsize=(8, 10))
        sns.heatmap(pivot[["raw", "invariant"]], annot=True, fmt=".3f", cmap="YlGnBu", cbar=True)
        plt.title("Random Forest LOCO Accuracy across 15 PVT Corners")
        plt.xlabel("Feature Representation")
        plt.ylabel("Held-Out PVT Corner")

        heat_path = out_dir / "fig3_rf_pvt_corner_heatmap.png"
        plt.savefig(heat_path, bbox_inches="tight")
        plt.close()
        print(f"Saved corner heatmap plot: {heat_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Cross-PVT Analysis and Figures")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_DIR / "sram_fault_dataset.csv",
        help="Path to generated CSV dataset",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found at {args.dataset}.")
        return

    df = pd.read_csv(args.dataset)
    print(f"Running Cross-PVT Analysis on {len(df)} samples...")
    analyze_feature_spread(df)
    plot_loco_comparison()
    print("Cross-PVT Analysis complete.")


if __name__ == "__main__":
    main()
