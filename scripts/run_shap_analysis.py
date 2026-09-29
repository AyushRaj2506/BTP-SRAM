"""SHAP (SHapley Additive exPlanations) Analysis for SRAM Fault Diagnosis.

Implements Week 9 Explainability requirement from PLAN (3).md:
- Explains the best invariant-feature Random Forest model using TreeExplainer.
- Identifies feature importance rankings globally and per fault class.
- Generates publication-ready figures in reports/figures/ and summary table in reports/final_results/.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_pipeline import INVARIANT_FEATURES
DATA_PATH = ROOT / "data" / "sram_fault_dataset.csv"
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "reports" / "final_results"

CLASS_LABELS = {
    0: "Class 0: Healthy",
    1: "Class 1: Resistive Open",
    2: "Class 2: Bridging Fault",
    3: "Class 3: Vth Drift Storage",
    4: "Class 4: Vth Drift Access",
}

FEATURE_DISPLAY_NAMES = {
    "iddq_norm": "Normalized IDDQ (iddq_norm)",
    "t_write_norm": "Normalized Write Delay (t_write_norm)",
    "dv_bl_norm": "Normalized Bitline Swing (dv_bl_norm)",
    "snm_hold_norm": "Normalized Hold SNM (snm_hold_norm)",
    "snm_read_norm": "Normalized Read SNM (snm_read_norm)",
    "read_write_delay_ratio": "Read/Write Delay Ratio",
    "snm_ratio": "Read/Hold SNM Ratio",
    "dv_bl_vdd_ratio": "Sense Swing / VDD Ratio",
    "hold_pass": "Hold Retention Pass Flag",
    "write_pass": "Write Functional Pass Flag",
    "read_stable_pass": "Read Stability Pass Flag",
    "is_bistable": "Bistability Flag",
}

# Typography and styling
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


def main():
    print("========================================================")
    print("SRAM FAULT DIAGNOSIS - WEEK 9: SHAP EXPLAINABILITY")
    print("========================================================")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[INVARIANT_FEATURES]
    y = df["fault_class"]

    print(f"Loaded dataset with {len(df)} samples across 12 invariant features.")

    # Train optimal Random Forest model (consistent with pipeline seed=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    print("Fitted Random Forest model successfully.")

    # Compute SHAP values via TreeExplainer
    print("Computing TreeExplainer SHAP values...")
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X)

    # shap_vals has shape (n_samples, n_features, n_classes) or list of (n_samples, n_features)
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        # Array shape: (450, 12, 5)
        n_classes = shap_vals.shape[2]
        class_shap = [shap_vals[:, :, c] for c in range(n_classes)]
    elif isinstance(shap_vals, list):
        class_shap = shap_vals
        n_classes = len(class_shap)
    else:
        raise ValueError(f"Unexpected SHAP values type: {type(shap_vals)}")

    # 1. Global Mean Absolute SHAP values
    mean_abs_per_class = np.array([np.mean(np.abs(cs), axis=0) for cs in class_shap])  # (5, 12)
    global_mean_abs = np.mean(mean_abs_per_class, axis=0)  # (12,)

    # Build importance dataframe
    importance_df = pd.DataFrame({
        "feature": INVARIANT_FEATURES,
        "display_name": [FEATURE_DISPLAY_NAMES[f] for f in INVARIANT_FEATURES],
        "global_importance": global_mean_abs,
    })
    for c in range(n_classes):
        importance_df[f"class_{c}_importance"] = mean_abs_per_class[c]

    importance_df = importance_df.sort_values("global_importance", ascending=False).reset_index(drop=True)

    # Save CSV table
    csv_out = RES_DIR / "shap_feature_importance.csv"
    importance_df.to_csv(csv_out, index=False)
    print(f"Saved SHAP importance table to: {csv_out}")
    print("\nTop 5 Features by Global SHAP Importance:")
    print(importance_df[["feature", "global_importance"]].head(5).to_string(index=False))

    # --- FIGURE 4: Overall SHAP Global Importance Bar Plot ---
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        y="display_name",
        x="global_importance",
        palette="Blues_r",
    )
    plt.title("Global Feature Importance via TreeExplainer SHAP (Invariant Features)")
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output)")
    plt.ylabel("Physics-Normalized Invariant Feature")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig4_path = FIG_DIR / "fig4_shap_summary.png"
    plt.savefig(fig4_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4: {fig4_path}")

    # --- FIGURE 4b: Stacked / Grouped Class Breakdown Bar Chart ---
    # Reshape for seaborn
    melt_cols = [f"class_{c}_importance" for c in range(n_classes)]
    melt_df = importance_df.melt(
        id_vars=["display_name"],
        value_vars=melt_cols,
        var_name="class_col",
        value_name="shap_val",
    )
    melt_df["fault_class"] = melt_df["class_col"].apply(lambda x: CLASS_LABELS[int(x.split("_")[1])])

    plt.figure(figsize=(12, 8))
    # Horizontal grouped barplot
    sns.barplot(
        data=melt_df,
        y="display_name",
        x="shap_val",
        hue="fault_class",
        palette="tab10",
    )
    plt.title("SHAP Feature Importance Disaggregated by Fault Class")
    plt.xlabel("Mean |SHAP Value| within Class")
    plt.ylabel("Invariant Feature")
    plt.legend(title="Target Class", loc="lower right")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig4b_path = FIG_DIR / "fig4b_shap_class_importance.png"
    plt.savefig(fig4b_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4b: {fig4b_path}")

    print("\n========================================================")
    print("WEEK 9 SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
    print("========================================================")


if __name__ == "__main__":
    main()
