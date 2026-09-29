"""Final original-vs-cleaned comparison table and figure.

Reads (never modifies):
  reports/final_results/loco_summary.csv      original 450-row results
  reports/audit/clean_loco_summary_noSVM.csv  cleaned LOCO results (no LinearSVM)
  reports/audit/dataset_clean_main.csv, dataset_clean_strict.csv

Writes only to reports/audit/:
  final_comparison_table.csv
  final_comparison_figure.png

LinearSVM on the cleaned sets is a SENSITIVITY experiment run with
probability=False (probability=True hangs on the cleaned C_75C_0P9V split).
Predictions matched probability=True on all 900 original-data test
predictions. It is labelled as such in the table and the figure.

std = sample standard deviation (ddof=1) across held-out corners.
Run from the project root:  set PYTHONPATH=.  &&  python scripts/make_final_comparison.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "reports" / "audit"
FINAL_DIR = PROJECT_ROOT / "reports" / "final_results"

RAW_FEATURES = [
    "i_ddq_uA", "t_write_ps", "i_write_peak_mA", "dv_bl_strobe_mV",
    "t_sense_ps", "v_bump_mV", "hold_pass", "write_pass", "read_stable_pass",
    "snm_hold_v", "snm_read_v", "snm_asym_hold_v", "snm_asym_read_v",
    "snm_drop_v", "is_bistable",
]
INVARIANT_FEATURES = [
    "iddq_norm", "t_write_norm", "dv_bl_norm", "snm_hold_norm",
    "snm_read_norm", "read_write_delay_ratio", "snm_ratio",
    "dv_bl_vdd_ratio", "hold_pass", "write_pass", "read_stable_pass",
    "is_bistable",
]
FEATURE_SETS = {"raw": RAW_FEATURES, "invariant": INVARIANT_FEATURES}

MODEL_ORDER = [
    "RandomForest",
    "HistGradientBoosting",
    "LogisticRegression",
    "LinearSVM",
    "NonML_Baseline",
]
DATASET_ORDER = ["original_450", "clean_main_428", "clean_strict_302"]
DATASET_NOTES = {
    "original_450": "5-class, original results",
    "clean_main_428": "5-class, 22 rows with missing waveform output removed",
    "clean_strict_302": "4-class (no bridge class); converged==1 only; sensitivity, not directly comparable",
}


def linear_svm_sensitivity(dataset_name, path):
    """LinearSVM LOCO with probability=False (sensitivity experiment)."""
    df = pd.read_csv(path)
    rows = []
    for fs, cols in FEATURE_SETS.items():
        accs = []
        for corner in sorted(df["corner_id"].unique()):
            train = df[df["corner_id"] != corner]
            test = df[df["corner_id"] == corner]
            model = make_pipeline(
                StandardScaler(),
                SVC(kernel="linear", random_state=42, probability=False),
            )
            model.fit(train[cols], train["fault_class"])
            accs.append(accuracy_score(test["fault_class"], model.predict(test[cols])))
        rows.append(
            {
                "dataset": dataset_name,
                "model": "LinearSVM",
                "feature_set": fs,
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs, ddof=1)),
                "min": float(np.min(accs)),
                "max": float(np.max(accs)),
                "n_corners": len(accs),
            }
        )
    return pd.DataFrame(rows)


def main():
    orig = pd.read_csv(FINAL_DIR / "loco_summary.csv")
    orig.insert(0, "dataset", "original_450")
    orig["n_corners"] = 15

    clean = pd.read_csv(AUDIT_DIR / "clean_loco_summary_noSVM.csv")
    clean["n_corners"] = 15

    svm_clean = pd.concat(
        [
            linear_svm_sensitivity("clean_main_428", AUDIT_DIR / "dataset_clean_main.csv"),
            linear_svm_sensitivity("clean_strict_302", AUDIT_DIR / "dataset_clean_strict.csv"),
        ],
        ignore_index=True,
    )

    cols = ["dataset", "model", "feature_set", "mean", "std", "min", "max", "n_corners"]
    long = pd.concat([orig[cols], clean[cols], svm_clean[cols]], ignore_index=True)

    wide = long.pivot_table(
        index=["dataset", "model"],
        columns="feature_set",
        values=["mean", "std", "min", "max"],
    )
    wide.columns = [f"{feat}_{stat}" for stat, feat in wide.columns]
    wide = wide.reset_index()
    wide["gain_pp"] = (wide["invariant_mean"] - wide["raw_mean"]) * 100.0

    def note(r):
        parts = [DATASET_NOTES[r["dataset"]]]
        if r["model"] == "LinearSVM":
            if r["dataset"] == "original_450":
                parts.append("SVC(probability=True), original configuration")
            else:
                parts.append("SENSITIVITY: SVC(probability=False), not the original configuration")
        if r["model"] == "RandomForest" and r["invariant_std"] == 0:
            parts.append("invariant accuracy is 100% in every corner (ceiling)")
        return "; ".join(parts)

    wide["note"] = wide.apply(note, axis=1)
    wide["dataset"] = pd.Categorical(wide["dataset"], DATASET_ORDER, ordered=True)
    wide["model"] = pd.Categorical(wide["model"], MODEL_ORDER, ordered=True)
    wide = wide.sort_values(["dataset", "model"]).reset_index(drop=True)

    out_cols = [
        "dataset", "model",
        "raw_mean", "raw_std", "raw_min", "raw_max",
        "invariant_mean", "invariant_std", "invariant_min", "invariant_max",
        "gain_pp", "note",
    ]
    table = wide[out_cols].copy()
    table_path = AUDIT_DIR / "final_comparison_table.csv"
    table.to_csv(table_path, index=False, float_format="%.4f")

    # ---- figure: original_450 vs clean_main_428 -------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = []
    for m in MODEL_ORDER:
        labels.append(m + ("*" if m == "LinearSVM" else ""))
    x = np.arange(len(MODEL_ORDER))
    width = 0.2
    series = [
        ("original_450", "raw", "Original raw", "#c9c9c9"),
        ("original_450", "invariant", "Original invariant", "#7a9cc6"),
        ("clean_main_428", "raw", "Cleaned raw", "#e0a86b"),
        ("clean_main_428", "invariant", "Cleaned invariant", "#2c5f9e"),
    ]
    for i, (ds, fs, lab, colr) in enumerate(series):
        sub = wide[wide["dataset"] == ds].set_index("model")
        means = [sub.loc[m, f"{fs}_mean"] * 100 for m in MODEL_ORDER]
        stds = [sub.loc[m, f"{fs}_std"] * 100 for m in MODEL_ORDER]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=3,
               label=lab, color=colr, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean LOCO accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Raw vs PVT-invariant features: original (450 rows) vs cleaned (428 rows)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.01, 0.01,
        "Error bars: sample std (ddof=1) across 15 held-out PVT corners.  "
        "*LinearSVM cleaned bars: SVC(probability=False) sensitivity run.  "
        "RF invariant (cleaned) = 100% in every corner (ceiling).",
        fontsize=8, ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig_path = AUDIT_DIR / "final_comparison_figure.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(table.drop(columns=["note"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved table:  {table_path}")
    print(f"Saved figure: {fig_path}")
    print("No files under src/, data/ or reports/final_results/ were written.")


if __name__ == "__main__":
    main()
