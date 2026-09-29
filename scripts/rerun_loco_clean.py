"""Cleaned LOCO evaluation for SRAM fault diagnosis.

Runs the same four ML models + 3-sigma baseline as src/ml_pipeline.py
on the approved cleaned datasets, using the exact raw/invariant feature
lists and model hyperparameters from src/ml_pipeline.py.

Outputs are written only under reports/audit/.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.baseline_detector import BaselineRuleDetector


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "reports" / "audit"
FINAL_RESULTS_DIR = PROJECT_ROOT / "reports" / "final_results"

RAW_FEATURES = [
    "i_ddq_uA",
    "t_write_ps",
    "i_write_peak_mA",
    "dv_bl_strobe_mV",
    "t_sense_ps",
    "v_bump_mV",
    "hold_pass",
    "write_pass",
    "read_stable_pass",
    "snm_hold_v",
    "snm_read_v",
    "snm_asym_hold_v",
    "snm_asym_read_v",
    "snm_drop_v",
    "is_bistable",
]

INVARIANT_FEATURES = [
    "iddq_norm",
    "t_write_norm",
    "dv_bl_norm",
    "snm_hold_norm",
    "snm_read_norm",
    "read_write_delay_ratio",
    "snm_ratio",
    "dv_bl_vdd_ratio",
    "hold_pass",
    "write_pass",
    "read_stable_pass",
    "is_bistable",
]

FEATURE_SETS = {
    "raw": RAW_FEATURES,
    "invariant": INVARIANT_FEATURES,
}


def get_model_instances(random_state=42):
    """Exact model definitions from src/ml_pipeline.py."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=random_state),
        ),
        "LinearSVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", random_state=random_state, probability=True),
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            random_state=random_state,
        ),
        "NonML_Baseline": BaselineRuleDetector(n_sigma=3.0),
    }


def compute_metrics(y_true, y_pred):
    """Same multiclass/binary baseline scoring logic as src/ml_pipeline.py."""
    if np.max(y_pred) <= 1 and np.max(y_true) > 1:
        y_true_eval = (y_true > 0).astype(int)
    else:
        y_true_eval = y_true

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true_eval, y_pred)),
        "precision": float(
            precision_score(y_true_eval, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_eval, y_pred, average="macro", zero_division=0)
        ),
        "f1": float(
            f1_score(y_true_eval, y_pred, average="macro", zero_division=0)
        ),
    }


def load_clean_dataset(path):
    df = pd.read_csv(path)

    required = {"sample_id", "fault_class", "corner_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )

    missing_features = [
        c for cols in FEATURE_SETS.values() for c in cols if c not in df.columns
    ]
    if missing_features:
        raise ValueError(
            f"{path.name} is missing feature columns: {sorted(set(missing_features))}"
        )

    if df[RAW_FEATURES + INVARIANT_FEATURES].isna().any().any():
        bad = df[RAW_FEATURES + INVARIANT_FEATURES].isna().sum()
        bad = bad[bad > 0]
        raise ValueError(f"NaNs found in features:\n{bad}")

    return df


def run_loco(df, dataset_name):
    corners = sorted(df["corner_id"].unique())
    results = []

    print(f"\n{'=' * 72}")
    print(f"{dataset_name}: {len(df)} rows, {len(corners)} PVT corners")
    print(f"{'=' * 72}")

    for corner in corners:
        train_df = df[df["corner_id"] != corner].copy()
        test_df = df[df["corner_id"] == corner].copy()

        train_ids = set(train_df["sample_id"])
        test_ids = set(test_df["sample_id"])
        overlap = train_ids.intersection(test_ids)
        assert not overlap, f"LOCO leakage on {corner}: {overlap}"

        healthy_test = int((test_df["fault_class"] == 0).sum())

        print(
            f"{corner}: test_rows={len(test_df):2d}, "
            f"healthy_test={healthy_test:2d}, "
            f"train_rows={len(train_df):3d}"
        )

        y_train = train_df["fault_class"].to_numpy()
        y_test = test_df["fault_class"].to_numpy()

        for feature_set, feature_cols in FEATURE_SETS.items():
            X_train = train_df[feature_cols]
            X_test = test_df[feature_cols]

            for model_name, model in get_model_instances().items():
                
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred)

                results.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "feature_set": feature_set,
                        "protocol": "leave_one_corner_out",
                        "corner": corner,
                        "test_rows": len(test_df),
                        "healthy_test_rows": healthy_test,
                        **metrics,
                    }
                )

    return pd.DataFrame(results)


def summarize(results):
    """Mean/std/min/max across held-out corners; std uses ddof=1."""
    summary = (
        results.groupby(["dataset", "model", "feature_set"])["accuracy"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    return summary


def load_old_summary():
    path = FINAL_RESULTS_DIR / "loco_summary.csv"
    if not path.exists():
        return None

    old = pd.read_csv(path)
    expected = {"model", "feature_set", "mean", "std", "min", "max"}
    if not expected.issubset(old.columns):
        return None

    old = old.copy()
    old.insert(0, "dataset", "original_450")
    return old[["dataset", "model", "feature_set", "mean", "std", "min", "max"]]


def print_comparison(summary, old_summary):
    print(f"\n{'=' * 100}")
    print("CLEANED LOCO SUMMARY — ACCURACY")
    print("std = sample standard deviation (ddof=1)")
    print(f"{'=' * 100}")

    if old_summary is not None:
        print("\nOLD ORIGINAL 450-ROW RESULTS:")
        print(old_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nNEW CLEANED RESULTS:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if old_summary is not None:
        old_keyed = old_summary.set_index(["model", "feature_set"])
        new_keyed = summary.set_index(["dataset", "model", "feature_set"])

        rows = []
        for _, r in summary.iterrows():
            key = (r["model"], r["feature_set"])
            if key in old_keyed.index:
                old_r = old_keyed.loc[key]
                rows.append(
                    {
                        "dataset": r["dataset"],
                        "model": r["model"],
                        "feature_set": r["feature_set"],
                        "old_mean": old_r["mean"],
                        "old_std": old_r["std"],
                        "old_min": old_r["min"],
                        "old_max": old_r["max"],
                        "new_mean": r["mean"],
                        "new_std": r["std"],
                        "new_min": r["min"],
                        "new_max": r["max"],
                    }
                )

        if rows:
            comparison = pd.DataFrame(rows)
            print("\nOLD vs NEW:")
            print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            return comparison

    return None


def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "clean_main_428": AUDIT_DIR / "dataset_clean_main.csv",
        "clean_strict_302": AUDIT_DIR / "dataset_clean_strict.csv",
    }

    all_results = []

    for name, path in datasets.items():
        if not path.exists():
            raise FileNotFoundError(f"Cleaned dataset not found: {path}")

        df = load_clean_dataset(path)
        results = run_loco(df, name)

        out_path = AUDIT_DIR / f"{name}_loco_by_corner.csv"
        results.to_csv(out_path, index=False)
        print(f"Saved per-corner results: {out_path}")

        all_results.append(results)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_path = AUDIT_DIR / "clean_loco_by_corner_all.csv"
    all_results_df.to_csv(all_results_path, index=False)

    summary = summarize(all_results_df)
    summary_path = AUDIT_DIR / "clean_loco_summary.csv"
    summary.to_csv(summary_path, index=False)

    old_summary = load_old_summary()
    comparison = print_comparison(summary, old_summary)

    if comparison is not None:
        comparison_path = AUDIT_DIR / "clean_loco_vs_original.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"\nSaved comparison: {comparison_path}")

    print(f"Saved combined per-corner results: {all_results_path}")
    print(f"Saved summary: {summary_path}")
    print("\nNOTE: Reference-cell provenance is probably fine, not proven.")
    print("No files under reports/final_results/ were written by this script.")


if __name__ == "__main__":
    main()
