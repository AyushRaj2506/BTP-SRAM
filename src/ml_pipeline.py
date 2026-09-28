"""Machine Learning Pipeline for SRAM Fault Diagnosis.

Implements Section 6.7 of PLAN (3).md:
- Classical ML models:
  1. RandomForestClassifier
  2. LogisticRegression
  3. SVC(kernel="linear")
  4. HistGradientBoostingClassifier
  5. BaselineRuleDetector (Non-ML baseline)
- Two evaluation protocols:
  1. Standard Split (Nominal 1.0V, 27°C corner, leakage-checked train/test)
  2. Leave-One-Corner-Out (LOCO across all 15 PVT corners)
- Feature sets:
  - Raw Features
  - Physics-Normalized Invariant Features
- Artifacts:
  - Saved model pickles in `models/{model}_{feature_set}_{protocol}.pkl`
  - Comparison table in `reports/final_results/comparison_table.csv`
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.baseline_detector import BaselineRuleDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "final_results"

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


def load_dataset(csv_path: Path = DATA_DIR / "sram_fault_dataset.csv") -> pd.DataFrame:
    """Load the SRAM fault simulation dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run generate_dataset.py first.")
    df = pd.read_csv(csv_path)
    return df


def get_model_instances(random_state: int = 42) -> Dict[str, object]:
    """Instantiate classical classifiers per Section 6.7."""
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics (multiclass macro-averaged)."""
    # For NonML_Baseline binary predictions on multiclass labels:
    # If y_pred is binary (0 vs 1) and y_true is multiclass (0..4),
    # map y_true to binary for fair baseline detection scoring.
    if np.max(y_pred) <= 1 and np.max(y_true) > 1:
        y_true_eval = (y_true > 0).astype(int)
    else:
        y_true_eval = y_true

    acc = float(accuracy_score(y_true_eval, y_pred))
    prec = float(precision_score(y_true_eval, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true_eval, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true_eval, y_pred, average="macro", zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def run_standard_split_protocol(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    random_state: int = 42,
) -> Tuple[List[Dict], Dict[str, object]]:
    """Protocol 1: Standard Split on Nominal Corner (1.0V, 27°C).
    
    Validates zero sample_id leakage between train and test splits.
    """
    nom_df = df[df["corner_id"] == "1P0V_27C"].copy()
    if len(nom_df) == 0:
        raise ValueError("Nominal corner 1P0V_27C not found in dataset.")

    results = []
    trained_models = {}

    # Only stratify if test split will have at least 1 sample per class
    n_classes = nom_df["fault_class"].nunique()
    test_size = 0.20
    if len(nom_df) * test_size >= n_classes and nom_df["fault_class"].value_counts().min() >= 2:
        stratify = nom_df["fault_class"]
    else:
        stratify = None

    train_df, test_df = train_test_split(
        nom_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Checkpoint Section 8: Split leakage assert
    train_ids = set(train_df["sample_id"])
    test_ids = set(test_df["sample_id"])
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: {overlap}"

    y_train = train_df["fault_class"].values
    y_test = test_df["fault_class"].values

    for feat_name, feat_cols in feature_sets.items():
        X_train = train_df[feat_cols]
        X_test = test_df[feat_cols]

        models = get_model_instances(random_state)
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)

            record = {
                "model": model_name,
                "feature_set": feat_name,
                "protocol": "standard_split",
                "corner": "1P0V_27C",
                **metrics,
            }
            results.append(record)

            save_key = f"{model_name.lower()}_{feat_name}_standard"
            trained_models[save_key] = model

    return results, trained_models


def run_leave_one_corner_out_protocol(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    random_state: int = 42,
) -> Tuple[List[Dict], Dict[str, object]]:
    """Protocol 2: Leave-One-Corner-Out (LOCO) Cross-PVT Validation.
    
    Trains on 14 corners, tests on the 1 held-out corner; repeats for all 15 corners.
    """
    unique_corners = sorted(df["corner_id"].unique())
    results = []
    trained_models = {}

    for held_out_corner in unique_corners:
        train_df = df[df["corner_id"] != held_out_corner].copy()
        test_df = df[df["corner_id"] == held_out_corner].copy()

        # Leakage assertion
        train_ids = set(train_df["sample_id"])
        test_ids = set(test_df["sample_id"])
        overlap = train_ids.intersection(test_ids)
        assert len(overlap) == 0, f"LOCO LEAKAGE DETECTED on corner {held_out_corner}: {overlap}"

        y_train = train_df["fault_class"].values
        y_test = test_df["fault_class"].values

        for feat_name, feat_cols in feature_sets.items():
            X_train = train_df[feat_cols]
            X_test = test_df[feat_cols]

            models = get_model_instances(random_state)
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred)

                record = {
                    "model": model_name,
                    "feature_set": feat_name,
                    "protocol": "leave_one_corner_out",
                    "corner": held_out_corner,
                    **metrics,
                }
                results.append(record)

                # Keep the model trained with nominal corner held-in (or latest representative)
                save_key = f"{model_name.lower()}_{feat_name}_leaveonecorner"
                if save_key not in trained_models:
                    trained_models[save_key] = model

    return results, trained_models


def main():
    parser = argparse.ArgumentParser(description="Run SRAM Fault Diagnosis ML Pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_DIR / "sram_fault_dataset.csv",
        help="Path to generated CSV dataset",
    )
    args = parser.parse_args()

    print("========================================================")
    print("SRAM FAULT DIAGNOSIS - STAGE 6: ML CLASSIFICATION")
    print("========================================================")

    df = load_dataset(args.dataset)
    print(f"Loaded dataset: {len(df)} samples across {df['corner_id'].nunique()} PVT corners.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_sets = {
        "raw": RAW_FEATURES,
        "invariant": INVARIANT_FEATURES,
    }

    # Protocol 1: Standard Split
    print("\n[1/2] Executing Protocol 1: Standard Split (Nominal Corner)...")
    res_std, models_std = run_standard_split_protocol(df, feature_sets)
    print(f"  Standard split completed: {len(res_std)} evaluations.")

    # Protocol 2: Leave-One-Corner-Out
    print("\n[2/2] Executing Protocol 2: Leave-One-Corner-Out (LOCO)...")
    res_loco, models_loco = run_leave_one_corner_out_protocol(df, feature_sets)
    print(f"  LOCO completed: {len(res_loco)} evaluations across 15 corners.")

    # Combine results
    all_results = res_std + res_loco
    res_df = pd.DataFrame(all_results)

    # Save Comparison Table per Section 6.7
    out_table_path = REPORTS_DIR / "comparison_table.csv"
    res_df.to_csv(out_table_path, index=False)
    print(f"\nSaved comparison table to: {out_table_path}")

    # Save Models per Section 9
    all_models = {**models_std, **models_loco}
    for model_key, model_obj in all_models.items():
        model_path = MODELS_DIR / f"{model_key}.pkl"
        joblib.dump(model_obj, model_path)
    print(f"Saved {len(all_models)} model pickles to: {MODELS_DIR}")

    # Generate Summary Aggregation
    print("\n========================================================")
    print("SUMMARY RESULTS: RAW vs. INVARIANT (LOCO Macro Accuracy)")
    print("========================================================")
    loco_df = res_df[res_df["protocol"] == "leave_one_corner_out"]
    summary = (
        loco_df.groupby(["model", "feature_set"])["accuracy"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    print(summary.to_string(index=False))

    summary_path = REPORTS_DIR / "loco_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved LOCO summary to: {summary_path}")


if __name__ == "__main__":
    main()
