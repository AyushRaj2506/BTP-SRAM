"""Comprehensive Physics and Data Quality Audit for SRAM Fault Dataset."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif

DATA_PATH = Path("data/sram_fault_dataset.csv")

def main():
    if not DATA_PATH.exists():
        print("Dataset not found!")
        return

    df = pd.read_csv(DATA_PATH)
    print("=================================================================")
    print("      SRAM FAULT DIAGNOSIS - DATASET PHYSICAL & STATISTICAL AUDIT")
    print("=================================================================")
    print(f"Total Rows:    {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Total Missing / NaN Values: {df.isnull().sum().sum()}")
    print(f"Total Infinite Values:      {np.isinf(df.select_dtypes(include=np.number)).sum().sum()}")

    # 1. Physics: Temperature Scaling
    print("\n--- 1. SEMICONDUCTOR SCALING: LEAKAGE VS TEMPERATURE (CLASS 0) ---")
    h_df = df[df["fault_class"] == 0]
    t_summary = h_df.groupby("temperature_c")["i_ddq_uA"].agg(["mean", "std", "min", "max"])
    print(t_summary.round(4))

    # 2. Physics: Voltage Scaling
    print("\n--- 2. TRANSISTOR OVERDRIVE: WRITE DELAY VS VDD (CLASS 0) ---")
    v_summary = h_df.groupby("vdd_v")["t_write_ps"].agg(["mean", "std", "min", "max"])
    print(v_summary.round(2))

    # 3. Physics: Resistive Open Fault Scaling
    print("\n--- 3. CLASS 1 (RESISTIVE OPEN) DELAY SCALING ---")
    f1_df = df[df["fault_class"] == 1]
    f1_summary = f1_df.groupby("severity_value")[["t_write_ps", "dv_bl_strobe_mV", "write_pass"]].mean()
    print(f1_summary.round(2))

    # 4. Physics: Bridging Fault Leakage & Bistability
    print("\n--- 4. CLASS 2 (BRIDGING) OHMIC LEAKAGE SCALING ---")
    f2_df = df[df["fault_class"] == 2]
    f2_summary = f2_df.groupby("severity_value")[["i_ddq_uA", "snm_hold_v", "write_pass", "is_bistable"]].mean()
    print(f2_summary.round(3))

    # 5. Physics: Vth Drift Inverters vs Access
    print("\n--- 5. CLASS 3 & 4 (Vth DRIFT) SELECTIVE SENSITIVITY ---")
    f3_df = df[df["fault_class"] == 3]
    f4_df = df[df["fault_class"] == 4]
    f3_s = f3_df.groupby("severity_value")[["snm_hold_v", "t_write_ps"]].mean().round(3)
    f4_s = f4_df.groupby("severity_value")[["snm_hold_v", "t_sense_ps"]].mean().round(3)
    print("Class 3 (Storage pair drift) -> Monotonic SNM Degradation:")
    print(f3_s)
    print("Class 4 (Access pair drift) -> Sense Delay Increases, Hold SNM unaffected:")
    print(f4_s)

    # 6. Feature Dispersion Reduction: Raw vs Invariant (CV %)
    print("\n--- 6. PVT DISPERSION REDUCTION (COEFFICIENT OF VARIATION) ---")
    metrics = [
        ("i_ddq_uA", "iddq_norm"),
        ("t_write_ps", "t_write_norm"),
        ("snm_hold_v", "snm_hold_norm"),
        ("snm_read_v", "snm_read_norm"),
        ("dv_bl_strobe_mV", "dv_bl_norm"),
    ]
    print(f"{'Feature':<18} | {'Raw Mean':<10} | {'Raw Std':<10} | {'Raw CV (%)':<10} | {'Norm Mean':<10} | {'Norm Std':<10} | {'Norm CV (%)':<10}")
    print("-" * 88)
    for r_col, n_col in metrics:
        r_vals = h_df[r_col].values
        n_vals = h_df[n_col].values
        r_cv = (np.std(r_vals) / np.mean(r_vals)) * 100
        n_cv = (np.std(n_vals) / np.mean(n_vals)) * 100
        print(f"{r_col:<18} | {np.mean(r_vals):<10.3f} | {np.std(r_vals):<10.3f} | {r_cv:<10.1f} | {np.mean(n_vals):<10.3f} | {np.std(n_vals):<10.3f} | {n_cv:<10.1f}")

    # 7. Fisher Discriminant & Mutual Info for ML Separability
    print("\n--- 7. DISCRIMINATORY POWER & MUTUAL INFORMATION (5 CLASSES) ---")
    eval_cols = [
        "i_ddq_uA", "t_write_ps", "i_write_peak_mA", "dv_bl_strobe_mV",
        "t_sense_ps", "v_bump_mV", "hold_pass", "write_pass", "read_stable_pass",
        "snm_hold_v", "snm_read_v", "snm_drop_v", "is_bistable",
        "iddq_norm", "t_write_norm", "dv_bl_norm", "snm_hold_norm", "snm_read_norm",
        "read_write_delay_ratio", "snm_ratio", "dv_bl_vdd_ratio"
    ]
    f_vals, p_vals = f_classif(df[eval_cols], df["fault_class"])
    mi = mutual_info_classif(df[eval_cols], df["fault_class"], random_state=42)
    feat_df = pd.DataFrame({
        "Feature": eval_cols,
        "F-Stat": [round(f, 1) for f in f_vals],
        "p-val": [f"{p:.2e}" for p in p_vals],
        "Mutual_Info": [round(m, 3) for m in mi]
    }).sort_values("Mutual_Info", ascending=False)
    print(feat_df.head(10).to_string(index=False))
    print("=================================================================")

if __name__ == "__main__":
    main()
