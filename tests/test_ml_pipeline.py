import numpy as np
import pandas as pd
import pytest
from src.ml_pipeline import (
    RAW_FEATURES,
    INVARIANT_FEATURES,
    get_model_instances,
    compute_metrics,
    run_standard_split_protocol,
    run_leave_one_corner_out_protocol,
)


@pytest.fixture
def mock_dataset():
    """Create a minimal synthetic dataset matching the schema across 3 corners."""
    np.random.seed(42)
    corners = ["C_27C_1P0V", "0P9V_M40C", "1P1V_125C"]
    classes = [0, 1, 2, 3, 4]
    rows = []
    sample_counter = 1

    for c in corners:
        parts = c.split("_")
        vdd = 1.0 if parts[0] == "1P0V" else (0.9 if parts[0] == "0P9V" else 1.1)
        temp = 27.0 if parts[1] == "27C" else (-40.0 if parts[1] == "M40C" else 125.0)

        for fc in classes:
            for s_idx in range(6):
                row = {
                    "sample_id": f"SRAM_F{fc}_S{s_idx}_{parts[1]}_{parts[0]}_{sample_counter:04d}",
                    "corner_id": c,
                    "pvt_type": "in_distribution" if c == "C_27C_1P0V" else "ood",
                    "vdd_v": vdd,
                    "temperature_c": temp,
                    "fault_class": fc,
                    "fault_name": f"Fault_{fc}",
                    "fault_target": "target",
                    "severity_value": float(s_idx + 1),
                    # Raw features
                    "i_ddq_uA": float(np.random.normal(5.0, 0.5)),
                    "t_write_ps": float(np.random.normal(250.0, 20.0)),
                    "i_write_peak_mA": float(np.random.normal(1.2, 0.1)),
                    "dv_bl_strobe_mV": float(np.random.normal(150.0, 10.0)),
                    "t_sense_ps": float(np.random.normal(180.0, 15.0)),
                    "v_bump_mV": float(np.random.normal(30.0, 5.0)),
                    "hold_pass": 1,
                    "write_pass": 1,
                    "read_stable_pass": 1,
                    "snm_hold_v": float(np.random.normal(0.25, 0.02)),
                    "snm_read_v": float(np.random.normal(0.12, 0.02)),
                    "snm_asym_hold_v": float(np.random.normal(0.01, 0.005)),
                    "snm_asym_read_v": float(np.random.normal(0.01, 0.005)),
                    "snm_drop_v": float(np.random.normal(0.13, 0.02)),
                    "is_bistable": 1,
                    # Invariant features
                    "iddq_norm": float(np.random.normal(1.0, 0.1)),
                    "t_write_norm": float(np.random.normal(1.0, 0.1)),
                    "dv_bl_norm": float(np.random.normal(1.0, 0.1)),
                    "snm_hold_norm": float(np.random.normal(1.0, 0.1)),
                    "snm_read_norm": float(np.random.normal(1.0, 0.1)),
                    "read_write_delay_ratio": 0.72,
                    "snm_ratio": 0.48,
                    "dv_bl_vdd_ratio": 0.15,
                    "converged": 1,
                }
                rows.append(row)
                sample_counter += 1

    return pd.DataFrame(rows)


def test_model_instantiation():
    models = get_model_instances(random_state=42)
    expected_keys = {"RandomForest", "LogisticRegression", "LinearSVM", "HistGradientBoosting", "NonML_Baseline"}
    assert set(models.keys()) == expected_keys


def test_compute_metrics():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0


def test_standard_split_protocol(mock_dataset):
    feature_sets = {"raw": RAW_FEATURES, "invariant": INVARIANT_FEATURES}
    results, models = run_standard_split_protocol(mock_dataset, feature_sets, random_state=42)
    assert len(results) > 0
    for r in results:
        assert r["protocol"] == "standard_split"
        assert r["corner"] == "C_27C_1P0V"
        assert 0.0 <= r["accuracy"] <= 1.0


def test_leave_one_corner_out_protocol(mock_dataset):
    feature_sets = {"raw": RAW_FEATURES, "invariant": INVARIANT_FEATURES}
    results, models = run_leave_one_corner_out_protocol(mock_dataset, feature_sets, random_state=42)
    # 3 corners * 2 feature sets * 5 models = 30 evaluations
    assert len(results) == 3 * 2 * 5
    corners_evaluated = set(r["corner"] for r in results)
    assert corners_evaluated == {"C_27C_1P0V", "0P9V_M40C", "1P1V_125C"}
