import numpy as np
import pandas as pd
import pytest
from src.baseline_detector import (
    snm_threshold_detector,
    standby_current_detector,
    BaselineRuleDetector,
)


def test_snm_threshold_detector():
    threshold = 0.20
    # Above threshold -> Healthy (False)
    assert snm_threshold_detector(0.25, threshold) is False
    # Below threshold -> Faulty (True)
    assert snm_threshold_detector(0.15, threshold) is True
    # NaN -> Faulty (True)
    assert snm_threshold_detector(np.nan, threshold) is True


def test_standby_current_detector():
    mean = 10.0
    std = 1.0
    # Nominal -> Healthy (False)
    assert standby_current_detector(11.0, mean, std, n_sigma=3.0) is False
    # Extreme leakage (> 13.0) -> Faulty (True)
    assert standby_current_detector(14.5, mean, std, n_sigma=3.0) is True
    # NaN -> Faulty (True)
    assert standby_current_detector(np.nan, mean, std) is True


def test_baseline_rule_detector_fit_and_predict():
    # Construct synthetic training set
    np.random.seed(42)
    n_healthy = 30
    n_faulty = 30

    healthy_data = {
        "hold_snm_v": np.random.normal(0.25, 0.01, n_healthy),
        "standby_leakage_current_na": np.random.normal(5.0, 0.2, n_healthy),
        "write_time_ps": np.random.normal(250.0, 10.0, n_healthy),
        "read_access_time_ps": np.random.normal(180.0, 5.0, n_healthy),
        "write_pass": np.ones(n_healthy, dtype=int),
    }

    faulty_data = {
        # Some degraded SNM, high leakage, failed writes
        "hold_snm_v": np.concatenate([np.random.normal(0.10, 0.02, 15), np.random.normal(0.25, 0.01, 15)]),
        "standby_leakage_current_na": np.concatenate([np.random.normal(50.0, 5.0, 15), np.random.normal(5.0, 0.2, 15)]),
        "write_time_ps": np.random.normal(300.0, 50.0, n_faulty),
        "read_access_time_ps": np.random.normal(200.0, 20.0, n_faulty),
        "write_pass": np.array([0] * 10 + [1] * 20),
    }

    df_train = pd.DataFrame({k: np.concatenate([healthy_data[k], faulty_data[k]]) for k in healthy_data})
    y_train = np.array([0] * n_healthy + [1] * n_faulty)

    detector = BaselineRuleDetector(n_sigma=3.0)
    detector.fit(df_train, y_train)

    assert "snm_min" in detector.thresholds_
    assert "leakage_max" in detector.thresholds_
    assert detector.thresholds_["snm_min"] < 0.25

    preds = detector.predict(df_train)
    assert len(preds) == len(df_train)
    # Healthy samples should have zero or minimal false positives
    fp_rate = np.mean(preds[:n_healthy] == 1)
    assert fp_rate < 0.10  # Low false positive rate

    metrics = detector.evaluate(df_train, y_train)
    assert metrics["accuracy"] > 0.70
    assert metrics["recall"] > 0.50
