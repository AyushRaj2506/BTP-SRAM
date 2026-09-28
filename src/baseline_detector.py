"""Non-ML Baseline Detector for SRAM Fault Diagnosis.

Implements Section 6.6 of PLAN (3).md:
- snm_threshold_detector
- standby_current_detector
- BaselineRuleDetector (scikit-learn compatible classifier)

Derives detection thresholds strictly from the healthy-class distribution
in the training split to prevent data leakage.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def snm_threshold_detector(hold_snm: float, threshold: float) -> bool:
    """Detect fault based on hold-SNM threshold.
    
    A healthy 6T cell has a nominal SNM around 0.15 - 0.35 V. If hold_snm
    falls below threshold, the cell is flagged as faulty (True).
    
    Args:
        hold_snm: Measured static noise margin in volts.
        threshold: Minimum acceptable SNM in volts.
        
    Returns:
        True if fault detected (hold_snm < threshold), False otherwise.
    """
    if np.isnan(hold_snm):
        return True
    return bool(hold_snm < threshold)


def standby_current_detector(
    standby_current: float,
    mean: float,
    std: float,
    n_sigma: float = 3.0,
    two_sided: bool = False,
) -> bool:
    """Detect fault based on standby leakage current statistics.
    
    Faults like bridging or gate oxide degradation increase leakage
    significantly. If standby current exceeds mean + n_sigma * std,
    the cell is flagged as faulty (True).
    
    Args:
        standby_current: Measured standby leakage current in nA.
        mean: Training healthy-class mean standby leakage in nA.
        std: Training healthy-class std standby leakage in nA.
        n_sigma: Number of standard deviations for threshold (default: 3.0).
        two_sided: If True, flags values outside [mean - n*std, mean + n*std].
        
    Returns:
        True if fault detected, False otherwise.
    """
    if np.isnan(standby_current):
        return True
    
    # Safe guard for zero std
    effective_std = max(float(std), 1e-12)
    upper_threshold = mean + n_sigma * effective_std
    
    if two_sided:
        lower_threshold = max(0.0, mean - n_sigma * effective_std)
        return bool(standby_current < lower_threshold or standby_current > upper_threshold)
    return bool(standby_current > upper_threshold)


class BaselineRuleDetector(BaseEstimator, ClassifierMixin):
    """Rule-based non-ML fault detector trained on healthy-class statistics.
    
    Thresholds are derived strictly from Class 0 (Healthy) samples in the
    training set, ensuring no test-split information leakage.
    
    Flags a cell as faulty (label 1) if:
    1. write_pass == 0 (functional write failure)
    2. hold_snm < mean_snm - n_sigma * std_snm
    3. standby_leakage > mean_leakage + n_sigma * std_leakage
    4. read_access_time > mean_read + n_sigma * std_read
    5. write_time > mean_write + n_sigma * std_write
    """

    def __init__(self, n_sigma: float = 3.0, binary_mode: bool = True):
        self.n_sigma = n_sigma
        self.binary_mode = binary_mode
        self.thresholds_: Dict[str, float] = {}
        self.healthy_stats_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None):
        """Fit baseline thresholds using healthy (y == 0) training samples.
        
        Args:
            X: Features dataframe or numpy array.
            y: Ground-truth fault classes (0: healthy, 1..4: faults).
               If y is None, assumes all samples in X are healthy.
        """
        if not isinstance(X, pd.DataFrame):
            # If numpy array, convert to DataFrame with default names
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        if y is not None:
            y_arr = np.asarray(y)
            healthy_mask = (y_arr == 0)
            if np.sum(healthy_mask) == 0:
                raise ValueError("No healthy (y == 0) samples found in training set to establish baseline.")
            healthy_df = df[healthy_mask]
        else:
            healthy_df = df

        self.healthy_stats_ = {}
        self.thresholds_ = {}

        # 1. Hold SNM threshold (lower bound)
        snm_col = "hold_snm_v" if "hold_snm_v" in healthy_df.columns else None
        if snm_col:
            m = float(healthy_df[snm_col].mean())
            s = float(healthy_df[snm_col].std(ddof=1) if len(healthy_df) > 1 else 0.0)
            self.healthy_stats_["hold_snm"] = (m, s)
            self.thresholds_["snm_min"] = max(0.0, m - self.n_sigma * s)

        # 2. Standby leakage threshold (upper bound)
        leakage_col = None
        for col in ["standby_leakage_current_na", "standby_leakage_na", "i_standby_leakage_na"]:
            if col in healthy_df.columns:
                leakage_col = col
                break
        if leakage_col:
            m = float(healthy_df[leakage_col].mean())
            s = float(healthy_df[leakage_col].std(ddof=1) if len(healthy_df) > 1 else 0.0)
            self.healthy_stats_["leakage"] = (m, s)
            self.thresholds_["leakage_max"] = m + self.n_sigma * s

        # 3. Write time threshold (upper bound)
        write_col = None
        for col in ["write_time_ps", "write_time_ns", "t_write_ps"]:
            if col in healthy_df.columns:
                write_col = col
                break
        if write_col:
            m = float(healthy_df[write_col].mean())
            s = float(healthy_df[write_col].std(ddof=1) if len(healthy_df) > 1 else 0.0)
            self.healthy_stats_["write_time"] = (m, s)
            self.thresholds_["write_time_max"] = m + self.n_sigma * s

        # 4. Read access time threshold (upper bound)
        read_col = None
        for col in ["read_access_time_ps", "read_access_time_ns", "t_read_access_ps"]:
            if col in healthy_df.columns:
                read_col = col
                break
        if read_col:
            m = float(healthy_df[read_col].mean())
            s = float(healthy_df[read_col].std(ddof=1) if len(healthy_df) > 1 else 0.0)
            self.healthy_stats_["read_time"] = (m, s)
            self.thresholds_["read_time_max"] = m + self.n_sigma * s

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict whether each sample is Healthy (0) or Faulty (1).
        
        Args:
            X: Features dataframe or numpy array.
            
        Returns:
            np.ndarray of predicted labels (0: Healthy, 1: Faulty).
        """
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        n_samples = len(df)
        preds = np.zeros(n_samples, dtype=int)

        # Check write pass failure
        if "write_pass" in df.columns:
            preds[df["write_pass"] == 0] = 1

        # Check SNM violation
        if "snm_min" in self.thresholds_:
            snm_col = "hold_snm_v" if "hold_snm_v" in df.columns else None
            if snm_col:
                val = df[snm_col].values
                viol = np.isnan(val) | (val < self.thresholds_["snm_min"])
                preds[viol] = 1

        # Check Standby leakage violation
        if "leakage_max" in self.thresholds_:
            leakage_col = None
            for col in ["standby_leakage_current_na", "standby_leakage_na", "i_standby_leakage_na"]:
                if col in df.columns:
                    leakage_col = col
                    break
            if leakage_col:
                val = df[leakage_col].values
                viol = np.isnan(val) | (val > self.thresholds_["leakage_max"])
                preds[viol] = 1

        # Check Write time violation
        if "write_time_max" in self.thresholds_:
            write_col = None
            for col in ["write_time_ps", "write_time_ns", "t_write_ps"]:
                if col in df.columns:
                    write_col = col
                    break
            if write_col:
                val = df[write_col].values
                viol = np.isnan(val) | (val > self.thresholds_["write_time_max"])
                preds[viol] = 1

        # Check Read time violation
        if "read_time_max" in self.thresholds_:
            read_col = None
            for col in ["read_access_time_ps", "read_access_time_ns", "t_read_access_ps"]:
                if col in df.columns:
                    read_col = col
                    break
            if read_col:
                val = df[read_col].values
                viol = np.isnan(val) | (val > self.thresholds_["read_time_max"])
                preds[viol] = 1

        return preds

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Compute standard detection metrics against ground truth.
        
        If binary_mode is True, converts multiclass y (0 vs 1..4) to binary (0 vs 1).
        
        Returns:
            Dict containing accuracy, precision, recall, f1.
        """
        y_true = np.asarray(y)
        if self.binary_mode and np.max(y_true) > 1:
            y_true = (y_true > 0).astype(int)

        y_pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
