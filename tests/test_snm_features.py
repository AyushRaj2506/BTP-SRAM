"""
tests/test_snm_features.py -- Unit tests for SNM feature extraction pipeline (Task 4).

Covers:
  - Healthy cell run through full pipeline:
    core -> compose snm_read / snm_hold -> LTspice simulation -> extract_snm_features
  - Read SNM = 263.7 mV +/- 1 mV
  - Hold SNM = 470.0 mV +/- 5 mV
  - Verification of all feature keys, value types, read/hold drop, and _ok flag
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from testbench_composer import compose_testbench
from simulation_runner import run_simulation
from feature_extractor import extract_snm_features, compute_read_hold_snm_drop

HEALTHY_CORE = PROJECT_ROOT / "circuits" / "core" / "cell_core_healthy.net"


def test_snm_read_feature_extraction_pipeline(tmp_path):
    """
    Healthy cell run through the real pipeline gives read SNM = 263.7 mV +/- 1 mV,
    _ok is True, and all required keys are present.
    """
    tb_path = str(tmp_path / "tb_healthy_snm_read.net")
    compose_testbench(str(HEALTHY_CORE), "snm_read", tb_path)

    raw_dir = tmp_path / "raw"
    log_dir = tmp_path / "logs"
    sim_res = run_simulation(tb_path, str(raw_dir), str(log_dir))
    assert sim_res["converged"] is True

    feats = extract_snm_features(sim_res["raw_path"], mode="read", vdd=1.0)

    expected_keys = {
        "read_snm_v",
        "read_snm_state0_v",
        "read_snm_state1_v",
        "read_snm_asym_v",
        "read_snm_ok",
        "read_bistable",
    }
    assert set(feats.keys()) == expected_keys

    assert feats["read_snm_ok"] is True
    assert feats["read_bistable"] is True
    assert isinstance(feats["read_snm_v"], float)
    assert isinstance(feats["read_snm_asym_v"], float)

    # 263.7 mV +/- 1 mV -> 0.2627 to 0.2647 V
    assert 0.2627 <= feats["read_snm_v"] <= 0.2647
    assert 0.2627 <= feats["read_snm_state0_v"] <= 0.2647
    assert 0.2627 <= feats["read_snm_state1_v"] <= 0.2647
    assert feats["read_snm_asym_v"] < 0.0010


def test_snm_hold_feature_extraction_and_drop(tmp_path):
    """
    Test snm_hold features and compute_read_hold_snm_drop.
    """
    tb_hold = str(tmp_path / "tb_healthy_snm_hold.net")
    tb_read = str(tmp_path / "tb_healthy_snm_read.net")
    compose_testbench(str(HEALTHY_CORE), "snm_hold", tb_hold)
    compose_testbench(str(HEALTHY_CORE), "snm_read", tb_read)

    raw_dir = tmp_path / "raw"
    log_dir = tmp_path / "logs"
    res_hold = run_simulation(tb_hold, str(raw_dir), str(log_dir))
    res_read = run_simulation(tb_read, str(raw_dir), str(log_dir))

    f_hold = extract_snm_features(res_hold["raw_path"], mode="hold", vdd=1.0)
    f_read = extract_snm_features(res_read["raw_path"], mode="read", vdd=1.0)

    assert f_hold["hold_snm_ok"] is True
    assert f_hold["hold_bistable"] is True
    # 470 mV +/- 5 mV
    assert 0.465 <= f_hold["hold_snm_v"] <= 0.475

    drop = compute_read_hold_snm_drop(f_hold, f_read)
    # Expected drop ~ 470 - 263.7 = ~206 mV
    assert 0.200 <= drop <= 0.212


def test_bridge_fault_flags_nonmonotonic(tmp_path):
    """
    Severe bridging fault (2k) destroys inverter monotonicity and must be flagged
    with read_snm_ok == False (reason: snm_nonmonotonic).
    """
    from fault_injection import inject_bridging_fault
    from utils import load_cell_core

    lines = load_cell_core(str(HEALTHY_CORE))
    core_p = str(tmp_path / "core_bridge_2k.net")
    inject_bridging_fault(list(lines), 2000, core_p)

    tb_p = str(tmp_path / "tb_bridge_snm_read.net")
    compose_testbench(core_p, "snm_read", tb_p)

    res = run_simulation(tb_p, str(tmp_path / "raw"), str(tmp_path / "logs"))
    assert res["converged"] is False
    assert res["warning"] == "snm_nonmonotonic"

    feats = extract_snm_features(res["raw_path"], mode="read", vdd=1.0)
    assert feats["read_snm_ok"] is False
