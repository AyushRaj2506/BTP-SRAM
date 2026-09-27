"""
tests/test_feature_extractor.py -- Unit tests for dynamic and static feature extraction.
"""

from pathlib import Path
import pytest
from feature_extractor import (
    extract_hold_features,
    extract_write_features,
    extract_read_features,
)

RAW_DIR = Path(__file__).parent.parent / "data" / "pilot_batch"


def _get_or_run_raw(stim_type: str, tmp_path: Path) -> Path:
    """Find an existing raw file or run a single simulation on-the-fly."""
    raw_candidates = list(RAW_DIR.rglob(f"*healthy_{stim_type}*.raw"))
    if raw_candidates:
        return raw_candidates[0]

    from testbench_composer import compose_testbench
    from simulation_runner import run_simulation

    core = Path(__file__).parent.parent / "circuits" / "core" / "cell_core_healthy.net"
    tb = str(tmp_path / f"tb_healthy_{stim_type}.net")
    compose_testbench(str(core), stim_type, tb)
    res = run_simulation(tb, str(tmp_path / "raw"), str(tmp_path / "logs"))
    return Path(res["raw_path"])


def test_hold_feature_extraction(tmp_path):
    raw_path = _get_or_run_raw("hold", tmp_path)
    feats = extract_hold_features(str(raw_path))
    assert feats["hold_pass"] is True
    assert feats["v_q_final"] == 1.0000
    assert feats["v_qb_final"] == 0.0000
    assert feats["i_ddq_uA"] >= 0.0


def test_write_feature_extraction(tmp_path):
    raw_path = _get_or_run_raw("write", tmp_path)
    feats = extract_write_features(str(raw_path), target_state=1)
    assert feats["write_pass"] is True
    assert feats["v_q_final"] == 1.0000
    assert feats["v_qb_final"] == 0.0000
    assert feats["t_write_ps"] > 0
    assert feats["i_peak_mA"] > 0


def test_read_feature_extraction(tmp_path):
    raw_path = _get_or_run_raw("read", tmp_path)
    feats = extract_read_features(str(raw_path), stored_state=1)
    assert feats["read_stable_pass"] is True
    assert feats["v_q_final"] == 1.0000
    assert feats["v_qb_final"] == 0.0000
    assert feats["dv_bl_strobe_mV"] > 0
    assert feats["t_sense_ps"] > 0
    assert feats["v_bump_mV"] >= 0
