"""
tests/test_snm_extraction.py -- Unit tests for butterfly-curve SNM extraction (Task 1).

Covers:
  1. Read reference (Appendix B deck & Appendix C checkpoints): 263.7 mV +/- 1 mV
  2. Hold reference (Appendix B deck): 470 mV +/- 5 mV
  3. Symmetric analytic VTC: lobe1 == lobe2 within 0.5 mV
  4. Asymmetric synthetic curves: module >= brute force (10 mV grid) and within 20 mV
  5. Monostable input (curves crossing once): returns 0 for missing lobe, no exception, no NaN
  6. Step-size robustness: subsampling to 2-5 mV changes SNM by < 0.1 mV
"""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from snm_extraction import lobe_sides, snm_from_curves, snm_from_raw
from simulation_runner import run_simulation
from spicelib.raw.raw_read import RawRead


def _get_or_run_reference_raw(mode: str, tmp_path: Path) -> Path:
    """Load existing reference raw or simulate on the fly."""
    val_raw = PROJECT_ROOT / "validation" / "snm_manual_check" / "raw"
    matches = list(val_raw.glob(f"snm_{mode}_healthy_*.raw"))
    if matches and matches[0].exists() and matches[0].stat().st_size > 0:
        return matches[0]

    net_file = PROJECT_ROOT / "circuits" / "reference" / f"snm_{mode}_healthy.net"
    raw_dir = tmp_path / "raw"
    log_dir = tmp_path / "logs"
    res = run_simulation(str(net_file), str(raw_dir), str(log_dir))
    return Path(res["raw_path"])


def _extract_dc_waves(raw_path: Path):
    raw = RawRead(str(raw_path))
    def _find(name):
        for tname in raw.get_trace_names():
            if tname.lower() == name.lower():
                return np.asarray(raw.get_trace(tname).get_wave(), float)
        raise KeyError(f"Trace {name} not found in {raw.get_trace_names()}")

    return _find("v(q)"), _find("v(qb)"), _find("v(q_c2)"), _find("v(qb_c2)")


def test_no_module_level_imports_beyond_numpy():
    """Module must have no imports beyond numpy at module level."""
    import snm_extraction
    # Check module imports: sys.modules contains libraries, but snm_extraction.__dict__
    # should only expose numpy and functions
    for k, v in snm_extraction.__dict__.items():
        if isinstance(v, type(sys)) and not k.startswith("__"):
            assert v.__name__ in ("numpy", "np"), f"Unexpected module-level import: {v.__name__}"


def test_read_reference(tmp_path):
    """
    1. Read reference: Appendix B read deck -> both lobes 263.7 mV +/- 1 mV.
       Check Appendix C read VTC points within 2 mV.
    """
    raw_path = _get_or_run_reference_raw("read", tmp_path)
    vq, vqb, vq_c2, vqb_c2 = _extract_dc_waves(raw_path)

    l1, l2 = lobe_sides(vq, vqb, vq_c2, vqb_c2, 1.0)
    snm = snm_from_curves(vq, vqb, vq_c2, vqb_c2, 1.0)

    # 263.7 mV +/- 1 mV
    assert 0.2627 <= l1 <= 0.2647, f"Read lobe1 = {l1*1e3:.2f} mV out of range"
    assert 0.2627 <= l2 <= 0.2647, f"Read lobe2 = {l2*1e3:.2f} mV out of range"
    assert 0.2627 <= snm <= 0.2647, f"Read SNM = {snm*1e3:.2f} mV out of range"
    assert abs(l1 - l2) < 0.0005, f"Healthy read SNM should be symmetric, diff={abs(l1-l2)*1e3:.2f} mV"

    # Appendix C checkpoints
    checkpoints = [
        (0.0, 1.0000),
        (0.3, 1.0000),
        (0.4, 1.0000),
        (0.45, 1.0000),
        (0.5, 0.4118),
        (0.55, 0.3588),
        (0.6, 0.3116),
        (0.7, 0.2204),
        (1.0, 0.1268),
    ]
    for q_val, expected_qb in checkpoints:
        measured_qb = float(np.interp(q_val, vq, vqb))
        assert abs(measured_qb - expected_qb) < 0.002, (
            f"VTC point at Q={q_val}V: measured {measured_qb:.4f}V != expected {expected_qb:.4f}V"
        )


def test_hold_reference(tmp_path):
    """
    2. Hold reference: Appendix B hold deck -> both lobes 470 mV +/- 5 mV.
    """
    raw_path = _get_or_run_reference_raw("hold", tmp_path)
    vq, vqb, vq_c2, vqb_c2 = _extract_dc_waves(raw_path)

    l1, l2 = lobe_sides(vq, vqb, vq_c2, vqb_c2, 1.0)
    snm = snm_from_curves(vq, vqb, vq_c2, vqb_c2, 1.0)

    # 470 mV +/- 5 mV (465 to 475 mV)
    assert 0.465 <= l1 <= 0.475, f"Hold lobe1 = {l1*1e3:.2f} mV out of range"
    assert 0.465 <= l2 <= 0.475, f"Hold lobe2 = {l2*1e3:.2f} mV out of range"
    assert 0.465 <= snm <= 0.475, f"Hold SNM = {snm*1e3:.2f} mV out of range"
    assert abs(l1 - l2) < 0.0005, f"Healthy hold SNM should be symmetric, diff={abs(l1-l2)*1e3:.2f} mV"


def test_symmetric_analytic_vtc():
    """
    3. Symmetric analytic VTC (sigmoid, trip 0.5) -> lobe1 == lobe2 within 0.5 mV.
    """
    x = np.linspace(0.0, 1.0, 2001)
    y = 1.0 / (1.0 + np.exp(30.0 * (x - 0.5)))
    l1, l2 = lobe_sides(x, y, y, x, 1.0)

    assert abs(l1 - l2) < 0.0005, f"Symmetric curves produced asymmetric lobes: {l1} vs {l2}"


def test_asymmetric_synthetic_curves():
    """
    4. Asymmetric synthetic curves (trips 0.40/0.58, gains 40/25):
       module result vs independent brute-force on 10 mV grid -> module >= brute and within 20 mV.
    """
    fa = lambda x: 1.0 / (1.0 + np.exp(40.0 * (x - 0.40)))
    gb = lambda y: 1.0 / (1.0 + np.exp(25.0 * (y - 0.58)))

    ax = np.linspace(0.0, 1.0, 2001)
    ay = fa(ax)
    by = np.linspace(0.0, 1.0, 2001)
    bx = gb(by)

    ml1, ml2 = lobe_sides(ax, ay, bx, by, 1.0)

    # Independent brute-force search on a 10 mV grid
    grid_s = np.arange(0.0, 1.001, 0.010)
    brute_l1 = 0.0
    for t in np.arange(0.0, 1.001, 0.001):
        x0, y0 = gb(t), t
        for s in grid_s:
            if x0 + s <= 1.0 and y0 + s <= 1.0:
                if fa(x0 + s) >= y0 + s:
                    if s > brute_l1:
                        brute_l1 = s

    brute_l2 = 0.0
    for t in np.arange(0.0, 1.001, 0.001):
        x0, y0 = t, fa(t)
        for s in grid_s:
            if x0 + s <= 1.0 and y0 + s <= 1.0:
                if gb(y0 + s) >= x0 + s:
                    if s > brute_l2:
                        brute_l2 = s

    assert ml1 >= brute_l1, f"Module lobe1 ({ml1}) < brute force ({brute_l1})"
    assert (ml1 - brute_l1) <= 0.020, f"Module lobe1 ({ml1}) differs from brute ({brute_l1}) by > 20 mV"

    assert ml2 >= brute_l2, f"Module lobe2 ({ml2}) < brute force ({brute_l2})"
    assert (ml2 - brute_l2) <= 0.020, f"Module lobe2 ({ml2}) differs from brute ({brute_l2}) by > 20 mV"


def test_monostable_input():
    """
    5. Monostable input (curves crossing once): returns 0 for the missing lobe, no exception, no NaN.
    """
    fa = lambda x: 1.0 / (1.0 + np.exp(40.0 * (x - 0.10)))
    gb = lambda y: 1.0 / (1.0 + np.exp(40.0 * (y - 0.90)))
    x = np.linspace(0.0, 1.0, 2001)
    ax, ay = x, fa(x)
    by, bx = x, gb(x)

    l1, l2 = lobe_sides(ax, ay, bx, by, 1.0)
    snm = snm_from_curves(ax, ay, bx, by, 1.0)

    assert not np.isnan(l1) and not np.isnan(l2)
    assert not np.isnan(snm)
    assert l1 == 0.0 or l2 == 0.0
    assert snm == 0.0


def test_step_size_robustness(tmp_path):
    """
    6. Step-size robustness: subsampling the read-reference curves changes SNM by < 0.1 mV.
    """
    raw_path = _get_or_run_reference_raw("read", tmp_path)
    vq, vqb, vq_c2, vqb_c2 = _extract_dc_waves(raw_path)

    l1_orig, l2_orig = lobe_sides(vq, vqb, vq_c2, vqb_c2, 1.0)
    l1_sub2, l2_sub2 = lobe_sides(vq[::2], vqb[::2], vq_c2[::2], vqb_c2[::2], 1.0)
    l1_sub5, l2_sub5 = lobe_sides(vq[::5], vqb[::5], vq_c2[::5], vqb_c2[::5], 1.0)

    assert abs(l1_orig - l1_sub2) < 0.0001, f"Step 2 change: {abs(l1_orig - l1_sub2)*1e3:.4f} mV >= 0.1 mV"
    assert abs(l1_orig - l1_sub5) < 0.0001, f"Step 5 change: {abs(l1_orig - l1_sub5)*1e3:.4f} mV >= 0.1 mV"
