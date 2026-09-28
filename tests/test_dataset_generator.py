"""
tests/test_dataset_generator.py -- Unit tests for dataset generator functions.
"""

from pathlib import Path
import pytest

from generate_dataset import (
    format_temp_str,
    format_vdd_str,
    build_sample_id,
    get_default_fault_specs,
)


def test_format_temp_str():
    assert format_temp_str(27.0) == "27C"
    assert format_temp_str(-40.0) == "M40C"
    assert format_temp_str(0.0) == "0C"
    assert format_temp_str(125.0) == "125C"


def test_format_vdd_str():
    assert format_vdd_str(1.0) == "1P0V"
    assert format_vdd_str(0.9) == "0P9V"
    assert format_vdd_str(1.1) == "1P1V"


def test_build_sample_id_healthy():
    sid = build_sample_id(0, "none", 0, 27.0, 1.0, 1)
    assert sid == "SRAM_F0_S0_27C_1P0V_0001"


def test_build_sample_id_faulty():
    sid = build_sample_id(1, "M5", 3, -40.0, 0.9, 42)
    assert sid == "SRAM_F1_M5_S3_M40C_0P9V_0042"

    sid2 = build_sample_id(2, "Q_Qb", 1, 125.0, 1.1, 105)
    assert sid2 == "SRAM_F2_Q_Qb_S1_125C_1P1V_0105"


def test_fault_specs_taxonomy():
    specs = get_default_fault_specs()
    classes = {s["fault_class"] for s in specs}
    assert classes == {0, 1, 2, 3, 4}

    # Healthy spec
    healthy = [s for s in specs if s["fault_class"] == 0]
    assert len(healthy) == 1

    # Class 1 (Open): M5 and M6 both present
    opens = [s for s in specs if s["fault_class"] == 1]
    targets = {s["target"] for s in opens}
    assert targets == {"M5", "M6"}

    # Class 2 (Bridge)
    bridges = [s for s in specs if s["fault_class"] == 2]
    assert len(bridges) == 5

    # Class 3 and 4 (Vth drift)
    vth_sp = [s for s in specs if s["fault_class"] == 3]
    vth_ac = [s for s in specs if s["fault_class"] == 4]
    assert len(vth_sp) == 6
    assert len(vth_ac) == 6
