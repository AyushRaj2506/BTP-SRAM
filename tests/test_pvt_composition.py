"""
tests/test_pvt_composition.py -- Unit tests for PVT adaptation in testbench composer.
"""

from pathlib import Path
import pytest

from testbench_composer import compose_testbench

PROJECT_ROOT = Path(__file__).parent.parent
HEALTHY_CORE = PROJECT_ROOT / "circuits" / "core" / "cell_core_healthy.net"


def test_pvt_defaults_produce_nominal_deck(tmp_path):
    out_nominal = str(tmp_path / "tb_hold_nominal.net")
    compose_testbench(str(HEALTHY_CORE), "hold", out_nominal)
    content = Path(out_nominal).read_text(encoding="utf-8")

    assert "V4 Vdd 0 1V" in content
    assert ".temp" not in content
    assert ".end" in content


def test_pvt_voltage_and_temperature_applied(tmp_path):
    out_pvt = str(tmp_path / "tb_write_pvt.net")
    compose_testbench(str(HEALTHY_CORE), "write", out_pvt, vdd=0.9, temperature_c=-40.0)
    content = Path(out_pvt).read_text(encoding="utf-8")

    assert "V4 Vdd 0 0.9V" in content
    assert "PULSE(0 0.9 " in content
    assert "V1 BL 0 0.9V" in content
    assert ".temp -40.0" in content


def test_pvt_snm_dc_sweep_scaled(tmp_path):
    out_snm = str(tmp_path / "tb_snm_read_pvt.net")
    compose_testbench(str(HEALTHY_CORE), "snm_read", out_snm, vdd=1.1, temperature_c=125.0)
    content = Path(out_snm).read_text(encoding="utf-8")

    assert "V4 Vdd 0 1.1V" in content
    assert "V2 WL 0 1.1V" in content
    assert ".dc V5 0 1.1 1m" in content
    assert ".temp 125.0" in content
