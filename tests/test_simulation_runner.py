"""
tests/test_simulation_runner.py -- Unit tests for simulation runner and batch executor.
"""

from pathlib import Path
import pytest
from simulation_runner import _stable_stem, prune_raw_files, run_batch

DATA_DIR = Path(__file__).parent.parent / "data" / "demo_features"


def test_stable_stem_deterministic():
    p1 = "circuits/testbenches/stimulus_hold.net"
    stem1 = _stable_stem(p1)
    stem2 = _stable_stem(p1)
    assert stem1 == stem2
    assert "stimulus_hold" in stem1


def test_prune_raw_files(tmp_path):
    f1 = tmp_path / "sim1.raw"
    f2 = tmp_path / "sim2.raw"
    f3 = tmp_path / "sim3.log"
    f1.write_text("dummy raw 1", encoding="utf-8")
    f2.write_text("dummy raw 2", encoding="utf-8")
    f3.write_text("dummy log", encoding="utf-8")

    deleted = prune_raw_files([str(f1), str(f2), str(f3)])
    assert deleted == 2
    assert not f1.exists()
    assert not f2.exists()
    assert f3.exists()


def test_run_batch_parallel_execution(tmp_path):
    from testbench_composer import compose_testbench

    healthy_core = Path(__file__).parent.parent / "circuits" / "core" / "cell_core_healthy.net"
    tb1 = str(tmp_path / "tb_test_hold.net")
    tb2 = str(tmp_path / "tb_test_write.net")
    compose_testbench(str(healthy_core), "hold", tb1)
    compose_testbench(str(healthy_core), "write", tb2)

    raw_dir = tmp_path / "raw"
    log_dir = tmp_path / "logs"
    flag_csv = tmp_path / "flags.csv"

    df = run_batch(
        net_paths=[tb1, tb2],
        raw_output_dir=str(raw_dir),
        log_output_dir=str(log_dir),
        flagged_csv_path=str(flag_csv),
        max_workers=2,
    )

    assert len(df) == 2
    assert df["converged"].all()


def test_run_simulation_catches_failed_simulation(tmp_path):
    from simulation_runner import run_simulation

    # Deliberately broken netlist with invalid model
    broken_net = tmp_path / "broken_syntax.net"
    broken_net.write_text("* Broken netlist\nM1 Q Qb Vdd Vdd NON_EXISTENT_MODEL\n.end\n", encoding="utf-8")

    res = run_simulation(
        net_path=str(broken_net),
        raw_output_dir=str(tmp_path / "raw"),
        log_output_dir=str(tmp_path / "logs"),
    )

    assert res["converged"] is False
    assert res["warning"] is not None


def test_check_dc_traces_integrity_fixtures():
    import numpy as np
    from simulation_runner import check_dc_traces_integrity

    x = np.linspace(0.0, 1.0, 1001)
    y = 1.0 - x

    # 1. Valid healthy traces
    valid_traces = {"v(q)": x, "v(qb)": y, "v(q_c2)": y, "v(qb_c2)": x}
    ok, warn = check_dc_traces_integrity(valid_traces)
    assert ok is True
    assert warn is None

    # 2. Point-count mismatch (500 pts instead of 1001)
    short_x = np.linspace(0.0, 1.0, 500)
    short_traces = {"v(q)": short_x, "v(qb)": short_x, "v(q_c2)": short_x, "v(qb_c2)": short_x}
    ok, warn = check_dc_traces_integrity(short_traces)
    assert ok is False
    assert "point count mismatch" in warn

    # 3. Voltage out of range
    high_y = y.copy()
    high_y[500] = 1.20  # > 1.05 V
    bad_range = {"v(q)": x, "v(qb)": high_y, "v(q_c2)": y, "v(qb_c2)": x}
    ok, warn = check_dc_traces_integrity(bad_range)
    assert ok is False
    assert "out of range" in warn

    # 4. Monotonicity violation in curve A (vqb increases by > 5 mV)
    nonmono_y = y.copy()
    nonmono_y[500] = nonmono_y[499] + 0.050  # +50 mV jump
    nonmono_traces_a = {"v(q)": x, "v(qb)": nonmono_y, "v(q_c2)": y, "v(qb_c2)": x}
    ok, warn = check_dc_traces_integrity(nonmono_traces_a)
    assert ok is False
    assert warn == "snm_nonmonotonic"

    # 5. Monotonicity violation in curve B (vq_c2 increases by > 5 mV)
    nonmono_traces_b = {"v(q)": x, "v(qb)": y, "v(q_c2)": nonmono_y, "v(qb_c2)": x}
    ok, warn = check_dc_traces_integrity(nonmono_traces_b)
    assert ok is False
    assert warn == "snm_nonmonotonic"


def test_healthy_snm_simulation_passes_dc_integrity(tmp_path):
    from simulation_runner import run_simulation

    deck = Path(__file__).parent.parent / "circuits" / "reference" / "snm_read_healthy.net"
    res = run_simulation(
        net_path=str(deck),
        raw_output_dir=str(tmp_path / "raw"),
        log_output_dir=str(tmp_path / "logs"),
    )
    assert res["converged"] is True
    assert res["warning"] is None


