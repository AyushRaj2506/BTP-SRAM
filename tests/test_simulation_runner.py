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
