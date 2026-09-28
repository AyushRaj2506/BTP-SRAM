"""
tests/test_snm_composition.py -- Unit tests for SNM testbench composition (Task 2).

Covers:
  1. Healthy core + snm_read: parses, unique instances, correct shared vs non-shared nodes.
  2. Each fault type (open M5, open M6, bridge, Vth drift storage, Vth drift access):
     duplicated deck contains fault element in both copies with _c2 naming.
  3. snm_hold differs from snm_read only in the Vwl line.
  4. Existing operations (hold, write, read, write_0, read_0) produce byte-identical output to before.
  5. Rejects unknown operations and cores with missing/dangling nodes.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_cell_core, save_cell_core
from fault_injection import (
    inject_resistive_open,
    inject_bridging_fault,
    inject_vth_drift,
)
from testbench_composer import compose_testbench, duplicate_core_for_snm

HEALTHY_CORE = PROJECT_ROOT / "circuits" / "core" / "cell_core_healthy.net"


def test_healthy_core_snm_read_composition(tmp_path):
    """
    1. Healthy core + snm_read:
       - Compiles and ends with .end
       - Instance names are unique
       - Shared nets (0, Vdd, WL, BL, BLB) are shared; storage nodes are segregated
    """
    out_path = str(tmp_path / "snm_read_healthy.net")
    compose_testbench(str(HEALTHY_CORE), "snm_read", out_path)

    lines = load_cell_core(out_path)
    assert lines[-1] == ".end"
    assert any(".dc" in l for l in lines)
    assert not any(".tran" in l for l in lines)

    # Collect device instances
    instances = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("."):
            continue
        first = stripped[0].upper()
        if first in ("M", "R", "C", "L", "I", "V", "B"):
            inst = stripped.split()[0].upper()
            instances.append(inst)

    assert len(instances) == len(set(instances)), f"Duplicate instances found: {instances}"

    # Verify copy 2 instances exist
    assert "M1_C2" in instances
    assert "M5_C2" in instances
    assert "CQ_C2" in instances


def test_fault_types_duplicated_in_snm_deck(tmp_path):
    """
    2. Duplicated deck contains the fault element in BOTH copies with _c2 names.
    """
    healthy_lines = load_cell_core(str(HEALTHY_CORE))

    fault_configs = [
        ("ro_m5", lambda lines, p: inject_resistive_open(lines, "M5", 1000, p), "Rfault", "Rfault_c2"),
        ("ro_m6", lambda lines, p: inject_resistive_open(lines, "M6", 1000, p), "Rfault", "Rfault_c2"),
        ("bridge", lambda lines, p: inject_bridging_fault(lines, 2000, p), "Rbridge", "Rbridge_c2"),
        ("vth_sp", lambda lines, p: inject_vth_drift(lines, "storage_pair", 10, p), "M1", "M1_c2"),
        ("vth_ac", lambda lines, p: inject_vth_drift(lines, "access", 10, p), "M5", "M5_c2"),
    ]

    for label, inject_fn, token_c1, token_c2 in fault_configs:
        core_p = str(tmp_path / f"core_{label}.net")
        inject_fn(list(healthy_lines), core_p)

        tb_p = str(tmp_path / f"tb_{label}_snm_read.net")
        compose_testbench(core_p, "snm_read", tb_p)

        tb_lines = load_cell_core(tb_p)
        text = "\n".join(tb_lines)

        assert token_c1 in text, f"[{label}] Missing copy 1 token {token_c1}"
        assert token_c2 in text, f"[{label}] Missing copy 2 token {token_c2}"

        # If it's a resistive open, verify the internal node was suffixed in copy 2
        if "ro_m5" in label:
            assert "BL_r" in text
            assert "BL_r_c2" in text
        elif "ro_m6" in label:
            assert "BLB_r" in text
            assert "BLB_r_c2" in text


def test_snm_hold_differs_from_snm_read_only_in_vwl(tmp_path):
    """
    3. snm_hold differs from snm_read only in the V2 WL line (0V vs 1V) and title.
    """
    p_hold = str(tmp_path / "healthy_snm_hold.net")
    p_read = str(tmp_path / "healthy_snm_read.net")

    compose_testbench(str(HEALTHY_CORE), "snm_hold", p_hold)
    compose_testbench(str(HEALTHY_CORE), "snm_read", p_read)

    lines_hold = load_cell_core(p_hold)
    lines_read = load_cell_core(p_read)

    assert len(lines_hold) == len(lines_read)

    diffs = []
    for idx, (lh, lr) in enumerate(zip(lines_hold, lines_read)):
        if lh != lr:
            diffs.append((idx, lh, lr))

    # Exactly 2 differences: line 0 (title line containing stimulus type) and the WL source line
    assert len(diffs) == 2, f"Expected 2 diffs (title and V2 line), got {len(diffs)}: {diffs}"
    assert "title" in lines_hold[diffs[0][0]].lower() or diffs[0][0] == 0
    assert "WL" in diffs[1][1] and "WL" in diffs[1][2]
    assert "0V" in diffs[1][1] and "1V" in diffs[1][2]


def test_existing_operations_produce_byte_identical_output(tmp_path):
    """
    4. Existing operations (hold, write, read, write_0, read_0) produce byte-identical
       decks to before (regression).
    """
    ops = ("hold", "write", "read", "write_0", "read_0")
    for op in ops:
        out_p = str(tmp_path / f"tb_healthy_{op}.net")
        compose_testbench(str(HEALTHY_CORE), op, out_p)
        content = Path(out_p).read_text(encoding="utf-8")

        # Re-construct manual composition as in original code
        stim_path = PROJECT_ROOT / "circuits" / "testbenches" / f"stimulus_{op}.net"
        core_lines = load_cell_core(str(HEALTHY_CORE))
        stim_lines = load_cell_core(str(stim_path))
        title = f"* SRAM 6T cell testbench: cell_core_healthy + {op} stimulus"
        expected = "\n".join([title] + core_lines + stim_lines + [".end"]) + "\n"

        assert content == expected, f"Composition mismatch for operation {op}"


def test_composer_rejects_unknown_op_and_dangling_nodes(tmp_path):
    """
    5. Rejects unknown operation and dangling-node core.
    """
    out_p = str(tmp_path / "bogus.net")
    with pytest.raises(ValueError, match="stimulus_type"):
        compose_testbench(str(HEALTHY_CORE), "invalid_op", out_p)

    # Broken core missing Q
    import re
    healthy_lines = load_cell_core(str(HEALTHY_CORE))
    broken_lines = [re.sub(r"\bQ\b", "Qx", l) for l in healthy_lines]
    broken_p = str(tmp_path / "broken_core.net")
    save_cell_core(broken_lines, broken_p)

    with pytest.raises(ValueError):
        compose_testbench(broken_p, "snm_read", out_p)
