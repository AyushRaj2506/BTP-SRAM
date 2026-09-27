"""
tests/test_testbench_composer.py

Pytest test for src/testbench_composer.py.

  4. test_composer_rejects_dangling_nodes — calling compose_testbench with a
       core where "Q" has been globally renamed to "Qx" must raise ValueError
       rather than silently producing a broken netlist.

Run with:  pytest tests/
"""

import re
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_cell_core, save_cell_core
from testbench_composer import compose_testbench

HEALTHY_CORE = Path(__file__).parent.parent / "circuits" / "core" / "cell_core_healthy.net"
GENERATED = Path(__file__).parent.parent / "data" / "generated_netlists"


# ---------------------------------------------------------------------------
# Test 4 — composer rejects dangling (mismatched) nodes
# ---------------------------------------------------------------------------

class TestComposerRejectsDanglingNodes:
    """
    compose_testbench must raise ValueError when the core's node names do not
    match what the stimulus expects, rather than silently writing a broken deck.
    """

    def test_global_Q_rename_raises_value_error(self, tmp_path):
        """
        Rename every occurrence of the node 'Q' (whole-word) in the healthy
        core to 'Qx'.  The HOLD stimulus's .ic refers to V(Q), and the sanity
        check must catch that 'Q' is no longer a core node.
        """
        lines = load_cell_core(str(HEALTHY_CORE))
        # Global whole-word replacement of Q → Qx in all M-line fields
        broken = [re.sub(r"\bQ\b", "Qx", l) for l in lines]

        broken_path = str(tmp_path / "broken_Qx_core.net")
        out_path = str(tmp_path / "should_not_be_created.net")
        save_cell_core(broken, broken_path)

        with pytest.raises(ValueError, match="Q"):
            compose_testbench(broken_path, "hold", out_path)

        # Output file must NOT have been created
        assert not Path(out_path).exists(), (
            "compose_testbench wrote a broken netlist instead of raising ValueError"
        )

    def test_healthy_core_composes_successfully(self, tmp_path):
        """
        Positive control: healthy core + hold stimulus should produce a .net
        with a .end line at the end and both core and stimulus sections present.
        """
        out_path = str(tmp_path / "healthy_hold.net")
        result = compose_testbench(str(HEALTHY_CORE), "hold", out_path)

        assert result == out_path
        assert Path(out_path).exists()
        combined = load_cell_core(out_path)
        assert combined[-1] == ".end", f"Last line should be '.end', got {combined[-1]!r}"

        # Line 0 is the SPICE title comment; M1 should be on line 1
        assert combined[0].startswith("*"), \
            f"First line should be the SPICE title comment ('* ...'), got {combined[0]!r}"
        assert combined[1].startswith("M1 "), \
            f"Second line should be M1..., got {combined[1]!r}"

        # Stimulus .tran and .ic should be present
        has_tran = any(".tran" in l for l in combined)
        has_ic = any(".ic" in l for l in combined)
        assert has_tran, "Composed netlist missing .tran line"
        assert has_ic, "Composed netlist missing .ic line"

    def test_invalid_stimulus_type_raises_value_error(self, tmp_path):
        """
        Passing an unrecognised stimulus_type must raise ValueError immediately.
        """
        out_path = str(tmp_path / "bogus.net")
        with pytest.raises(ValueError, match="stimulus_type"):
            compose_testbench(str(HEALTHY_CORE), "bogus_type", out_path)

    def test_all_stimulus_types_compose_with_healthy_core(self, tmp_path):
        """
        All three stimulus types should compose without error against the
        healthy core, and each composed file should end with '.end'.
        """
        for stype in ("hold", "write", "read"):
            out_path = str(tmp_path / f"healthy_{stype}.net")
            result = compose_testbench(str(HEALTHY_CORE), stype, out_path)
            combined = load_cell_core(result)
            assert combined[-1] == ".end", \
                f"[{stype}] Last line should be '.end', got {combined[-1]!r}"
