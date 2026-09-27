"""
tests/test_fault_injection.py

Pytest tests for src/fault_injection.py.
These checks are derived from real bugs caught during manual circuit validation:

  1. test_gate_never_equals_drain   — a self-gated transistor from accidental
       schematic edits broke the HOLD test twice during development.
  2. test_node_mapping_preserved   — catches reintroduction of the swapped-node
       bug (M5 drain / M6 drain got mixed up multiple times).
  3. test_fault_netlist_differs_minimally — detects unintended side-effects
       (fault functions touching lines they should not touch).

Run with:  pytest tests/
"""

import re
import sys
import pytest
from pathlib import Path

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_cell_core, save_cell_core
from fault_injection import inject_resistive_open, inject_bridging_fault, inject_vth_drift

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

HEALTHY_CORE = Path(__file__).parent.parent / "circuits" / "core" / "cell_core_healthy.net"
GENERATED = Path(__file__).parent.parent / "data" / "generated_netlists"
GENERATED.mkdir(parents=True, exist_ok=True)


def _healthy_lines():
    return load_cell_core(str(HEALTHY_CORE))


def _parse_m_line(line: str):
    """
    Return (name, drain, gate, source, bulk, model) for an M-device line.
    Returns None if the line does not start with 'M'.
    """
    stripped = line.strip()
    if not stripped.upper().startswith("M"):
        return None
    fields = stripped.split()
    if len(fields) < 6:
        return None
    return fields[0], fields[1], fields[2], fields[3], fields[4], fields[5]


def _all_m_lines(lines):
    """Filter a list of core lines down to only the M-device lines."""
    return [l for l in lines if l.strip() and l.strip()[0].upper() == "M"
            and not l.strip().upper().startswith("MODEL")]


# ---------------------------------------------------------------------------
# Helpers that generate the four variants used across tests
# ---------------------------------------------------------------------------

def _make_all_variants(tmp_path=None):
    """
    Returns a dict mapping a fault_label → list of lines for that variant.
    All files are written to data/generated_netlists/ so they persist for
    manual inspection; the test itself operates on in-memory line lists.
    """
    lines = _healthy_lines()

    ro_path = str(GENERATED / "test_ro_M5_1000.net")
    br_path = str(GENERATED / "test_bridging_2000.net")
    vd_sp_path = str(GENERATED / "test_vth_drift_storage_pair_10pct.net")
    vd_ac_path = str(GENERATED / "test_vth_drift_access_10pct.net")

    inject_resistive_open(list(lines), "M5", 1000, ro_path)
    inject_bridging_fault(list(lines), 2000, br_path)
    inject_vth_drift(list(lines), "storage_pair", 10, vd_sp_path)
    inject_vth_drift(list(lines), "access", 10, vd_ac_path)

    return {
        "healthy": lines,
        "resistive_open_M5": load_cell_core(ro_path),
        "bridging_2000": load_cell_core(br_path),
        "vth_drift_storage_pair": load_cell_core(vd_sp_path),
        "vth_drift_access": load_cell_core(vd_ac_path),
    }


# ---------------------------------------------------------------------------
# Test 1 — gate ≠ drain for every M-line in every variant
# ---------------------------------------------------------------------------

class TestGateNeverEqualsDrain:
    """
    For every M-line in the healthy core and every fault-injected variant,
    assert drain ≠ gate.  A self-gated transistor (drain == gate) is physically
    wrong and broke the HOLD test twice during manual development.
    """

    def _check_no_self_gate(self, lines, label):
        for line in lines:
            parsed = _parse_m_line(line)
            if parsed is None:
                continue
            name, drain, gate, source, bulk, model = parsed
            assert drain != gate, (
                f"[{label}] Transistor {name} has drain == gate == '{drain}'.  "
                f"Self-gated transistor detected.  Offending line: {line!r}"
            )

    def test_healthy(self):
        variants = _make_all_variants()
        self._check_no_self_gate(variants["healthy"], "healthy")

    def test_resistive_open_M5(self):
        variants = _make_all_variants()
        self._check_no_self_gate(variants["resistive_open_M5"], "resistive_open_M5")

    def test_bridging(self):
        variants = _make_all_variants()
        self._check_no_self_gate(variants["bridging_2000"], "bridging_2000")

    def test_vth_drift_storage_pair(self):
        variants = _make_all_variants()
        self._check_no_self_gate(variants["vth_drift_storage_pair"], "vth_drift_storage_pair")

    def test_vth_drift_access(self):
        variants = _make_all_variants()
        self._check_no_self_gate(variants["vth_drift_access"], "vth_drift_access")


# ---------------------------------------------------------------------------
# Test 2 — Node mapping preserved (M5 drain=Q, M6 drain=BLB, etc.)
# ---------------------------------------------------------------------------

class TestNodeMappingPreserved:
    """
    M5 drain must be 'Q'; M6 drain must be 'BLB'.
    M5 source must be 'BL' unless a resistive-open on M5 renamed it to 'BL_r'
    (in which case verify the drain side is still 'Q').
    M6 source must be 'Qb' unless a resistive-open on M6 renamed it to 'Qb_r'.

    This catches reintroduction of the swapped-node bug found multiple times.
    """

    def _find_transistor(self, lines, tname):
        for line in lines:
            parsed = _parse_m_line(line)
            if parsed and parsed[0].upper() == tname.upper():
                return parsed
        return None

    def _check_mapping(self, lines, label,
                       m5_source_exact=None, m6_source_exact=None):
        m5 = self._find_transistor(lines, "M5")
        m6 = self._find_transistor(lines, "M6")

        assert m5 is not None, f"[{label}] M5 not found in netlist"
        assert m6 is not None, f"[{label}] M6 not found in netlist"

        _, m5_drain, _, m5_source, _, _ = m5
        _, m6_drain, _, m6_source, _, _ = m6

        # Drains are always fixed regardless of fault type
        assert m5_drain == "Q",   f"[{label}] M5 drain should be 'Q', got '{m5_drain}'"
        assert m6_drain == "BLB", f"[{label}] M6 drain should be 'BLB', got '{m6_drain}'"

        # Sources: check exact value if specified
        if m5_source_exact is not None:
            assert m5_source == m5_source_exact, (
                f"[{label}] M5 source should be '{m5_source_exact}', got '{m5_source}'"
            )
        if m6_source_exact is not None:
            assert m6_source == m6_source_exact, (
                f"[{label}] M6 source should be '{m6_source_exact}', got '{m6_source}'"
            )

    def test_healthy(self):
        v = _make_all_variants()
        self._check_mapping(v["healthy"], "healthy", "BL", "Qb")

    def test_resistive_open_M5_source_renamed(self):
        """
        inject_resistive_open on M5 renames M5's source BL→BL_r.
        Drain should still be 'Q'; M6 should be completely unmodified.
        """
        v = _make_all_variants()
        self._check_mapping(v["resistive_open_M5"], "resistive_open_M5",
                            m5_source_exact="BL_r", m6_source_exact="Qb")

    def test_bridging(self):
        v = _make_all_variants()
        self._check_mapping(v["bridging_2000"], "bridging_2000", "BL", "Qb")

    def test_vth_drift_storage_pair(self):
        v = _make_all_variants()
        self._check_mapping(v["vth_drift_storage_pair"], "vth_drift_storage_pair", "BL", "Qb")

    def test_vth_drift_access(self):
        v = _make_all_variants()
        self._check_mapping(v["vth_drift_access"], "vth_drift_access", "BL", "Qb")


# ---------------------------------------------------------------------------
# Test 3 — Fault netlists differ minimally from the healthy core
# ---------------------------------------------------------------------------

class TestFaultNetlistDiffersMinimally:
    """
    For each inject_* function, generate a faulty netlist and diff it
    line-by-line against the healthy core.  Assert only the expected line(s)
    changed (or a single new line was added).  Unrelated changes → test fails.
    """

    def _lines_changed(self, healthy, faulty):
        """
        Use difflib.SequenceMatcher to compute an insertion-aware diff.

        Returns:
        - replaced: list of (healthy_idx, faulty_idx) pairs where the line
          *changed* (replace opcode — one line swapped for another).
        - inserted: list of lines in faulty that are pure insertions with
          no counterpart in healthy (insert opcode).
        - deleted: list of lines in healthy that were deleted (delete opcode).

        Positional comparison is inadequate here because inject_resistive_open
        *inserts* a new Rfault line, which shifts all subsequent lines in faulty
        by one index — making every downstream line look "changed" when compared
        at the same position.  SequenceMatcher handles this correctly.
        """
        import difflib
        sm = difflib.SequenceMatcher(None, healthy, faulty, autojunk=False)
        replaced = []
        inserted = []
        deleted = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "replace":
                h_count = i2 - i1
                f_count = j2 - j1
                # Pair up as many lines as possible as replacements
                for hi, fi in zip(range(i1, i2), range(j1, j2)):
                    replaced.append((hi, fi))
                # Any extra faulty lines beyond the healthy range are insertions
                if f_count > h_count:
                    inserted.extend(faulty[j1 + h_count: j2])
                # Any extra healthy lines beyond the faulty range are deletions
                elif h_count > f_count:
                    deleted.extend(healthy[i1 + f_count: i2])
            elif tag == "insert":
                inserted.extend(faulty[j1:j2])
            elif tag == "delete":
                deleted.extend(healthy[i1:i2])
            # "equal" → no action
        return replaced, inserted, deleted

    def test_resistive_open_M5_minimal_diff(self):
        v = _make_all_variants()
        healthy = v["healthy"]
        faulty = v["resistive_open_M5"]

        replaced, inserted, deleted = self._lines_changed(healthy, faulty)

        # Exactly one line should be replaced: M5 line (source BL→BL_r)
        assert len(replaced) == 1, (
            f"resistive_open_M5: expected 1 replaced line, got {len(replaced)}: "
            f"{[(healthy[hi] + ' → ' + faulty[fi]) for hi, fi in replaced]}"
        )
        hi, fi = replaced[0]
        assert hi == 4, (
            f"resistive_open_M5: replaced healthy line index should be 4 (M5), got {hi}"
        )
        assert "BL_r" in faulty[fi], (
            f"resistive_open_M5: M5 source should become BL_r, line: {faulty[fi]!r}"
        )

        # Exactly one new line inserted: the Rfault resistor
        assert len(inserted) == 1, (
            f"resistive_open_M5: expected 1 inserted line (Rfault), got {len(inserted)}: {inserted}"
        )
        assert "Rfault" in inserted[0] and "BL" in inserted[0] and "BL_r" in inserted[0], (
            f"resistive_open_M5: inserted line should be Rfault BL BL_r ..., got: {inserted[0]!r}"
        )

        # No deletions
        assert len(deleted) == 0, f"resistive_open_M5: unexpected deletions: {deleted}"

    def test_bridging_fault_minimal_diff(self):
        v = _make_all_variants()
        healthy = v["healthy"]
        faulty = v["bridging_2000"]

        replaced, inserted, deleted = self._lines_changed(healthy, faulty)

        # No lines should be replaced; one line should be inserted (Rbridge)
        assert len(replaced) == 0, (
            f"bridging_2000: expected 0 replaced lines, got {len(replaced)}: "
            f"{[(healthy[hi], faulty[fi]) for hi, fi in replaced]}"
        )
        assert len(inserted) == 1, (
            f"bridging_2000: expected 1 inserted line (Rbridge), got {len(inserted)}: {inserted}"
        )
        assert inserted[0].startswith("Rbridge Q Qb"), (
            f"bridging_2000: inserted line should start with 'Rbridge Q Qb', got: {inserted[0]!r}"
        )
        assert len(deleted) == 0, f"bridging_2000: unexpected deletions: {deleted}"

    def test_vth_drift_storage_pair_minimal_diff(self):
        v = _make_all_variants()
        healthy = v["healthy"]
        faulty = v["vth_drift_storage_pair"]

        replaced, inserted, deleted = self._lines_changed(healthy, faulty)

        # Exactly two lines should be replaced: the two .model lines for SRAM_NMOS and SRAM_PMOS
        assert len(replaced) == 2, (
            f"vth_drift_storage_pair: expected 2 replaced lines (the two .model lines), "
            f"got {len(replaced)}: {[(healthy[hi], faulty[fi]) for hi, fi in replaced]}"
        )
        for hi, _ in replaced:
            assert healthy[hi].startswith(".model SRAM_"), (
                f"vth_drift_storage_pair: replaced line should be an SRAM .model line, got: {healthy[hi]!r}"
            )
        assert len(inserted) == 0, f"vth_drift_storage_pair: unexpected insertions: {inserted}"
        assert len(deleted) == 0, f"vth_drift_storage_pair: unexpected deletions: {deleted}"

        # VTO in both replaced .model lines should have changed
        for hi, fi in replaced:
            assert "VTO=" in faulty[fi], f"Replaced .model line missing VTO=: {faulty[fi]!r}"
            assert faulty[fi] != healthy[hi], f"Replaced line is identical to original: {faulty[fi]!r}"

    def test_vth_drift_access_minimal_diff(self):
        v = _make_all_variants()
        healthy = v["healthy"]
        faulty = v["vth_drift_access"]

        replaced, inserted, deleted = self._lines_changed(healthy, faulty)

        # Expected replacements: M5 and M6 lines
        # model field re-pointed from SRAM_NMOS → SRAM_NMOS_ACCESS_DRIFT
        replaced_healthy_lines = [healthy[hi] for hi, fi in replaced]
        assert any(l.startswith("M5") for l in replaced_healthy_lines), (
            f"vth_drift_access: M5 line should have been replaced; got {replaced_healthy_lines}"
        )
        assert any(l.startswith("M6") for l in replaced_healthy_lines), (
            f"vth_drift_access: M6 line should have been replaced; got {replaced_healthy_lines}"
        )
        # Only M5 and M6 should be replaced (no other lines touched)
        assert len(replaced) == 2, (
            f"vth_drift_access: expected exactly 2 replaced lines (M5 and M6), got {len(replaced)}"
        )

        # No deletions
        assert len(deleted) == 0, f"vth_drift_access: unexpected deletions: {deleted}"

        # One new .model line inserted (SRAM_NMOS_ACCESS_DRIFT)
        drift_model_lines = [l for l in inserted if "SRAM_NMOS_ACCESS_DRIFT" in l]
        assert len(drift_model_lines) == 1, (
            f"vth_drift_access: expected exactly 1 'SRAM_NMOS_ACCESS_DRIFT' .model line "
            f"to be inserted, got {len(drift_model_lines)}: {drift_model_lines}"
        )
        assert "VTO=0.495" in drift_model_lines[0], (
            f"vth_drift_access: drifted model VTO should be 0.495, got: {drift_model_lines[0]!r}"
        )
