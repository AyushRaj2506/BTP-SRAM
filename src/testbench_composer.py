"""
src/testbench_composer.py — Compose a runnable SPICE netlist from a cell-core
fragment and a named stimulus file.

Usage
-----
    from testbench_composer import compose_testbench
    out = compose_testbench("data/generated_netlists/fault_bridging_2000.net",
                            "hold",
                            "data/generated_netlists/tb_bridging_hold.net")
"""

import re
from pathlib import Path
from typing import List

from utils import load_cell_core, save_cell_core

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_STIMULUS_DIR = _PROJECT_ROOT / "circuits" / "testbenches"


# ---------------------------------------------------------------------------
# Node extraction helpers
# ---------------------------------------------------------------------------

def _nodes_in_core(lines: List[str]):
    """
    Return the set of node names that appear in device lines of the core.

    Scans:
    - M-device lines: collects fields 1–4 (drain, gate, source, bulk)
    - R/C/L/I-device lines: collects fields 1 and 2 (the two terminal nodes)

    This is necessary because inject_resistive_open adds an Rfault line that
    connects the original BL node — if we only scan M-lines, BL appears to
    vanish from the core even though it's still wired through Rfault.
    """
    nodes = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        first = stripped[0].upper()
        fields = stripped.split()
        if first == "M":
            # fields: name drain gate source bulk model [param=val ...]
            for f in fields[1:5]:
                nodes.add(f)
        elif first in ("R", "C", "L", "I"):
            # fields: name node+ node- [value ...]
            if len(fields) >= 3:
                nodes.add(fields[1])
                nodes.add(fields[2])
    return nodes


def _nodes_in_stimulus(lines: List[str]):
    """
    Return the set of node names that stimulus V/C sources reference.

    Checks every line that starts with 'V' or 'C' (voltage/current source or
    capacitor) and collects the two node arguments (fields[1] and fields[2]).
    """
    nodes = set()
    for line in lines:
        stripped = line.strip()
        first = stripped[0].upper() if stripped else ""
        if first not in ("V", "C"):
            continue
        fields = stripped.split()
        if len(fields) >= 3:
            nodes.add(fields[1])
            nodes.add(fields[2])
    return nodes


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _sanity_check(core_lines: List[str], stimulus_lines: List[str]) -> None:
    """
    Raise ValueError if the stimulus references nodes not present in the core,
    or if storage nodes Q/Qb are absent from the core.

    Specifically:
    - Every node referenced by a stimulus V/C line (except the ground node "0")
      must appear somewhere in the core M-device lines.
    - The core must expose Q and Qb (the two storage nodes the stimulus .ic
      lines always reference, even if they're not in a V/C line).
    """
    core_nodes = _nodes_in_core(core_lines)
    stim_nodes = _nodes_in_stimulus(stimulus_lines)

    # Ground node "0" is implicitly defined everywhere — skip it
    stim_nodes.discard("0")

    missing_in_core = stim_nodes - core_nodes
    if missing_in_core:
        raise ValueError(
            f"compose_testbench: stimulus references node(s) not found in core: "
            f"{sorted(missing_in_core)}.  Check for typos in node names."
        )

    # Q and Qb must be in the core so the stimulus .ic lines are meaningful
    for required in ("Q", "Qb"):
        if required not in core_nodes:
            raise ValueError(
                f"compose_testbench: required storage node '{required}' is missing "
                f"from core M-device lines.  The core may have been corrupted."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose_testbench(
    core_path: str,
    stimulus_type: str,
    output_path: str,
) -> str:
    """
    Combine a cell-core fragment with a stimulus file into a runnable SPICE deck.

    Parameters
    ----------
    core_path : str
        Path to a healthy or fault-injected cell-core .net fragment.
    stimulus_type : str
        One of "hold", "write", "read".  Selects
        circuits/testbenches/stimulus_{stimulus_type}.net.
    output_path : str
        Destination path for the composed, runnable .net file.

    Returns
    -------
    str
        The value of output_path (for chaining).

    Raises
    ------
    ValueError
        If stimulus_type is not one of the three valid strings, or if the
        combined netlist has dangling / mismatched nodes.
    """
    valid_types = ("hold", "write", "read", "write_0", "read_0")
    if stimulus_type not in valid_types:
        raise ValueError(
            f"compose_testbench: stimulus_type must be one of {valid_types}, "
            f"got {stimulus_type!r}."
        )

    stimulus_path = _STIMULUS_DIR / f"stimulus_{stimulus_type}.net"
    if not stimulus_path.exists():
        raise FileNotFoundError(
            f"compose_testbench: stimulus file not found: {stimulus_path}"
        )

    core_lines = load_cell_core(str(core_path))
    stimulus_lines = load_cell_core(str(stimulus_path))

    # Sanity check before writing
    _sanity_check(core_lines, stimulus_lines)

    # SPICE requires a title comment as the very first line of any netlist.
    # SpiceEditor (PyLTSpice 6 / spicelib) enforces this: it raises an error
    # if line 1 does not match r'^(?:\*|\.title)'. We derive the title from
    # the core filename and stimulus type so the .net is self-describing.
    core_stem = Path(core_path).stem
    title_line = f"* SRAM 6T cell testbench: {core_stem} + {stimulus_type} stimulus"

    # Compose: title, then core, then stimulus, then .end
    combined = [title_line] + core_lines + stimulus_lines + [".end"]

    save_cell_core(combined, output_path)
    return output_path
