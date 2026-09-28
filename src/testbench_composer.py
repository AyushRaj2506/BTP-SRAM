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
_SHARED_NETS = {"0", "GND", "VDD", "WL", "BL", "BLB"}


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
    Return the set of node names that stimulus V/C/B sources reference.

    Checks lines starting with 'V', 'C', or 'B' and collects the terminal
    node arguments (fields[1] and fields[2]), plus any node referenced in
    V(node) behavioral source expressions.
    """
    nodes = set()
    for line in lines:
        stripped = line.strip()
        first = stripped[0].upper() if stripped else ""
        if first not in ("V", "C", "B"):
            continue
        fields = stripped.split()
        if len(fields) >= 3:
            nodes.add(fields[1])
            nodes.add(fields[2])
        # Also catch referenced nodes in behavioral formulas, e.g. V=V(Q)
        if first == "B":
            for ref in re.findall(r"V\(([A-Za-z0-9_]+)\)", stripped, re.IGNORECASE):
                nodes.add(ref)
    return nodes


# ---------------------------------------------------------------------------
# Core duplication for SNM
# ---------------------------------------------------------------------------

def duplicate_core_for_snm(core_lines: List[str]) -> List[str]:
    """
    Duplicate a cell-core line list into a two-cell butterfly SNM topology.

    Parameters
    ----------
    core_lines : list of str
        Lines of the single cell core (healthy or faulty).

    Returns
    -------
    list of str
        Copy 1 lines (verbatim) + divider comment + Copy 2 lines with instances
        and non-shared nodes suffixed with '_c2'.
    """
    c2_lines = []
    c2_lines.append("* ---- copy 2 (same cell, nodes suffixed _c2) ----")

    for line in core_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("."):
            continue

        first = stripped[0].upper()
        fields = stripped.split()

        if first == "M":
            # fields: [0]=name [1]=drain [2]=gate [3]=source [4]=bulk [5]=model [6+]=params
            name = f"{fields[0]}_c2"
            drain = fields[1] if fields[1].upper() in _SHARED_NETS else f"{fields[1]}_c2"
            gate = fields[2] if fields[2].upper() in _SHARED_NETS else f"{fields[2]}_c2"
            source = fields[3] if fields[3].upper() in _SHARED_NETS else f"{fields[3]}_c2"
            bulk = fields[4] if fields[4].upper() in _SHARED_NETS else f"{fields[4]}_c2"
            rest = fields[5:]
            c2_lines.append(" ".join([name, drain, gate, source, bulk] + rest))

        elif first in ("R", "C", "L", "I"):
            # fields: [0]=name [1]=n1 [2]=n2 [3+]=value / params
            name = f"{fields[0]}_c2"
            n1 = fields[1] if fields[1].upper() in _SHARED_NETS else f"{fields[1]}_c2"
            n2 = fields[2] if fields[2].upper() in _SHARED_NETS else f"{fields[2]}_c2"
            rest = fields[3:]
            c2_lines.append(" ".join([name, n1, n2] + rest))

    return list(core_lines) + c2_lines


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _sanity_check(core_lines: List[str], stimulus_lines: List[str]) -> None:
    """
    Raise ValueError if the stimulus references nodes not present in the core,
    or if storage nodes Q/Qb are absent from the core.
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


def _sanity_check_snm(duplicated_core_lines: List[str], stimulus_lines: List[str]) -> None:
    """
    Sanity checks for butterfly SNM composed decks:
    - No dangling nodes (stimulus nodes exist in duplicated core)
    - Both (Q, Qb) and (Q_c2, Qb_c2) exist in core
    - Device count of copy 2 == copy 1 (fault elements present in both)
    - .model cards appear exactly once
    - Exactly one .dc command and no .tran
    """
    core_nodes = _nodes_in_core(duplicated_core_lines)
    stim_nodes = _nodes_in_stimulus(stimulus_lines)
    stim_nodes.discard("0")

    missing = stim_nodes - core_nodes
    if missing:
        raise ValueError(
            f"compose_testbench (SNM): stimulus references node(s) not found in core: {sorted(missing)}"
        )

    for req in ("Q", "Qb", "Q_c2", "Qb_c2"):
        if req not in core_nodes:
            raise ValueError(
                f"compose_testbench (SNM): required node '{req}' is missing from duplicated core."
            )

    # Device count comparison
    c1_devs = []
    c2_devs = []
    in_c2 = False
    for line in duplicated_core_lines:
        stripped = line.strip()
        if "copy 2" in stripped:
            in_c2 = True
            continue
        if not stripped or stripped.startswith("*") or stripped.startswith("."):
            continue
        fields = stripped.split()
        if fields and fields[0][0].upper() in ("M", "R", "C", "L", "I"):
            if in_c2:
                c2_devs.append(fields[0])
            else:
                c1_devs.append(fields[0])

    if len(c1_devs) != len(c2_devs):
        raise ValueError(
            f"compose_testbench (SNM): device count mismatch between copy 1 ({len(c1_devs)}) "
            f"and copy 2 ({len(c2_devs)})."
        )

    expected_c2 = {f"{d}_c2" for d in c1_devs}
    actual_c2 = set(c2_devs)
    if expected_c2 != actual_c2:
        raise ValueError(
            f"compose_testbench (SNM): copy 2 devices do not mirror copy 1: "
            f"diff={expected_c2 ^ actual_c2}"
        )

    # .model cards appear exactly once per model name
    model_cards = [
        l.strip().split()[1].upper()
        for l in duplicated_core_lines
        if l.strip().upper().startswith(".MODEL ")
    ]
    if len(model_cards) != len(set(model_cards)):
        raise ValueError(
            f"compose_testbench (SNM): duplicate .model cards found: {model_cards}"
        )

    # Exactly one .dc and no .tran
    combined_stimulus = duplicated_core_lines + stimulus_lines
    dc_count = sum(1 for l in combined_stimulus if l.strip().lower().startswith(".dc"))
    tran_count = sum(1 for l in combined_stimulus if l.strip().lower().startswith(".tran"))
    if dc_count != 1:
        raise ValueError(f"compose_testbench (SNM): deck must contain exactly one .dc, found {dc_count}")
    if tran_count != 0:
        raise ValueError(f"compose_testbench (SNM): deck must contain no .tran, found {tran_count}")


# ---------------------------------------------------------------------------
# PVT Stimulus Adaptation
# ---------------------------------------------------------------------------

def _apply_pvt_to_stimulus(
    stimulus_lines: List[str],
    vdd: float = 1.0,
    temperature_c: float = 27.0,
) -> List[str]:
    """
    Adjust stimulus lines for non-nominal supply voltage and temperature.

    If vdd == 1.0 and temperature_c == 27.0, returns lines unchanged.
    Otherwise:
    - Scales VDD supply (V4), bitline drivers (V1, V3), WL pulse amplitude (V2)
    - Updates initial conditions (.ic) and bitline cap ICs
    - Updates .dc sweep voltage upper bound to vdd
    - Adds .temp directive for SPICE
    """
    if abs(vdd - 1.0) < 1e-6 and abs(temperature_c - 27.0) < 1e-6:
        return list(stimulus_lines)

    pvt_lines = []
    for line in stimulus_lines:
        s = line.strip()
        if not s:
            pvt_lines.append(line)
            continue

        upper = s.upper()

        # VDD supply: V4 Vdd 0 1V -> V4 Vdd 0 {vdd}V
        if upper.startswith("V4 ") and "VDD" in upper:
            pvt_lines.append(f"V4 Vdd 0 {vdd}V")
            continue

        # WL pulse: V2 WL 0 PULSE(0 1 2n 0.1n 0.1n 5n 20n) -> PULSE(0 {vdd} ...)
        if upper.startswith("V2 ") and "PULSE(0 1 " in upper:
            new_l = re.sub(r"PULSE\(0\s+1\s+", f"PULSE(0 {vdd} ", line, flags=re.IGNORECASE)
            pvt_lines.append(new_l)
            continue

        # WL static high (SNM read): V2 WL 0 1V -> V2 WL 0 {vdd}V
        if upper.startswith("V2 ") and "WL 0 1V" in upper:
            pvt_lines.append(f"V2 WL 0 {vdd}V")
            continue

        # Bitline supplies: V1 BL 0 1V -> {vdd}V, V3 BLB 0 1V -> {vdd}V
        if upper.startswith("V1 ") and "BL 0 1V" in upper:
            pvt_lines.append(f"V1 BL 0 {vdd}V")
            continue
        if upper.startswith("V3 ") and "BLB 0 1V" in upper:
            pvt_lines.append(f"V3 BLB 0 {vdd}V")
            continue

        # Bitline capacitors: Cbl BL 0 20fF IC=1 -> IC={vdd}
        if (upper.startswith("CBL ") or upper.startswith("CBLB ")) and "IC=1" in upper:
            new_l = re.sub(r"IC=1(\.0)?", f"IC={vdd}", line, flags=re.IGNORECASE)
            pvt_lines.append(new_l)
            continue

        # Initial conditions: .ic V(Q)=1V ...
        if upper.startswith(".IC "):
            new_l = re.sub(r"=1V", f"={vdd}V", line, flags=re.IGNORECASE)
            pvt_lines.append(new_l)
            continue

        # DC sweep: .dc V5 0 1 1m -> .dc V5 0 {vdd} 1m
        if upper.startswith(".DC "):
            new_l = re.sub(
                r"^(\.dc\s+\S+\s+0\s+)(1|1\.0)(\s+.*)$",
                rf"\g<1>{vdd}\g<3>",
                line,
                flags=re.IGNORECASE,
            )
            pvt_lines.append(new_l)
            continue

        pvt_lines.append(line)

    if abs(temperature_c - 27.0) >= 1e-6:
        pvt_lines.append(f".temp {temperature_c}")

    return pvt_lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose_testbench(
    core_path: str,
    stimulus_type: str,
    output_path: str,
    vdd: float = 1.0,
    temperature_c: float = 27.0,
) -> str:
    """
    Combine a cell-core fragment with a stimulus file into a runnable SPICE deck.

    Parameters
    ----------
    core_path : str
        Path to a healthy or fault-injected cell-core .net fragment.
    stimulus_type : str
        One of "hold", "write", "read", "write_0", "read_0", "snm_hold", "snm_read".
    output_path : str
        Destination path for the composed, runnable .net file.
    vdd : float, optional
        Supply voltage in volts (default 1.0).
    temperature_c : float, optional
        Simulation temperature in Celsius (default 27.0).

    Returns
    -------
    str
        The value of output_path (for chaining).

    Raises
    ------
    ValueError
        If stimulus_type is invalid, or if the combined netlist has dangling /
        mismatched nodes.
    """
    valid_types = ("hold", "write", "read", "write_0", "read_0", "snm_hold", "snm_read")
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
    raw_stimulus_lines = load_cell_core(str(stimulus_path))
    stimulus_lines = _apply_pvt_to_stimulus(raw_stimulus_lines, vdd=vdd, temperature_c=temperature_c)

    if stimulus_type in ("snm_hold", "snm_read"):
        duplicated_core = duplicate_core_for_snm(core_lines)
        _sanity_check_snm(duplicated_core, stimulus_lines)
        final_core_lines = duplicated_core
    else:
        _sanity_check(core_lines, stimulus_lines)
        final_core_lines = core_lines

    core_stem = Path(core_path).stem
    title_line = f"* SRAM 6T cell testbench: {core_stem} + {stimulus_type} stimulus"

    # Compose: title, then core, then stimulus, then .end
    combined = [title_line] + final_core_lines + stimulus_lines + [".end"]

    save_cell_core(combined, output_path)
    return output_path
