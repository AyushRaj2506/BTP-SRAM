"""
src/fault_injection.py — Fault injection functions for the SRAM fault-diagnosis pipeline.

Each function operates on the cell-core template at circuits/core/cell_core_healthy.net
and produces a new faulty core file, never overwriting the healthy template.

Fault classes follow the taxonomy in PLAN.md Section 4:
  Class 1 — Resistive Open (inject_resistive_open)
  Class 2 — Bridging Fault (inject_bridging_fault)
  Class 3 — Vth Drift, storage pair (inject_vth_drift, target="storage_pair")
  Class 4 — Vth Drift, access transistors (inject_vth_drift, target="access")
"""

import re
from pathlib import Path
from typing import List

from utils import load_cell_core, save_cell_core

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEALTHY_CORE_PATH = Path(__file__).parent.parent / "circuits" / "core" / "cell_core_healthy.net"


def _parse_m_fields(line: str):
    """
    Split an M-device line into its whitespace-separated fields.
    Returns None if the line does not start with 'M' (case-insensitive).

    SPICE M-line field order:
      [0] name  [1] drain  [2] gate  [3] source  [4] bulk  [5] model  [6+] params
    """
    stripped = line.strip()
    if not stripped.upper().startswith("M"):
        return None
    return stripped.split()


def _replace_field(line: str, field_index: int, new_value: str) -> str:
    """
    Replace a single whitespace-delimited field in *line* at *field_index*,
    preserving original inter-field spacing as a single space (SPICE is
    whitespace-insensitive between fields).
    """
    fields = line.strip().split()
    fields[field_index] = new_value
    return " ".join(fields)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_resistive_open(
    core_lines: List[str],
    transistor: str = "M5",
    resistance_ohms: int = 1000,
    output_path: str = None,
) -> str:
    """
    PLAN.md Fault Class 1 — Resistive Open.

    Inserts a series resistor between the named transistor's source terminal
    and the node it currently connects to, breaking the direct connection.

    For M5 (source field = BL):
      - Renames M5's source field from BL  to BL_r
      - Inserts  "Rfault BL BL_r {resistance_ohms}"

    For M6 (source field = Qb):
      - Renames M6's source field from Qb  to Qb_r
      - Inserts  "Rfault Qb Qb_r {resistance_ohms}"

    Works generically for whichever of M5 / M6 is passed in.

    Parameters
    ----------
    core_lines : list of str
        Lines from load_cell_core() — the healthy (or already-modified) core.
    transistor : str
        "M5" or "M6" — the transistor whose source terminal gets the series R.
    resistance_ohms : int or float
        Resistance value in ohms to insert.
    output_path : str
        Destination .net path.  Defaults to
        data/generated_netlists/fault_resistive_open_{transistor}_{resistance_ohms}.net
        relative to the project root.

    Returns
    -------
    str
        Path to the written faulty core file.
    """
    project_root = Path(__file__).parent.parent
    if output_path is None:
        output_path = (
            project_root
            / "data"
            / "generated_netlists"
            / f"fault_resistive_open_{transistor}_{resistance_ohms}.net"
        )
    output_path = Path(output_path)

    tgt = transistor.upper()
    modified = False
    result = []

    for line in core_lines:
        fields = _parse_m_fields(line)
        if fields is not None and fields[0].upper() == tgt:
            # SPICE M-line: name drain gate source bulk model [params...]
            # field index:   0     1     2    3      4    5     6+
            source_node = fields[3]
            new_node = source_node + "_r"
            new_line = _replace_field(line, 3, new_node)
            result.append(new_line)
            # Insert the series resistor immediately after the modified M-line
            result.append(f"Rfault {source_node} {new_node} {resistance_ohms}")
            modified = True
        else:
            result.append(line)

    if not modified:
        raise ValueError(
            f"inject_resistive_open: transistor '{transistor}' not found in core_lines."
        )

    save_cell_core(result, str(output_path))
    return str(output_path)


def inject_bridging_fault(
    core_lines: List[str],
    resistance_ohms: int = 2000,
    output_path: str = None,
) -> str:
    """
    PLAN.md Fault Class 2 — Bridging Fault.

    Adds a single new line "Rbridge Q Qb {resistance_ohms}" to the core lines,
    with no other changes.  This creates a resistive path directly between the
    two storage nodes, degrading the cell's static noise margin.

    Parameters
    ----------
    core_lines : list of str
        Lines from load_cell_core().
    resistance_ohms : int or float
        Resistance of the bridging path in ohms.
    output_path : str
        Destination .net path.  Defaults to
        data/generated_netlists/fault_bridging_{resistance_ohms}.net

    Returns
    -------
    str
        Path to the written faulty core file.
    """
    project_root = Path(__file__).parent.parent
    if output_path is None:
        output_path = (
            project_root
            / "data"
            / "generated_netlists"
            / f"fault_bridging_{resistance_ohms}.net"
        )
    output_path = Path(output_path)

    result = list(core_lines)  # shallow copy — do not mutate caller's list
    result.append(f"Rbridge Q Qb {resistance_ohms}")

    save_cell_core(result, str(output_path))
    return str(output_path)


def inject_vth_drift(
    core_lines: List[str],
    target: str = "storage_pair",
    drift_pct: float = 10,
    output_path: str = None,
) -> str:
    """
    PLAN.md Fault Class 3/4 — Vth Drift.

    Modifies VTO in the SRAM_NMOS / SRAM_PMOS .model lines to simulate
    threshold-voltage ageing or process variation.

    Drift is always computed from the ORIGINAL magnitude (0.45 V), never
    compounded from a previously-drifted file.

    target="storage_pair"  (Fault Class 3)
        Modifies the shared SRAM_NMOS and SRAM_PMOS .model lines used by
        ALL four transistors M1–M4.  M5/M6 continue to use the original model.

    target="access"  (Fault Class 4)
        Leaves M1–M4's model lines untouched.  Duplicates the SRAM_NMOS model
        under the name SRAM_NMOS_ACCESS_DRIFT and re-points M5 and M6's model
        reference to it.  (PMOS is irrelevant for M5/M6 which are both NMOS.)

    Parameters
    ----------
    core_lines : list of str
        Lines from load_cell_core().
    target : str
        "storage_pair" or "access".
    drift_pct : float
        Percentage by which VTO magnitude is increased (e.g. 10 → +10%).
    output_path : str
        Destination .net path.

    Returns
    -------
    str
        Path to the written faulty core file.
    """
    if target not in ("storage_pair", "access"):
        raise ValueError(f"inject_vth_drift: target must be 'storage_pair' or 'access', got {target!r}")

    project_root = Path(__file__).parent.parent
    if output_path is None:
        safe_target = target.replace("_", "-")
        output_path = (
            project_root
            / "data"
            / "generated_netlists"
            / f"fault_vth_drift_{safe_target}_{drift_pct}pct.net"
        )
    output_path = Path(output_path)

    # Original VTO magnitudes (from the validated healthy template)
    VTO_NMOS_ORIG = 0.45
    VTO_PMOS_ORIG = 0.45  # magnitude; sign will be re-applied below

    drifted_nmos_vto = VTO_NMOS_ORIG * (1 + drift_pct / 100.0)
    drifted_pmos_vto = -VTO_PMOS_ORIG * (1 + drift_pct / 100.0)  # more negative

    def _patch_vto(model_line: str, new_vto: float) -> str:
        """Replace VTO=<value> in a .model line with the drifted value."""
        return re.sub(
            r"VTO=[-+]?\d+\.?\d*",
            f"VTO={new_vto:.6g}",
            model_line,
        )

    if target == "storage_pair":
        result = []
        for line in core_lines:
            stripped = line.strip()
            if stripped.startswith(".model SRAM_NMOS "):
                result.append(_patch_vto(line, drifted_nmos_vto))
            elif stripped.startswith(".model SRAM_PMOS "):
                result.append(_patch_vto(line, drifted_pmos_vto))
            else:
                result.append(line)

    else:  # target == "access"
        # 1. Find the SRAM_NMOS .model line and create a drifted ACCESS variant
        access_model_name = "SRAM_NMOS_ACCESS_DRIFT"
        access_model_line = None

        for line in core_lines:
            if line.strip().startswith(".model SRAM_NMOS "):
                # Build the access-drift model by renaming and patching VTO
                access_model_line = _patch_vto(
                    line.replace("SRAM_NMOS", access_model_name, 1),
                    drifted_nmos_vto,
                )
                break

        if access_model_line is None:
            raise ValueError("inject_vth_drift(access): could not find '.model SRAM_NMOS' line in core_lines.")

        result = []
        for line in core_lines:
            fields = _parse_m_fields(line)
            if fields is not None and fields[0].upper() in ("M5", "M6"):
                # Re-point model reference (field index 5) to the new ACCESS model
                result.append(_replace_field(line, 5, access_model_name))
            else:
                result.append(line)

        # Append the new ACCESS model line after the existing SRAM_NMOS line
        final = []
        for line in result:
            final.append(line)
            if line.strip().startswith(".model SRAM_NMOS "):
                final.append(access_model_line)
        result = final

    save_cell_core(result, str(output_path))
    return str(output_path)
