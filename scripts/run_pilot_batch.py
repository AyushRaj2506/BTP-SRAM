"""
scripts/run_pilot_batch.py -- 5-simulation pilot batch (Task 7).

Generates one cell core per fault class (healthy + 4 fault types), each at the
middle-of-range severity from PLAN.md's severity list, composes each with the
HOLD stimulus only, runs all 5 through simulation_runner.run_batch(), and prints
a table of results.

PLAN.md severity lists (middle value selected for each):
  Resistive open           -- [100, 500, 1000, 5000, 10000] ohm  -> 1000 ohm
  Bridging                 -- [500, 1000, 2000, 5000, 10000] ohm  -> 2000 ohm
  Vth drift (storage pair) -- [5, 10, 20, 30] %                   -> 10 % (index 1 of 4)
  Vth drift (access)       -- [5, 10, 20, 30] %                   -> 10 % (index 1 of 4)

Outputs saved to data/pilot_batch/.

Usage:
    cd <project_root>
    python scripts/run_pilot_batch.py
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_cell_core, save_cell_core
from fault_injection import inject_resistive_open, inject_bridging_fault, inject_vth_drift
from testbench_composer import compose_testbench
from simulation_runner import run_batch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
HEALTHY_CORE = PROJECT_ROOT / "circuits" / "core" / "cell_core_healthy.net"
PILOT_DIR = PROJECT_ROOT / "data" / "pilot_batch"
PILOT_DIR.mkdir(parents=True, exist_ok=True)

CORE_DIR = PILOT_DIR / "cores"
TB_DIR = PILOT_DIR / "testbenches"
RAW_DIR = PILOT_DIR / "raw"
LOG_DIR = PILOT_DIR / "logs"

for d in (CORE_DIR, TB_DIR, RAW_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

FLAGGED_CSV = PROJECT_ROOT / "logs" / "convergence_flags.csv"

# ---------------------------------------------------------------------------
# Step 1 -- Generate all 5 cell cores
# ---------------------------------------------------------------------------

print("=" * 60)
print("Step 1: Generating faulty cell cores")
print("=" * 60)

healthy_lines = load_cell_core(str(HEALTHY_CORE))

fault_specs = [
    {
        "label": "healthy",
        "core_path": str(CORE_DIR / "core_healthy.net"),
    },
    {
        "label": "resistive_open_M5_1000ohm",
        "core_path": str(CORE_DIR / "core_ro_M5_1000.net"),
    },
    {
        "label": "bridging_2000ohm",
        "core_path": str(CORE_DIR / "core_bridging_2000.net"),
    },
    {
        "label": "vth_drift_storage_pair_10pct",
        "core_path": str(CORE_DIR / "core_vth_sp_10pct.net"),
    },
    {
        "label": "vth_drift_access_10pct",
        "core_path": str(CORE_DIR / "core_vth_ac_10pct.net"),
    },
]

# Write the healthy core into pilot_batch/cores/ (original in circuits/core/ untouched)
save_cell_core(list(healthy_lines), fault_specs[0]["core_path"])
print(f"  [healthy]                    -> {fault_specs[0]['core_path']}")

inject_resistive_open(list(healthy_lines), "M5", 1000, fault_specs[1]["core_path"])
print(f"  [resistive_open M5 1000 ohm] -> {fault_specs[1]['core_path']}")

inject_bridging_fault(list(healthy_lines), 2000, fault_specs[2]["core_path"])
print(f"  [bridging Q-Qb 2000 ohm]     -> {fault_specs[2]['core_path']}")

inject_vth_drift(list(healthy_lines), "storage_pair", 10, fault_specs[3]["core_path"])
print(f"  [Vth drift storage pair 10%] -> {fault_specs[3]['core_path']}")

inject_vth_drift(list(healthy_lines), "access", 10, fault_specs[4]["core_path"])
print(f"  [Vth drift access 10%]       -> {fault_specs[4]['core_path']}")

# ---------------------------------------------------------------------------
# Step 2 -- Compose each core with the HOLD stimulus
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Step 2: Composing testbenches (HOLD stimulus only)")
print("=" * 60)

net_paths = []
for spec in fault_specs:
    lbl = spec["label"]
    tb_path = str(TB_DIR / f"tb_{lbl}_hold.net")
    compose_testbench(spec["core_path"], "hold", tb_path)
    spec["tb_path"] = tb_path
    net_paths.append(tb_path)
    print(f"  {lbl:35s} -> {Path(tb_path).name}")

# ---------------------------------------------------------------------------
# Step 3 -- Run all 5 through run_batch()
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Step 3: Running batch simulation (5 netlists)")
print("=" * 60)

df = run_batch(
    net_paths=net_paths,
    raw_output_dir=str(RAW_DIR),
    log_output_dir=str(LOG_DIR),
    flagged_csv_path=str(FLAGGED_CSV),
)

# ---------------------------------------------------------------------------
# Step 4 -- Read V(Q) and V(Qb) at t=20ns from each .raw file
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Step 4: Reading final voltages from .raw files at t=20ns")
print("=" * 60)

# PyLTSpice 6 ships RawRead via spicelib
try:
    from spicelib.raw.raw_read import RawRead
except ImportError:
    try:
        from PyLTSpice import RawRead
    except ImportError:
        RawRead = None
        print("WARNING: RawRead not available -- cannot read .raw files.")


def _read_final_voltages(raw_path: str):
    """
    Read V(Q) and V(Qb) at the final time point from a .raw file.
    Returns (vq, vqb) as floats, or (None, None) if the file cannot be read.
    """
    if RawRead is None:
        return None, None
    try:
        raw = RawRead(raw_path)
        vq_trace = raw.get_trace("V(Q)")
        vqb_trace = raw.get_trace("V(Qb)")
        vq_final = float(vq_trace.get_wave()[-1])
        vqb_final = float(vqb_trace.get_wave()[-1])
        return vq_final, vqb_final
    except Exception as e:
        print(f"    Warning: could not read {raw_path}: {e}")
        return None, None


# Build the results table (ltspice_version populated by run_simulation)
rows = []
for spec, (_, row) in zip(fault_specs, df.iterrows()):
    vq, vqb = _read_final_voltages(row["raw_path"])
    rows.append({
        "fault_class": spec["label"],
        "converged": row["converged"],
        "final V(Q) at t=20ns": f"{vq:.4f} V" if vq is not None else "N/A",
        "final V(Qb) at t=20ns": f"{vqb:.4f} V" if vqb is not None else "N/A",
        "ltspice_version": row.get("ltspice_version") or "N/A",
    })

# ---------------------------------------------------------------------------
# Step 5 -- Print the pilot table
# ---------------------------------------------------------------------------

print()
print("=" * 105)
print("PILOT BATCH RESULTS TABLE")
print("=" * 105)

col_widths = [35, 10, 14, 14, 30]
headers = ["fault_class", "converged", "final V(Q)", "final V(Qb)", "ltspice_version"]
header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
print(header_line)
print("-" * len(header_line))

for r in rows:
    ver = r["ltspice_version"]
    line = "  ".join([
        r["fault_class"].ljust(col_widths[0]),
        str(r["converged"]).ljust(col_widths[1]),
        r["final V(Q) at t=20ns"].ljust(col_widths[2]),
        r["final V(Qb) at t=20ns"].ljust(col_widths[3]),
        ver[:col_widths[4]].ljust(col_widths[4]),
    ])
    print(line)

print()
print(f"All outputs saved under: {PILOT_DIR}")
print(f"Convergence flags CSV:   {FLAGGED_CSV}")
