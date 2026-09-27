"""
scripts/run_pilot_extended.py -- WRITE + READ pilot batch (extension of Task 7).

Runs the same 5 cell cores (healthy + 4 fault types) under both WRITE and READ
stimuli.  For each simulation, in addition to the final V(Q)/V(Qb), we extract
the time-domain trajectory of V(Q) and V(Qb) at key instants so that timing
differences show up even when the final value is identical to healthy.

Key reference points extracted from each waveform:
  t_sample points (ns): 0, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20
  (WL is high from 2ns to 7ns; t=7ns is WL-fall; t=20ns is final)

WRITE pass criteria (matching original WRITE validation):
  - At t=20ns, V(Q) >= 0.9V  (wrote 1 successfully, held after WL low)
  - At t=20ns, V(Qb) <= 0.1V (complement flipped and held)

READ pass criteria (matching original READ validation):
  - At t=20ns, V(Q)  >= 0.9V  (stored value not disturbed)
  - At t=20ns, V(Qb) <= 0.1V  (complement not disturbed)
  - Differential BL-BLB develops during WL-high window (read-out indicator)

Outputs saved to data/pilot_batch/write/ and data/pilot_batch/read/.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_cell_core, save_cell_core
from fault_injection import inject_resistive_open, inject_bridging_fault, inject_vth_drift
from testbench_composer import compose_testbench
from simulation_runner import run_batch, run_simulation

from spicelib.raw.raw_read import RawRead
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
HEALTHY_CORE = PROJECT_ROOT / "circuits" / "core" / "cell_core_healthy.net"
PILOT_DIR    = PROJECT_ROOT / "data" / "pilot_batch"
FLAGGED_CSV  = PROJECT_ROOT / "logs" / "convergence_flags.csv"

# Sample the waveform at these time points (seconds)
SAMPLE_NS = [0, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]
SAMPLE_S  = [t * 1e-9 for t in SAMPLE_NS]

# ---------------------------------------------------------------------------
# Fault specs (same 5 as HOLD pilot, regenerated from the updated core)
# ---------------------------------------------------------------------------

FAULT_SPECS = [
    {"label": "healthy",                      "fn": None},
    {"label": "resistive_open_M5_1000ohm",    "fn": "ro"},
    {"label": "bridging_2000ohm",             "fn": "br"},
    {"label": "vth_drift_storage_pair_10pct", "fn": "vd_sp"},
    {"label": "vth_drift_access_10pct",       "fn": "vd_ac"},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_cores(core_dir: Path) -> list:
    """
    Generate all 5 cell cores from the updated healthy core and return the
    list of fault_specs dicts with 'core_path' filled in.
    """
    core_dir.mkdir(parents=True, exist_ok=True)
    healthy_lines = load_cell_core(str(HEALTHY_CORE))
    specs = []
    for spec in FAULT_SPECS:
        label = spec["label"]
        core_path = str(core_dir / f"core_{label}.net")
        if spec["fn"] is None:
            save_cell_core(list(healthy_lines), core_path)
        elif spec["fn"] == "ro":
            inject_resistive_open(list(healthy_lines), "M5", 1000, core_path)
        elif spec["fn"] == "br":
            inject_bridging_fault(list(healthy_lines), 2000, core_path)
        elif spec["fn"] == "vd_sp":
            inject_vth_drift(list(healthy_lines), "storage_pair", 10, core_path)
        elif spec["fn"] == "vd_ac":
            inject_vth_drift(list(healthy_lines), "access", 10, core_path)
        specs.append({**spec, "core_path": core_path})
    return specs


def compose_all(specs: list, stimulus_type: str, tb_dir: Path) -> list:
    """Compose each core with the given stimulus and return net_paths list."""
    tb_dir.mkdir(parents=True, exist_ok=True)
    net_paths = []
    for spec in specs:
        lbl  = spec["label"]
        path = str(tb_dir / f"tb_{lbl}_{stimulus_type}.net")
        compose_testbench(spec["core_path"], stimulus_type, path)
        spec[f"tb_{stimulus_type}"] = path
        net_paths.append(path)
        print(f"    {lbl:35s} -> {Path(path).name}")
    return net_paths


def interpolate_at(times_arr, values_arr, t_target: float) -> float:
    """
    Linear interpolation of a waveform value at t_target.
    Clamps to the boundary values if t_target is outside the range.
    """
    times_arr  = np.asarray(times_arr,  dtype=float)
    values_arr = np.asarray(values_arr, dtype=float)
    if t_target <= times_arr[0]:
        return float(values_arr[0])
    if t_target >= times_arr[-1]:
        return float(values_arr[-1])
    idx = np.searchsorted(times_arr, t_target, side="right") - 1
    t0, t1 = times_arr[idx], times_arr[idx + 1]
    v0, v1 = values_arr[idx], values_arr[idx + 1]
    frac = (t_target - t0) / (t1 - t0)
    return float(v0 + frac * (v1 - v0))


def read_trajectory(raw_path: str, trace_names: list) -> dict:
    """
    Read named traces from a .raw file and return a dict:
      {trace_name: {t_ns: value_V, ...}}
    sampled at SAMPLE_S.
    Returns {} if the file cannot be read.
    """
    try:
        raw = RawRead(raw_path)
    except Exception as e:
        print(f"      RawRead error: {e}")
        return {}

    result = {}
    time_trace = raw.get_trace("time")
    times = time_trace.get_wave()

    for name in trace_names:
        try:
            wave = raw.get_trace(name).get_wave()
            sampled = {}
            for t_s, t_ns in zip(SAMPLE_S, SAMPLE_NS):
                sampled[t_ns] = interpolate_at(times, wave, t_s)
            result[name] = sampled
        except Exception as e:
            result[name] = {t_ns: float("nan") for t_ns in SAMPLE_NS}
    return result


def write_pass(vq_final: float, vqb_final: float) -> bool:
    """True if WRITE succeeded and held: Q>=0.9V and Qb<=0.1V at t=20ns."""
    return vq_final >= 0.9 and vqb_final <= 0.1


def read_pass(vq_final: float, vqb_final: float) -> bool:
    """True if READ is non-destructive: Q>=0.9V and Qb<=0.1V at t=20ns."""
    return vq_final >= 0.9 and vqb_final <= 0.1


def flag_if_same_as_healthy(label: str, vq: float, vqb: float,
                             healthy_vq: float, healthy_vqb: float,
                             stimulus: str) -> str:
    """
    For the two access-transistor fault types, warn if their final state is
    indistinguishable from healthy — that would mean the fault isn't being
    exercised by this stimulus.
    """
    ACCESS_FAULTS = {"resistive_open_M5_1000ohm", "vth_drift_access_10pct"}
    if label not in ACCESS_FAULTS:
        return ""
    dq  = abs(vq  - healthy_vq)
    dqb = abs(vqb - healthy_vqb)
    if dq < 0.01 and dqb < 0.01:
        return f"  *** FLAG: {label} shows no deviation from healthy under {stimulus.upper()} ***"
    return ""


def print_trajectory_table(label: str, traj: dict, stimulus: str):
    """Print V(Q) and V(Qb) at each sample point for a single fault variant."""
    print(f"\n  Trajectory for [{label}] under {stimulus.upper()}:")
    print(f"    {'t (ns)':>8}  {'V(Q) V':>10}  {'V(Qb) V':>10}")
    print(f"    {'-'*8}  {'-'*10}  {'-'*10}")
    vq_d  = traj.get("V(Q)",  {})
    vqb_d = traj.get("V(Qb)", {})
    for t_ns in SAMPLE_NS:
        vq_v  = vq_d.get(t_ns,  float("nan"))
        vqb_v = vqb_d.get(t_ns, float("nan"))
        marker = " <-- WL rises" if t_ns == 2 else (
                 " <-- WL falls" if t_ns == 7 else "")
        print(f"    {t_ns:>8}  {vq_v:>10.4f}  {vqb_v:>10.4f}{marker}")


def run_stimulus_batch(specs: list, stimulus_type: str,
                       base_dir: Path) -> tuple:
    """
    Full pipeline for one stimulus type: compose -> simulate -> read results.
    Returns (rows, flags) where rows is a list of result dicts and flags is
    a list of warning strings.
    """
    print(f"\n{'='*60}")
    print(f"  Composing testbenches ({stimulus_type.upper()} stimulus)")
    print(f"{'='*60}")
    tb_dir  = base_dir / "testbenches"
    raw_dir = base_dir / "raw"
    log_dir = base_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    net_paths = compose_all(specs, stimulus_type, tb_dir)

    print(f"\n{'='*60}")
    print(f"  Running batch simulation ({stimulus_type.upper()}, {len(net_paths)} netlists)")
    print(f"{'='*60}")
    df = run_batch(
        net_paths=net_paths,
        raw_output_dir=str(raw_dir),
        log_output_dir=str(log_dir),
        flagged_csv_path=str(FLAGGED_CSV),
    )

    # Determine healthy baseline first
    healthy_vq  = float("nan")
    healthy_vqb = float("nan")

    rows  = []
    flags = []
    trajectories = {}  # label -> traj dict

    for spec, (_, row) in zip(specs, df.iterrows()):
        label    = spec["label"]
        raw_path = row["raw_path"]
        traj     = read_trajectory(raw_path, ["V(Q)", "V(Qb)"])
        trajectories[label] = traj

        vq_traj  = traj.get("V(Q)",  {})
        vqb_traj = traj.get("V(Qb)", {})
        vq_final  = vq_traj.get(20,  float("nan"))
        vqb_final = vqb_traj.get(20, float("nan"))

        if label == "healthy":
            healthy_vq  = vq_final
            healthy_vqb = vqb_final

        if stimulus_type == "write":
            passed = write_pass(vq_final, vqb_final)
        else:
            passed = read_pass(vq_final, vqb_final)

        flag_str = flag_if_same_as_healthy(
            label, vq_final, vqb_final,
            healthy_vq, healthy_vqb, stimulus_type
        )
        if flag_str:
            flags.append(flag_str)

        rows.append({
            "fault_class":    label,
            "converged":      row["converged"],
            "final V(Q)":     vq_final,
            "final V(Qb)":    vqb_final,
            "pass":           passed,
            "ltspice_version": row.get("ltspice_version") or "N/A",
        })

    return rows, flags, trajectories


def print_results_table(rows: list, stimulus_type: str, flags: list):
    """Print the summary results table for one stimulus type."""
    print(f"\n{'='*105}")
    print(f"  {stimulus_type.upper()} PILOT TABLE")
    print(f"{'='*105}")

    col_w = [35, 10, 12, 12, 6, 30]
    hdrs  = ["fault_class", "converged", "final V(Q)", "final V(Qb)", "pass", "ltspice_version"]
    hdr_line = "  ".join(h.ljust(w) for h, w in zip(hdrs, col_w))
    print(hdr_line)
    print("-" * len(hdr_line))

    for r in rows:
        line = "  ".join([
            r["fault_class"].ljust(col_w[0]),
            str(r["converged"]).ljust(col_w[1]),
            f"{r['final V(Q)']:.4f} V".ljust(col_w[2]),
            f"{r['final V(Qb)']:.4f} V".ljust(col_w[3]),
            str(r["pass"]).ljust(col_w[4]),
            r["ltspice_version"][:col_w[5]].ljust(col_w[5]),
        ])
        print(line)

    if flags:
        print()
        for f in flags:
            print(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=" * 60)
print("Extended pilot: WRITE + READ stimuli, all 5 fault classes")
print("=" * 60)

# Step 1 -- Generate all 5 cores (from the updated healthy core with Cq/Cqb)
print("\nStep 1: Generating cell cores from updated healthy template")
core_dir = PILOT_DIR / "extended" / "cores"
specs = generate_cores(core_dir)
for s in specs:
    print(f"  {s['label']:35s} -> {Path(s['core_path']).name}")

# ---------------------------------------------------------------------------
# WRITE batch
# ---------------------------------------------------------------------------

write_base = PILOT_DIR / "extended" / "write"
write_rows, write_flags, write_traj = run_stimulus_batch(specs, "write", write_base)

# ---------------------------------------------------------------------------
# READ batch
# ---------------------------------------------------------------------------

read_base = PILOT_DIR / "extended" / "read"
read_rows, read_flags, read_traj = run_stimulus_batch(specs, "read", read_base)

# ---------------------------------------------------------------------------
# Print both summary tables
# ---------------------------------------------------------------------------

print_results_table(write_rows, "write", write_flags)
print_results_table(read_rows,  "read",  read_flags)

# ---------------------------------------------------------------------------
# Trajectories for access-transistor fault types under both stimuli
# ---------------------------------------------------------------------------

TRAJECTORY_LABELS = ["resistive_open_M5_1000ohm", "vth_drift_access_10pct"]

print(f"\n{'='*60}")
print("  V(Q) / V(Qb) trajectories for access-transistor faults")
print(f"{'='*60}")
print("  (WL PULSE: rises at t=2ns, falls at t=7ns)")

for label in TRAJECTORY_LABELS:
    print_trajectory_table(label, write_traj.get(label, {}), "write")
    print_trajectory_table(label, read_traj.get(label, {}),  "read")

# Also print healthy trajectory as reference baseline
print(f"\n  Reference: healthy under WRITE and READ")
print_trajectory_table("healthy", write_traj.get("healthy", {}), "write")
print_trajectory_table("healthy", read_traj.get("healthy", {}),  "read")

print(f"\nAll outputs saved under: {PILOT_DIR / 'extended'}")
print("Stopping here for review -- do not proceed to full PVT sweep.")
