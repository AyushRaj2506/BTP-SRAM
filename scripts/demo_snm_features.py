"""
scripts/demo_snm_features.py -- Demonstration of butterfly-curve Hold and Read SNM extraction.

Evaluates 6 cell configurations:
  1. Healthy cell
  2. Bridging fault (2 kOhm)
  3. Resistive open M5 (1 kOhm)
  4. Resistive open M6 (1 kOhm)
  5. Vth drift storage pair (+10%)
  6. Vth drift access (+10%)

For each configuration:
  - Injects fault into core
  - Composes snm_hold and snm_read testbenches
  - Runs batch simulation in LTspice
  - Extracts Hold & Read SNM, lobe sides, asymmetry, bistability, and hold-to-read drop
  - Exports data/demo_features/snm_demo.csv and prints summary tables
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_cell_core, save_cell_core
from fault_injection import inject_resistive_open, inject_bridging_fault, inject_vth_drift
from testbench_composer import compose_testbench
from simulation_runner import run_batch
from feature_extractor import extract_snm_features, compute_read_hold_snm_drop

DEMO_DIR = PROJECT_ROOT / "data" / "demo_features"
HEALTHY_CORE = PROJECT_ROOT / "circuits" / "core" / "cell_core_healthy.net"

FAULT_SPECS = [
    {"label": "healthy",                      "type": "none",   "arg": None},
    {"label": "bridging_2000ohm",             "type": "bridge", "arg": 2000},
    {"label": "resistive_open_M5_1000ohm",    "type": "ro_m5",  "arg": 1000},
    {"label": "resistive_open_M6_1000ohm",    "type": "ro_m6",  "arg": 1000},
    {"label": "vth_drift_storage_pair_10pct", "type": "vth_sp", "arg": 10},
    {"label": "vth_drift_access_10pct",       "type": "vth_ac", "arg": 10},
]


def generate_cores(core_dir: Path) -> list:
    core_dir.mkdir(parents=True, exist_ok=True)
    healthy_lines = load_cell_core(str(HEALTHY_CORE))
    specs = []

    for spec in FAULT_SPECS:
        lbl = spec["label"]
        cpath = str(core_dir / f"core_{lbl}.net")
        ftype = spec["type"]
        arg = spec["arg"]

        if ftype == "none":
            save_cell_core(list(healthy_lines), cpath)
        elif ftype == "bridge":
            inject_bridging_fault(list(healthy_lines), arg, cpath)
        elif ftype == "ro_m5":
            inject_resistive_open(list(healthy_lines), "M5", arg, cpath)
        elif ftype == "ro_m6":
            inject_resistive_open(list(healthy_lines), "M6", arg, cpath)
        elif ftype == "vth_sp":
            inject_vth_drift(list(healthy_lines), "storage_pair", arg, cpath)
        elif ftype == "vth_ac":
            inject_vth_drift(list(healthy_lines), "access", arg, cpath)

        specs.append({**spec, "core_path": cpath})

    return specs


def run_snm_demo():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    core_dir = DEMO_DIR / "snm_cores"
    tb_dir = DEMO_DIR / "snm_testbenches"
    raw_dir = DEMO_DIR / "snm_raw"
    log_dir = DEMO_DIR / "snm_logs"

    for d in (core_dir, tb_dir, raw_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    specs = generate_cores(core_dir)

    all_tbs = []
    tb_map = {}

    for spec in specs:
        lbl = spec["label"]
        tb_map[lbl] = {}
        for op in ("snm_hold", "snm_read"):
            tb_path = str(tb_dir / f"tb_{lbl}_{op}.net")
            compose_testbench(spec["core_path"], op, tb_path)
            all_tbs.append(tb_path)
            tb_map[lbl][op] = tb_path

    print(f"Launching batch simulation for {len(all_tbs)} SNM testbenches (6 cells x 2 modes)...")
    flags_csv = DEMO_DIR / "snm_convergence_flags.csv"
    df_sim = run_batch(all_tbs, str(raw_dir), str(log_dir), str(flags_csv))

    net_to_raw = dict(zip(df_sim["net_path"], df_sim["raw_path"]))

    results = []
    for spec in specs:
        lbl = spec["label"]
        raw_hold = net_to_raw[tb_map[lbl]["snm_hold"]]
        raw_read = net_to_raw[tb_map[lbl]["snm_read"]]

        f_hold = extract_snm_features(raw_hold, mode="hold", vdd=1.0)
        f_read = extract_snm_features(raw_read, mode="read", vdd=1.0)
        drop_v = compute_read_hold_snm_drop(f_hold, f_read)

        row = {
            "fault_class": lbl,
            "hold_snm_mV": round(f_hold["hold_snm_v"] * 1e3, 1),
            "hold_lobe1_mV": round(f_hold["hold_snm_state0_v"] * 1e3, 1),
            "hold_lobe2_mV": round(f_hold["hold_snm_state1_v"] * 1e3, 1),
            "hold_asym_mV": round(f_hold["hold_snm_asym_v"] * 1e3, 1),
            "hold_bistable": f_hold["hold_bistable"],
            "read_snm_mV": round(f_read["read_snm_v"] * 1e3, 1),
            "read_lobe1_mV": round(f_read["read_snm_state0_v"] * 1e3, 1),
            "read_lobe2_mV": round(f_read["read_snm_state1_v"] * 1e3, 1),
            "read_asym_mV": round(f_read["read_snm_asym_v"] * 1e3, 1),
            "read_bistable": f_read["read_bistable"],
            "snm_drop_mV": round(drop_v * 1e3, 1),
            "hold_ok": f_hold["hold_snm_ok"],
            "read_ok": f_read["read_snm_ok"],
        }
        results.append(row)

    df_out = pd.DataFrame(results)
    csv_path = DEMO_DIR / "snm_demo.csv"
    df_out.to_csv(str(csv_path), index=False)
    print(f"\nSaved SNM demo features table to: {csv_path}\n")

    # Formatted console output
    print("=" * 115)
    print("  STATIC NOISE MARGIN (SNM) DIAGNOSTIC MATRIX")
    print("=" * 115)
    header = f"{'Fault Class':32s} | {'Hold SNM':>9} | {'H Lobe0':>8} | {'H Lobe1':>8} | {'Read SNM':>9} | {'R Lobe0':>8} | {'R Lobe1':>8} | {'R Asym':>7} | {'Drop':>7}"
    print(header)
    print("-" * 115)
    for r in results:
        print(
            f"{r['fault_class']:32s} | "
            f"{r['hold_snm_mV']:8.1f}m | "
            f"{r['hold_lobe1_mV']:7.1f}m | "
            f"{r['hold_lobe2_mV']:7.1f}m | "
            f"{r['read_snm_mV']:8.1f}m | "
            f"{r['read_lobe1_mV']:7.1f}m | "
            f"{r['read_lobe2_mV']:7.1f}m | "
            f"{r['read_asym_mV']:6.1f}m | "
            f"{r['snm_drop_mV']:6.1f}m"
        )
    print("=" * 115)

    return df_out


if __name__ == "__main__":
    run_snm_demo()
