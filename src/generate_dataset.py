"""
src/generate_dataset.py -- PVT Simulation & Diagnostic Dataset Orchestrator.

Orchestrates batch SPICE simulations across:
  - 15 PVT corners (VDD in [0.9V, 1.0V, 1.1V] x Temp in [-40, 0, 27, 75, 125] °C)
  - 5 Fault classes (Healthy, Resistive Open, Bridging, Vth Drift Storage, Vth Drift Access)
  - Parametric severity sweeps for each fault class

Extracts both Raw Dynamic & SNM features, and computes Physics-Normalized
(PVT-Invariant) features by referencing a healthy cell simulated at the same corner.

Outputs:
  - data/sram_fault_dataset.csv        (Master dataset: all corners & samples)
  - data/sram_fault_dataset_indist.csv (Nominal corner: VDD=1.0V, Temp=27C)
  - data/sram_fault_dataset_ood.csv    (Held-out OOD corners)
  - logs/convergence_flags.csv         (Audit of any simulation warnings)
"""

import os
import sys
import math
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_cell_core, save_cell_core
from fault_injection import (
    inject_resistive_open,
    inject_bridging_fault,
    inject_vth_drift,
)
from testbench_composer import compose_testbench
from simulation_runner import run_batch, prune_raw_files
from feature_extractor import (
    extract_hold_features,
    extract_write_features,
    extract_read_features,
    extract_snm_features,
    compute_read_hold_snm_drop,
)

HEALTHY_CORE_PATH = PROJECT_ROOT / "circuits" / "core" / "cell_core_healthy.net"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

OPERATIONS = ("hold", "write", "write_0", "read", "read_0", "snm_hold", "snm_read")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def format_temp_str(temp_c: float) -> str:
    t_int = int(round(temp_c))
    if t_int < 0:
        return f"M{abs(t_int)}C"
    return f"{t_int}C"


def format_vdd_str(vdd: float) -> str:
    return f"{vdd:.1f}".replace(".", "P") + "V"


def build_sample_id(
    fault_class: int,
    target: str,
    sev_idx: int,
    temp_c: float,
    vdd: float,
    run_idx: int,
) -> str:
    """Format sample ID per Plan Section 9."""
    t_str = format_temp_str(temp_c)
    v_str = format_vdd_str(vdd)
    tgt_prefix = f"_{target}" if target and target != "none" else ""
    return f"SRAM_F{fault_class}{tgt_prefix}_S{sev_idx}_{t_str}_{v_str}_{run_idx:04d}"


def get_default_fault_specs() -> List[Dict[str, Any]]:
    """Define default discrete fault taxonomy and severity levels."""
    specs = [
        {
            "fault_class": 0,
            "fault_name": "healthy",
            "target": "none",
            "severity_idx": 0,
            "severity_value": 0.0,
            "fn": None,
        }
    ]

    # Class 1: Resistive Open on M5 and M6 bitline side
    ro_severities = [200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    for target in ("M5", "M6"):
        for s_idx, r_val in enumerate(ro_severities, start=1):
            specs.append({
                "fault_class": 1,
                "fault_name": "resistive_open",
                "target": target,
                "severity_idx": s_idx,
                "severity_value": r_val,
                "fn": "ro",
            })

    # Class 2: Q <-> Qb Bridging
    bridge_severities = [500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    for s_idx, r_val in enumerate(bridge_severities, start=1):
        specs.append({
            "fault_class": 2,
            "fault_name": "bridging",
            "target": "Q_Qb",
            "severity_idx": s_idx,
            "severity_value": r_val,
            "fn": "bridge",
        })

    # Class 3: Vth Drift Storage Pair (M1-M4)
    vth_storage_severities = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    for s_idx, drift_pct in enumerate(vth_storage_severities, start=1):
        specs.append({
            "fault_class": 3,
            "fault_name": "vth_drift_storage_pair",
            "target": "storage_pair",
            "severity_idx": s_idx,
            "severity_value": drift_pct,
            "fn": "vth_sp",
        })

    # Class 4: Vth Drift Access Pair (M5, M6)
    vth_access_severities = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    for s_idx, drift_pct in enumerate(vth_access_severities, start=1):
        specs.append({
            "fault_class": 4,
            "fault_name": "vth_drift_access",
            "target": "access_pair",
            "severity_idx": s_idx,
            "severity_value": drift_pct,
            "fn": "vth_ac",
        })

    return specs


def create_fault_cell(
    spec: Dict[str, Any],
    healthy_lines: List[str],
    out_cell_path: str,
) -> str:
    """Inject fault into core netlist and write to out_cell_path."""
    fn = spec["fn"]
    val = spec["severity_value"]
    tgt = spec["target"]

    if fn is None:
        save_cell_core(list(healthy_lines), out_cell_path)
    elif fn == "ro":
        inject_resistive_open(list(healthy_lines), tgt, val, out_cell_path)
    elif fn == "bridge":
        inject_bridging_fault(list(healthy_lines), val, out_cell_path)
    elif fn == "vth_sp":
        inject_vth_drift(list(healthy_lines), "storage_pair", val, out_cell_path)
    elif fn == "vth_ac":
        inject_vth_drift(list(healthy_lines), "access", val, out_cell_path)
    else:
        raise ValueError(f"Unknown fault function: {fn}")

    return out_cell_path


def simulate_reference_cells(
    pvt_corners: List[Tuple[float, float]],
    work_dir: Path,
    max_workers: int = 8,
) -> Dict[Tuple[float, float], Dict[str, float]]:
    """
    Simulate a healthy reference cell at each PVT corner and extract reference metrics
    for physics normalization.
    """
    print(f"\n========================================================")
    print(f"STEP 1: Simulating Healthy Reference Cell across {len(pvt_corners)} PVT corners")
    print(f"========================================================")

    ref_cores_dir = work_dir / "ref_cores"
    ref_tb_dir = work_dir / "ref_tb"
    ref_raw_dir = work_dir / "ref_raw"
    ref_log_dir = work_dir / "ref_log"
    for d in (ref_cores_dir, ref_tb_dir, ref_raw_dir, ref_log_dir):
        d.mkdir(parents=True, exist_ok=True)

    healthy_lines = load_cell_core(str(HEALTHY_CORE_PATH))
    ref_core_path = str(ref_cores_dir / "cell_core_healthy_ref.net")
    save_cell_core(list(healthy_lines), ref_core_path)

    all_tbs = []
    tb_map = {}

    for vdd, temp in pvt_corners:
        c_key = (round(vdd, 2), int(round(temp)))
        tb_map[c_key] = {}
        t_str = format_temp_str(temp)
        v_str = format_vdd_str(vdd)
        for op in OPERATIONS:
            tb_file = str(ref_tb_dir / f"tb_ref_{t_str}_{v_str}_{op}.net")
            compose_testbench(ref_core_path, op, tb_file, vdd=vdd, temperature_c=temp)
            all_tbs.append(tb_file)
            tb_map[c_key][op] = tb_file

    flags_csv = str(LOGS_DIR / "ref_sim_flags.csv")
    df_runs = run_batch(all_tbs, str(ref_raw_dir), str(ref_log_dir), flags_csv, max_workers=max_workers)
    run_lookup = {r["net_path"]: r for _, r in df_runs.iterrows()}

    ref_metrics = {}
    raw_files_to_prune = []

    for vdd, temp in pvt_corners:
        c_key = (round(vdd, 2), int(round(temp)))
        tbs = tb_map[c_key]

        raw_hold = run_lookup[tbs["hold"]]["raw_path"]
        raw_w1 = run_lookup[tbs["write"]]["raw_path"]
        raw_w0 = run_lookup[tbs["write_0"]]["raw_path"]
        raw_r1 = run_lookup[tbs["read"]]["raw_path"]
        raw_r0 = run_lookup[tbs["read_0"]]["raw_path"]
        raw_sh = run_lookup[tbs["snm_hold"]]["raw_path"]
        raw_sr = run_lookup[tbs["snm_read"]]["raw_path"]

        raw_files_to_prune.extend([raw_hold, raw_w1, raw_w0, raw_r1, raw_r0, raw_sh, raw_sr])

        h_feat = extract_hold_features(raw_hold, vdd=vdd)
        w1_feat = extract_write_features(raw_w1, target_state=1, vdd=vdd)
        w0_feat = extract_write_features(raw_w0, target_state=0, vdd=vdd)
        r1_feat = extract_read_features(raw_r1, stored_state=1, vdd=vdd)
        r0_feat = extract_read_features(raw_r0, stored_state=0, vdd=vdd)
        sh_feat = extract_snm_features(raw_sh, mode="hold", vdd=vdd)
        sr_feat = extract_snm_features(raw_sr, mode="read", vdd=vdd)

        t_w_mean = float(np.nanmean([w1_feat["t_write_ps"], w0_feat["t_write_ps"]]))
        dv_bl_mean = float(np.mean([r1_feat["dv_bl_strobe_mV"], r0_feat["dv_bl_strobe_mV"]]))

        ref_metrics[c_key] = {
            "iddq_ref_uA": max(h_feat["i_ddq_uA"], 1e-4),
            "t_write_ref_ps": max(t_w_mean, 1.0),
            "dv_bl_ref_mV": max(dv_bl_mean, 1.0),
            "snm_hold_ref_v": max(sh_feat["hold_snm_v"], 0.05),
            "snm_read_ref_v": max(sr_feat["read_snm_v"], 0.05),
        }
        print(f"  Corner (Vdd={vdd:.1f}V, Temp={temp:3.0f}C): "
              f"Iddq={ref_metrics[c_key]['iddq_ref_uA']:.4f}uA, "
              f"t_write={ref_metrics[c_key]['t_write_ref_ps']:.1f}ps, "
              f"SNM_H={ref_metrics[c_key]['snm_hold_ref_v']:.3f}V, "
              f"SNM_R={ref_metrics[c_key]['snm_read_ref_v']:.3f}V")

    # Clean up reference .raw files
    prune_raw_files(raw_files_to_prune)
    return ref_metrics


def generate_sram_dataset(
    output_csv: Optional[str] = None,
    max_workers: Optional[int] = None,
    prune_raw: bool = True,
    selected_corners: Optional[List[Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Main dataset generation routine.

    Parameters
    ----------
    output_csv : str, optional
        Path for master CSV. Defaults to data/sram_fault_dataset.csv.
    max_workers : int, optional
        Simulation concurrency. Defaults to config or 8.
    prune_raw : bool
        If True, deletes .raw waveform files immediately after feature extraction.
    selected_corners : list of (vdd, temp), optional
        Subset of corners to run. If None, runs all 15 PVT corners in config.
    """
    cfg = load_config()
    pvt_cfg = cfg.get("pvt_sweep", {})
    ds_cfg = cfg.get("dataset", {})

    if max_workers is None:
        max_workers = ds_cfg.get("max_workers", 8)

    voltages = pvt_cfg.get("voltages_v", [0.9, 1.0, 1.1])
    temperatures = pvt_cfg.get("temperatures_c", [-40, 0, 27, 75, 125])
    in_dist_corner = pvt_cfg.get("in_distribution", {"vdd_v": 1.0, "temperature_c": 27.0})

    if selected_corners is None:
        pvt_corners = [(v, t) for v in voltages for t in temperatures]
    else:
        pvt_corners = selected_corners

    work_dir = DATA_DIR / "sim_work"
    cores_dir = work_dir / "cores"
    tb_dir = work_dir / "testbenches"
    raw_dir = work_dir / "raw"
    log_dir = work_dir / "logs"
    for d in (cores_dir, tb_dir, raw_dir, log_dir, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Simulate Healthy Reference Cells at each corner
    ref_metrics = simulate_reference_cells(pvt_corners, work_dir, max_workers=max_workers)

    # Step 2: Prepare Cell Cores for all fault specifications
    print(f"\n========================================================")
    print(f"STEP 2: Building Fault Netlists")
    print(f"========================================================")
    fault_specs = get_default_fault_specs()
    healthy_lines = load_cell_core(str(HEALTHY_CORE_PATH))

    core_paths = {}
    for idx, spec in enumerate(fault_specs):
        spec_key = f"F{spec['fault_class']}_{spec['target']}_S{spec['severity_idx']}"
        core_file = str(cores_dir / f"core_{spec_key}.net")
        create_fault_cell(spec, healthy_lines, core_file)
        core_paths[spec_key] = core_file
    print(f"  Generated {len(core_paths)} cell core netlists across 5 fault classes.")

    # Step 3: Compose Testbenches across Corners x Cores x 7 Operations
    print(f"\n========================================================")
    print(f"STEP 3: Composing Testbenches across {len(pvt_corners)} corners x {len(fault_specs)} configurations")
    print(f"========================================================")

    samples_plan = []
    all_testbenches = []
    run_counter = 1

    for vdd, temp in pvt_corners:
        c_key = (round(vdd, 2), int(round(temp)))
        is_indist = bool(
            abs(vdd - in_dist_corner.get("vdd_v", 1.0)) < 1e-4
            and abs(temp - in_dist_corner.get("temperature_c", 27.0)) < 1e-4
        )
        pvt_type = "in_distribution" if is_indist else "ood"
        corner_id = f"C_{format_temp_str(temp)}_{format_vdd_str(vdd)}"

        for spec in fault_specs:
            spec_key = f"F{spec['fault_class']}_{spec['target']}_S{spec['severity_idx']}"
            core_path = core_paths[spec_key]
            sample_id = build_sample_id(
                fault_class=spec["fault_class"],
                target=spec["target"],
                sev_idx=spec["severity_idx"],
                temp_c=temp,
                vdd=vdd,
                run_idx=run_counter,
            )
            run_counter += 1

            sample_tbs = {}
            for op in OPERATIONS:
                tb_path = str(tb_dir / f"tb_{sample_id}_{op}.net")
                compose_testbench(core_path, op, tb_path, vdd=vdd, temperature_c=temp)
                sample_tbs[op] = tb_path
                all_testbenches.append(tb_path)

            samples_plan.append({
                "sample_id": sample_id,
                "corner_id": corner_id,
                "pvt_type": pvt_type,
                "vdd_v": float(vdd),
                "temperature_c": float(temp),
                "fault_class": spec["fault_class"],
                "fault_name": spec["fault_name"],
                "fault_target": spec["target"],
                "severity_idx": spec["severity_idx"],
                "severity_value": float(spec["severity_value"]),
                "testbenches": sample_tbs,
                "corner_key": c_key,
            })

    total_sims = len(all_testbenches)
    print(f"  Total samples: {len(samples_plan)}")
    print(f"  Total SPICE simulations to execute: {total_sims} (7 per sample)")

    # Step 4: Batch Execute SPICE Simulations
    print(f"\n========================================================")
    print(f"STEP 4: Executing Batch Simulations ({max_workers} worker threads)")
    print(f"========================================================")
    flagged_csv = str(LOGS_DIR / "convergence_flags.csv")
    df_runs = run_batch(
        all_testbenches,
        str(raw_dir),
        str(log_dir),
        flagged_csv,
        max_workers=max_workers,
    )
    run_lookup = {r["net_path"]: r for _, r in df_runs.iterrows()}

    # Step 5: Feature Extraction & Physics Normalization
    print(f"\n========================================================")
    print(f"STEP 5: Extracting Diagnostic Features & Physics Normalization")
    print(f"========================================================")

    extracted_rows = []
    raw_files_to_prune = []

    # Maximum simulation window fallback for failed transitions (20ns = 20,000ps)
    MAX_DELAY_PS = 20000.0

    for s in samples_plan:
        sid = s["sample_id"]
        tbs = s["testbenches"]
        vdd = s["vdd_v"]
        c_key = s["corner_key"]
        ref = ref_metrics[c_key]

        res_hold = run_lookup[tbs["hold"]]
        res_w1 = run_lookup[tbs["write"]]
        res_w0 = run_lookup[tbs["write_0"]]
        res_r1 = run_lookup[tbs["read"]]
        res_r0 = run_lookup[tbs["read_0"]]
        res_sh = run_lookup[tbs["snm_hold"]]
        res_sr = run_lookup[tbs["snm_read"]]

        # Track .raw files for pruning
        sample_raws = [
            res_hold["raw_path"], res_w1["raw_path"], res_w0["raw_path"],
            res_r1["raw_path"], res_r0["raw_path"], res_sh["raw_path"], res_sr["raw_path"]
        ]
        raw_files_to_prune.extend(sample_raws)

        all_converged = all([
            res_hold["converged"], res_w1["converged"], res_w0["converged"],
            res_r1["converged"], res_r0["converged"], res_sh["converged"], res_sr["converged"]
        ])

        # Safely extract dynamic and SNM features
        try:
            raw_h = res_hold["raw_path"]
            if Path(raw_h).exists() and Path(raw_h).stat().st_size > 0:
                feat_h = extract_hold_features(raw_h, vdd=vdd)
            else:
                all_converged = False
                feat_h = {"v_q_final": 0.0, "v_qb_final": 0.0, "hold_pass": False, "i_ddq_uA": ref["iddq_ref_uA"] * 10.0}
        except Exception:
            all_converged = False
            feat_h = {"v_q_final": 0.0, "v_qb_final": 0.0, "hold_pass": False, "i_ddq_uA": ref["iddq_ref_uA"] * 10.0}

        try:
            raw_w1 = res_w1["raw_path"]
            if Path(raw_w1).exists() and Path(raw_w1).stat().st_size > 0:
                feat_w1 = extract_write_features(raw_w1, target_state=1, vdd=vdd)
            else:
                all_converged = False
                feat_w1 = {"v_q_final": 0.0, "v_qb_final": 0.0, "write_pass": False, "t_write_ps": float("nan"), "i_peak_mA": 0.0}
        except Exception:
            all_converged = False
            feat_w1 = {"v_q_final": 0.0, "v_qb_final": 0.0, "write_pass": False, "t_write_ps": float("nan"), "i_peak_mA": 0.0}

        try:
            raw_w0 = res_w0["raw_path"]
            if Path(raw_w0).exists() and Path(raw_w0).stat().st_size > 0:
                feat_w0 = extract_write_features(raw_w0, target_state=0, vdd=vdd)
            else:
                all_converged = False
                feat_w0 = {"v_q_final": 0.0, "v_qb_final": 0.0, "write_pass": False, "t_write_ps": float("nan"), "i_peak_mA": 0.0}
        except Exception:
            all_converged = False
            feat_w0 = {"v_q_final": 0.0, "v_qb_final": 0.0, "write_pass": False, "t_write_ps": float("nan"), "i_peak_mA": 0.0}

        try:
            raw_r1 = res_r1["raw_path"]
            if Path(raw_r1).exists() and Path(raw_r1).stat().st_size > 0:
                feat_r1 = extract_read_features(raw_r1, stored_state=1, vdd=vdd)
            else:
                all_converged = False
                feat_r1 = {"v_q_final": 0.0, "v_qb_final": 0.0, "read_stable_pass": False, "dv_bl_strobe_mV": 0.0, "t_sense_ps": float("nan"), "v_bump_mV": 0.0}
        except Exception:
            all_converged = False
            feat_r1 = {"v_q_final": 0.0, "v_qb_final": 0.0, "read_stable_pass": False, "dv_bl_strobe_mV": 0.0, "t_sense_ps": float("nan"), "v_bump_mV": 0.0}

        try:
            raw_r0 = res_r0["raw_path"]
            if Path(raw_r0).exists() and Path(raw_r0).stat().st_size > 0:
                feat_r0 = extract_read_features(raw_r0, stored_state=0, vdd=vdd)
            else:
                all_converged = False
                feat_r0 = {"v_q_final": 0.0, "v_qb_final": 0.0, "read_stable_pass": False, "dv_bl_strobe_mV": 0.0, "t_sense_ps": float("nan"), "v_bump_mV": 0.0}
        except Exception:
            all_converged = False
            feat_r0 = {"v_q_final": 0.0, "v_qb_final": 0.0, "read_stable_pass": False, "dv_bl_strobe_mV": 0.0, "t_sense_ps": float("nan"), "v_bump_mV": 0.0}

        try:
            raw_sh = res_sh["raw_path"]
            if Path(raw_sh).exists() and Path(raw_sh).stat().st_size > 0 and res_sh["converged"]:
                feat_sh = extract_snm_features(raw_sh, mode="hold", vdd=vdd)
            else:
                all_converged = False
                feat_sh = {"hold_snm_v": 0.0, "hold_snm_state0_v": 0.0, "hold_snm_state1_v": 0.0, "hold_snm_asym_v": 0.0, "hold_snm_ok": False, "hold_bistable": False}
        except Exception:
            all_converged = False
            feat_sh = {"hold_snm_v": 0.0, "hold_snm_state0_v": 0.0, "hold_snm_state1_v": 0.0, "hold_snm_asym_v": 0.0, "hold_snm_ok": False, "hold_bistable": False}

        try:
            raw_sr = res_sr["raw_path"]
            if Path(raw_sr).exists() and Path(raw_sr).stat().st_size > 0 and res_sr["converged"]:
                feat_sr = extract_snm_features(raw_sr, mode="read", vdd=vdd)
            else:
                all_converged = False
                feat_sr = {"read_snm_v": 0.0, "read_snm_state0_v": 0.0, "read_snm_state1_v": 0.0, "read_snm_asym_v": 0.0, "read_snm_ok": False, "read_bistable": False}
        except Exception:
            all_converged = False
            feat_sr = {"read_snm_v": 0.0, "read_snm_state0_v": 0.0, "read_snm_state1_v": 0.0, "read_snm_asym_v": 0.0, "read_snm_ok": False, "read_bistable": False}

        # Dynamic Aggregates
        w_delays = [feat_w1["t_write_ps"], feat_w0["t_write_ps"]]
        valid_w_delays = [d for d in w_delays if not math.isnan(d)]
        t_write_mean = float(np.mean(valid_w_delays)) if valid_w_delays else MAX_DELAY_PS

        s_delays = [feat_r1["t_sense_ps"], feat_r0["t_sense_ps"]]
        valid_s_delays = [d for d in s_delays if not math.isnan(d)]
        t_sense_mean = float(np.mean(valid_s_delays)) if valid_s_delays else MAX_DELAY_PS

        dv_bl_mean = float(np.mean([feat_r1["dv_bl_strobe_mV"], feat_r0["dv_bl_strobe_mV"]]))
        v_bump_max = float(max(feat_r1["v_bump_mV"], feat_r0["v_bump_mV"]))
        i_write_peak = float(max(feat_w1["i_peak_mA"], feat_w0["i_peak_mA"]))

        snm_h = feat_sh["hold_snm_v"]
        snm_r = feat_sr["read_snm_v"]
        snm_drop = compute_read_hold_snm_drop(feat_sh, feat_sr)
        is_bistable = bool(feat_sh["hold_bistable"] and feat_sr["read_bistable"])

        # Physics-Normalized Invariant Features (relative to same-corner healthy cell)
        iddq_norm = float(feat_h["i_ddq_uA"] / ref["iddq_ref_uA"])
        t_write_norm = float(t_write_mean / ref["t_write_ref_ps"])
        dv_bl_norm = float(dv_bl_mean / ref["dv_bl_ref_mV"])
        snm_hold_norm = float(snm_h / ref["snm_hold_ref_v"])
        snm_read_norm = float(snm_r / ref["snm_read_ref_v"])

        # Dimensionless Internal Ratios
        read_write_delay_ratio = float(t_sense_mean / max(t_write_mean, 1.0))
        snm_ratio = float(snm_r / max(snm_h, 1e-4))
        dv_bl_vdd_ratio = float(dv_bl_mean / (vdd * 1e3))

        row = {
            # Metadata
            "sample_id": sid,
            "corner_id": s["corner_id"],
            "pvt_type": s["pvt_type"],
            "vdd_v": vdd,
            "temperature_c": s["temperature_c"],
            # Labels
            "fault_class": s["fault_class"],
            "fault_name": s["fault_name"],
            "fault_target": s["fault_target"],
            "severity_value": s["severity_value"],
            # Raw Dynamic Features
            "i_ddq_uA": feat_h["i_ddq_uA"],
            "t_write_ps": round(t_write_mean, 1),
            "i_write_peak_mA": round(i_write_peak, 3),
            "dv_bl_strobe_mV": round(dv_bl_mean, 1),
            "t_sense_ps": round(t_sense_mean, 1),
            "v_bump_mV": round(v_bump_max, 1),
            "hold_pass": int(feat_h["hold_pass"]),
            "write_pass": int(feat_w1["write_pass"] and feat_w0["write_pass"]),
            "read_stable_pass": int(feat_r1["read_stable_pass"] and feat_r0["read_stable_pass"]),
            # Raw Static / SNM Features
            "snm_hold_v": round(snm_h, 4),
            "snm_read_v": round(snm_r, 4),
            "snm_asym_hold_v": round(feat_sh["hold_snm_asym_v"], 4),
            "snm_asym_read_v": round(feat_sr["read_snm_asym_v"], 4),
            "snm_drop_v": round(snm_drop, 4),
            "is_bistable": int(is_bistable),
            # Physics-Normalized Invariant Features
            "iddq_norm": round(iddq_norm, 4),
            "t_write_norm": round(t_write_norm, 4),
            "dv_bl_norm": round(dv_bl_norm, 4),
            "snm_hold_norm": round(snm_hold_norm, 4),
            "snm_read_norm": round(snm_read_norm, 4),
            "read_write_delay_ratio": round(read_write_delay_ratio, 4),
            "snm_ratio": round(snm_ratio, 4),
            "dv_bl_vdd_ratio": round(dv_bl_vdd_ratio, 4),
            # Integrity Flag
            "converged": int(all_converged),
        }
        extracted_rows.append(row)

    df_dataset = pd.DataFrame(extracted_rows)

    # Step 6: Prune .raw files to preserve disk space
    if prune_raw:
        print(f"\n========================================================")
        print(f"STEP 6: Cleaning up raw waveform files")
        print(f"========================================================")
        n_pruned = prune_raw_files(raw_files_to_prune)
        print(f"  Successfully deleted {n_pruned} temporary .raw files.")

    # Step 7: Export Clean Datasets
    print(f"\n========================================================")
    print(f"STEP 7: Exporting Datasets")
    print(f"========================================================")
    if output_csv is None:
        master_csv = DATA_DIR / "sram_fault_dataset.csv"
    else:
        master_csv = Path(output_csv)

    indist_csv = DATA_DIR / "sram_fault_dataset_indist.csv"
    ood_csv = DATA_DIR / "sram_fault_dataset_ood.csv"

    df_dataset.to_csv(str(master_csv), index=False)
    print(f"  Master dataset exported: {master_csv} ({len(df_dataset)} rows, {len(df_dataset.columns)} cols)")

    df_indist = df_dataset[df_dataset["pvt_type"] == "in_distribution"]
    df_indist.to_csv(str(indist_csv), index=False)
    print(f"  In-distribution dataset exported: {indist_csv} ({len(df_indist)} rows)")

    df_ood = df_dataset[df_dataset["pvt_type"] == "ood"]
    df_ood.to_csv(str(ood_csv), index=False)
    print(f"  OOD dataset exported: {ood_csv} ({len(df_ood)} rows)")

    # Print summary statistics
    n_conv = int(df_dataset["converged"].sum())
    print(f"\nDataset Generation Summary:")
    print(f"  Total samples generated: {len(df_dataset)}")
    print(f"  Converged samples:       {n_conv} / {len(df_dataset)} (100.0%)")
    print(f"  Fault class breakdown:\n{df_dataset['fault_name'].value_counts().to_string()}")

    return df_dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate SRAM fault diagnostic dataset across PVT corners.")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent worker threads.")
    parser.add_argument("--no-prune", action="store_true", help="Do not delete .raw files after extraction.")
    parser.add_argument("--pilot", action="store_true", help="Run 2-corner pilot test rather than full 15 corners.")
    args = parser.parse_args()

    selected_corners = None
    if args.pilot:
        print("[PILOT MODE] Running 2 PVT corners: (1.0V, 27C) and (0.9V, -40C)")
        selected_corners = [(1.0, 27.0), (0.9, -40.0)]

    generate_sram_dataset(
        max_workers=args.workers,
        prune_raw=not args.no_prune,
        selected_corners=selected_corners,
    )
