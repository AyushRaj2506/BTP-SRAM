"""
src/feature_extractor.py -- Dynamic and Static Diagnostic Feature Extraction.

Extracts physically meaningful diagnostic features from LTspice .raw simulation
files for SRAM 6T cells under HOLD, WRITE, and READ operations.

Extracted Features:
  - Static: V(Q)_final, V(Qb)_final, Pass/Fail booleans
  - Hold:   I_ddq (quiescent supply current -- strongly detects bridging)
  - Write:  t_write (time from WL 50% rise until target storage node crosses 0.5V),
            I_write_peak (peak dynamic supply current)
  - Read:   dV_BL_strobe (|V(BL) - V(BLB)| at sense-amplifier strobe time),
            t_sense (time from WL rise until dV_BL >= 100mV),
            V_bump (peak voltage bump on low storage node during read)
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from spicelib.raw.raw_read import RawRead


def _load_traces(raw_path: str) -> Dict[str, np.ndarray]:
    """Load all standard analog waveform traces into numpy arrays."""
    raw = RawRead(str(raw_path))
    traces = {}
    traces["time"] = raw.get_trace("time").get_wave()
    for name in raw.get_trace_names():
        n_clean = name.lower()
        if n_clean.startswith("v(") or n_clean.startswith("i("):
            traces[n_clean] = raw.get_trace(name).get_wave()
    return traces


def _interp(time_arr: np.ndarray, val_arr: np.ndarray, t_target: float) -> float:
    """Safely interpolate scalar value at t_target."""
    return float(np.interp(t_target, time_arr, val_arr))


def extract_hold_features(raw_path: str) -> Dict[str, Any]:
    """
    Extract diagnostic features from a HOLD simulation (.raw).

    Features:
      - v_q_final: V(Q) at t=20ns
      - v_qb_final: V(Qb) at t=20ns
      - hold_pass: True if cell retained state (Q >= 0.9V, Qb <= 0.1V)
      - i_ddq_uA: Quiescent supply current (absolute |I(Vdd)|) at t=20ns in microamps
    """
    traces = _load_traces(raw_path)
    t = traces["time"]
    vq = traces["v(q)"]
    vqb = traces["v(qb)"]

    vq_final = _interp(t, vq, 20e-9)
    vqb_final = _interp(t, vqb, 20e-9)
    hold_pass = bool(vq_final >= 0.9 and vqb_final <= 0.1)

    # Supply current trace is typically i(v4) or i(vdd)
    iddq = 0.0
    for key in ("i(v4)", "i(vdd)"):
        if key in traces:
            iddq = abs(_interp(t, traces[key], 20e-9)) * 1e6  # uA
            break

    return {
        "v_q_final": round(vq_final, 4),
        "v_qb_final": round(vqb_final, 4),
        "hold_pass": hold_pass,
        "i_ddq_uA": round(iddq, 4),
    }


def extract_write_features(raw_path: str, target_state: int = 1) -> Dict[str, Any]:
    """
    Extract diagnostic features from a WRITE simulation (.raw).

    Parameters
    ----------
    raw_path : str
        Path to .raw simulation file.
    target_state : int
        1 if writing '1' to Q (BL=1, BLB=0), 0 if writing '0' to Q (BL=0, BLB=1).

    Features:
      - v_q_final, v_qb_final: voltages at t=20ns
      - write_pass: True if final state matches target
      - t_write_ps: Write switching delay in picoseconds (from WL 50% rise to node crossing 0.5V)
      - i_peak_mA: Peak dynamic supply current during write window (2ns to 7ns)
    """
    traces = _load_traces(raw_path)
    t = traces["time"]
    vq = traces["v(q)"]
    vqb = traces["v(qb)"]

    vq_final = _interp(t, vq, 20e-9)
    vqb_final = _interp(t, vqb, 20e-9)

    if target_state == 1:
        write_pass = bool(vq_final >= 0.9 and vqb_final <= 0.1)
        target_wave = vq
        cross_up = True
    else:
        write_pass = bool(vq_final <= 0.1 and vqb_final >= 0.9)
        target_wave = vq
        cross_up = False

    # WL 50% rise reference time: pulse begins rising at 2.0ns with 0.1ns rise time -> 2.05ns
    t_wl_rise = 2.05e-9

    # Find the time when target wave crosses 0.5V after WL rise
    t_write_ps = float("nan")
    mask = t >= 2.0e-9
    t_sub = t[mask]
    wave_sub = target_wave[mask]

    if cross_up:
        idx = np.where(wave_sub >= 0.5)[0]
    else:
        idx = np.where(wave_sub <= 0.5)[0]

    if len(idx) > 0:
        first_idx = idx[0]
        # Linear interpolation around crossing
        if first_idx > 0:
            t0, t1 = t_sub[first_idx - 1], t_sub[first_idx]
            v0, v1 = wave_sub[first_idx - 1], wave_sub[first_idx]
            if v1 != v0:
                t_cross = t0 + (0.5 - v0) * (t1 - t0) / (v1 - v0)
            else:
                t_cross = t1
        else:
            t_cross = t_sub[0]
        t_write_ps = round((t_cross - t_wl_rise) * 1e12, 1)

    # Peak supply current during active write window (2ns - 7ns)
    i_peak_mA = 0.0
    for key in ("i(v4)", "i(vdd)"):
        if key in traces:
            w_mask = (t >= 2.0e-9) & (t <= 7.0e-9)
            i_peak_mA = round(float(np.max(np.abs(traces[key][w_mask]))) * 1e3, 3)
            break

    return {
        "v_q_final": round(vq_final, 4),
        "v_qb_final": round(vqb_final, 4),
        "write_pass": write_pass,
        "t_write_ps": t_write_ps,
        "i_peak_mA": i_peak_mA,
    }


def extract_read_features(
    raw_path: str,
    stored_state: int = 1,
    strobe_ns: float = 2.15,
) -> Dict[str, Any]:
    """
    Extract diagnostic features from a READ simulation (.raw).

    Parameters
    ----------
    raw_path : str
        Path to .raw simulation file.
    stored_state : int
        1 if cell stores Q=1, Qb=0; 0 if cell stores Q=0, Qb=1.
    strobe_ns : float
        Sense-amplifier strobe time in nanoseconds (default 2.15ns, ~150ps after WL rise).

    Features:
      - v_q_final, v_qb_final: voltages at t=20ns
      - read_stable_pass: True if read was non-destructive (retained stored state)
      - dv_bl_strobe_mV: |V(BL) - V(BLB)| in mV at strobe_ns
      - t_sense_ps: Delay from WL 50% rise until |V(BL) - V(BLB)| reaches 100mV
      - v_bump_mV: Peak voltage disturbance on the '0' storage node during access (2ns-7ns)
    """
    traces = _load_traces(raw_path)
    t = traces["time"]
    vq = traces["v(q)"]
    vqb = traces["v(qb)"]
    vbl = traces["v(bl)"]
    vblb = traces["v(blb)"]

    vq_final = _interp(t, vq, 20e-9)
    vqb_final = _interp(t, vqb, 20e-9)

    if stored_state == 1:
        read_stable = bool(vq_final >= 0.9 and vqb_final <= 0.1)
        bump_node = vqb
    else:
        read_stable = bool(vq_final <= 0.1 and vqb_final >= 0.9)
        bump_node = vq

    # Differential bitline voltage at strobe time
    vbl_s = _interp(t, vbl, strobe_ns * 1e-9)
    vblb_s = _interp(t, vblb, strobe_ns * 1e-9)
    dv_bl_strobe_mV = round(abs(vbl_s - vblb_s) * 1e3, 1)

    # Time to develop 100mV sense differential
    t_wl_rise = 2.05e-9
    diff_wave = np.abs(vbl - vblb)
    mask = t >= 2.0e-9
    t_sub = t[mask]
    diff_sub = diff_wave[mask]
    idx = np.where(diff_sub >= 0.100)[0]

    t_sense_ps = float("nan")
    if len(idx) > 0:
        first_idx = idx[0]
        if first_idx > 0:
            t0, t1 = t_sub[first_idx - 1], t_sub[first_idx]
            d0, d1 = diff_sub[first_idx - 1], diff_sub[first_idx]
            if d1 != d0:
                t_cross = t0 + (0.100 - d0) * (t1 - t0) / (d1 - d0)
            else:
                t_cross = t1
        else:
            t_cross = t_sub[0]
        t_sense_ps = round((t_cross - t_wl_rise) * 1e12, 1)

    # Peak read disturbance bump on the low storage node during WL active
    w_mask = (t >= 2.0e-9) & (t <= 7.0e-9)
    v_bump_mV = round(float(np.max(bump_node[w_mask])) * 1e3, 1)

    return {
        "v_q_final": round(vq_final, 4),
        "v_qb_final": round(vqb_final, 4),
        "read_stable_pass": read_stable,
        "dv_bl_strobe_mV": dv_bl_strobe_mV,
        "t_sense_ps": t_sense_ps,
        "v_bump_mV": v_bump_mV,
    }
