"""
src/simulation_runner.py — Batch LTspice simulation runner with convergence checking.

LTspice path resolution order (see _resolve_ltspice_exe for full details):
  1. config.yaml: ltspice_path  — machine-specific, never committed to git
  2. PyLTSpice/spicelib built-in LTspice.is_available() (after trying create_from)
  3. System PATH  (shutil.which)
  4. Drive-root scan across all available drive letters (C:, D:, etc.)
  5. On success, path is written back into config.yaml for future runs.
  6. On failure, raises a clear EnvironmentError with instructions.

Convergence checking: always parses the .log file — never trusts .raw existence alone.
Determinism: output filenames are derived from a stable MD5 of the input path,
             not from random state, so reruns are reproducible.
"""

import hashlib
import os
import shutil
import string
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Project root — one level above src/
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

# ---------------------------------------------------------------------------
# Convergence-failure substrings to scan for in .log files (case-insensitive)
# ---------------------------------------------------------------------------

_CONVERGENCE_FAILURE_MARKERS = [
    "time step too small",
    "singular matrix",
    "gmin stepping failed",
    "no convergence",
]

# ---------------------------------------------------------------------------
# config.yaml helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """
    Load config.yaml from the project root.  Returns an empty dict if the
    file does not exist or is empty.
    """
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _save_config(data: dict) -> None:
    """
    Write *data* back to config.yaml, preserving the file's header comment
    if it already exists (we only append/update, never clobber comments).
    We do a full rewrite here since PyYAML doesn't support comment-preserving
    round-trips — the header comment is re-emitted from a fixed template.
    """
    header = (
        "# config.yaml — machine-specific settings for btp-sram-fault-diagnosis\n"
        "# This file is listed in .gitignore.  Do not commit it.\n"
        "# See config.example.yaml for the template.\n"
        "#\n"
        "# ltspice_path: path to the LTspice executable on THIS machine.\n"
        "#   - Set it manually if auto-detection fails.\n"
        "#   - It is written automatically on the first successful auto-detect.\n"
        "#   - Example (LTspice 24 on D: drive): D:\\LTspice\\LTspice.exe\n"
        "#   - Example (LTspice XVII on C:):      C:\\Program Files\\LTC\\LTspiceXVII\\XVIIx64.exe\n"
        "#\n"
    )
    with open(_CONFIG_PATH, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(data, fh, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# LTspice path resolution — priority-ordered
# ---------------------------------------------------------------------------

def _candidate_paths_for_drive(drive: str) -> List[Path]:
    """
    Return a list of candidate LTspice executable paths for a given drive root
    (e.g. "C:" or "D:").  Covers both LTspice 24 and XVII naming conventions,
    plus bare root-level installs (common on secondary drives).
    """
    d = drive.rstrip("\\/")  # normalise to bare "C:" or "D:"
    root = Path(d + "\\")
    return [
        root / "Program Files" / "ADI" / "LTspice" / "LTspice.exe",
        root / "Program Files" / "LTC" / "LTspiceXVII" / "XVIIx64.exe",
        root / "Program Files (x86)" / "LTC" / "LTspiceXVII" / "XVIIx64.exe",
        root / "LTspice" / "LTspice.exe",           # bare root install (D:\LTspice\...)
        root / "LTspice" / "XVIIx64.exe",
        root / "ADI" / "LTspice" / "LTspice.exe",
    ]


def _resolve_ltspice_exe() -> str:
    """
    Resolve the LTspice executable path using the following priority order:

    1. config.yaml: ltspice_path  — used directly if set and the file exists.
    2. spicelib LTspice.is_available() — the library-maintained detection path.
    3. System PATH via shutil.which.
    4. Drive-root scan across all available drive letters + %LOCALAPPDATA%.
    5. On auto-detect success: prints the found path and writes it to config.yaml.
    6. On failure: raises EnvironmentError with a clear manual-setup message.

    Returns
    -------
    str
        Absolute path to the LTspice executable.
    """
    # ---- 1. config.yaml -------------------------------------------------------
    config = _load_config()
    configured = config.get("ltspice_path", None)
    if configured and configured != "null" and Path(configured).exists():
        return str(configured)

    # ---- 2. spicelib / PyLTSpice built-in detection ---------------------------
    try:
        from spicelib.simulators.ltspice_simulator import LTspice as LTspiceSim
        if LTspiceSim.is_available():
            exe = LTspiceSim.spice_exe[0]
            _record_detected_path(exe)
            return exe
    except Exception:
        pass  # library not structured as expected; fall through

    # ---- 3. System PATH -------------------------------------------------------
    for name in ("LTspice.exe", "XVIIx64.exe", "LTspice", "XVIIx64"):
        found = shutil.which(name)
        if found and Path(found).exists():
            _record_detected_path(found)
            return found

    # ---- 4. Drive-root scan (all available drive letters + AppData) -----------
    available_drives = [
        f"{letter}:" for letter in string.ascii_uppercase
        if Path(f"{letter}:\\").exists()
    ]

    # Also check %LOCALAPPDATA% and %APPDATA% (per-user installs)
    appdata_candidates = []
    for env_var in ("LOCALAPPDATA", "APPDATA"):
        base = os.environ.get(env_var, "")
        if base:
            appdata_candidates += [
                Path(base) / "Programs" / "ADI" / "LTspice" / "LTspice.exe",
                Path(base) / "ADI" / "LTspice" / "LTspice.exe",
            ]

    for drive in available_drives:
        for candidate in _candidate_paths_for_drive(drive):
            if candidate.exists():
                _record_detected_path(str(candidate))
                return str(candidate)

    for candidate in appdata_candidates:
        if candidate.exists():
            _record_detected_path(str(candidate))
            return str(candidate)

    # ---- 5. Failure -----------------------------------------------------------
    raise EnvironmentError(
        "LTspice executable not found.\n"
        "Auto-detection searched:\n"
        f"  - Drives: {available_drives}\n"
        "  - Paths: Program Files/ADI/LTspice, Program Files/LTC/LTspiceXVII,\n"
        "           root-level LTspice/ directories, %LOCALAPPDATA%/Programs/ADI/LTspice\n"
        "  - System PATH\n\n"
        "Add the path manually to config.yaml under the key 'ltspice_path', e.g.:\n"
        "  ltspice_path: 'D:\\\\LTspice\\\\LTspice.exe'\n"
        "Then re-run — no Python files need to be edited."
    )


def _record_detected_path(exe_path: str) -> None:
    """
    Print the auto-detected LTspice path and persist it in config.yaml so
    future runs on this machine skip the detection scan entirely.
    """
    print(f"[auto-detect] LTspice found at: {exe_path}")
    print(f"[auto-detect] Writing to config.yaml for future runs.")
    config = _load_config()
    config["ltspice_path"] = str(exe_path)
    _save_config(config)


# ---------------------------------------------------------------------------
# Stable filename stem
# ---------------------------------------------------------------------------

def _stable_stem(net_path: str) -> str:
    """
    Return a deterministic short identifier for a .net file based on its
    absolute path, so output filenames are reproducible across reruns.
    """
    abs_path = str(Path(net_path).resolve())
    h = hashlib.md5(abs_path.encode()).hexdigest()[:8]
    stem = Path(net_path).stem
    return f"{stem}_{h}"


# ---------------------------------------------------------------------------
# LTspice version extraction from log
# ---------------------------------------------------------------------------

def _extract_ltspice_version(log_text: str) -> Optional[str]:
    """
    Extract the LTspice version string from the first line of a .log file.

    LTspice typically writes something like:
      "LTspice 24.1.9 for Windows"  or  "LTspice XVII"
    on the first non-empty line of the log.

    Returns the version string if found, else None.
    """
    for line in log_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("ltspice"):
            return stripped
    return None


# ---------------------------------------------------------------------------
# Core: run a single simulation
# ---------------------------------------------------------------------------

def run_simulation(
    net_path: str,
    raw_output_dir: str,
    log_output_dir: str,
) -> dict:
    """
    Run LTspice on *net_path* via PyLTSpice and return a result dict.

    Parameters
    ----------
    net_path : str
        Path to a fully-composed, runnable SPICE .net file.
    raw_output_dir : str
        Directory where the .raw waveform file will be written.
    log_output_dir : str
        Directory where the .log file will be written.

    Returns
    -------
    dict with keys:
        net_path       : str        -- input path (echoed back)
        raw_path       : str        -- path to the .raw file
        log_path       : str        -- path to the .log file
        converged      : bool       -- True iff no convergence-failure markers in log
        warning        : str|None   -- matched failure marker string, or None
        ltspice_version: str|None   -- version string from first line of .log
    """
    from PyLTSpice import SimRunner
    from spicelib.simulators.ltspice_simulator import LTspice as LTspiceSim
    from spicelib.editor.spice_editor import SpiceEditor

    raw_output_dir = Path(raw_output_dir)
    log_output_dir = Path(log_output_dir)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    log_output_dir.mkdir(parents=True, exist_ok=True)

    stem = _stable_stem(net_path)

    # Resolve LTspice path (config.yaml -> auto-detect -> error)
    ltspice_exe = _resolve_ltspice_exe()

    # Build a simulator class from the resolved executable path
    simulator_cls = LTspiceSim.create_from(ltspice_exe)

    runner = SimRunner(
        output_folder=str(raw_output_dir),
        simulator=simulator_cls,
        verbose=False,
    )

    # run_now() blocks until completion and returns (raw_path, log_path) as
    # pathlib.Path objects (or None on crash).
    result_raw, result_log = runner.run_now(
        SpiceEditor(net_path),
        run_filename=f"{stem}.net",
    )

    # Normalise to strings; fall back to constructed paths if LTspice crashed
    raw_path = str(result_raw) if result_raw is not None else str(raw_output_dir / f"{stem}.raw")
    log_path = str(result_log) if result_log is not None else str(raw_output_dir / f"{stem}.log")

    # Copy .log to log_output_dir if it was written elsewhere
    result_log_path = Path(log_path)
    target_log_path = log_output_dir / result_log_path.name
    if result_log_path.exists() and result_log_path.parent != log_output_dir:
        shutil.copy2(str(result_log_path), str(target_log_path))
        log_path = str(target_log_path)

    # -----------------------------------------------------------------------
    # Read log text
    # -----------------------------------------------------------------------
    log_text = ""
    for candidate_log in (Path(log_path), result_log_path):
        if candidate_log.exists():
            log_text = candidate_log.read_text(encoding="utf-8", errors="replace")
            log_path = str(candidate_log)
            break

    # -----------------------------------------------------------------------
    # Parse convergence markers
    # -----------------------------------------------------------------------
    converged = True
    warning = None
    log_lower = log_text.lower()
    for marker in _CONVERGENCE_FAILURE_MARKERS:
        if marker.lower() in log_lower:
            converged = False
            warning = marker
            break

    # -----------------------------------------------------------------------
    # Extract LTspice version string from first log line
    # -----------------------------------------------------------------------
    ltspice_version = _extract_ltspice_version(log_text)

    return {
        "net_path": str(net_path),
        "raw_path": raw_path,
        "log_path": log_path,
        "converged": converged,
        "warning": warning,
        "ltspice_version": ltspice_version,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Batch runner & cleanup
# ---------------------------------------------------------------------------

def prune_raw_files(raw_paths: List[str]) -> int:
    """
    Delete large .raw waveform files to conserve disk space once features
    have been extracted. Returns count of files deleted.
    """
    deleted = 0
    for p in raw_paths:
        pp = Path(p)
        if pp.exists() and pp.suffix.lower() == ".raw":
            try:
                pp.unlink()
                deleted += 1
            except Exception:
                pass
    return deleted


def run_batch(
    net_paths: List[str],
    raw_output_dir: str,
    log_output_dir: str,
    flagged_csv_path: str,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Batch-run a list of composed .net files and return a results DataFrame.
    Supports multi-threaded parallel execution across CPU cores.

    Parameters
    ----------
    net_paths : list of str
        Paths to runnable SPICE .net files.
    raw_output_dir : str
        Directory for .raw outputs.
    log_output_dir : str
        Directory for .log outputs.
    flagged_csv_path : str
        CSV file to which non-converged rows are appended (append mode).
    max_workers : int, optional
        Maximum number of concurrent LTspice simulation threads.
        Defaults to min(8, os.cpu_count() or 4). Set to 1 for serial execution.

    Returns
    -------
    pandas.DataFrame
        Columns: [net_path, raw_path, log_path, converged, warning, ltspice_version]
    """
    from concurrent.futures import ThreadPoolExecutor

    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 4)

    results = []

    if max_workers <= 1 or len(net_paths) <= 1:
        # Serial execution
        for net_path in net_paths:
            print(f"  Running: {Path(net_path).name} ... ", end="", flush=True)
            result = run_simulation(net_path, raw_output_dir, log_output_dir)
            status = "OK" if result["converged"] else f"FAILED ({result['warning']})"
            print(status)
            results.append(result)
    else:
        # Multi-threaded parallel execution
        print(f"  Launching {len(net_paths)} simulations across {max_workers} worker threads...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_net = [
                executor.submit(run_simulation, net_path, raw_output_dir, log_output_dir)
                for net_path in net_paths
            ]
            for future in future_to_net:
                result = future.result()
                status = "OK" if result["converged"] else f"FAILED ({result['warning']})"
                print(f"  Completed: {Path(result['net_path']).name} ... {status}")
                results.append(result)

    df = pd.DataFrame(
        results,
        columns=["net_path", "raw_path", "log_path", "converged", "warning", "ltspice_version"],
    )

    # Append non-converged rows to the flags CSV (append mode, header only on creation)
    flagged = df[~df["converged"]]
    if not flagged.empty:
        flagged_path = Path(flagged_csv_path)
        flagged_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not flagged_path.exists()
        flagged.to_csv(str(flagged_path), mode="a", header=write_header, index=False)

    total = len(df)
    n_converged = int(df["converged"].sum())
    n_flagged = total - n_converged
    print(
        f"\nBatch complete: {total} runs, "
        f"{n_converged} converged, "
        f"{n_flagged} flagged (non-converged)."
    )

    return df
