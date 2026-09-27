# 6T SRAM Fault Injection, Simulation & Diagnostic Pipeline — Complete Project Documentation

**Project:** BTP — Automated SRAM Fault Injection & Diagnosis  
**Technology Node:** 180nm CMOS ($V_{DD} = 1.0\,\text{V}$)  
**Simulator Engine:** LTspice (via PyLTSpice / spicelib batch execution)  
**Status:** All Core Tasks (1–7), Issues 1–6, and Dynamic Feature Extraction Fully Implemented and Verified  
**Test Suite:** 24/24 Tests Passing (`pytest`)

---

## 1. Executive Summary & Architecture

This repository contains an end-to-end automated pipeline for injecting physical and parametric manufacturing defects into a 6T SRAM cell, composing runnable SPICE testbenches for multiple memory operations, executing batch simulations in parallel with convergence monitoring, and extracting rich dynamic/static diagnostic feature vectors for downstream Machine Learning fault classification.

### Core Architectural Principle: Decoupled Composition
A central design decision of this pipeline is the **strict separation between the Cell Core and the Test Stimulus**:
- **Cell Core Fragment** (`circuits/core/`): Contains exclusively the 6 transistors ($M_1$ through $M_6$), storage node capacitances, and device model cards. Fault injection modifies **only** this template fragment.
- **Stimulus Fragments** (`circuits/testbenches/`): Agnostic to whether a cell is healthy or faulty. Provides power supplies, wordline pulses, bitline precharge capacitors or drivers, initial conditions (`.ic`), and transient analysis commands (`.tran`).
- **Composer Engine** (`src/testbench_composer.py`): Combines *any* cell core with *any* stimulus on demand into a self-describing, runnable SPICE netlist.

```
       +----------------------------+
       | cell_core_healthy.net      |
       +----------------------------+
                     |
         [ src/fault_injection.py ]
                     |
       +----------------------------+       +------------------------------------+
       |   Faulty Cell Core         |  +    |  Stimulus Fragment                 |
       | (Open / Bridge / Vth Drift)|       | (HOLD / WRITE 1/0 / READ 1/0)      |
       +----------------------------+       +------------------------------------+
                     \                                 /
                      \                               /
                       [ src/testbench_composer.py ]
                                      |
                     +----------------------------------+
                     | Composed Runnable Testbench (.net)|
                     +----------------------------------+
                                      |
                     [ src/simulation_runner.py ]
                        (Multi-Threaded LTspice)
                                      |
                     +----------------------------------+
                     | Raw Waveforms (.raw) & Logs (.log)|
                     +----------------------------------+
                                      |
                     [ src/feature_extractor.py ]
                                      |
                     +----------------------------------+
                     | Machine Learning Feature Matrix  |
                     |  (t_write, dV_BL, I_DDQ, V_bump) |
                     +----------------------------------+
```

---

## 2. Directory Structure

```
btp-sram-fault-diagnosis/
├── circuits/
│   ├── sram_6t_healthy.asc              # Reference validated schematic (read-only)
│   ├── core/
│   │   └── cell_core_healthy.net        # Validated M1-M6 core template + Cq/Cqb + models
│   └── testbenches/
│       ├── stimulus_hold.net            # HOLD stimulus (retention & IDDQ test)
│       ├── stimulus_write.net           # WRITE 1 stimulus (BL=1V, BLB=0V; exercises M6)
│       ├── stimulus_write_0.net         # WRITE 0 stimulus (BL=0V, BLB=1V; exercises M5)
│       ├── stimulus_read.net            # READ 1 stimulus (precharged BL/BLB; exercises M6)
│       └── stimulus_read_0.net          # READ 0 stimulus (precharged BL/BLB; exercises M5)
├── src/
│   ├── utils.py                         # File I/O helpers (load_cell_core, save_cell_core)
│   ├── fault_injection.py               # Fault injection algorithms (Open, Bridge, Vth Drift)
│   ├── testbench_composer.py            # Netlist assembly, title generation, node sanity check
│   ├── simulation_runner.py             # Path auto-detection, parallel batch execution, log validation
│   └── feature_extractor.py             # Sub-nanosecond timing, differential voltage & IDDQ extraction
├── tests/
│   ├── test_fault_injection.py          # Unit tests for fault injection topologies & minimal diffs
│   ├── test_testbench_composer.py       # Unit tests for composition & dangling node checks
│   ├── test_simulation_runner.py        # Unit tests for determinism, parallel batch & pruning
│   └── test_feature_extractor.py        # Unit tests for feature extraction from simulation waveforms
├── scripts/
│   ├── run_pilot_batch.py               # Task 7 HOLD pilot (5 cell cores)
│   ├── run_pilot_extended.py            # WRITE & READ pilot with trajectory sampling
│   └── demo_feature_extraction.py       # 25-sim symmetric benchmark demonstrating 100% fault separation
├── data/
│   ├── pilot_batch/                     # Task 7 pilot netlists and convergence logs
│   └── demo_features/                   # 25-run symmetric diagnostic demonstration outputs
├── config.example.yaml                  # Template for local LTspice path configuration
├── config.yaml                          # Machine-specific configuration (auto-populated)
├── pytest.ini                           # Test suite configuration
└── README.md                            # Project overview & quickstart
```

---

## 3. Circuit Implementation & Node Definitions

### 3.1 6T SRAM Cell Topology ([`circuits/core/cell_core_healthy.net`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/circuits/core/cell_core_healthy.net))

The cell consists of two cross-coupled CMOS inverters ($M_1/M_3$ and $M_2/M_4$) and two NMOS pass-gates ($M_5$ and $M_6$).

```spice
M1 Qb Q Vdd Vdd SRAM_PMOS l=180nm w=360nm
M2 Q Qb Vdd Vdd SRAM_PMOS l=180nm w=360nm
M3 Q Qb 0 0 SRAM_NMOS l=180nm w=720nm
M4 Qb Q 0 0 SRAM_NMOS l=180nm w=720nm
M5 Q WL BL 0 SRAM_NMOS l=180nm w=540nm
M6 BLB WL Qb 0 SRAM_NMOS l=180nm w=540nm
Cq  Qb 0 1fF
Cqb Q  0 1fF
.model NMOS NMOS
.model PMOS PMOS
.model SRAM_NMOS NMOS (LEVEL=1 VTO=0.45 KP=250u GAMMA=0.4 LAMBDA=0.08 PHI=0.7 TOX=4n)
.model SRAM_PMOS PMOS (LEVEL=1 VTO=-0.45 KP=100u GAMMA=0.4 LAMBDA=0.1 PHI=0.7 TOX=4n)
```

- **Sizing Ratios:**
  - Pull-down NMOS ($M_3, M_4$): $W = 720\,\text{nm}$
  - Access NMOS ($M_5, M_6$): $W = 540\,\text{nm}$ ($\beta$-ratio $= 720/540 = 1.33$ ensures read stability)
  - Pull-up PMOS ($M_1, M_2$): $W = 360\,\text{nm}$ ($\gamma$-ratio $= 540/360 = 1.50$ ensures writability)
  - Gate Length ($L$): $180\,\text{nm}$ across all devices
- **Storage Node Capacitors:**
  - `Cq  Qb 0 1fF` and `Cqb Q  0 1fF` represent intrinsic diffusion and wiring parasitics on the storage nodes, validated against ground-truth transient simulations. (Preserved with verbatim naming as confirmed).

### 3.2 Stimulus Fragments

1. **HOLD (`stimulus_hold.net`):**
   - $V_{DD} = 1.0\,\text{V}$, $WL = 0\,\text{V}$, $BL = BLB = 1.0\,\text{V}$.
   - Initial conditions: $V(Q) = 1.0\,\text{V}$, $V(Qb) = 0.0\,\text{V}$.
   - Evaluates static data retention and quiescent supply leakage current ($I_{DDQ}$).
2. **WRITE 1 (`stimulus_write.net`):**
   - Initial state: $V(Q) = 0\,\text{V}, V(Qb) = 1\,\text{V}$.
   - Inputs: $BL = 1\,\text{V}, BLB = 0\,\text{V}$. $WL$ pulses high ($0 \to 1\,\text{V}$) from $t = 2.0\,\text{ns}$ to $7.0\,\text{ns}$ ($t_r = t_f = 0.1\,\text{ns}$).
   - **Active switching path:** NMOS $M_6$ pulls down $Qb$, flipping the latch to $Q=1$.
3. **WRITE 0 (`stimulus_write_0.net`):**
   - Initial state: $V(Q) = 1\,\text{V}, V(Qb) = 0\,\text{V}$.
   - Inputs: $BL = 0\,\text{V}, BLB = 1\,\text{V}$. $WL$ pulses high.
   - **Active switching path:** NMOS $M_5$ pulls down $Q$, flipping the latch to $Q=0$.
4. **READ 1 (`stimulus_read.net`):**
   - Initial state: $V(Q) = 1\,\text{V}, V(Qb) = 0\,\text{V}$.
   - Precharged bitline capacitors: $C_{BL} = C_{BLB} = 20\,\text{fF}$ initialized to $1.0\,\text{V}$.
   - $WL$ pulses high. $BLB$ discharges through $M_6$ and $M_4$ to ground.
5. **READ 0 (`stimulus_read_0.net`):**
   - Initial state: $V(Q) = 0\,\text{V}, V(Qb) = 1\,\text{V}$.
   - Precharged bitline capacitors: $C_{BL} = C_{BLB} = 20\,\text{fF}$ initialized to $1.0\,\text{V}$.
   - $WL$ pulses high. $BL$ discharges through $M_5$ and $M_3$ to ground.

---

## 4. Source Code Modules

### 4.1 Utilities ([`src/utils.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/utils.py))
- `load_cell_core(path: str) -> List[str]`: Reads a netlist fragment, strips trailing newlines and empty lines, and returns a clean list of line strings.
- `save_cell_core(lines: List[str], path: str) -> None`: Writes line strings out to file with standard Unix/LF newlines, ensuring parent directories exist.

### 4.2 Fault Injection Engine ([`src/fault_injection.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/fault_injection.py))
Implements precise, minimal text surgery on cell core templates:
- **`inject_resistive_open(lines, transistor_name, r_fault_ohm, output_path)`:**
  Inserts an open defect in series with the transistor's source terminal:
  - Renames the source node on the transistor line (e.g. `BL` $\to$ `BL_r` on $M_5$).
  - Inserts `Rfault BL BL_r <value>` immediately following the modified device.
  - Leaves drain, gate, and bulk nodes strictly untouched.
- **`inject_bridging_fault(lines, r_bridge_ohm, output_path)`:**
  Appends `Rbridge Q Qb <value>` between storage nodes $Q$ and $Qb$.
- **`inject_vth_drift(lines, target, pct_drift, output_path)`:**
  - For `storage_pair`: Modifies $V_{TO}$ in-place on both `.model SRAM_NMOS` and `.model SRAM_PMOS` ($V_{TO} \to V_{TO} \times (1 + \text{pct}/100)$).
  - For `access`: Clones `SRAM_NMOS` into `SRAM_NMOS_ACCESS_DRIFT` with shifted $V_{TO}$, appends the new model card, and re-points $M_5$ and $M_6$ to use it while leaving $M_3$ and $M_4$ on nominal parameters.

### 4.3 Testbench Composer ([`src/testbench_composer.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/testbench_composer.py))
- Combines core lines and stimulus lines into a production SPICE deck.
- Prepends a SPICE title comment on line 1 (`* SRAM 6T cell testbench: <core> + <stimulus>`), required by LTspice / spicelib.
- Appends `.end` as the final line.
- **Pre-execution Sanity Check (`_sanity_check`)**: Parses all device lines and validates that no dangling nodes exist, every stimulus node is accounted for in the core, and critical storage nodes $Q$ and $Qb$ are exposed.
- Supported operations: `("hold", "write", "read", "write_0", "read_0")`.

### 4.4 Simulation Runner Engine ([`src/simulation_runner.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/simulation_runner.py))
- **Auto-Detection Engine (`_resolve_ltspice_exe`)**:
  1. Checks `config.yaml` for `ltspice_path`.
  2. Probes `spicelib.simulators.ltspice_simulator.LTspice.is_available()`.
  3. Checks system `PATH` via `shutil.which()`.
  4. Scans all system drive roots (`C:`, `D:`, etc.) for standard install directories.
  5. Automatically writes back resolved path to `config.yaml`.
- **Deterministic Hashing (`_stable_stem`)**: Filenames encode an MD5 hash of netlist content to prevent race conditions or collisions during parallel execution.
- **Parallel Batch Execution (`run_batch`)**:
  - Leverages Python's `concurrent.futures.ThreadPoolExecutor` with configurable `max_workers` (defaults to `min(8, os.cpu_count())`).
  - Releases the GIL during LTspice subprocess execution, achieving linear multi-core speedup.
- **Robust Convergence Checking**:
  - Parses the generated `.log` file for fatal simulation markers (`"time step too small"`, `"singular matrix"`, `"gmin stepping failed"`, `"no convergence"`).
  - Appends failing runs to `logs/convergence_flags.csv`.
- **Waveform Cleanup (`prune_raw_files`)**: Automatically deletes bulky `.raw` files after feature extraction to conserve disk space.

### 4.5 Feature Extractor Module ([`src/feature_extractor.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/feature_extractor.py))
Translates raw analog simulation waveforms into high-dimensional tabular ML features:
- **`extract_hold_features(raw_path)`**:
  - $V_{final}(Q), V_{final}(Qb)$ (retention check)
  - $I_{DDQ}$ (quiescent supply leakage current $|I(V_{DD})|$ at $t=20\,\text{ns}$)
- **`extract_write_features(raw_path, target_state)`**:
  - $V_{final}(Q), V_{final}(Qb)$ (flip check)
  - $t_{write}$: Write delay in picoseconds (time elapsed from WL 50% rise at $2.05\,\text{ns}$ to target storage node crossing $V_{DD}/2 = 0.5\,\text{V}$)
  - $I_{peak}$: Peak dynamic supply current during write window ($2.0\,\text{ns}$ to $7.0\,\text{ns}$)
- **`extract_read_features(raw_path, stored_state, strobe_ns=2.15)`**:
  - $\Delta V_{BL}$: Bitline differential $|V(BL) - V(BLB)|$ at sense-amp strobe instant ($t=2.15\,\text{ns}$)
  - $t_{sense}$: Delay in picoseconds from WL rise until $\Delta V_{BL} \ge 100\,\text{mV}$
  - $V_{bump}$: Peak read disturbance voltage on the low storage node during access

---

## 5. Summary of Verified Experimental Results

The table below demonstrates the diagnostic differentiation achieved across all 5 fault classes evaluated across 25 simulations:

| Fault Class | HOLD $I_{DDQ}$ | WRITE 1 Delay ($t_{w1}$) | WRITE 0 Delay ($t_{w0}$) | Delay Asymmetry ($|t_{w1} - t_{w0}|$) | READ 1 $\Delta V_{BL}$ ($2.15\,\text{ns}$) | READ Sense Delay ($t_{sense}$) | Physical Signature & Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Healthy Baseline** | $0.004\,\mu\text{A}$ | $75.1\,\text{ps}$ | $45.4\,\text{ps}$ | $29.7\,\text{ps}$ | $224.8\,\text{mV}$ | $60.0\,\text{ps}$ | Nominal baseline reference |
| **Bridging ($2000\,\Omega$)** | **$4.338\,\mu\text{A}$** | **FAIL** | **FAIL** | N/A | **$0.0\,\text{mV}$** | **FAIL** | **$>1000\times$ $I_{DDQ}$ leakage jump, static rail collapse ($0.48\,\text{V}$)** |
| **Resistive Open $M_5$ ($1000\,\Omega$)** | $0.004\,\mu\text{A}$ | $75.1\,\text{ps}$ | **$53.1\,\text{ps}$** | **$22.0\,\text{ps}$** | $224.8\,\text{mV}$ | $60.0\,\text{ps}$ | **Asymmetric $+17\%$ delay penalty strictly on $M_5$ pull-down (W0)** |
| **$V_{th}$ Drift Storage Pair ($+10\%$)** | $0.004\,\mu\text{A}$ | $81.9\,\text{ps}$ | $49.1\,\text{ps}$ | $32.8\,\text{ps}$ | **$185.3\,\text{mV}$** | **$67.6\,\text{ps}$** | **Symmetric write delay penalty & $-39.5\,\text{mV}$ read differential drop** |
| **$V_{th}$ Drift Access ($+10\%$)** | $0.004\,\mu\text{A}$ | $82.2\,\text{ps}$ | $50.2\,\text{ps}$ | $32.0\,\text{ps}$ | **$192.4\,\text{mV}$** | **$66.3\,\text{ps}$** | **Read current starvation & slower bitline discharge ($+6.3\,\text{ps}$ sense delay)** |

### Key Physical Insights Validated
1. **Write Asymmetry:** SRAM writing is an NMOS pull-down operation. Writing '1' pulls down $Qb$ through $M_6$, leaving $M_5$ dormant. Writing '0' pulls down $Q$ through $M_5$. Single-ended tests hide half of all access faults; symmetric testing exposes them completely.
2. **Dynamic Margins vs. DC State:** At moderate severities ($1\,\text{k}\Omega, 10\%$), SRAM cells still complete their flip if given an unconstrained $5\,\text{ns}$ pulse. Dynamic delay ($t_{write}$), bitline discharge rates ($\Delta V_{BL}$), and quiescent leakage ($I_{DDQ}$) provide 100% classification separability without needing artificial clock scaling.

---

## 6. Automated Test Suite & Coverage

The project includes 24 automated unit tests running under `pytest`:

```bash
python -m pytest tests/ -v
```

### Test Breakdown
- **`tests/test_fault_injection.py` (14 tests):**
  - Verifies gate $\neq$ drain rules across all injected variants.
  - Verifies node mapping preservation (drain/gate/bulk nodes unchanged).
  - Verifies minimal diff properties (only intended lines modified/inserted).
  - Uses semantic device matching (`.model SRAM_`, `M5`, `M6`) to guarantee resilience against circuit changes.
- **`tests/test_testbench_composer.py` (4 tests):**
  - Verifies composition with HOLD, WRITE, and READ stimuli.
  - Verifies rejection of invalid stimulus types.
  - Verifies rejection of corrupted/dangling nodes.
  - Verifies correct placement of SPICE title comment and `.end`.
- **`tests/test_simulation_runner.py` (3 tests):**
  - Verifies deterministic stem generation (`_stable_stem`).
  - Verifies raw waveform file pruning (`prune_raw_files`).
  - Verifies multi-threaded parallel batch execution (`run_batch` with `max_workers=2`).
- **`tests/test_feature_extractor.py` (3 tests):**
  - Verifies feature extraction from real simulation `.raw` waveforms for HOLD, WRITE, and READ.

---

## 7. Operational Scripts & Usage Guide

### 7.1 Running the Demonstration
To run the 25-simulation parallel diagnostic demo and view the complete feature table:
```bash
python scripts/demo_feature_extraction.py
```
*Executes all 25 simulations in ~6.7 seconds on 6 workers and prints the full diagnostic matrix.*

### 7.2 Running Pilot Batches
- **HOLD Pilot (Task 7):**
  ```bash
  python scripts/run_pilot_batch.py
  ```
- **Extended Pilot (Task 7 Extension):**
  ```bash
  python scripts/run_pilot_extended.py
  ```

---

## 8. Scalability & Path to Full PVT Dataset Generation

With the completion of Issue 6 and the feature extraction engine, the pipeline is fully equipped for large-scale dataset generation:
1. **Parameter Sweeps:**
   - Resistive Opens: $100\,\Omega, 500\,\Omega, 1000\,\Omega, 2500\,\Omega, 5000\,\Omega, 10000\,\Omega$ across $M_1 - M_6$.
   - Bridging Faults: $500\,\Omega, 1000\,\Omega, 2000\,\Omega, 5000\,\Omega, 10000\,\Omega$ across $Q-Qb$, $BL-BLB$, $Q-V_{DD}$, etc.
   - $V_{th}$ Drift: $\pm 5\%, \pm 10\%, \pm 20\%, \pm 30\%$ across storage and access pairs.
   - PVT Corners: SS / TT / FF corners, $V_{DD} \in [0.9\,\text{V}, 1.0\,\text{V}, 1.1\,\text{V}]$, Temp $\in [-40^\circ\text{C}, 27^\circ\text{C}, 125^\circ\text{C}]$.
2. **Throughput:**
   - With 8–12 worker threads, 1,000 simulations complete in $\approx 4$ minutes.
   - On-the-fly extraction appends features directly to a compact CSV/Parquet dataset and removes raw files, preventing gigabyte-scale disk bloat.
