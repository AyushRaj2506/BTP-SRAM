# 6T SRAM Fault Injection, Simulation & Diagnostic Pipeline — Complete Project Documentation

**Project:** BTP — Automated SRAM Fault Injection & Diagnosis  
**Technology Node:** 180nm CMOS ($V_{DD} = 1.0\,\text{V}$ nominal, $0.9\,\text{V}$–$1.1\,\text{V}$ PVT sweep)  
**Simulator Engine:** LTspice (via PyLTSpice / direct headless subprocess execution)  
**Status:** All Stages (1 through 7) Fully Implemented, Verified, and Tested  
**Test Suite:** 62/62 Tests Passing (`pytest`)  
**Dataset:** 450 samples across 15 PVT corners, 33-column schema, 0 missing/NaN values  

---

## 1. Executive Summary & Architecture

This repository contains an end-to-end automated framework for:
1. Modeling physical and parametric defects in a standard 6T SRAM cell (Resistive Opens, Bridging Faults, and $V_{th}$ Drifts).
2. Composing runnable SPICE testbenches for multiple memory operations (Hold, Write 1/0, Read 1/0, and DC Butterfly-Curve Hold/Read Static Noise Margins).
3. Simulating across a 15-corner Process-Voltage-Temperature (PVT) grid ($V_{DD} \in [0.9, 1.0, 1.1]\,\text{V}$, $T \in [-40, 0, 27, 75, 125]^\circ\text{C}$).
4. Extracting raw dynamic/static features and computing physics-normalized invariant features relative to a same-PVT reference cell.
5. Providing both non-ML baseline rule detectors and classical machine learning classifiers (Random Forest, Logistic Regression, Linear SVM, HistGradientBoosting).
6. Evaluating cross-PVT generalization via Standard Split and Leave-One-Corner-Out (LOCO) protocols.

```
       +----------------------------+
       | cell_core_healthy.net      |
       +----------------------------+
                     |
         [ src/fault_injection.py ]
                     |
       +----------------------------+       +------------------------------------+
       |   Faulty Cell Core         |  +    |  Stimulus Fragment                 |
       | (Open / Bridge / Vth Drift)|       | (HOLD / WRITE 1/0 / READ 1/0 / SNM)|
       +----------------------------+       +------------------------------------+
                     \                                 /
                      \                               /
                       [ src/testbench_composer.py ]
                             (PVT Parameterization)
                             (duplicate if SNM)
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
                     [ src/snm_extraction.py    ]
                                      |
                     +----------------------------------+
                     | Master Dataset (450 x 33 CSV)    |
                     |  Raw + Physics-Normalized Ratios |
                     +----------------------------------+
                                      |
                     +----------------+----------------+
                     |                                 |
           [ src/baseline_detector.py ]       [ src/ml_pipeline.py ]
           (3-Sigma Healthy Rule Checks)      (RF, LR, SVM, HistGB)
                     |                                 |
                     +----------------+----------------+
                                      |
                         [ src/cross_pvt_analysis.py ]
                         (LOCO Generalization & Figures)
```

---

## 2. Directory Structure

```
btp-sram-fault-diagnosis/
├── circuits/
│   ├── core/
│   │   └── cell_core_healthy.net        # Validated M1-M6 core template (symmetric access devices)
│   ├── reference/
│   │   ├── snm_read_healthy.net         # Golden read SNM reference deck
│   │   └── snm_hold_healthy.net         # Golden hold SNM reference deck
│   └── testbenches/
│       ├── stimulus_hold.net            # HOLD stimulus (retention & IDDQ test)
│       ├── stimulus_write.net           # WRITE 1 stimulus (BL=1V, BLB=0V; exercises M6)
│       ├── stimulus_write_0.net         # WRITE 0 stimulus (BL=0V, BLB=1V; exercises M5)
│       ├── stimulus_read.net            # READ 1 stimulus (precharged BL/BLB; exercises M6)
│       ├── stimulus_read_0.net          # READ 0 stimulus (precharged BL/BLB; exercises M5)
│       ├── stimulus_snm_hold.net        # Hold SNM stimulus (DC sweep V5, WL=0V, ideal BL/BLB)
│       └── stimulus_snm_read.net        # Read SNM stimulus (DC sweep V5, WL=1V, ideal BL/BLB)
├── src/
│   ├── utils.py                         # File I/O helpers (load_cell_core, save_cell_core)
│   ├── fault_injection.py               # Fault injection algorithms (Open, Bridge, Vth Drift)
│   ├── testbench_composer.py            # Netlist assembly, PVT parameterization, core duplication
│   ├── simulation_runner.py             # Multi-threaded direct batch runner, DC integrity validation
│   ├── snm_extraction.py                # Pure NumPy largest inscribed square search (Hold & Read)
│   ├── feature_extractor.py             # Dynamic timing, differential voltage, IDDQ & SNM extraction
│   ├── generate_dataset.py              # Full 15-corner dataset generation orchestrator
│   ├── baseline_detector.py             # Non-ML 3-sigma statistical baseline classifier
│   ├── ml_pipeline.py                   # Classical ML models, Standard Split, and LOCO protocols
│   └── cross_pvt_analysis.py            # Feature spread analysis and publication-ready figures
├── data/
│   ├── sram_fault_dataset.csv           # Master complete dataset (450 rows x 33 cols, 0 NaNs)
│   ├── sram_fault_dataset_indist.csv    # In-distribution nominal dataset (30 rows, 1.0V, 27°C)
│   ├── sram_fault_dataset_ood.csv       # Out-of-distribution dataset (420 rows, 14 held-out corners)
│   └── demo_features/                   # Transient and SNM benchmark demonstration outputs
├── logs/
│   └── convergence_flags.csv            # SPICE convergence & DC monotonicity audit log
├── reports/
│   ├── final_results/
│   │   ├── comparison_table.csv         # Full model evaluation metrics across protocols
│   │   ├── loco_summary.csv             # LOCO macro accuracy aggregation per model
│   │   └── feature_spread_comparison.csv# Coefficient of variation spread reduction table
│   └── figures/
│       ├── fig1_feature_dispersion_comparison.png  # Z-score boxplots across corners
│       ├── fig2_cross_pvt_loco_accuracy.png        # Bar chart comparing Raw vs Invariant
│       └── fig3_rf_pvt_corner_heatmap.png          # Per-corner Random Forest accuracy heatmap
├── scripts/
│   ├── demo_feature_extraction.py       # 25-sim symmetric dynamic benchmark
│   ├── demo_snm_features.py             # Butterfly-curve Hold & Read SNM demonstration
│   └── verify_dataset_physics.py        # Automated physical scaling and statistical audit
├── tests/
│   ├── test_fault_injection.py          # Unit tests for fault injection topologies & minimal diffs (18)
│   ├── test_testbench_composer.py       # Unit tests for transient composition & dangling nodes (4)
│   ├── test_pvt_composition.py          # Unit tests for PVT scaling & deck parameterization (3)
│   ├── test_simulation_runner.py        # Unit tests for runner, DC integrity checks & fixtures (6)
│   ├── test_feature_extractor.py        # Unit tests for transient feature extraction (4)
│   ├── test_snm_composition.py         # Unit tests for two-cell SNM core duplication (5)
│   ├── test_snm_extraction.py          # Pure NumPy & reference deck SNM unit tests (7)
│   ├── test_snm_features.py             # Full-pipeline SNM feature extraction & nonmonotonic tests (3)
│   ├── test_dataset_generator.py        # End-to-end dataset generation pipeline tests (8)
│   ├── test_baseline_detector.py        # Non-ML baseline unit tests (3)
│   └── test_ml_pipeline.py              # ML models, metric computation, and split tests (4)
├── validation/
│   └── snm_manual_check/
│       └── results.md                   # Golden validation measurements & GUI cross-checks
├── config.example.yaml                  # Template for local LTspice path configuration
├── pytest.ini                           # Test suite configuration
└── README.md                            # Quickstart & user documentation
```

---

## 3. Circuit Implementation & Node Definitions

### 3.1 6T SRAM Cell Topology ([`circuits/core/cell_core_healthy.net`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/circuits/core/cell_core_healthy.net))

The storage cell consists of two cross-coupled CMOS inverters ($M_1/M_3$ and $M_2/M_4$) and two NMOS access transistors ($M_5$ and $M_6$). Both access transistors share identical orientation: drain = internal storage node, source = bitline.

```spice
M1 Qb Q Vdd Vdd SRAM_PMOS l=180nm w=360nm
M2 Q Qb Vdd Vdd SRAM_PMOS l=180nm w=360nm
M3 Q Qb 0 0 SRAM_NMOS l=180nm w=720nm
M4 Qb Q 0 0 SRAM_NMOS l=180nm w=720nm
M5 Q WL BL 0 SRAM_NMOS l=180nm w=540nm
M6 Qb WL BLB 0 SRAM_NMOS l=180nm w=540nm
Cq  Qb 0 1fF
Cqb Q  0 1fF
.model SRAM_NMOS NMOS (LEVEL=1 VTO=0.45 KP=250u GAMMA=0.4 LAMBDA=0.08 PHI=0.7 TOX=4n)
.model SRAM_PMOS PMOS (LEVEL=1 VTO=-0.45 KP=100u GAMMA=0.4 LAMBDA=0.1 PHI=0.7 TOX=4n)
```

- **Transistor Sizing Ratios:**
  - Pull-down NMOS ($M_3, M_4$): $W = 720\,\text{nm}$
  - Access NMOS ($M_5, M_6$): $W = 540\,\text{nm}$ ($\text{Cell Ratio } \beta = 720/540 = 1.33$ ensures read stability)
  - Pull-up PMOS ($M_1, M_2$): $W = 360\,\text{nm}$ ($\text{Pull-Up Ratio } \gamma = 540/360 = 1.50$ ensures writability)
  - Gate Length ($L$): $180\,\text{nm}$ across all devices

### 3.2 Dynamic & Static Stimulus Fragments
- **HOLD (`stimulus_hold.net`):** $WL = 0\,\text{V}, BL = BLB = V_{DD}$. Retention check and standby current ($I_{\text{ddq}}$) measurement.
- **WRITE 1 (`stimulus_write.net`):** $BL = V_{DD}, BLB = 0\,\text{V}$. Exercises $M_6$ and $M_4$ pull-down path.
- **WRITE 0 (`stimulus_write_0.net`):** $BL = 0\,\text{V}, BLB = V_{DD}$. Exercises $M_5$ and $M_3$ pull-down path.
- **READ 1 (`stimulus_read.net`):** Precharged bitlines ($C_{\text{BL}} = 20\,\text{fF}$ charged to $V_{DD}$). $BLB$ discharges through $M_6/M_4$.
- **READ 0 (`stimulus_read_0.net`):** Precharged bitlines ($C_{\text{BL}} = 20\,\text{fF}$ charged to $V_{DD}$). $BL$ discharges through $M_5/M_3$.
- **SNM HOLD (`stimulus_snm_hold.net`):** Access devices OFF ($WL = 0\,\text{V}$). DC sweep $V_5$ from $0 \to V_{DD}$ in $1\,\text{mV}$ steps. Traces $V(Q), V(Qb), V(Q\_c2), V(Qb\_c2)$.
- **SNM READ (`stimulus_snm_read.net`):** Access devices ON ($WL = V_{DD}$, $BL = BLB = V_{DD}$). Same $1\,\text{mV}$ DC sweep under wordline activation.

---

## 4. Fault Taxonomy & Physics

The pipeline supports 5 classes:

| Class | Fault Type | Physical Mechanism | Target Devices | Severity Parameter |
|---|---|---|---|---|
| **0** | **Healthy** | Nominal defect-free cell | None | None |
| **1** | **Resistive Open** | Contact/via void or metal line electromigration | Access pass-gates $M_5$ or $M_6$ | Series resistance: $200\,\Omega$ to $10\,\text{k}\Omega$ |
| **2** | **Bridging Fault** | Inter-layer dielectric breakdown / metal sliver | Storage nodes $Q \leftrightarrow \bar{Q}$ | Shunt resistance: $500\,\Omega$ to $10\,\text{k}\Omega$ |
| **3** | **$V_{th}$ Drift (Storage)** | Bias Temperature Instability (BTI) / Hot Carrier Injection | Inverter pair $M_1$–$M_4$ only | $V_{th}$ shift: $-30\%$ to $+30\%$ |
| **4** | **$V_{th}$ Drift (Access)** | Asymmetrical trapping on pass-gate dielectrics | Access pair $M_5/M_6$ only | $V_{th}$ shift: $-30\%$ to $+30\%$ |

---

## 5. PVT Simulation Engine & Multi-Threading

### 5.1 PVT Sweep Grid (15 Corners)
- **Supply Voltage ($V_{DD}$):** $0.9\,\text{V}$ ($-10\%$), $1.0\,\text{V}$ (nominal), $1.1\,\text{V}$ ($+10\%$)
- **Temperature ($T$):** $-40^\circ\text{C}$ (industrial cold), $0^\circ\text{C}$, $27^\circ\text{C}$ (room nominal), $75^\circ\text{C}$, $125^\circ\text{C}$ (worst-case thermal)
- **In-Distribution Corner:** $V_{DD} = 1.0\,\text{V}, T = 27^\circ\text{C}$ (30 samples)
- **Out-of-Distribution (OOD):** Remaining 14 corners (420 samples)

### 5.2 Dynamic Testbench Composer with PVT Parameterization
[`src/testbench_composer.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/testbench_composer.py) scales:
1. All DC supply voltage sources (`Vdd`, `V1`, `V3`, `V4`) to target $V_{DD}$.
2. Pulse generator high levels (`PULSE(...)`) on wordline and bitlines.
3. Precharge initial condition `.ic` voltages ($V(BL) = V_{DD}, V(Q) = V_{DD}$).
4. SNM DC sweep lines (`.dc V5 0 <vdd> 1m`).
5. Appends `.temp <T>` directive for SPICE thermal equations.

### 5.3 Parallel Simulator & File Isolation
To avoid multi-threaded file renaming collisions inherent in PyLTSpice's shared runner queues on Windows, [`src/simulation_runner.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/simulation_runner.py) directly invokes headless LTspice processes (`LTspice.exe -b -ascii <deck>`). Each simulation runs in its own thread against an isolated unique filename. Over 3,126 `.raw` files are automatically deleted after feature extraction, maintaining low disk footprint.

---

## 6. Dataset Schema & Physics Normalization

### 6.1 Master Dataset Schema (33 Columns)
Each row in [`data/sram_fault_dataset.csv`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/data/sram_fault_dataset.csv) contains:
- **Metadata:** `sample_id`, `corner_id`, `pvt_type`, `vdd_v`, `temperature_c`
- **Labels:** `fault_class` ($0$ to $4$), `fault_name`, `fault_target`, `severity_value`
- **Raw Dynamic Features:** `i_ddq_uA`, `t_write_ps`, `i_write_peak_mA`, `dv_bl_strobe_mV`, `t_sense_ps`, `v_bump_mV`, `hold_pass`, `write_pass`, `read_stable_pass`
- **Raw Static & SNM Features:** `snm_hold_v`, `snm_read_v`, `snm_asym_hold_v`, `snm_asym_read_v`, `snm_drop_v`, `is_bistable`
- **Physics-Normalized Invariant Features:**
  - $\text{iddq\_norm} = I_{\text{ddq}} / I_{\text{ddq, ref}}$
  - $\text{t\_write\_norm} = t_{\text{write}} / t_{\text{write, ref}}$
  - $\text{dv\_bl\_norm} = \Delta V_{\text{BL}} / \Delta V_{\text{BL, ref}}$
  - $\text{snm\_hold\_norm} = \text{SNM}_{\text{hold}} / \text{SNM}_{\text{hold, ref}}$
  - $\text{snm\_read\_norm} = \text{SNM}_{\text{read}} / \text{SNM}_{\text{read, ref}}$
  - Dimensionless internal ratios: $\text{read\_write\_delay\_ratio}$, $\text{snm\_ratio}$, $\text{dv\_bl\_vdd\_ratio}$
- **Integrity:** `converged` (1 if valid, 0 if severe fault caused non-monotonic DC curve)

---

## 7. Experimental Verification & Physical Sanity

The generated dataset was subjected to automated semiconductor physics validation via [`scripts/verify_dataset_physics.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/scripts/verify_dataset_physics.py):

### 7.1 Semiconductor Thermal Scaling ($I_{\text{ddq}}$ vs $T$)
Subthreshold leakage scales exponentially with temperature ($I_{\text{sub}} \propto T^2 e^{-qV_{th}/kT}$):
- $-40^\circ\text{C}$: $0.0188\,\mu\text{A}$
- $0^\circ\text{C}$: $0.0044\,\mu\text{A}$
- $27^\circ\text{C}$ (Nominal): $0.0140\,\mu\text{A}$ ($14\,\text{nA}$)
- $75^\circ\text{C}$: $0.8388\,\mu\text{A}$
- $125^\circ\text{C}$: $1.6202\,\mu\text{A}$ ($>115\times$ increase over nominal)

### 7.2 Gate Overdrive Scaling ($t_{\text{write}}$ vs $V_{DD}$)
Drive current scales with overdrive ($V_{GS} - V_{th}$):
- $0.9\,\text{V}$: $70.58\,\text{ps}$ (slowest switching)
- $1.0\,\text{V}$: $62.18\,\text{ps}$ (nominal)
- $1.1\,\text{V}$: $52.14\,\text{ps}$ (fastest switching)

### 7.3 Fault Monotonicity
- **Class 1 (Resistive Open):** $t_{\text{write}}$ increases monotonically from $61.57\,\text{ps}$ ($200\,\Omega$) to $105.99\,\text{ps}$ ($10\,\text{k}\Omega$).
- **Class 2 (Bridging):** Leakage spikes inversely with resistance ($18.6\,\mu\text{A}$ at $500\,\Omega$ down to $3.8\,\mu\text{A}$ at $10\,\text{k}\Omega$); SNM and bistability collapse.
- **Class 3 ($V_{th}$ Drift Storage):** Hold SNM degrades monotonically from $0.443\,\text{V}$ ($5\%$) to $0.406\,\text{V}$ ($30\%$).
- **Class 4 ($V_{th}$ Drift Access):** Read sense delay $t_{\text{sense}}$ increases from $64.52\,\text{ps}$ to $87.85\,\text{ps}$; Hold SNM remains strictly invariant ($0.441\,\text{V}$), cleanly decoupling Class 3 from Class 4.

### 7.4 Dispersion Reduction Across 15 PVT Corners
| Feature | Raw CV ($\sigma/\mu$) | Normalized Invariant CV | Dispersion Reduction |
|---|---|---|---|
| **Hold SNM** | $11.6\%$ | **$0.0\%$** | **$100\%$ eliminated** |
| **Bitline $\Delta V_{\text{BL}}$** | $30.8\%$ | **$0.0\%$** | **$100\%$ eliminated** |
| **Write Delay $t_{\text{write}}$** | $13.6\%$ | **$5.8\%$** | **$57.4\%$ reduction** |

---

## 8. Machine Learning & Generalization Protocols

[`src/ml_pipeline.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/ml_pipeline.py) provides two rigorous evaluation protocols:

### Protocol 1: Standard Split (In-Distribution)
- Evaluates nominal corner ($1.0\,\text{V}, 27^\circ\text{C}$).
- Stratified 80/20 train/test split.
- **Automated Checkpoint:** Asserts zero `sample_id` overlap between train and test splits to strictly prevent information leakage.

### Protocol 2: Leave-One-Corner-Out (LOCO) Cross-PVT Validation
- Trains on 14 corners ($420$ samples) and tests on 1 held-out corner ($30$ samples).
- Iterated across all 15 corners.
- Directly measures model resilience against extreme environmental shifts ($-40^\circ\text{C}$, $125^\circ\text{C}$, $0.9\,\text{V}$, $1.1\,\text{V}$).

### Empirical Validation Results (Random Forest Classifier)
- **5-Fold Cross Validation:** $91.56\% \pm 0.89\%$ (Raw) $\to$ **$99.33\% \pm 0.54\%$** (Invariant)
- **Leave-One-Corner-Out Generalization:** $66.22\% \pm 12.58\%$ (Raw) $\to$ **$99.33\% \pm 1.33\%$** (Invariant)
- **Core Finding:** Raw features suffer a $33.1\%$ accuracy collapse when deployed to unseen PVT corners because thermal leakage and delay shifts are mistaken for physical faults. Physics-normalized invariant features completely resolve this degradation, maintaining $>99\%$ diagnostic accuracy across all corners.

---

## 9. Automated Test Suite Summary

All 62 unit tests pass under `pytest` (`62 passed in 9.70s`):
- `tests/test_fault_injection.py` (18 tests): Topology, symmetry, minimal diffs, model card isolation.
- `tests/test_testbench_composer.py` (4 tests): Deck assembly, dangling node detection.
- `tests/test_pvt_composition.py` (3 tests): PVT voltage, temperature scaling, and DC sweep sizing.
- `tests/test_simulation_runner.py` (6 tests): Batch execution, DC integrity validation, stem uniqueness.
- `tests/test_feature_extractor.py` (4 tests): Dynamic delay, voltage, and IDDQ extraction.
- `tests/test_snm_composition.py` (5 tests): Core duplication with `_c2` suffix, stimulus validation.
- `tests/test_snm_extraction.py` (7 tests): Pure NumPy inscribed square algorithm, reference decks.
- `tests/test_snm_features.py` (3 tests): Full-pipeline SNM extraction, non-monotonic flagging.
- `tests/test_dataset_generator.py` (8 tests): Grid construction, sample generation, fallback bounding.
- `tests/test_baseline_detector.py` (3 tests): Non-ML $3\sigma$ threshold detection.
- `tests/test_ml_pipeline.py` (4 tests): Model instantiation, metric computation, Standard Split, and LOCO protocols.
