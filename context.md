# Context Dossier: PVT-Invariant Parametric Fault Diagnosis in 6T SRAM Cells Using Machine Learning

---

## 1. Project Overview & Administrative Details

- **Official Project Title:**  
  *PVT-Invariant Parametric Fault Diagnosis in 6T SRAM Cells Using Machine Learning*
- **Degree & Branch:**  
  Bachelor of Technology (B.Tech) in Electronics & Communication Engineering (ECE)
- **Institution:**  
  Netaji Subhas University of Technology (NSUT), New Delhi – 110078
- **Department:**  
  Department of Electronics & Communication Engineering
- **Supervisor:**  
  Dr. Shweta Gautam, Associate Professor, Department of ECE, NSUT
- **Student Team:**
  1. **Archit Gupta** — Roll No: `2023UEC2558`
  2. **Ayush Raj** — Roll No: `2023UEC2575`
  3. **Debasish Dhungana** — Roll No: `2023UEC2571`
  4. **Vansh Singh** — Roll No: `2023UEC2550`
- **Academic Milestone:**  
  Mid-Semester Evaluation (BTP Phase 1 / Mid-Sem Review, September–October 2026)
- **Target Mid-Semester Completion Scope:**  
  **50%–60% of total work** (Full circuit sizing + 4 fault models + 15 PVT corners + Non-ML $3\sigma$ Baseline + Random Forest Classifier under Nominal and LOCO evaluation). The remaining models (HistGB, SVM, LogReg) and SHAP explainability are scheduled for End-Semester.

---

## 2. Core Engineering Problem & Motivation

1. **SRAM Scale:** Embedded SRAM caches occupy over 70% of the silicon area in modern SoCs, making memory defect tolerance the single largest driver of manufacturing yield and field reliability.
2. **Nanoscale Parametric Faults:** In deep sub-micron nodes, subtle defects (resistive opens in contacts, weak resistive bridges, oxide breakdown, and NBTI/HCI threshold voltage shifts) degrade electrical margins without causing complete functional stuck-at failures at nominal conditions. Consequently, traditional digital March tests mark these defective cells as healthy.
3. **The Cross-PVT Domain Shift Challenge:**
   - Chips operate from $-40^\circ\text{C}$ to $+125^\circ\text{C}$ and $0.9\,\text{V}$ to $1.1\,\text{V}$.
   - Subthreshold leakage current ($I_{\text{DDQ}}$) increases non-linearly by **$>400\%$** with temperature ($I_{\text{sub}} \propto e^{\frac{V_{GS}-V_{th}}{kT/q}}$).
   - Write switching delay ($t_{\text{write}}$) increases by **$>50\%$** under reduced supply voltage due to reduced gate overdrive.
   - **The Failure:** Machine Learning models trained on raw parameters at a nominal lab condition ($27^\circ\text{C}, 1.0\,\text{V}$) misinterpret thermal and voltage drift as defect signatures, causing severe out-of-distribution (OOD) diagnostic collapse ($<83\%$ accuracy, dropping to $50\%$ on extreme corners). Traditional fixed $3\sigma$ threshold baselines fail even worse, dropping to $25.78\%$.
4. **Our Solution:** A **Physics-Informed Domain Adaptation** methodology that normalizes test cell measurements against a concurrent on-die reference cell simulated under identical environmental conditions ($X_{\text{inv}} = X_{\text{meas}} / X_{\text{ref}}$). This cancels common-mode environmental shifts mathematically and restores diagnostic accuracy to **$98.89\%$–$100.0\%$**.

---

## 3. Circuit Sizing & Operational Physics

### Symmetrical 6T SRAM Cell Topology
- **Storage Inverters:** Cross-coupled PMOS pull-up ($M_1, M_3$) and NMOS pull-down driver ($M_2, M_4$).
- **Access Transistors:** NMOS pass-gates ($M_5, M_6$) gated by Wordline ($WL$), coupling internal nodes $Q$ and $Q_b$ to Bitlines ($BL, \overline{BL}$).
- **Aspect Ratios & Dimensions:**
  - Pull-up PMOS ($M_1, M_3$): $W = 360\,\text{nm}, L = 180\,\text{nm} \implies (W/L) = 2.0$
  - Driver NMOS ($M_2, M_4$): $W = 720\,\text{nm}, L = 180\,\text{nm} \implies (W/L) = 4.0$
  - Access NMOS ($M_5, M_6$): $W = 540\,\text{nm}, L = 180\,\text{nm} \implies (W/L) = 3.0$
- **Stability Ratio Criteria:**
  - **Cell Ratio (CR, $\beta$):** $\beta = \frac{(W/L)_{\text{driver}}}{(W/L)_{\text{access}}} = \frac{4.0}{3.0} = 1.333$ (Specification: $>1.2$ for non-destructive read).
  - **Pull-up Ratio (PR, $\gamma$):** $\gamma = \frac{(W/L)_{\text{pullup}}}{(W/L)_{\text{access}}} = \frac{2.0}{3.0} = 0.667$ (Specification: $<1.0$ for write flip capability).

### Operational Modes & Verified Baselines (at Nominal $27^\circ\text{C}, 1.0\,\text{V}$)
1. **Hold Mode ($WL=0\,\text{V}$):** Latched state is stable. Measured $V(Q)=1.0000\,\text{V}, V(Q_b)=0.0000\,\text{V}$. Quiescent leakage $I_{\text{DDQ}} = 14.2\,\text{pA}$. Hold Static Noise Margin ($\text{HSNM}$) $= 470.1\,\text{mV}$ (Spec: $>200\,\text{mV}$).
2. **Read Mode ($BL = \overline{BL} = 1.0\,\text{V}, WL=1.0\,\text{V}$):** Non-destructive read. Minimum internal node voltage during read $V(Q)_{\min} = 0.9845\,\text{V} > 0.5\,\text{V}$. Voltage bump on '0' node $V(Q_b)_{\text{bump}} = 0.1279\,\text{V} < 0.5\,\text{V}$. Read SNM ($\text{RSNM}$) $= 263.7\,\text{mV}$. Bitline differential $\Delta V_{\text{BL}} = 235.4\,\text{mV}$.
3. **Write Mode ($BL=0\,\text{V}, \overline{BL}=1.0\,\text{V}, WL=1.0\,\text{V}$):** Successful latch flip. Switching delay $t_{\text{write}} = 62.2\,\text{ps}$. Full post-write retention after $WL$ returns to $0\,\text{V}$.

---

## 4. Parametric Fault Injection Taxonomy

Faults are programmatically injected into SPICE netlists via pure text replacement in `src/fault_injection.py`:

| Class | Fault Name | Physical Root Cause | SPICE Netlist Injection Mechanism | Injected Severity Sweep |
|:---:|---|---|---|---|
| **0** | **Healthy (Baseline)** | Nominal defect-free cell | None (Baseline reference netlist) | Nominal |
| **1** | **Resistive Open** | Contact/via void in access path | Series resistor `Rfault` inserted at $M_5$ or $M_6$ source/drain node | $200\,\Omega, 500\,\Omega, 1\,\text{k}\Omega, 2\,\text{k}\Omega, 5\,\text{k}\Omega, 10\,\text{k}\Omega$ |
| **2** | **$Q \leftrightarrow \overline{Q}$ Bridging** | Particle bridging between storage nodes | Parallel resistor `Rbridge` inserted across nodes $Q$ and $Q_b$ | $500\,\Omega, 1\,\text{k}\Omega, 2\,\text{k}\Omega, 5\,\text{k}\Omega, 10\,\text{k}\Omega$ |
| **3** | **$V_{\text{th}}$ Drift (Storage)** | NBTI/HCI aging on cross-coupled latch | Override `.model` parameter `VTO` on $M_1$–$M_4$ | $+5\%, +10\%, +15\%, +20\%, +25\%, +30\%$ shift |
| **4** | **$V_{\text{th}}$ Drift (Access)** | HCI degradation on pass-gates | Override `.model` parameter `VTO` on $M_5/M_6$ | $+5\%, +10\%, +15\%, +20\%, +25\%, +30\%$ shift |

*Verification:* Unit test suite in `tests/test_fault_injection.py` asserts all injected netlists parse and simulate with zero syntax errors (**22/22 unit tests passing**).

---

## 5. Simulation Matrix & Dataset Infrastructure

- **Headless SPICE Automation:** Automated subprocess execution of LTspice with automatic parsing of `.log` files for convergence strings ("time step too small", "singular matrix", "gmin stepping failed", "no convergence").
- **15-Corner Operating Grid:**
  - Voltages: $0.9\,\text{V}, 1.0\,\text{V}, 1.1\,\text{V}$
  - Temperatures: $-40^\circ\text{C}, 0^\circ\text{C}, 27^\circ\text{C}, 75^\circ\text{C}, 125^\circ\text{C}$
  - Process: Typical-Typical (TT), Fast-Fast (FF), Slow-Slow (SS)
- **Dataset Scale:** 450 total multi-corner simulation runs ($30$ samples per corner $\times 15$ corners).
- **Data Integrity:** Exactly 450 rows in `data/sram_fault_dataset.csv`, **0 missing values, 0 NaNs, 0 infinite values**. 295 simulator convergence warning flags audited and logged.

---

## 6. Multi-Domain Feature Engineering

### Dynamic Electrical Metrics
- $I_{\text{DDQ}}$: Quiescent standby power supply leakage current ($\mu\text{A}$).
- $t_{\text{write}}$: Write switching flip delay from $50\%$ $WL$ rise to internal node crossing $0.5 V_{\text{DD}}$ ($\text{ps}$).
- $I_{\text{write, peak}}$: Peak dynamic current during write switching ($\text{mA}$).
- $\Delta V_{\text{BL}}$: Bitline discharge voltage differential at sense strobe ($\text{mV}$).
- $t_{\text{sense}}$: Sense amplifier strobe delay ($\text{ps}$).
- $V_{\text{bump}}$: Voltage divider bump on storing '0' node during read ($\text{mV}$).

### Static Noise Margin (SNM) Extraction via Butterfly Curves
- Overlapping Voltage Transfer Characteristics (VTCs) of inverters: $V(Q_b) = f(V_Q)$ and $V(Q) = g(V_{Q_b})$.
- **Inscribed Square Algorithm:** Implemented in pure NumPy (`src/snm_extraction.py`). Computes the side length of the largest axis-aligned square inscribed in the smaller butterfly curve lobe:
  $$\text{SNM} = \min(\text{Lobe}_1, \text{Lobe}_2)$$
- Verified against reference golden measurements with $<1.0\,\text{mV}$ numerical error. Avoids the flawed $45^\circ$ curve rotation shortcut that overestimates SNM on steep rail-to-rail CMOS slopes.

### Physics-Normalized Feature Invariants (The Core Innovation)
Normalized relative to a concurrent on-die healthy reference cell simulated at the identical operating corner:
1. `iddq_norm` $= I_{\text{DDQ, meas}} / I_{\text{DDQ, ref}}$
2. `t_write_norm` $= t_{\text{write, meas}} / t_{\text{write, ref}}$
3. `dv_bl_norm` $= \Delta V_{\text{BL, meas}} / \Delta V_{\text{BL, ref}}$
4. `snm_hold_norm` $= \text{HSNM}_{\text{meas}} / \text{HSNM}_{\text{ref}}$
5. `snm_read_norm` $= \text{RSNM}_{\text{meas}} / \text{RSNM}_{\text{ref}}$
6. `read_write_delay_ratio` $= t_{\text{sense}} / t_{\text{write}}$
7. `snm_ratio` $= \text{RSNM} / \text{HSNM}$
8. `dv_bl_vdd_ratio` $= \Delta V_{\text{BL}} / V_{\text{DD}}$

**Dispersion Impact:** Raw features exhibit coefficient of variation $\text{CV} > 60\%$ across temperatures; invariant features collapse healthy-cell variation to **$\text{CV} < 5\%$** while preserving defect separation.

---

## 7. Experimental Results & Mid-Semester Benchmarking

### Evaluation Protocols
1. **Protocol 1 (Standard Split, Nominal Corner $27^\circ\text{C}, 1.0\,\text{V}$):** 80/20 train/test split. Automated assertion confirms zero `sample_id` leakage.
2. **Protocol 2 (Leave-One-Corner-Out / LOCO):** Train on 14 corners, evaluate on the 15th held-out unseen corner. Repeated for all 15 corners.

### Mid-Semester Performance Benchmark Table

| Diagnostic Model | Feature Representation | Protocol 1: Nominal Corner ($27^\circ\text{C}, 1.0\text{V}$) | Protocol 2: Cross-PVT LOCO (15 Corners) | Technical Behavior & Notes |
|---|---|:---:|:---:|---|
| **Non-ML Baseline** ($3\sigma$ Threshold Detector) | Raw Absolute Metrics | $93.3\%$ (6.7% FP) | **$25.78\%$** | Fails completely under thermal drift; static thresholds cannot adapt. |
| **Random Forest** (Classical ML) | Raw Features | **$100.0\%$** | **$82.67\%$** | Suffers severe OOD domain shift (drops to $50.0\%$ on extreme corners). |
| **Random Forest + Proposed Invariants** | **Physics-Normalized Ratios** | **$100.0\%$** | **$98.89\%$** | **$+16.22\%$ Generalization Gain**; robust across all 15 corners. |

*(Note: On clean simulation runs without transient integration timeouts, the invariant Random Forest achieves **100.0% accuracy across all 15 held-out PVT corners**).*

---

## 8. Division of Work: Mid-Sem (Completed) vs. End-Sem (Remaining)

### Mid-Semester Scope (50%–60% Completed)
- [x] Full 6T cell sizing, ratio validation ($\beta=1.333, \gamma=0.667$), and hold/read/write operational baselines.
- [x] Automated text-based netlist fault injection engine for all 4 defect classes (22/22 unit tests passing).
- [x] Full 450-sample simulation matrix generated across 15 PVT corners with convergence logging.
- [x] Pure-NumPy inscribed-square SNM butterfly extraction.
- [x] Formulation and statistical validation of physics-normalized invariant ratios.
- [x] Implementation of non-ML $3\sigma$ baseline rule detector.
- [x] Training and LOCO evaluation of primary ML model (**Random Forest Classifier**), proving $+16.22\%$ cross-PVT generalization gain.

### End-Semester Scope (Remaining 40%–50%)
- [ ] Multi-Model Benchmark Expansion: Training and evaluating the remaining classical classifiers:
  * `HistGradientBoostingClassifier` (binned histogram decision trees)
  * `SVC(kernel="linear")` (Linear Support Vector Machine)
  * `LogisticRegression` (L2-regularized multi-class logistic regression)
- [ ] Tree SHAP (SHapley Additive exPlanations) attribution analysis:
  * Computing global feature importance values (demonstrating `dv_bl_norm` and `t_write_norm` dominate decisions).
  * Class-specific beeswarm and waterfall diagnostic plots.
- [ ] Hardware BIST Implementation:
  * Developing a lightweight embedded C / MicroPython inference routine for on-chip test co-processors.
- [ ] Final Research Publication & Thesis:
  * Writing full IEEE-format conference/journal manuscript and complete BTP thesis.

---

## 9. Figures & Graphics Included in the Report

1. **Figure 1 (Chapter 1):**  
   `sram_architecture_faults.png` — Standard 6T SRAM Cell Circuit Topology with Parametric Defect Injection Sites ($F_1$ bridge, $F_2$ open, $F_3$ oxide leakage, $F_4$ $V_{\text{th}}$ mismatch).
2. **Figure 2 (Chapter 3):**  
   `snm_butterfly_diagram.png` — Overlapping Voltage Transfer Characteristics (Butterfly Curve) and Maximum Inscribed Square for Static Noise Margin ($\text{SNM}$) Extraction.
3. **Figure 3 (Chapter 3):**  
   `methodology_flowchart.png` — Proposed Automated Simulation, Feature Normalization, and Diagnostic Machine Learning Workflow.
4. **Figure 4 (Chapter 4):**  
   `fig1_feature_dispersion_comparison.png` — Parametric Feature Dispersion Comparison: Raw vs. Physics-Normalized Invariant Features Across 15 PVT Corners.
5. **Figure 5 (Chapter 4):**  
   `fig3_rf_pvt_corner_heatmap.png` — Random Forest Diagnostic Accuracy Across 15 Held-Out PVT Corners under Leave-One-Corner-Out (LOCO) Evaluation.

---

## 10. Key References (IEEE Style)

1. J. Seevinck, F. J. List, and J. Lohstroh, "Static-noise margin analysis of MOS SRAM cells," *IEEE Journal of Solid-State Circuits*, vol. 22, no. 5, pp. 748–754, Oct. 1987.
2. S. Mukhopadhyay, H. Mahmoodi-Meimand, and K. Roy, "Modeling and analysis of failure in sub-100-nm ultra-low-power SRAM design," *IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, vol. 13, no. 5, pp. 586–595, May 2005.
3. P. Marchal et al., "SRAM dynamic fault modeling and testing in deep-submicron technologies," *IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, vol. 16, no. 6, pp. 627–640, Jun. 2008.
4. Arizona State University, "Predictive Technology Model (PTM) 45nm/180nm Low-Power CMOS Model Cards," Nanoscale Integration and Modeling (NIM) Group.
5. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
6. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 4765–4774.
