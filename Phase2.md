# Phase 2: Machine Learning Classification, Cross-PVT Analysis & Research Publication Guide

**Project:** BTP — Automated 6T SRAM Fault Injection, PVT Simulation & Diagnostic Framework  
**Document Purpose:** Complete, step-by-step handover and execution guide for collaborators taking over Phase 2.  
**Repository State:** Phase 1 Complete (SPICE Netlists, SNM Extraction, Fault Injection, 15-Corner PVT Simulations, 450-Sample Dataset, 62/62 Unit Tests Passing).

---

## 1. Project Context & Current Status (Handover Summary)

### 1.1 What Has Been Completed (Phase 1)
1. **Healthy 6T SRAM Cell** ([`circuits/core/cell_core_healthy.net`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/circuits/core/cell_core_healthy.net)):
   - Sized for 180nm CMOS ($V_{DD} = 1.0\,\text{V}$ nominal).
   - Cell ratio $\beta = 1.33$ (read stability), pull-up ratio $\gamma = 1.50$ (writability).
2. **Butterfly-Curve Static Noise Margin (SNM) Extraction** ([`src/snm_extraction.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/snm_extraction.py)):
   - True largest-inscribed-square search in pure NumPy (both Hold and Read SNM).
3. **4-Class Fault Injection Engine** ([`src/fault_injection.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/fault_injection.py)):
   - Class 1: Resistive Open on pass-gates $M_5/M_6$ ($200\,\Omega$ to $10\,\text{k}\Omega$).
   - Class 2: Bridging Fault $Q \leftrightarrow \bar{Q}$ ($500\,\Omega$ to $10\,\text{k}\Omega$).
   - Class 3: $V_{th}$ Drift on cross-coupled inverters $M_1$–$M_4$ ($\pm 5\%$ to $\pm 30\%$).
   - Class 4: $V_{th}$ Drift on access transistors $M_5/M_6$ ($\pm 5\%$ to $\pm 30\%$).
4. **PVT Simulation Engine & Multi-Threading** ([`src/simulation_runner.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/simulation_runner.py), [`src/testbench_composer.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/testbench_composer.py)):
   - 15-corner grid: $V_{DD} \in [0.9, 1.0, 1.1]\,\text{V}$, $T \in [-40, 0, 27, 75, 125]^\circ\text{C}$.
   - 3,255 SPICE simulations executed with zero file collisions and on-the-fly binary waveform pruning.
5. **Verified Clean Datasets** (in `data/`):
   - [`data/sram_fault_dataset.csv`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/data/sram_fault_dataset.csv): Master dataset (**450 rows $\times$ 33 columns**, **0 NaNs**).
   - [`data/sram_fault_dataset_indist.csv`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/data/sram_fault_dataset_indist.csv): In-distribution nominal split (30 rows, $1.0\,\text{V}, 27^\circ\text{C}$).
   - [`data/sram_fault_dataset_ood.csv`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/data/sram_fault_dataset_ood.csv): Out-of-distribution split (420 rows, 14 held-out PVT corners).
   - [`logs/convergence_flags.csv`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/logs/convergence_flags.csv): Traceability and DC monotonicity audit log.
6. **Codebase Quality**:
   - Full regression test suite passing: `pytest` passes **62/62 tests**.

---

## 2. Phase 2 Scope & Roadmap

Phase 2 consists of **4 main objectives**:

```
+---------------------------------------------------------------------------------+
|                               PHASE 2 ROADMAP                                   |
+---------------------------------------------------------------------------------+
|  STAGE 5: Feature Analysis & Dispersion Validation                             |
|  - Verify Coefficient of Variation (CV) reduction on healthy cells.            |
|  - Confirm invariance of physics-normalized features across 15 PVT corners.     |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|  STAGE 6: Machine Learning Pipeline & Baseline Evaluation                       |
|  - Execute Non-ML 3-Sigma Rule Baseline (src/baseline_detector.py).             |
|  - Train 4 Classical Models: RF, Logistic Regression, Linear SVM, HistGB.       |
|  - Protocol 1: Standard Split (Nominal 1.0V, 27°C, leakage-checked).            |
|  - Protocol 2: Leave-One-Corner-Out (LOCO across all 15 corners).               |
|  - Export reports/final_results/comparison_table.csv and trained models (.pkl). |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|  STAGE 7: Cross-PVT Analysis & Publication Figures                              |
|  - Generate publication-grade plots in reports/figures/:                        |
|    * Fig 1: Feature dispersion boxplots (Raw vs Invariant).                     |
|    * Fig 2: Cross-PVT generalization bar chart (Raw vs Invariant).              |
|    * Fig 3: Corner degradation heatmap across 15 corners.                       |
|  - (Optional / Stretch): SHAP feature importance analysis on best model.        |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|  STAGE 8: Research Paper & Thesis Integration                                   |
|  - Write Results, Discussion, and Methodology sections.                         |
|  - Compile LaTeX tables from generated CSVs.                                    |
+---------------------------------------------------------------------------------+
```

---

## 3. Step-by-Step Execution Guide for Collaborators

### Step 1: Environment Setup & Sanity Check
Before writing any code, clone the repository, activate your virtual environment, and ensure all existing unit tests pass:

```bash
# 1. Pull latest code
git pull origin main

# 2. Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib pyyaml pytest

# 3. Verify test suite (Expected: 62 passed)
pytest -v
```

---

### Step 2: Audit Dataset Physical Scaling
Execute the audit script to verify that the dataset obeys physical semiconductor scaling laws:

```bash
python scripts/verify_dataset_physics.py
```

**What to check in the output:**
- **Temperature Leakage Scaling:** $I_{\text{ddq}}$ should rise exponentially from $-40^\circ\text{C}$ ($18\,\text{nA}$) to $125^\circ\text{C}$ ($1.62\,\mu\text{A}$).
- **Overdrive Sizing:** Write delay $t_{\text{write}}$ should decrease monotonically as $V_{DD}$ increases ($70.6\,\text{ps}$ at $0.9\,\text{V} \to 52.1\,\text{ps}$ at $1.1\,\text{V}$).
- **Fault Signatures:**
  - Class 1 (Open): $t_{\text{write}}$ increases monotonically with resistance ($61.6\,\text{ps} \to 106.0\,\text{ps}$).
  - Class 2 (Bridge): Standby leakage jumps to $18.6\,\mu\text{A}$, bistability collapses (`is_bistable = 0`).
  - Class 3 ($V_{th}$ Storage): Hold SNM degrades monotonically ($0.443\,\text{V} \to 0.406\,\text{V}$).
  - Class 4 ($V_{th}$ Access): Sense delay increases ($64.5\,\text{ps} \to 87.9\,\text{ps}$), while Hold SNM remains strictly invariant ($0.441\,\text{V}$).

---

### Step 3: Run the Machine Learning Pipeline

The pipeline is implemented in [`src/ml_pipeline.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/ml_pipeline.py). It runs both protocols and trains all models on both **Raw** and **Physics-Normalized Invariant** feature sets.

Run the ML pipeline with:
```bash
python src/ml_pipeline.py
```

#### Core Components Executed by `ml_pipeline.py`:
1. **Models Evaluated:**
   - `RandomForest`: `RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)`
   - `LogisticRegression`: `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))`
   - `LinearSVM`: `make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=42, probability=True))`
   - `HistGradientBoosting`: `HistGradientBoostingClassifier(max_iter=100, random_state=42)`
   - `NonML_Baseline`: `BaselineRuleDetector(n_sigma=3.0)` from [`src/baseline_detector.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/baseline_detector.py).
2. **Feature Sets:**
   - **Raw Features:** `i_ddq_uA`, `t_write_ps`, `i_write_peak_mA`, `dv_bl_strobe_mV`, `t_sense_ps`, `v_bump_mV`, `hold_pass`, `write_pass`, `read_stable_pass`, `snm_hold_v`, `snm_read_v`, `snm_asym_hold_v`, `snm_asym_read_v`, `snm_drop_v`, `is_bistable`.
   - **Physics-Normalized Invariant Features:** `iddq_norm`, `t_write_norm`, `dv_bl_norm`, `snm_hold_norm`, `snm_read_norm`, `read_write_delay_ratio`, `snm_ratio`, `dv_bl_vdd_ratio`, `hold_pass`, `write_pass`, `read_stable_pass`, `is_bistable`.
3. **Protocols:**
   - **Protocol 1 (Standard Split):** In-distribution nominal corner (`1.0V`, `27°C`). Train/test 80/20 stratified split. Asserts **zero sample ID overlap** (leakage-free).
   - **Protocol 2 (Leave-One-Corner-Out / LOCO):** Iteratively trains on 14 corners (420 samples) and tests on 1 held-out corner (30 samples) across all 15 corners.
4. **Outputs Generated:**
   - Saved model weights in `models/*.pkl`
   - Master comparison table: [`reports/final_results/comparison_table.csv`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/reports/final_results/comparison_table.csv)
   - Aggregated LOCO summary: [`reports/final_results/loco_summary.csv`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/reports/final_results/loco_summary.csv)

---

### Step 4: Generate Publication Figures & Dispersion Analysis

The analysis script is implemented in [`src/cross_pvt_analysis.py`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/src/cross_pvt_analysis.py).

Run the figure generator with:
```bash
python src/cross_pvt_analysis.py
```

#### Generated Artifacts:
1. [`reports/figures/fig1_feature_dispersion_comparison.png`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/reports/figures/fig1_feature_dispersion_comparison.png):
   - Standardized Z-score boxplots for Healthy cells across all 15 corners comparing Raw vs Invariant features.
   - Shows that Hold SNM and Bitline $\Delta V_{\text{BL}}$ dispersion are reduced to **0.0%**, and write delay variation is cut by **57.4%**.
2. [`reports/figures/fig2_cross_pvt_loco_accuracy.png`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/reports/figures/fig2_cross_pvt_loco_accuracy.png):
   - Grouped bar chart comparing LOCO accuracy (Mean $\pm$ 1 Std Dev) between Raw and Invariant features across all 4 ML models.
   - Visualizes the central empirical finding: **$+33.1\%$ accuracy improvement** under cross-PVT testing.
3. [`reports/figures/fig3_rf_pvt_corner_heatmap.png`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/reports/figures/fig3_rf_pvt_corner_heatmap.png):
   - Heatmap displaying test accuracy across each of the 15 held-out corners.
   - Highlights how raw features collapse under extreme temperature corners ($-40^\circ\text{C}$ and $125^\circ\text{C}$), while invariant features remain robust ($>96.7\%$).
4. [`reports/final_results/feature_spread_comparison.csv`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/reports/final_results/feature_spread_comparison.csv):
   - Numerical table of Coefficient of Variation ($CV = \frac{\sigma}{\mu} \times 100\%$) and percentage dispersion reductions.

---

### Step 5 (Optional / Stretch Goal): Hyperparameter Tuning & SHAP Explainability

If you wish to optimize the models further or add explainability for the paper:

#### A. Hyperparameter Tuning with Optuna (Optional)
To tune Random Forest (`n_estimators`, `max_depth`, `min_samples_split`) or SVM (`C`, `gamma`):
1. Install Optuna: `pip install optuna`
2. Create `scripts/tune_hyperparameters.py` using 5-fold CV on `data/sram_fault_dataset_indist.csv`.
3. Optimize macro F1-score. Use `seed=42`.

#### B. SHAP Feature Attribution (Optional)
To generate SHAP summary and waterfall plots for the thesis:
1. Install SHAP: `pip install shap`
2. Run TreeExplainer on the trained Random Forest model (`models/randomforest_invariant_standard.pkl`):
   ```python
   import shap, joblib, pandas as pd
   model = joblib.load("models/randomforest_invariant_standard.pkl")
   df = pd.read_csv("data/sram_fault_dataset.csv")
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(df[INVARIANT_FEATURES])
   shap.summary_plot(shap_values, df[INVARIANT_FEATURES], show=False)
   plt.savefig("reports/figures/fig4_shap_summary.png", bbox_inches="tight")
   ```

---

## 4. Key Results & Expected Numbers

Collaborators can verify their training outputs against these expected benchmarks:

### 4.1 Feature Dispersion (Healthy Cells across 15 Corners)
| Feature | Raw Mean $\pm$ Std | Raw CV (%) | Invariant Mean $\pm$ Std | Invariant CV (%) | Spread Reduction |
|---|---|---|---|---|---|
| **Hold SNM** | $0.441 \pm 0.051\,\text{V}$ | $11.6\%$ | $1.000 \pm 0.000$ | **$0.0\%$** | **$100\%$ eliminated** |
| **Bitline $\Delta V_{\text{BL}}$** | $235.4 \pm 72.4\,\text{mV}$ | $30.8\%$ | $1.000 \pm 0.000$ | **$0.0\%$** | **$100\%$ eliminated** |
| **Write Delay $t_{\text{write}}$** | $61.6 \pm 8.4\,\text{ps}$ | $13.6\%$ | $1.016 \pm 0.059$ | **$5.8\%$** | **$57.4\%$ reduction** |
| **Standby Current $I_{\text{ddq}}$** | $0.50 \pm 0.97\,\mu\text{A}$ | $194.1\%$ | $1.60 \pm 2.25$ | **$140.3\%$** | **$27.7\%$ reduction** |

### 4.2 Classification Performance Across Protocols (Random Forest)
| Evaluation Protocol | Raw Features Accuracy | Invariant Features Accuracy | Generalization Gain |
|---|---|---|---|
| **Standard Split (Nominal Corner)** | $91.56\% \pm 0.89\%$ | **$99.33\% \pm 0.54\%$** | **$+7.77\%$** |
| **Leave-One-Corner-Out (LOCO)** | $66.22\% \pm 12.58\%$ *(Min: 50.0%)* | **$99.33\% \pm 1.33\%$** *(Min: 96.7%)* | **$+33.11\%$** |

---

## 5. Research Paper / Thesis Writing Guide (Stage 8)

When writing the paper or thesis report, structure the content as follows:

### Section 1: Introduction & Problem Statement
- SRAM cells are increasingly susceptible to nanometer manufacturing defects (resistive opens, bridging) and parametric aging/variability ($V_{th}$ drift from BTI/HCI).
- Traditional BIST (Built-In Self-Test) only detects catastrophic functional go/no-go failures and misses subtle parametric degradation.
- **The Core Challenge:** Standard ML classifiers trained at nominal temperature/voltage fail when deployed across operating corners because environmental PVT drift masks or mimics fault signatures.
- **Proposed Solution:** A physics-normalized invariant feature representation referencing a co-located healthy cell on the same die.

### Section 2: 6T SRAM Cell & Defect Modeling
- Standard 180nm CMOS topology ($M_1$–$M_6$).
- Decoupled netlist architecture and 4 fault injection models.
- Mathematical definitions of butterfly-curve Static Noise Margins (Hold SNM and Read SNM) using the largest inscribed square method.

### Section 3: PVT Simulation Framework
- 15-corner grid: $V_{DD} \in \{0.9, 1.0, 1.1\}\,\text{V}$, $T \in \{-40, 0, 27, 75, 125\}^\circ\text{C}$.
- Multi-threaded headless simulation engine with transient convergence and DC monotonicity checks.
- Dataset construction: 450 balanced samples, 33-column schema, zero-NaN guarantee.

### Section 4: Physics-Normalized Invariant Feature Engineering
- Define the normalization equations:
  $$I_{\text{ddq, norm}} = \frac{I_{\text{ddq}}}{I_{\text{ddq, ref}}}, \quad t_{\text{write, norm}} = \frac{t_{\text{write}}}{t_{\text{write, ref}}}, \quad \text{SNM}_{\text{norm}} = \frac{\text{SNM}}{\text{SNM}_{\text{ref}}}$$
- Include Table 4.1 from this document showing $CV$ dispersion reduction. Include `fig1_feature_dispersion_comparison.png`.

### Section 5: Experimental Results & Discussion
- Present Table 4.2 comparing all 4 ML models (Random Forest, Logistic Regression, Linear SVM, HistGB) and the Non-ML Baseline.
- Include `fig2_cross_pvt_loco_accuracy.png` and `fig3_rf_pvt_corner_heatmap.png`.
- **Explain the Physics:** Why does raw accuracy drop to $66\%$ under LOCO? Because subthreshold leakage increases $>100\times$ between $-40^\circ\text{C}$ and $125^\circ\text{C}$, leading the classifier to misdiagnose high temperature as an internal bridging fault. Invariant normalization removes this thermal pedestal, restoring $>99\%$ accuracy.

### Section 6: Conclusion
- Summary of achievements: automated SPICE pipeline, 15-corner PVT dataset, $>33\%$ cross-PVT generalization improvement, and publication-ready diagnostic models.

---

## 6. Deliverables Checklist for Collaborators

Before submitting or wrapping up Phase 2, verify that all items below are complete:

- [ ] Run `python src/ml_pipeline.py` and verify `reports/final_results/comparison_table.csv` is populated.
- [ ] Verify trained model pickles exist in `models/` (`randomforest_invariant_standard.pkl`, etc.).
- [ ] Run `python src/cross_pvt_analysis.py` and verify PNG figures exist in `reports/figures/`.
- [ ] Run `pytest` and verify all 62 tests continue to pass with 0 failures.
- [ ] Review `reports/final_results/loco_summary.csv` to ensure mean LOCO accuracy on invariant features exceeds $98\%$.
- [ ] Insert figures and tables into the BTP report / LaTeX manuscript.
- [ ] Commit all generated reports and figures to Git and push to GitHub.

---

## 7. Contact & Assistance
If any unexpected test failures or SPICE anomalies occur:
- Inspect [`logs/convergence_flags.csv`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/logs/convergence_flags.csv) for any flagged netlists.
- Check that your local `config.yaml` points to a valid LTspice executable path.
- Review [`PROJECT_DOCUMENTATION.md`](file:///c:/Users/AYUSH/Desktop/btp-sram-fault-diagnosis/PROJECT_DOCUMENTATION.md) for deep circuit and algorithm specifications.
