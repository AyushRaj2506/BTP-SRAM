# PLAN.md — PVT-Invariant Fault Diagnosis in 6T SRAM Cells

**Project:** BTP — PVT-Invariant Feature Engineering for Parametric Fault Diagnosis in a 6T SRAM Cell
**Team size:** 4
**This file is the full engineering + team handoff.** It is written to be read by human team members and Antigravity (or any coding agent) as the working spec. If anything here conflicts with `BTP_SRAM_Proposal.docx`, the docx is the source of truth for research framing; this file is the source of truth for implementation detail and task ownership.

---

## 0. Non-negotiable ground rules

Read this section before writing any code. These exist because earlier project drafts hit exactly these failure modes.

1. **No result, number, or citation goes into any report unless it was actually produced by running code in this repo or independently verified by a human team member.** Do not let an agent pre-fill "expected results," fabricate literature citations, or invent numbers to fill a table. If a value isn't computed yet, write `TBD`, not a plausible-looking placeholder.
2. **The healthy SRAM cell and the SNM extraction script must be manually validated by a human before any batch automation runs.** This is Weeks 1–2 and it is the critical path for the entire project — see Section 7.
3. **Every batch simulation run must be checked for SPICE convergence**, not just "did a `.raw` file get written." See Section 6.5.
4. **No hardcoded absolute file paths** (e.g. `C:\Users\<name>\...`). Use `config.yaml` or relative paths so any team member or machine can run the pipeline.
5. **Every dataset generation run uses a fixed random seed** and logs it, so results are reproducible.
6. **If a milestone's acceptance criteria (Section 8) aren't met, do not proceed to the next milestone.** Flag it to the whole team and fix it first.

---

## 1. Research question and scope

**Research question:**
> How does PVT (voltage–temperature) variation degrade machine-learning-based parametric fault classification in a 6T SRAM cell, and can physics-normalized (PVT-invariant) features reduce that degradation without per-corner retraining?

**In scope:**
- One 6T SRAM cell, one technology assumption (generic CMOS model in LTspice).
- Fault taxonomy: healthy + 4 parametric fault classes (Section 4).
- PVT sweep across **voltage and temperature only** (process corner explicitly excluded — see Limitations).
- Four classical ML models, two evaluation protocols (standard split + leave-one-corner-out).
- One non-ML baseline detector.
- Raw-vs-invariant feature comparison as the paper's central result.

**Explicitly out of scope (do not let this creep back in):**
- Multiple circuits or topologies (SRAM only, no NAND/NOR/op-amp comparison in this phase).
- Process-corner variation (would require special model-card files; noted as a stated limitation instead).
- Deep learning models (dataset size doesn't justify it — see rationale in Section 6.7).
- Real silicon validation (simulation-only, stated as a limitation like the rest of the field).
- Full explainability suite — SHAP is a stretch goal only, first thing cut if behind schedule.

**Limitations to state explicitly in the paper, not discover during review:**
- "PVT" in the title/scope means voltage + temperature; process-corner variation is not covered.
- All fault labels come from SPICE simulation, not fabricated/measured silicon.
- PVT sweep is a discrete set of points (Section 6.3), not a continuous characterization.

---

## 2. Team and work division

| Member | Role | Primary ownership | Active from |
|---|---|---|---|
| **A** | Circuit & Measurement Lead | Healthy cell design, SNM extraction + validation (**critical path**) | Week 1 |
| **B** | Fault Injection & Simulation Automation Lead | Fault injection functions, batch simulation runner, convergence checking, dataset orchestration | Week 1 (parallel), critical from Week 3 |
| **C** | Feature Engineering & ML Pipeline Lead | Raw + invariant feature design, ML pipeline, baseline detector, central comparison result | Week 3 (baseline), critical from Week 4 |
| **D** | Explainability, Documentation & Paper Lead | Literature review (starts immediately), milestone reports, SHAP analysis, paper drafting | Week 1 (continuous) |

**Shared responsibility, every member:** be able to personally explain the SNM derivation and the leave-one-corner-out evaluation logic out loud before the defense, regardless of who wrote the code. Schedule one team walkthrough session for this in Week 9.

**Why this split:** A's work is deliberately slow and manual — it has to be, since it's the trust anchor the rest of the pipeline depends on. B's work is throughput/automation-focused and can start in parallel once a placeholder netlist exists. C owns the actual research contribution (the invariant features) and should be whoever on the team is strongest at connecting the circuits reasoning to the ML tooling. D starting the literature review in Week 1, instead of Week 9, is the single biggest schedule risk this division removes — citation verification has zero dependency on any simulation work finishing first.

---

## 3. Repository structure

```
btp-sram-fault-diagnosis/
├── PLAN.md                          <- this file
├── README.md
├── requirements.txt
├── config.yaml                      <- all paths, PVT sweep values, seeds (no hardcoded values in code)
├── circuits/
│   ├── sram_6t_healthy.asc          <- LTspice schematic (Member A, manual)
│   └── sram_6t_healthy.cir          <- exported netlist (template for fault injection)
├── src/
│   ├── fault_injection.py           <- Member B: generates faulty netlist variants
│   ├── simulation_runner.py         <- Member B: PyLTSpice batch execution + convergence checking
│   ├── snm_extraction.py            <- Member A: butterfly-curve DC sweep + SNM computation
│   ├── extract_features.py          <- Member C: parses .raw files into feature rows
│   ├── generate_dataset.py          <- Member B: orchestrates fault x severity x PVT sweep -> CSV
│   ├── baseline_detector.py         <- Member C: non-ML SNM/current-threshold detector
│   ├── ml_pipeline.py               <- Member C: train/val/test split, model training, LOCO eval
│   └── utils.py                     <- shared helpers (netlist parsing, logging)
├── data/
│   ├── raw_sim/                     <- .raw / .log files from LTspice (gitignored, large)
│   ├── sram_fault_dataset.csv       <- in-distribution dataset
│   ├── sram_fault_dataset_ood.csv   <- PVT out-of-distribution dataset
│   └── ml_splits/                   <- train/val/test ID lists (per protocol, per feature set)
├── logs/
│   ├── failed_simulations.csv
│   └── convergence_flags.csv
├── models/
│   └── <model_name>_<feature_set>_<split_protocol>.pkl
├── validation/
│   ├── snm_manual_check/            <- Member A: the 5–10 manually-run butterfly curves, kept for audit
│   └── sim_manual_crosscheck/       <- Member B: the random-sample manual cross-check runs
├── reports/
│   ├── milestone_1_healthy_cell.md      <- Member A
│   ├── milestone_2_snm_validation.md    <- Member A
│   ├── milestone_3_fault_taxonomy.md    <- Member B
│   ├── milestone_4_dataset.md           <- Member B
│   ├── milestone_5_features.md          <- Member C
│   ├── milestone_6_ml_results.md        <- Member C
│   ├── literature_review.md             <- Member D, continuous
│   └── final_results/
└── notebooks/
    └── exploratory/                 <- scratch analysis, not part of the pipeline
```

---

## 4. Fault taxonomy (implementation spec — Member B)

| Class | Name | Injection mechanism | Severity sweep |
|---|---|---|---|
| 0 | Healthy | — | — |
| 1 | Resistive open | Series resistor inserted at M5 or M6 drain/source node in the netlist text | 200Ω, 1kΩ, 5kΩ |
| 2 | Q↔QB bridging | Resistor added between the Q and QB storage nodes | 500Ω, 2kΩ, 10kΩ |
| 3 | Vth drift — storage pair | Override `.model` Vth parameter on M1–M4 | +5%, +10%, +20% shift |
| 4 | Vth drift — access transistors | Override `.model` Vth parameter on M5/M6 | +5%, +10%, +20% shift |

```python
# fault_injection.py — required function signatures (Member B)
def inject_resistive_open(netlist_path, transistor="M5", resistance_ohms=1000) -> str: ...
def inject_bridging_fault(netlist_path, resistance_ohms=2000) -> str: ...
def inject_vth_drift(netlist_path, target="storage_pair", drift_pct=10) -> str: ...
```
Pure text-editing functions on the netlist — no GUI interaction, ever, in automated code. Each must have a unit test asserting the output netlist parses in LTspice without error and differs from the healthy netlist at the expected line.

---

## 5. Environment setup

```
# requirements.txt
PyLTSpice
numpy
pandas
scikit-learn
optuna
shap
matplotlib
seaborn
pyyaml
pytest
```
- LTspice installed locally; PyLTSpice drives it headlessly for batch runs.
- All configurable values (paths, PVT sweep points, random seed, dataset target size) live in `config.yaml`.
- Run `pytest validation/` before pushing changes that touch `src/`.

---

## 6. Core technical components

### 6.1 Healthy SRAM cell — Member A, manual, Week 1
- Standard 6T topology: M1–M4 cross-coupled inverter pair, M5/M6 NMOS access transistors gated by wordline (WL).
- Node naming convention (use exactly these names, everything downstream depends on it): `VDD`, `GND`, `Q`, `QB`, `WL`, `BL`, `BLB`
- Size transistors against a documented reference ratio (typical cell ratio ~1.2–2, pull-up ratio ~0.5–1) rather than arbitrary values — avoids a fragile SNM budget that makes fault/PVT effects indistinguishable by construction.
- Manually confirm in the GUI: (a) cell holds state under DC bias, (b) read cycle doesn't disturb stored value, (c) write cycle correctly flips state. Save waveforms to `validation/`.

### 6.2 SNM extraction — Member A, `snm_extraction.py`
```
1. Run a DC sweep: sweep Q from 0->VDD while measuring QB, and QB from 0->VDD while measuring Q,
   with access transistors OFF (hold SNM) and separately ON with bitlines held at VDD/0 (read SNM).
2. This produces two voltage-transfer curves forming the "butterfly" shape.
3. For each of the two lobes, find the largest inscribed square:
   - Iterate candidate square side lengths
   - Check whether a square of that size fits between the two curves
   - SNM = side length of the largest square that fits
4. Return {hold_snm, read_snm} in volts.
```
**Validation requirement (Week 2 gate):** run against 5–10 manually-inspected GUI butterfly curves before trusting it for batch use. Store the comparison in `validation/snm_manual_check/results.md`. **Re-check on a handful of post-fault-injection netlists once fault injection exists (Week 3)** — a bridging fault can distort the curve shape in ways the healthy-cell validation alone won't catch.

### 6.3 PVT sweep — `config.yaml`, owned by Member B
```yaml
pvt_sweep:
  temperature_c: [-40, 0, 27, 75, 125]   # 27 is nominal/in-distribution
  vdd_v: [0.9, 1.0, 1.1]                 # 1.0 is nominal/in-distribution
in_distribution_corner:
  temperature_c: 27
  vdd_v: 1.0
```
In-distribution dataset = nominal corner only. OOD dataset = every other corner combination, held out entirely from training. Run a Week 1 timing pilot (20 DC sweeps, measured wall-clock) before committing to full dataset size — DC sweeps for SNM are heavier than a transient run.

### 6.4 Feature schema — Member C, `extract_features.py`

| Feature | Type | Notes |
|---|---|---|
| sample_id | string | unique ID, see Section 9 |
| fault_class | int (0–4) | ground truth label |
| fault_severity | float or null | resistance / drift % where applicable |
| temperature_c | float | |
| vdd_v | float | |
| hold_snm_v | float | raw |
| read_snm_v | float | raw |
| write_margin_v | float | raw |
| read_access_time_ns | float | raw |
| write_time_ns | float | raw |
| bl_discharge_current_peak_ua | float | raw |
| bl_discharge_current_mean_ua | float | raw |
| standby_leakage_current_na | float | raw |
| q_voltage_swing_v | float | raw |
| read_write_time_ratio | float | **invariant**: read_access_time / write_time |
| snm_normalized | float | **invariant**: hold_snm / reference_cell_hold_snm (same-run reference) |
| bl_current_normalized | float | **invariant**: bl_discharge_current_mean / standby_leakage_current |
| convergence_ok | bool | from log parsing, Section 6.5 |
| raw_file_path | string | for traceability |

Target: 15–20 columns, this table plus 2–3 additional invariant ratios designed during Week 6–7. Before trusting the invariant features: plot raw vs. invariant feature distributions across PVT corners for the healthy class — invariant features should show visibly less spread. If they don't, the normalization isn't working and needs redesign before it's used in training.

### 6.5 Convergence checking — Member B, `simulation_runner.py`
Do not accept a simulation based on `.raw` file existence alone. Parse the `.log` file for:
```
"time step too small"
"singular matrix"
"gmin stepping failed"
"no convergence"
```
Flagged samples go into `logs/convergence_flags.csv` with sample ID and matched warning, and are **excluded** from the training dataset by default.

### 6.6 Non-ML baseline — Member C, `baseline_detector.py`
```python
def snm_threshold_detector(hold_snm, threshold) -> bool: ...
def standby_current_detector(standby_current, mean, std, n_sigma=3) -> bool: ...
```
Threshold values derived from the healthy-class distribution in the **training split only** — never the full dataset, to avoid leaking test information.

### 6.7 ML pipeline — Member C, `ml_pipeline.py`
- Models: `RandomForestClassifier`, `LogisticRegression`, `SVC(kernel="linear")`, `HistGradientBoostingClassifier`. Classical models only — with ~1,500–2,000 rows, a deep net would underfit or memorize noise rather than outperform these.
- Two evaluation protocols, both required:
  1. **Standard split**: nominal-corner data only, train/val/test, leakage-checked (automated test asserts zero sample_id overlap between splits).
  2. **Leave-one-corner-out**: train on all corners except one held-out corner, test on it. Repeat per corner.
- Run every model on both raw and invariant feature sets, same split, same hyperparameters — only the feature columns change.
- Output: one `.pkl` per (model × feature_set × protocol) into `models/`, plus `reports/final_results/comparison_table.csv` with columns: `model, feature_set, protocol, corner, accuracy, precision, recall, f1`. **This table is the paper's central evidence** — a real effect should show up across all four models, not just one.

---

## 7. Weekly task breakdown (owner-tagged)

### Week 1 — Healthy cell + parallel setup
- [ ] **[A]** Build 6T SRAM schematic in LTspice GUI; size against reference ratios
- [ ] **[A]** Manually confirm hold/read/write behavior; save waveforms to `validation/`
- [ ] **[A]** Export netlist to `circuits/sram_6t_healthy.cir`; confirm node naming (Section 6.1)
- [ ] **[B]** Scaffold `fault_injection.py` and `simulation_runner.py` against a placeholder netlist
- [ ] **[B]** Run Week 1 timing pilot (20 DC sweeps) to validate dataset-size assumptions
- [ ] **[D]** Begin independent literature search (SRAM fault diagnosis, SNM methodology, PVT-aware ML testing)

### Week 2 — SNM extraction — CRITICAL PATH
- [ ] **[A]** Implement `snm_extraction.py` per Section 6.2
- [ ] **[A]** Run 5–10 manual GUI butterfly-curve sweeps; compare against script output
- [ ] **[A]** Write `validation/snm_manual_check/results.md`
- [ ] **[D]** Continue literature review; draft `reports/literature_review.md` skeleton
- [ ] **Gate (whole team): do not proceed to Week 3 until SNM validation passes.**

### Week 3 — Fault taxonomy
- [ ] **[B]** Implement all 4 `inject_*` functions (Section 4); unit tests for netlist validity
- [ ] **[A]** Re-validate SNM extraction against a handful of post-fault-injection netlists
- [ ] **[C]** Begin `baseline_detector.py` design (doesn't require full dataset yet)

### Week 3–4 — Non-ML baseline + dataset scaffolding
- [ ] **[C]** Finish `baseline_detector.py`, run on a ~50-sample pilot batch
- [ ] **[B]** Implement `generate_dataset.py` orchestration and convergence checking (Section 6.5)

### Week 4–6 — Dataset generation
- [ ] **[B]** Run full in-distribution batch (target ~1,500–2,000 rows)
- [ ] **[B]** Run full OOD batch (all other PVT corners)
- [ ] **[B]** Manually cross-check a random 15–20 sample subset against GUI reproduction
- [ ] **[C]** Begin raw feature extraction (`extract_features.py`) as data becomes available

### Week 6–7 — Feature engineering
- [ ] **[C]** Finalize raw feature extraction
- [ ] **[C]** Implement invariant/normalized features (Section 6.4); sanity-check distribution spread
- [ ] **[D]** Draft methodology section of the paper from finalized code

### Week 7–8 — Model training
- [ ] **[C]** Implement `ml_pipeline.py`, both protocols; train all 4 models on raw features, then invariant features
- [ ] **[C]** Produce `comparison_table.csv`
- [ ] **[D]** Draft results section skeleton, pending final numbers

### Week 9 — Explainability + consolidation
- [ ] **[D]** SHAP analysis on best invariant-feature model, if time permits (stretch goal — first to cut if behind)
- [ ] **[C]** Finalize all result tables/figures
- [ ] **Whole team:** walkthrough session — every member explains SNM derivation + leave-one-corner-out logic out loud

### Week 9–10 — Writing
- [ ] **[D]** Draft full paper from `reports/final_results/`
- [ ] **[D]** Independently verify every literature citation before inclusion — no agent-generated citation goes in unread
- [ ] **Whole team:** guide review

---

## 8. Validation checkpoints and definition of done

| Checkpoint | Owner | When | What to check |
|---|---|---|---|
| Healthy cell behavior | A | End of Week 1 | Manual waveform inspection confirms correct hold/read/write |
| SNM script accuracy | A | End of Week 2 | Script vs. 5–10 manual measurements, error understood and small |
| Fault netlist validity | B | Week 3 | Each injected netlist parses and simulates without syntax errors |
| Baseline sanity | C | Week 3–4 | Threshold detector flags healthy-class samples at a low, expected false-positive rate |
| Dataset integrity | B | Week 4–6 | Zero unexplained convergence failures; manual cross-check matches automated extraction |
| Split leakage | C | Week 7 | Automated test asserts zero sample_id overlap across train/val/test |
| Result consistency | C | Week 8 | Raw-vs-invariant comparison holds across all 4 models, not just one |

A milestone is **not** complete until: (1) the code runs end-to-end without manual intervention beyond the deliberately-manual Week 1–2 steps, (2) the relevant checkpoint above has passed and is documented in `reports/`, (3) a team member other than the author has reviewed the output for plausibility.

---

## 9. Naming conventions

- Sample IDs: `SRAM_F{fault_class}_S{severity_idx}_{temp}C_{vdd}V_{run_idx:04d}` e.g. `SRAM_F2_S1_M40C_0P9V_0032`
- Netlist files: `sram_fault_{class}_{severity}.cir`
- Model files: `{model_name}_{feature_set}_{protocol}.pkl` e.g. `random_forest_invariant_leaveonecorner.pkl`
- Branch naming: `feature/<component>` e.g. `feature/snm-extraction`, `feature/fault-injection`
- Commit messages reference the milestone: `[Week 2][A] snm_extraction: implement butterfly-curve script + manual validation`

---

## 10. Git workflow

- One branch per component, owned by the member listed in Section 2/3, merged to `main` only after its validation checkpoint passes.
- Every commit adding simulation or ML results must be reproducible from a clean checkout + `config.yaml` — no local-only state.
- `data/raw_sim/` is gitignored (too large); `data/*.csv` final datasets are committed so results stay auditable.

---

## 11. Final deliverable checklist (before paper submission)

- [ ] All validation checkpoints (Section 8) documented and passed
- [ ] `comparison_table.csv` — raw vs. invariant, all 4 models, both protocols — complete
- [ ] Non-ML baseline results included in every comparison
- [ ] Every citation independently verified by Member D (not agent-generated, not left unread)
- [ ] No hardcoded local paths remain anywhere in `src/`
- [ ] `README.md` updated with setup + reproduction instructions
- [ ] Target venue selected and formatting requirements checked
- [ ] Whole-team walkthrough of SNM + leave-one-corner-out logic completed before defense
