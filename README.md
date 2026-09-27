# btp-sram-fault-diagnosis

Automated SRAM 6T cell fault-injection and simulation pipeline for the BTP project.

## Quick start

```bash
pip install PyLTSpice pyyaml pandas
python scripts/run_pilot_batch.py
```

## LTspice configuration

LTspice must be installed separately (it is not a Python package).
Download from: https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html

**Path resolution is automatic on first run.** `simulation_runner.py` searches all
available drive letters (C:, D:, …) and common install locations including:

- `Program Files\ADI\LTspice\LTspice.exe`  (LTspice 24, default C: install)
- `Program Files\LTC\LTspiceXVII\XVIIx64.exe`  (LTspice XVII)
- `D:\LTspice\LTspice.exe`  (bare root install on secondary drive)
- `%LOCALAPPDATA%\Programs\ADI\LTspice\LTspice.exe`  (per-user install)
- Any location on the system PATH

When LTspice is found, the path is printed and **automatically written into
`config.yaml`** so the next run skips detection entirely.

### If auto-detection fails on your machine

Do **not** edit any Python file. Instead, open (or create) `config.yaml` in the
project root and set the single field:

```yaml
ltspice_path: 'D:\LTspice\LTspice.exe'   # ← replace with your actual path
```

`config.yaml` is gitignored — your local path will never overwrite a teammate's.
Use `config.example.yaml` (committed) as a template.

## Repository layout

```
circuits/
  core/            ← cell_core_healthy.net (M1–M6 template)
  testbenches/     ← stimulus_hold/write/read.net
src/
  fault_injection.py      ← inject_resistive_open, inject_bridging_fault, inject_vth_drift
  testbench_composer.py   ← compose_testbench()
  simulation_runner.py    ← run_simulation(), run_batch()
  utils.py                ← load_cell_core(), save_cell_core()
scripts/
  run_pilot_batch.py      ← 5-simulation pilot batch (Task 7)
tests/
  test_fault_injection.py
  test_testbench_composer.py
config.example.yaml   ← committed template (copy to config.yaml locally)
config.yaml           ← gitignored, machine-specific
```

## Running tests

```bash
pytest tests/ -v
```

All 18 tests should pass on any machine without LTspice installed
(the tests cover fault injection and testbench composition only,
not actual simulation).
