# FAULT_INJECTION_PLAN.md — Member B: Automated Fault Injection Pipeline

**Prerequisite (done):** `circuits/sram_6t_healthy.asc` — the 6T cell validated three ways: HOLD (retention confirmed both states), WRITE (flip confirmed + persists after WL low), READ (non-destructive, correct differential, ~2.11ns access time). All three used real `.raw`-verified data, not visual inspection.

**This document is written to be handed to Antigravity section by section.** Each numbered task below ends with a literal prompt block — paste it in as-is, in order. Do not skip ahead; each prompt assumes the previous one's output exists and is correct.

---

## 0. The key architectural decision — read this before prompting anything

Your three validated netlists (HOLD/WRITE/READ) each combine the same **cell core** (M1–M6 + `.model` lines) with a *different* stimulus setup around it (different `.ic`, different WL/BL/BLB source types). Fault injection should not touch the stimulus setup at all — it only ever modifies the cell core. So the pipeline has two separate, composable pieces:

1. **A fault-injection layer** that takes the healthy cell core and produces a *faulty cell core* (resistive open, bridging, Vth drift) — pure text editing on the M1–M6 lines only.
2. **A testbench-composition layer** that wraps *any* cell core (healthy or faulty) with a chosen stimulus (HOLD-style, WRITE-style, or READ-style), producing a runnable `.net` file.

This is what lets one faulty cell get measured under all three testbenches without duplicating fault-injection logic three times, and it's why you extract the cell core as its own template first, rather than editing three separate full netlists in three separate ways.

---

## 1. Folder structure (extends the structure in `PLAN.md`)

```
btp-sram-fault-diagnosis/
├── circuits/
│   ├── sram_6t_healthy.asc              <- Member A's validated schematic (reference only, don't edit)
│   ├── core/
│   │   └── cell_core_healthy.net        <- extracted M1–M6 + .model lines ONLY (Task 1 output)
│   └── testbenches/
│       ├── stimulus_hold.net            <- HOLD .ic + source lines, cell-core-agnostic (Task 2)
│       ├── stimulus_write.net           <- WRITE .ic + PULSE + driven-bitline lines (Task 2)
│       └── stimulus_read.net            <- READ .ic + PULSE + precharged-capacitor lines (Task 2)
├── src/
│   ├── fault_injection.py               <- Task 3: inject_* functions on cell_core
│   ├── testbench_composer.py            <- Task 4: combine core + stimulus -> runnable .net
│   ├── simulation_runner.py             <- Task 6: PyLTSpice batch execution + convergence check
│   └── utils.py                         <- shared netlist parsing helpers
├── tests/
│   ├── test_fault_injection.py          <- Task 5: gate≠drain check, node-mapping check, diff check
│   └── test_testbench_composer.py
├── data/
│   ├── generated_netlists/              <- output of fault_injection + composer, gitignored (large)
│   ├── raw_sim/                         <- .raw/.log outputs, gitignored
│   └── pilot_batch/                     <- Task 7 pilot results, small enough to commit
├── logs/
│   └── convergence_flags.csv
└── validation/
    └── fault_injection_manual_check/    <- Task 8: manual cross-check outputs
```

---

## 2. Task-by-task plan with Antigravity prompts

### Task 1 — Extract the cell core template

**Goal:** a standalone `.net` fragment containing only M1–M6 and the two `.model` lines, with no stimulus sources, no `.ic`, no `.tran`. This is the thing every fault-injection function will edit.

**Antigravity prompt:**
```
I have a validated SPICE netlist for a 6T SRAM cell. I need you to extract just the
transistor and model definitions into a standalone template file, with no stimulus,
no initial conditions, and no simulation commands.

Here is the confirmed-correct netlist to extract from (this exact topology has been
validated by HOLD, WRITE, and READ simulations — do not change any node names, pin
order, or transistor parameters):

M1 Qb Q Vdd Vdd SRAM_PMOS l=180nm w=360nm
M2 Q Qb Vdd Vdd SRAM_PMOS l=180nm w=360nm
M3 Q Qb 0 0 SRAM_NMOS l=180nm w=720nm
M4 Qb Q 0 0 SRAM_NMOS l=180nm w=720nm
M5 Q WL BL 0 SRAM_NMOS l=180nm w=540nm
M6 BLB WL Qb 0 SRAM_NMOS l=180nm w=540nm
.model NMOS NMOS
.model PMOS PMOS
.model SRAM_NMOS NMOS (LEVEL=1 VTO=0.45 KP=250u GAMMA=0.4 LAMBDA=0.08 PHI=0.7 TOX=4n)
.model SRAM_PMOS PMOS (LEVEL=1 VTO=-0.45 KP=100u GAMMA=0.4 LAMBDA=0.1 PHI=0.7 TOX=4n)

Tasks:
1. Save this exactly as circuits/core/cell_core_healthy.net, preserving every line
   and every field exactly as given above — do not reformat, reorder, or "clean up"
   the node names.
2. Write a short Python function in src/utils.py called `load_cell_core(path)` that
   reads this file and returns it as a list of strings (one per line), so other
   scripts can load and modify it without re-parsing SPICE syntax each time.
3. Write a second function `save_cell_core(lines, path)` that writes a list of
   strings back out to a .net file, one line per entry, no trailing modifications.

Do not add a V(Vdd) source, .ic, or .tran to this file — it is a template fragment
only, not a runnable netlist on its own.
```

---

### Task 2 — Extract the three stimulus fragments

**Goal:** separate, reusable stimulus blocks — the parts of HOLD/WRITE/READ that are *not* the cell core — so any cell core (healthy or faulty) can be dropped into any of the three.

**Antigravity prompt:**
```
I need to separate the "stimulus" portion of three validated SPICE testbenches from
the "cell" portion, so the cell definition can be swapped independently of the test
stimulus. The cell portion (M1–M6 and .model lines) is already extracted separately
and will NOT appear in these files — these three files should contain everything
else: the supply source, WL/BL/BLB sources, .ic, and .tran lines only.

Here is the confirmed HOLD stimulus (extract everything except the M1–M6 and .model
lines):
V4 Vdd 0 1V
V2 WL 0 0V
V1 BL 0 1V
V3 BLB 0 1V
.tran 0 20n 0 10p uic
.ic V(Q)=1V V(Qb)=0V

Here is the confirmed WRITE stimulus:
V4 Vdd 0 1V
V2 WL 0 PULSE(0 1 2n 0.1n 0.1n 5n 20n)
V1 BL 0 1V
V3 BLB 0 0V
.tran 0 20n 0 10p uic
.ic V(Q)=0V V(Qb)=1V

Here is the confirmed READ stimulus:
V4 Vdd 0 1V
V2 WL 0 PULSE(0 1 2n 0.1n 0.1n 5n 20n)
Cbl BL 0 20fF IC=1
Cblb BLB 0 20fF IC=1
.tran 0 20n 0 10p uic
.ic V(Q)=1V V(Qb)=0V V(BL)=1V V(BLB)=1V

Tasks:
1. Save these three blocks exactly as given into:
   circuits/testbenches/stimulus_hold.net
   circuits/testbenches/stimulus_write.net
   circuits/testbenches/stimulus_read.net
2. Do not change any node names, source values, or PULSE parameters — these have
   been validated against real simulation output and must not be "corrected" or
   reformatted.
3. Note for later: WRITE's .ic writes the OPPOSITE value to what BL/BLB are driving
   (testing a flip). READ's .ic matches what's already stored (testing retention +
   non-destructive readout). Keep this pairing intact if these are ever edited.
```

---

### Task 3 — Fault injection functions

**Goal:** the actual `inject_*` functions, operating only on the cell-core lines, per the taxonomy in `PLAN.md` Section 4.

**Antigravity prompt:**
```
Write src/fault_injection.py implementing three fault-injection functions that
operate on the SRAM cell core template at circuits/core/cell_core_healthy.net.
Each function must:
- Load the healthy core using load_cell_core() from src/utils.py
- Modify ONLY the specific line(s) needed for that fault — leave every other line
  byte-for-byte identical to the healthy template
- Save the result using save_cell_core() to a new file, never overwrite the
  healthy template
- Return the path to the new faulty netlist file

The healthy core's exact M-lines are:
M1 Qb Q Vdd Vdd SRAM_PMOS l=180nm w=360nm
M2 Q Qb Vdd Vdd SRAM_PMOS l=180nm w=360nm
M3 Q Qb 0 0 SRAM_NMOS l=180nm w=720nm
M4 Qb Q 0 0 SRAM_NMOS l=180nm w=720nm
M5 Q WL BL 0 SRAM_NMOS l=180nm w=540nm
M6 BLB WL Qb 0 SRAM_NMOS l=180nm w=540nm

Implement these three functions:

1. inject_resistive_open(core_lines, transistor="M5", resistance_ohms=1000,
   output_path=...) -> str
   Insert a series resistor between the named transistor's source terminal and
   the node it currently connects to, breaking that direct connection. For
   example, for M5 (currently "M5 Q WL BL 0 SRAM_NMOS ..."), this means:
   creating a new intermediate node (e.g. "BL_r"), changing M5's source field
   from BL to BL_r, and adding a new line "Rfault BL BL_r {resistance_ohms}"
   Do this generically so it works for M5 or M6, whichever is passed in.

2. inject_bridging_fault(core_lines, resistance_ohms=2000, output_path=...) -> str
   Add a single new line: "Rbridge Q Qb {resistance_ohms}" to the core lines,
   with no other changes. This creates a resistive path directly between the
   two storage nodes.

3. inject_vth_drift(core_lines, target="storage_pair", drift_pct=10,
   output_path=...) -> str
   target="storage_pair" modifies the SRAM_NMOS and SRAM_PMOS .model lines used
   by M1–M4 by adding drift_pct% to their VTO magnitude (increase VTO=0.45 for
   NMOS, make VTO=-0.45 more negative for PMOS, preserving sign).
   target="access" does the same but only to a NEW pair of .model lines used
   exclusively by M5/M6 (you will need to duplicate the SRAM_NMOS model under a
   new name like SRAM_NMOS_ACCESS_DRIFT and repoint M5/M6's model reference to
   it, leaving M1–M4's model untouched).
   Compute the drifted VTO value from the ORIGINAL 0.45 magnitude each time —
   do not compound drift from a previously-drifted file.

Add a short docstring to each function stating which PLAN.md fault class (1–4)
it implements. Do not add any GUI-interaction code, and do not modify anything
outside the cell core lines.
```

---

### Task 4 — Testbench composer

**Goal:** glue a (healthy or faulty) core to one of the three stimulus files, producing a runnable `.net`.

**Antigravity prompt:**
```
Write src/testbench_composer.py with one function:

compose_testbench(core_path, stimulus_type, output_path) -> str

Where stimulus_type is one of "hold", "write", "read", mapping to
circuits/testbenches/stimulus_{stimulus_type}.net.

The function should:
1. Read the core file at core_path (a fault-injected or healthy cell-core .net)
2. Read the matching stimulus file
3. Concatenate them into a single valid SPICE deck: core lines first, then
   stimulus lines, then a final ".end" line
4. Write the result to output_path
5. Return output_path

Add a basic sanity check before writing: scan the combined lines and confirm
every node referenced in a stimulus V/C source line (Vdd, WL, BL, BLB) also
appears somewhere in the core lines, and vice versa for Q/Qb — if a stimulus
file references a node the core doesn't define (or the reverse), raise a clear
ValueError naming the missing node, rather than silently writing a broken
netlist. This is meant to catch typos before a wasted simulation run.
```

---

### Task 5 — Unit tests (the two checks that caught every real bug so far)

**Antigravity prompt:**
```
Write tests/test_fault_injection.py and tests/test_testbench_composer.py using
pytest. Implement these specific checks, which are based on real bugs found
during manual circuit validation on this project — do not skip or simplify them:

1. test_gate_never_equals_drain: for every generated netlist (healthy core AND
   every fault-injected variant), parse every M-line and assert the 1st field
   (drain) is never identical to the 2nd field (gate). If any transistor has
   drain==gate, fail with a message naming the offending line — this exact bug
   (a self-gated transistor from an accidental schematic edit) broke our HOLD
   test twice during manual development.

2. test_node_mapping_preserved: assert that in every generated netlist, M5's
   drain is "Q" and its source is "BL", and M6's drain is "BLB" and its source
   is "Qb" — UNLESS the fault being tested is specifically a fault that's
   supposed to rename these nodes (e.g. inject_resistive_open on M5/M6 renames
   one endpoint to an intermediate node like "BL_r" — in that case, check the
   OTHER unmodified terminal still matches). This catches accidental
   reintroduction of the swapped-node bug we hit multiple times.

3. test_fault_netlist_differs_minimally: for each inject_* function, generate a
   faulty netlist and diff it line-by-line against the healthy core. Assert
   that ONLY the expected line(s) changed (or one new line was added, for
   bridging faults and resistive-open faults) — fail if unrelated lines
   differ, which would indicate an unintended side effect.

4. test_composer_rejects_dangling_nodes: call compose_testbench with a
   deliberately broken core (rename "Q" to "Qx" in a copy) and assert it
   raises ValueError rather than silently producing a broken netlist.

Run these with `pytest tests/` and make sure all four pass against the healthy
core and against a resistive-open, a bridging, and a Vth-drift variant before
moving on.
```

---

### Task 6 — Batch simulation runner with convergence checking

**Antigravity prompt:**
```
Write src/simulation_runner.py using PyLTSpice to batch-run a list of composed
.net files headlessly and collect results, with convergence checking — do not
accept a simulation as valid just because a .raw file exists.

Implement:

def run_simulation(net_path, raw_output_dir, log_output_dir) -> dict
  Runs LTspice on net_path via PyLTSpice, writes the .raw and .log to the given
  directories, and returns a dict: {"raw_path": ..., "log_path": ...,
  "converged": bool, "warning": str or None}

The "converged" check must parse the .log file text and set converged=False if
ANY of these substrings appear (case-insensitive):
  "time step too small"
  "singular matrix"
  "gmin stepping failed"
  "no convergence"
If found, set "warning" to the matched string. Otherwise converged=True,
warning=None. Do not rely on .raw file existence alone — a corrupted or
truncated .raw can still be written even when one of these warnings fires.

def run_batch(net_paths: list, raw_output_dir, log_output_dir,
              flagged_csv_path) -> pandas.DataFrame
  Calls run_simulation on each path, collects results into a DataFrame with
  columns [net_path, raw_path, log_path, converged, warning], and writes any
  row where converged=False to flagged_csv_path (append mode, so repeated runs
  don't overwrite earlier flags). Print a one-line summary at the end: total
  runs, how many converged, how many were flagged.

Use a fixed random seed anywhere randomness is involved (there shouldn't be any
in this file, but if PyLTSpice needs a temp-file naming scheme, make it
deterministic based on the input filename, not random, so reruns are
reproducible).
```

---

### Task 7 — Pilot batch (small, before the real dataset)

**Antigravity prompt:**
```
Write a short script scripts/run_pilot_batch.py that:
1. Generates all 5 fault classes from PLAN.md (healthy + 4 fault types) using
   the functions in src/fault_injection.py, using ONE representative severity
   per fault type (pick the middle value from PLAN.md's severity list for each)
2. Composes each of those 5 cell cores with the "hold" stimulus only (not write
   or read yet — keep this pilot small) using src/testbench_composer.py
3. Runs all 5 through src/simulation_runner.py's run_batch()
4. Prints a table: fault_class, converged, final V(Q), final V(Qb) at t=20ns
   (read these directly from each .raw using PyLTSpice's RawRead, the same way
   we verified HOLD/WRITE/READ manually)
5. Saves all outputs under data/pilot_batch/

This is a sanity check, not the real dataset — 5 simulations only. The purpose
is to visually confirm each fault type produces a plausibly different final
state before committing to the full PVT-swept batch in PLAN.md's Week 4–6.
```

---

## 3. What to do with the pilot batch output

Send me the printed table from Task 7, plus the `.net` for whichever fault (if any) looks physically implausible — e.g. a bridging fault that doesn't move `V(Q)`/`V(Qb)` toward each other at all, or a resistive-open that has zero effect. Same verification approach as HOLD/WRITE/READ: I'll check the actual numbers, not just whether the script ran without crashing.

## 4. Do not proceed to the full PVT-swept dataset (PLAN.md Week 4–6) until

- [ ] All four `tests/` checks (Task 5) pass on all 5 fault classes
- [ ] The pilot batch (Task 7) produces 5/5 converged simulations with plausible, physically-sensible final states
- [ ] At least one fault-injected netlist has been manually cross-checked by opening it directly in the LTspice GUI and confirming the injected fault is visibly present (same discipline as the HOLD/WRITE/READ manual checks)
