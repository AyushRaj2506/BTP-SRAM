"""Fig 1b: cross-corner dispersion of FAULT-class signatures, raw vs normalized.

For every fault condition (class, target, severity) the same fault is simulated at
15 PVT corners. We compute the coefficient of variation (CV, sample std, ddof=1)
of each feature across those 15 corners, then average per fault class.
Lower CV = the feature depends less on PVT.

Run from the repo root:  python scripts/redraw_fig1_faults.py
Reads:  data/sram_fault_dataset.csv     Writes: reports/figures/fig1b_fault_dispersion.png
        reports/final_results/fault_feature_spread.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sram_fault_dataset.csv"
FIG = ROOT / "reports" / "figures" / "fig1b_fault_dispersion.png"
CSV = ROOT / "reports" / "final_results" / "fault_feature_spread.csv"

PAIRS = [  # (title, raw column, invariant column)
    ("Standby leakage I_ddq", "i_ddq_uA", "iddq_norm"),
    ("Write time t_write", "t_write_ps", "t_write_norm"),
    ("Hold SNM", "snm_hold_v", "snm_hold_norm"),
    ("Sense voltage dV_BL", "dv_bl_strobe_mV", "dv_bl_norm"),
]
CLASS_NAMES = {1: "Resistive\nopen", 2: "Q-QB\nbridge", 3: "Vth drift\n(storage)", 4: "Vth drift\n(access)"}


def cv_percent(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    return np.nan if abs(m) < 1e-12 else 100.0 * x.std(ddof=1) / abs(m)


def main():
    df = pd.read_csv(DATA)
    df = df[df["fault_class"].isin([1, 2, 3, 4])]
    keys = ["fault_class", "fault_target", "severity_value"]

    rows = []
    for k, g in df.groupby(keys):
        if g["corner_id"].nunique() < 2:
            continue
        for title, raw, inv in PAIRS:
            rows.append({"fault_class": k[0], "fault_target": k[1], "severity_value": k[2],
                         "feature": title, "n_corners": g["corner_id"].nunique(),
                         "cv_raw_pct": cv_percent(g[raw]), "cv_inv_pct": cv_percent(g[inv])})
    per_cond = pd.DataFrame(rows)
    per_class = (per_cond.groupby(["feature", "fault_class"])[["cv_raw_pct", "cv_inv_pct"]]
                 .mean().reset_index())

    CSV.parent.mkdir(parents=True, exist_ok=True)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(CSV, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    w = 0.38
    for ax, (title, _, _) in zip(axes.ravel(), PAIRS):
        s = per_class[per_class["feature"] == title].set_index("fault_class").reindex([1, 2, 3, 4])
        x = np.arange(4)
        ax.bar(x - w / 2, s["cv_raw_pct"], w, label="Raw", color="#d95f02")
        ax.bar(x + w / 2, s["cv_inv_pct"], w, label="Invariant", color="#1b9e77")
        ax.set_xticks(x)
        ax.set_xticklabels([CLASS_NAMES[c] for c in [1, 2, 3, 4]], fontsize=9)
        ax.set_ylabel("Cross-corner CV (%)")
        ax.set_title(title)
        ax.grid(axis="y", ls="--", alpha=0.4)
    axes[0, 0].legend()
    fig.suptitle("Fault-signature dispersion across 15 PVT corners (mean CV per fault class, ddof=1)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=300)
    print(f"Saved {FIG}")
    print(f"Saved {CSV}")
    print(per_class.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
