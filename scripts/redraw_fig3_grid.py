"""Fig 3b: Random Forest LOCO accuracy as a VDD x temperature grid (raw vs invariant).

Run from the repo root:  python scripts/redraw_fig3_grid.py
Reads:  reports/final_results/comparison_table.csv
Writes: reports/figures/fig3b_rf_grid.png
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "reports" / "final_results" / "comparison_table.csv"
FIG = ROOT / "reports" / "figures" / "fig3b_rf_grid.png"

TEMPS = [-40, 0, 27, 75, 125]
VDDS = [0.9, 1.0, 1.1]


def parse_corner(cid):
    """'C_M40C_0P9V' -> (-40, 0.9);  'C_125C_1P1V' -> (125, 1.1)"""
    m = re.match(r"C_(M?)(\d+)C_(\d)P(\d)V", cid)
    if not m:
        raise ValueError(f"Unrecognised corner id: {cid}")
    temp = -int(m.group(2)) if m.group(1) == "M" else int(m.group(2))
    return temp, float(f"{m.group(3)}.{m.group(4)}")


def main():
    t = pd.read_csv(TABLE)
    t = t[(t["model"] == "RandomForest") & (t["protocol"] == "leave_one_corner_out")].copy()
    t[["temp", "vdd"]] = t["corner"].apply(lambda c: pd.Series(parse_corner(c)))

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, fs in zip(axes, ["raw", "invariant"]):
        s = t[t["feature_set"] == fs]
        grid = np.full((len(VDDS), len(TEMPS)), np.nan)
        for _, r in s.iterrows():
            grid[VDDS.index(r["vdd"]), TEMPS.index(r["temp"])] = r["accuracy"]
        im = ax.imshow(grid, vmin=0.5, vmax=1.0, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(TEMPS)))
        ax.set_xticklabels([f"{x}" for x in TEMPS])
        ax.set_yticks(range(len(VDDS)))
        ax.set_yticklabels([f"{v:.1f}" for v in VDDS])
        ax.set_xlabel("Held-out temperature (°C)")
        ax.set_title(f"{fs.capitalize()} features")
        for i in range(len(VDDS)):
            for j in range(len(TEMPS)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white" if v > 0.85 else "black", fontsize=9)
    axes[0].set_ylabel("Held-out VDD (V)")
    fig.colorbar(im, ax=axes, label="LOCO accuracy")
    fig.suptitle("Random Forest LOCO accuracy by held-out PVT corner")
    fig.savefig(FIG, dpi=300, bbox_inches="tight")
    print(f"Saved {FIG}")


if __name__ == "__main__":
    main()
