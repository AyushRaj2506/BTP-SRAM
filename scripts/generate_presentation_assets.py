"""Generate custom presentation diagram assets for BTP slides."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set high DPI and professional font styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 18,
})


def generate_flowchart():
    """Generate professional methodology workflow diagram."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.axis("off")

    # Box styles
    box_props = dict(boxstyle="round,pad=0.6,rounding_size=0.3", facecolor="#F8FAFC", edgecolor="#2563EB", linewidth=2)
    accent_box = dict(boxstyle="round,pad=0.6,rounding_size=0.3", facecolor="#EFF6FF", edgecolor="#1D4ED8", linewidth=2.5)
    result_box = dict(boxstyle="round,pad=0.6,rounding_size=0.3", facecolor="#ECFDF5", edgecolor="#059669", linewidth=2.5)

    # Stages
    ax.text(0.12, 0.78, "STAGE 1: NETLIST & FAULT INJECTION\n• 45nm PTM CMOS 6T SRAM Core\n• Physical Faults: F1 Bridge, F2 Open,\n  F3 Oxide Leakage, F4 Vth Mismatch", 
            ha="center", va="center", bbox=box_props, fontsize=11, fontweight="bold", color="#0F172A")

    ax.text(0.50, 0.78, "STAGE 2: MULTI-CORNER SPICE HARNESS\n• 15 PVT Corners (0.9V–1.1V, -40°C–125°C)\n• Process Corners: TT / FF / SS\n• Automated LTspice Batch Simulation\n• Convergence Log Failure Auditing", 
            ha="center", va="center", bbox=box_props, fontsize=11, fontweight="bold", color="#0F172A")

    ax.text(0.88, 0.78, "STAGE 3: MULTI-DOMAIN EXTRACTION\n• Dynamic: I_DDQ, t_write, dV_BL\n• Butterfly Curves: Hold & Read SNM\n• Inscribed Square Optimization\n• 450 Sample Matrix (0 NaNs / Infs)", 
            ha="center", va="center", bbox=box_props, fontsize=11, fontweight="bold", color="#0F172A")

    ax.text(0.28, 0.25, "STAGE 4: PHYSICS-INVARIANT NORMALIZATION\n• Dual-Cell Reference Normalization (X_meas / X_ref)\n• Invariant Ratios: dV_BL/VDD, Read/Write Delay\n• Eliminates Thermal & Supply Voltage Drift\n• Zero-Variance Domain Shift Mitigation", 
            ha="center", va="center", bbox=accent_box, fontsize=11, fontweight="bold", color="#1E3A8A")

    ax.text(0.72, 0.25, "STAGE 5: ML DIAGNOSIS & BENCHMARKING\n• Models: RF, HistGB, Linear SVM, LogReg\n• Protocol 1: Standard Split (Nominal Corner)\n• Protocol 2: Cross-PVT LOCO (15 Corners)\n• Explainability: Tree SHAP Attribution\n• Verdict: 98.89% - 100% Generalization", 
            ha="center", va="center", bbox=result_box, fontsize=11, fontweight="bold", color="#064E3B")

    # Arrows
    arrow_props = dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#475569", lw=2.5)
    ax.annotate("", xy=(0.31, 0.78), xytext=(0.25, 0.78), arrowprops=arrow_props)
    ax.annotate("", xy=(0.69, 0.78), xytext=(0.63, 0.78), arrowprops=arrow_props)
    
    # Downward arrow to Stage 4
    ax.annotate("", xy=(0.28, 0.42), xytext=(0.88, 0.62),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#2563EB", lw=2.5, connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    
    # Arrow Stage 4 to Stage 5
    ax.annotate("", xy=(0.53, 0.25), xytext=(0.48, 0.25), arrowprops=arrow_props)

    plt.title("End-to-End Proposed Methodology & System Architecture", fontsize=16, fontweight="bold", pad=20, color="#0F172A")
    plt.tight_layout()
    out_path = FIG_DIR / "methodology_flowchart.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def generate_sram_circuit_diagram():
    """Generate 6T SRAM cell architecture with fault injection points."""
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.axis("off")

    # Draw visual schematic diagram of 6T SRAM
    # Background card
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.5, transform=ax.transAxes, zorder=0)
    ax.add_patch(rect)

    # Inverter 1 & 2 representation
    inv1 = plt.Rectangle((0.26, 0.35), 0.16, 0.30, facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=2.5, transform=ax.transAxes)
    inv2 = plt.Rectangle((0.58, 0.35), 0.16, 0.30, facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=2.5, transform=ax.transAxes)
    ax.add_patch(inv1)
    ax.add_patch(inv2)

    ax.text(0.34, 0.50, "INV 1\n(P1 / N1)\nM2 / M1", ha="center", va="center", fontsize=13, fontweight="bold", color="#1E3A8A")
    ax.text(0.66, 0.50, "INV 2\n(P2 / N2)\nM4 / M3", ha="center", va="center", fontsize=13, fontweight="bold", color="#1E3A8A")

    # Access transistors
    ax.plot([0.10, 0.26], [0.50, 0.50], "k-", lw=2.5)  # Left access line
    ax.plot([0.74, 0.90], [0.50, 0.50], "k-", lw=2.5)  # Right access line

    # Cross coupling
    ax.plot([0.42, 0.50, 0.50, 0.58], [0.55, 0.55, 0.45, 0.45], color="#0F172A", lw=2.5)
    ax.plot([0.58, 0.50, 0.50, 0.42], [0.55, 0.55, 0.45, 0.45], color="#0F172A", lw=2.5, ls="--")

    # Wordline
    ax.plot([0.18, 0.18], [0.50, 0.85], color="#DC2626", lw=3)
    ax.plot([0.82, 0.82], [0.50, 0.85], color="#DC2626", lw=3)
    ax.plot([0.18, 0.82], [0.85, 0.85], color="#DC2626", lw=3)
    ax.text(0.50, 0.89, "Wordline (WL)", ha="center", va="center", fontsize=13, fontweight="bold", color="#DC2626")

    # Bitlines
    ax.plot([0.10, 0.10], [0.15, 0.85], color="#059669", lw=3)
    ax.plot([0.90, 0.90], [0.15, 0.85], color="#059669", lw=3)
    ax.text(0.10, 0.90, "BL", ha="center", va="center", fontsize=14, fontweight="bold", color="#059669")
    ax.text(0.90, 0.90, "BLB", ha="center", va="center", fontsize=14, fontweight="bold", color="#059669")

    # Nodes Q and QB
    ax.text(0.24, 0.53, "Node Q", ha="right", va="center", fontsize=12, fontweight="bold", color="#0F172A")
    ax.text(0.76, 0.53, "Node QB", ha="left", va="center", fontsize=12, fontweight="bold", color="#0F172A")

    # Fault Callouts
    f_box = dict(boxstyle="round,pad=0.3", facecolor="#FEF3C7", edgecolor="#D97706", lw=1.5)
    ax.text(0.50, 0.68, "F1: Resistive Bridge (Q-QB)", ha="center", va="center", bbox=f_box, fontsize=11, fontweight="bold", color="#92400E")
    ax.text(0.18, 0.38, "F2: Open Defect\n(Access Via)", ha="center", va="center", bbox=f_box, fontsize=11, fontweight="bold", color="#92400E")
    ax.text(0.34, 0.20, "F3: Oxide Breakdown\n(Gate-Source Leakage)", ha="center", va="center", bbox=f_box, fontsize=11, fontweight="bold", color="#92400E")
    ax.text(0.66, 0.20, "F4: Vth Mismatch\n(Doping / Asymmetry)", ha="center", va="center", bbox=f_box, fontsize=11, fontweight="bold", color="#92400E")

    plt.title("45nm 6T SRAM Cell Architecture & Physical Fault Injection Sites", fontsize=16, fontweight="bold", pad=15, color="#0F172A")
    plt.tight_layout()
    out_path = FIG_DIR / "sram_architecture_faults.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def generate_snm_butterfly_diagram():
    """Generate butterfly curve VTC plot with inscribed squares."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    v = np.linspace(0, 1.0, 500)
    # Realistic sigmoid VTCs
    k = 20.0
    vm = 0.50
    vtc1 = 1.0 / (1.0 + np.exp(k * (v - vm)))
    vtc2 = 1.0 / (1.0 + np.exp(k * (v - vm)))

    # Plot normal and mirrored curves
    ax.plot(v, vtc1, label="VTC 1: V(Qb) = f(V_Q)", color="#2563EB", lw=2.5)
    ax.plot(vtc2, v, label="VTC 2: V(Q) = g(V_Qb)", color="#DC2626", lw=2.5)

    # Inscribed squares for Hold SNM
    # Upper-left lobe square
    sq1_x = [0.08, 0.36, 0.36, 0.08, 0.08]
    sq1_y = [0.64, 0.64, 0.92, 0.92, 0.64]
    ax.plot(sq1_x, sq1_y, color="#059669", lw=2, ls="--", label="Max Inscribed Square (SNM)")
    ax.fill(sq1_x, sq1_y, color="#10B981", alpha=0.15)

    # Lower-right lobe square
    sq2_x = [0.64, 0.92, 0.92, 0.64, 0.64]
    sq2_y = [0.08, 0.08, 0.36, 0.36, 0.08]
    ax.plot(sq2_x, sq2_y, color="#059669", lw=2, ls="--")
    ax.fill(sq2_x, sq2_y, color="#10B981", alpha=0.15)

    ax.text(0.22, 0.78, "Lobe 1\n(SNM_H)", ha="center", va="center", fontsize=11, fontweight="bold", color="#065F46")
    ax.text(0.78, 0.22, "Lobe 2\n(SNM_L)", ha="center", va="center", fontsize=11, fontweight="bold", color="#065F46")

    ax.set_xlabel("V(Q) [Volts]", fontweight="bold")
    ax.set_ylabel("V(Qb) [Volts]", fontweight="bold")
    ax.set_title("Butterfly Curve & Maximum Inscribed Square SNM", fontweight="bold", color="#0F172A", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="center", framealpha=0.9, fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    out_path = FIG_DIR / "snm_butterfly_diagram.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_flowchart()
    generate_sram_circuit_diagram()
    generate_snm_butterfly_diagram()
    print("All presentation visual assets generated successfully!")
