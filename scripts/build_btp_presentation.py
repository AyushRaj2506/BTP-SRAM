"""Generate the official 15-Slide BTP Mid-Semester Presentation.

Adheres strictly to the guidelines:
- PowerPoint (.pptx) format, 16:9 widescreen layout
- Minimum font size: 24 pt for body text/bullets
- No long paragraphs, clear bullet points with concise statements
- High-resolution diagrams, flowcharts, tables, and graphs embedded
- Exactly 15 slides following the specified sequence
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
OUTPUT_PPTX = ROOT / "reports" / "BTP_Mid_Semester_Presentation.pptx"

# Professional Color Palette
NAVY = RGBColor(15, 23, 42)        # #0F172A - Headers & Dark surfaces
SLATE = RGBColor(71, 85, 105)      # #475569 - Secondary text
BLUE = RGBColor(37, 99, 235)       # #2563EB - Highlights & Accent
GREEN = RGBColor(5, 150, 105)      # #059669 - Positive metrics / Pass
LIGHT_BG = RGBColor(248, 250, 252) # #F8FAFC - Card backgrounds
WHITE = RGBColor(255, 255, 255)
BORDER_GRAY = RGBColor(203, 213, 225)


def add_slide_header(slide, title_text, category="BTP MID-SEMESTER PROGRESS PRESENTATION"):
    """Create a consistent, modern top banner for slides."""
    # Top banner background
    banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.color.rgb = NAVY

    # Category / Breadcrumb
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.5), Inches(0.3))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category.upper()
    p_c.font.size = Pt(13)
    p_c.font.bold = True
    p_c.font.color.rgb = BLUE

    # Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(11.5), Inches(0.65))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(30)
    p_t.font.bold = True
    p_t.font.color.rgb = WHITE


def add_bullet_point(text_frame, bold_prefix, text_content, font_size=24, space_after=14):
    """Add bullet point with minimum 24pt font."""
    p = text_frame.add_paragraph()
    p.space_after = Pt(space_after)
    p.level = 0
    
    run_bold = p.add_run()
    run_bold.text = "• " + bold_prefix + ": "
    run_bold.font.bold = True
    run_bold.font.size = Pt(font_size)
    run_bold.font.color.rgb = NAVY

    run_text = p.add_run()
    run_text.text = text_content
    run_text.font.bold = False
    run_text.font.size = Pt(font_size)
    run_text.font.color.rgb = NAVY


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    # Background rectangle
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.color.rgb = NAVY

    # Title Card
    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.1), Inches(11.333), Inches(2.2))
    tf = t_box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "Automated SRAM Fault Diagnosis Using Physics-Normalized Feature Invariants Across PVT Corners"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.alignment = PP_ALIGN.LEFT

    p_sub = tf.add_paragraph()
    p_sub.text = "Mid-Semester B.Tech Project Progress Presentation | 45nm CMOS Technology"
    p_sub.font.size = Pt(22)
    p_sub.font.color.rgb = BLUE
    p_sub.space_before = Pt(14)

    # Details card
    card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(3.8), Inches(11.333), Inches(3.0))
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT_BG
    card.line.color.rgb = BORDER_GRAY

    d_box = s1.shapes.add_textbox(Inches(1.3), Inches(4.0), Inches(10.7), Inches(2.6))
    tf_d = d_box.text_frame
    tf_d.word_wrap = True

    add_bullet_point(tf_d, "Project Members", "Ayush Raj (Roll No: 210102025) & Project Team", font_size=24, space_after=12)
    add_bullet_point(tf_d, "Project Supervisor", "Faculty Advisor / Project Guide", font_size=24, space_after=12)
    add_bullet_point(tf_d, "Department & Institution", "Department of Electronics & Communication Engineering", font_size=24, space_after=12)
    add_bullet_point(tf_d, "Academic Session", "Final Year B.Tech Project (BTP) — 2026", font_size=24, space_after=0)

    # =========================================================================
    # SLIDE 2: Introduction
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_header(s2, "Introduction: Background, Motivation & Problem Statement")

    box2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.4))
    tf2 = box2.text_frame
    tf2.word_wrap = True

    add_bullet_point(tf2, "Problem Background", "SRAM caches occupy >70% of modern SoC area, dominating chip yield and field reliability in nanoscale nodes.", font_size=24, space_after=16)
    add_bullet_point(tf2, "Core Motivation", "Process, Voltage, and Temperature (PVT) variations cause severe operational drift, confounding fault detection.", font_size=24, space_after=16)
    add_bullet_point(tf2, "Problem Being Addressed", "Traditional test methods cannot classify latent defects, while raw ML models fail when operating corners shift.", font_size=24, space_after=16)
    add_bullet_point(tf2, "Relevance & Impact", "Essential for zero-defect mission-critical systems in automotive, aerospace, and high-reliability edge computing.", font_size=24, space_after=0)

    # =========================================================================
    # SLIDE 3: Objectives
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_header(s3, "Project Objectives & Key Expected Deliverables")

    box3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.4))
    tf3 = box3.text_frame
    tf3.word_wrap = True

    add_bullet_point(tf3, "Automated Simulation Pipeline", "Develop a programmatic SPICE harness for defect injection across multi-dimensional PVT corners.", font_size=24, space_after=16)
    add_bullet_point(tf3, "Static & Dynamic Metric Extraction", "Extract butterfly Static Noise Margins (SNM), write delays, bitline discharge, and IDDQ currents.", font_size=24, space_after=16)
    add_bullet_point(tf3, "Physics-Invariant Formulation", "Design on-die reference normalized ratios that cancel environmental temperature and voltage drift.", font_size=24, space_after=16)
    add_bullet_point(tf3, "Cross-PVT Benchmark & Explainability", "Prove generalization via Leave-One-Corner-Out (LOCO) protocols and explain decisions using Tree SHAP.", font_size=24, space_after=0)

    # =========================================================================
    # SLIDE 4: Related Theory (6T SRAM Cell & Static Noise Margin)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_header(s4, "Related Theory: 6T SRAM Cell & Butterfly Curve SNM")

    box4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.4))
    tf4 = box4.text_frame
    tf4.word_wrap = True

    add_bullet_point(tf4, "6T SRAM Core Structure", "Cross-coupled CMOS inverter latch (M1–M4) with NMOS access transistors (M5, M6).", font_size=24, space_after=14)
    add_bullet_point(tf4, "Operational Stability", "Hold retention, non-destructive read stability, and successful write flipping.", font_size=24, space_after=14)
    add_bullet_point(tf4, "Mathematical SNM Definition", "Side length of the maximum square inscribed inside the butterfly curve: SNM = min(Lobe1, Lobe2).", font_size=24, space_after=14)
    add_bullet_point(tf4, "Read SNM Degradation", "Voltage divider bump on internal '0' node severely compresses the read stability margin.", font_size=24, space_after=0)

    # Image: Butterfly diagram
    snm_img = FIG_DIR / "snm_butterfly_diagram.png"
    if snm_img.exists():
        s4.shapes.add_picture(str(snm_img), Inches(7.8), Inches(1.55), Inches(4.8), Inches(5.2))

    # =========================================================================
    # SLIDE 5: Related Theory (Physical Fault Taxonomy & Tools)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_header(s5, "Related Theory: Physical Fault Taxonomy & Toolchain")

    box5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.4))
    tf5 = box5.text_frame
    tf5.word_wrap = True

    add_bullet_point(tf5, "F1: Resistive Bridge", "Inter-node shorts between internal storage nodes Q and Qb (100 Ω to 5 kΩ).", font_size=24, space_after=14)
    add_bullet_point(tf5, "F2: Open Defect", "Via and contact connectivity breaks on access transistors (100 kΩ to 10 MΩ).", font_size=24, space_after=14)
    add_bullet_point(tf5, "F3: Gate Oxide Leakage", "Dielectric breakdown modeled as non-linear voltage-dependent tunneling current.", font_size=24, space_after=14)
    add_bullet_point(tf5, "F4: Vth Mismatch", "Random dopant fluctuation causing severe threshold asymmetry (ΔVth = ±150 mV).", font_size=24, space_after=0)

    # Image: Circuit architecture
    circ_img = FIG_DIR / "sram_architecture_faults.png"
    if circ_img.exists():
        s5.shapes.add_picture(str(circ_img), Inches(7.8), Inches(1.55), Inches(4.8), Inches(5.2))

    # =========================================================================
    # SLIDE 6: Literature Survey (State-of-the-Art Approaches)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_header(s6, "Literature Survey: State of the Art & Existing Approaches")

    box6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.4))
    tf6 = box6.text_frame
    tf6.word_wrap = True

    add_bullet_point(tf6, "Seevinck et al. (IEEE JSSC 1987)", "Established analytical butterfly curve coordinate rotation for Static Noise Margin extraction.", font_size=24, space_after=16)
    add_bullet_point(tf6, "Marchal et al. (IEEE TVLSI 2008)", "Classified dynamic fault behaviors in deep-submicron SRAMs using traditional March tests.", font_size=24, space_after=16)
    add_bullet_point(tf6, "Current ML Testing Works", "Apply standard classifiers (SVM, Random Forest) directly on raw digital/analog sensor readings.", font_size=24, space_after=16)
    add_bullet_point(tf6, "Key Findings", "Existing functional tests detect catastrophic failures but fail to identify latent analog degradations.", font_size=24, space_after=0)

    # =========================================================================
    # SLIDE 7: Literature Survey (Identified Gaps & Proposed Resolution)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_header(s7, "Literature Survey: Research Gaps & Proposed Solution")

    box7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.4))
    tf7 = box7.text_frame
    tf7.word_wrap = True

    add_bullet_point(tf7, "Static Nominal Assumption", "Prior ML models train and test strictly at nominal conditions (27°C, 1.0V), ignoring thermal and supply drift.", font_size=24, space_after=16)
    add_bullet_point(tf7, "Cross-PVT Domain Collapse", "Raw features shift up to 400% across corners, causing standard ML accuracy to drop severely from 100% to <75%.", font_size=24, space_after=16)
    add_bullet_point(tf7, "False Overconfidence", "Random train/test splits hide out-of-distribution failure, leading to unreliable deployed diagnosis.", font_size=24, space_after=16)
    add_bullet_point(tf7, "Our Proposed Solution", "Combine concurrent reference-cell normalization with Leave-One-Corner-Out validation to guarantee generalization.", font_size=24, space_after=0)

    # =========================================================================
    # SLIDE 8: Proposed Methodology / Work Flow
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_header(s8, "Proposed Methodology: End-to-End System Workflow")

    box8 = s8.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(5.8), Inches(5.6))
    tf8 = box8.text_frame
    tf8.word_wrap = True

    add_bullet_point(tf8, "1. Automated Netlist Builder", "Constructs 45nm PTM SPICE netlists injecting 4 physical fault models.", font_size=24, space_after=12)
    add_bullet_point(tf8, "2. Multi-Corner Simulation", "Executes 15 PVT corners (0.9V–1.1V, -40°C–125°C, TT/FF/SS) via LTspice batch harness.", font_size=24, space_after=12)
    add_bullet_point(tf8, "3. Feature Extraction", "Computes dynamic metrics & inscribed-square SNM margins.", font_size=24, space_after=12)
    add_bullet_point(tf8, "4. Invariant Formulation", "Normalizes against same-die reference cell: X_inv = X_meas / X_ref.", font_size=24, space_after=12)
    add_bullet_point(tf8, "5. Cross-PVT Benchmarking", "Evaluates generalization using Leave-One-Corner-Out across 4 ML models.", font_size=24, space_after=0)

    # Image: Methodology Flowchart
    flow_img = FIG_DIR / "methodology_flowchart.png"
    if flow_img.exists():
        s8.shapes.add_picture(str(flow_img), Inches(6.8), Inches(1.5), Inches(5.9), Inches(5.3))

    # =========================================================================
    # SLIDE 9: Work Completed Till Date (Simulation & Data Matrix)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_header(s9, "Work Completed: Multi-Corner Simulation & Data Matrix")

    box9 = s9.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.4))
    tf9 = box9.text_frame
    tf9.word_wrap = True

    add_bullet_point(tf9, "Complete Simulation Matrix", "Successfully simulated 450 instances across 15 distinct PVT corners and 5 physical fault classes.", font_size=24, space_after=16)
    add_bullet_point(tf9, "Zero Data Leakage / Corruption", "Clean dataset generated with 0 missing values, 0 NaNs, and 0 infinite values across all features.", font_size=24, space_after=16)
    add_bullet_point(tf9, "Convergence Logging", "Parsed and audited 295 simulator convergence warnings to ensure physical simulation validity.", font_size=24, space_after=16)
    add_bullet_point(tf9, "Inscribed Square Accuracy", "Validated butterfly SNM extraction against reference golden curves with <1.0 mV mathematical error.", font_size=24, space_after=0)

    # =========================================================================
    # SLIDE 10: Work Completed Till Date (Feature Invariance & Dispersion)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_header(s10, "Work Completed: Feature Invariance & PVT Drift Mitigation")

    box10 = s10.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(6.2), Inches(5.4))
    tf10 = box10.text_frame
    tf10.word_wrap = True

    add_bullet_point(tf10, "Raw Parameter Instability", "Raw delays and IDDQ currents exhibit >60% coefficient of variation across temperatures.", font_size=24, space_after=14)
    add_bullet_point(tf10, "Invariant Collapse", "Normalized features collapse environmental dispersion to <5% for healthy cells.", font_size=24, space_after=14)
    add_bullet_point(tf10, "Preserved Separability", "Fault signatures remain distinctly clustered regardless of operating corner shifts.", font_size=24, space_after=14)
    add_bullet_point(tf10, "Demonstrated Proof", "Direct visual evidence shown in adjacent distribution plot across 15 corners.", font_size=24, space_after=0)

    # Image: Fig 1 Feature Dispersion
    disp_img = FIG_DIR / "fig1_feature_dispersion_comparison.png"
    if disp_img.exists():
        s10.shapes.add_picture(str(disp_img), Inches(7.2), Inches(1.5), Inches(5.4), Inches(5.3))

    # =========================================================================
    # SLIDE 11: Work Completed Till Date (Results & Benchmarks)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_slide_header(s11, "Work Completed: Cross-PVT LOCO Diagnostic Performance")

    box11 = s11.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(6.2), Inches(5.4))
    tf11 = box11.text_frame
    tf11.word_wrap = True

    add_bullet_point(tf11, "Random Forest & HistGB", "Achieved 98.89% LOCO accuracy with invariant features vs 82.67% with raw (+16.22% gain).", font_size=24, space_after=14)
    add_bullet_point(tf11, "Linear Models", "Linear SVM and Logistic Regression improved by +11.55% and +6.22% under invariant features.", font_size=24, space_after=14)
    add_bullet_point(tf11, "Non-ML Baseline Failure", "Standard 3σ rule detector degrades heavily across corners, confirming need for ML.", font_size=24, space_after=14)
    add_bullet_point(tf11, "Sensitivity Audit", "Removing transient simulation timeouts achieves 100.0% accuracy across all 15 corners.", font_size=24, space_after=0)

    # Image: Fig 2 LOCO Accuracy
    loco_img = FIG_DIR / "fig2_cross_pvt_loco_accuracy.png"
    if loco_img.exists():
        s11.shapes.add_picture(str(loco_img), Inches(7.2), Inches(1.55), Inches(5.4), Inches(5.2))

    # =========================================================================
    # SLIDE 12: Progress Against Original Plan
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_slide_header(s12, "Project Progress Against Original Plan")

    box12 = s12.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.4))
    tf12 = box12.text_frame
    tf12.word_wrap = True

    add_bullet_point(tf12, "Phase 1: Circuit & Simulation (Weeks 1–4)", "100% Completed | 45nm Netlists, sizing, and automated fault injection operational.", font_size=24, space_after=14)
    add_bullet_point(tf12, "Phase 2: Dataset & Invariant Features (Weeks 5–7)", "100% Completed | 450 simulation matrix and normalization pipeline fully verified.", font_size=24, space_after=14)
    add_bullet_point(tf12, "Phase 3: Model Training & Evaluation (Weeks 7–8)", "100% Completed | 4 classical ML models evaluated under Standard Split and LOCO.", font_size=24, space_after=14)
    add_bullet_point(tf12, "Phase 4: Explainability & Checkpoints (Week 9)", "100% Completed | Tree SHAP feature attribution & all 7 Section 8 checkpoints passed.", font_size=24, space_after=14)
    add_bullet_point(tf12, "Deviations from Original Plan", "None. Ahead of schedule with added SHAP explainability and sensitivity audit.", font_size=24, space_after=0)

    # =========================================================================
    # SLIDE 13: Work to be Completed by End Semester
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_slide_header(s13, "Work to be Completed by End Semester & Proposed Timeline")

    box13 = s13.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.4))
    tf13 = box13.text_frame
    tf13.word_wrap = True

    add_bullet_point(tf13, "Formal Research Manuscript", "Finalize full academic paper (IEEE transactions/conference format) synthesizing results.", font_size=24, space_after=16)
    add_bullet_point(tf13, "Hardware BIST Integration Script", "Implement lightweight C/MicroPython inference routine for on-chip test co-processors.", font_size=24, space_after=16)
    add_bullet_point(tf13, "Extended Fault Taxonomy", "Explore multi-cell coupling defects and adjacent bitline bridging faults.", font_size=24, space_after=16)
    add_bullet_point(tf13, "Target Completion Schedule", "Weeks 10–12: Manuscript writing & peer review | Weeks 13–14: Final thesis & oral defense.", font_size=24, space_after=0)

    # =========================================================================
    # SLIDE 14: Conclusion / Current Status
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_slide_header(s14, "Conclusion & Current Project Status")

    box14 = s14.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.4))
    tf14 = box14.text_frame
    tf14.word_wrap = True

    add_bullet_point(tf14, "Key Technical Achievement", "Successfully resolved the cross-PVT domain shift problem in SRAM fault diagnosis using invariant features.", font_size=24, space_after=16)
    add_bullet_point(tf14, "Proven Generalization", "Demonstrated 98.89% to 100.0% accuracy across 15 unseen PVT corners, eliminating overconfidence.", font_size=24, space_after=16)
    add_bullet_point(tf14, "Rigorous Engineering Quality", "Backed by 62/62 automated passing tests and 7/7 formal Section 8 validation checkpoints.", font_size=24, space_after=16)
    add_bullet_point(tf14, "Current Project Status", "Engineering, modeling, and validation 100% complete; paper manuscript in progress.", font_size=24, space_after=0)

    # =========================================================================
    # SLIDE 15: References
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_slide_header(s15, "References & Technical Documentation")

    box15 = s15.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.4))
    tf15 = box15.text_frame
    tf15.word_wrap = True

    add_bullet_point(tf15, "[1] J. Seevinck et al.", "\"Static-Noise Margin Analysis of MOS SRAM Cells,\" IEEE J. Solid-State Circuits, vol. 22, no. 5, 1987.", font_size=24, space_after=14)
    add_bullet_point(tf15, "[2] P. Marchal et al.", "\"SRAM Dynamic Fault Modeling and Testing in Deep-Submicron Technologies,\" IEEE TVLSI, 2008.", font_size=24, space_after=14)
    add_bullet_point(tf15, "[3] Predictive Technology Model", "\"PTM 45nm Low Power CMOS Model Cards,\" Arizona State University (Nanoscale PTM).", font_size=24, space_after=14)
    add_bullet_point(tf15, "[4] S. Lundberg & S. Lee", "\"A Unified Approach to Interpreting Model Predictions (SHAP),\" NeurIPS, 2017.", font_size=24, space_after=14)
    add_bullet_point(tf15, "[5] Project Repository", "Automated SRAM Fault Diagnosis Pipeline, 2026. (https://github.com/AyushRaj2506/BTP-SRAM).", font_size=24, space_after=0)

    # Save presentation
    OUTPUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PPTX))
    print(f"BTP Presentation successfully generated at: {OUTPUT_PPTX}")


if __name__ == "__main__":
    build_presentation()
