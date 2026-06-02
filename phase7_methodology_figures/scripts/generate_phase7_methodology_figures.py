#!/usr/bin/env python3

"""
Phase 7 Methodology Figure Generator

This script generates professor-facing, paper-style methodology diagrams
for the ArduPilot SITL software-sensor attack detection and recovery project.

Outputs:
- 5 PNG methodology figures
- 5 PDF methodology figures
- README.md
- phase7_latex_snippets.tex
- phase7_section_4_2_2_methodology_summary.md

Output folder:
    /home/tchowdh4/sensor_recovery_project/phase7_methodology_figures
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import matplotlib as mpl


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path("/home/tchowdh4/sensor_recovery_project")
OUT_DIR = PROJECT_ROOT / "phase7_methodology_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Matplotlib style
# =============================================================================

mpl.rcParams["figure.dpi"] = 160
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 10


# =============================================================================
# Drawing helpers
# =============================================================================

def setup_canvas(width=15, height=8.5, title=None):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if title:
        ax.text(
            0.5,
            0.965,
            title,
            ha="center",
            va="top",
            fontsize=15,
            fontweight="bold",
        )

    return fig, ax


def add_box(
    ax,
    xy,
    w,
    h,
    text,
    fontsize=9.2,
    facecolor="#FFFFFF",
    edgecolor="#222222",
    linewidth=1.3,
    radius=0.02,
    fontweight="normal",
    linestyle="-",
):
    x, y = xy

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        wrap=True,
    )

    return patch


def add_decision(ax, center, w, h, text, fontsize=9.2):
    cx, cy = center

    points = [
        (cx, cy + h / 2),
        (cx + w / 2, cy),
        (cx, cy - h / 2),
        (cx - w / 2, cy),
    ]

    patch = Polygon(
        points,
        closed=True,
        facecolor="#FFFFFF",
        edgecolor="#222222",
        linewidth=1.3,
    )
    ax.add_patch(patch)

    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )

    return patch


def add_arrow(
    ax,
    start,
    end,
    text=None,
    fontsize=8.3,
    connectionstyle="arc3,rad=0.0",
    linewidth=1.25,
    mutation_scale=14,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color="#222222",
        connectionstyle=connectionstyle,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)

    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2

        ax.text(
            mx,
            my + 0.018,
            text,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
        )

    return arrow


def add_group_label(ax, x, y, text):
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            facecolor="#F5F5F5",
            edgecolor="#BBBBBB",
            boxstyle="round,pad=0.25",
        ),
    )


def save_figure(fig, name):
    png_path = OUT_DIR / f"{name}.png"
    pdf_path = OUT_DIR / f"{name}.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved {png_path}")
    print(f"[OK] Saved {pdf_path}")


# =============================================================================
# Figure 1
# =============================================================================

def make_figure1():
    fig, ax = setup_canvas(
        title="Figure 1. Software-Sensor-Based Attack Detection and Recovery Pipeline"
    )

    add_group_label(ax, 0.04, 0.87, "Runtime detection and recovery architecture")

    add_box(
        ax,
        (0.04, 0.63),
        0.17,
        0.12,
        "Sensor measurement stream\nclean or attacked\n$x_{measured}[k+1]$",
        facecolor="#EAF2FF",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.04, 0.38),
        0.17,
        0.12,
        "Information available\nat time $k$\nhistory + aligned features",
        facecolor="#EAF2FF",
    )

    add_box(
        ax,
        (0.28, 0.38),
        0.18,
        0.12,
        "V9 trained hybrid\nsoftware sensor\npredicts $\\hat{x}_{V9}[k+1]$",
        facecolor="#ECFFEC",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.51, 0.53),
        0.18,
        0.14,
        "Residual calculation\n\n$r[k+1] = x_{measured}[k+1]\n- \\hat{x}_{V9}[k+1]$",
        facecolor="#FFF7E6",
        fontweight="bold",
        fontsize=8.8,
    )

    add_box(
        ax,
        (0.51, 0.31),
        0.18,
        0.11,
        "Clean residual threshold\n$p99\\_abs$",
        facecolor="#FFF7E6",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.74, 0.53),
        0.18,
        0.14,
        "Window detector\nW10_N3_mean1x\n\n10-sample window\nat least 3 violations",
        facecolor="#FFECEC",
        fontweight="bold",
        fontsize=8.8,
    )

    add_decision(
        ax,
        center=(0.83, 0.35),
        w=0.16,
        h=0.13,
        text="Attack\ndetected?",
    )

    add_decision(
        ax,
        center=(0.83, 0.17),
        w=0.17,
        h=0.12,
        text="Recovery\nswitch",
    )

    add_box(
        ax,
        (0.58, 0.08),
        0.17,
        0.09,
        "Use V9 prediction\n$\\hat{x}_{V9}[k+1]$",
        facecolor="#ECFFEC",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.86, 0.08),
        0.12,
        0.09,
        "Use measured\nvalue",
        facecolor="#EAF2FF",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.76, 0.78),
        0.20,
        0.10,
        "Recovered output\n$x_{recovered}[k+1]$\nfor evaluation/control",
        facecolor="#EFFFFA",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.36, 0.78),
        0.27,
        0.08,
        "Roll/Pitch residual: direct difference\nYaw residual: circular angular difference",
        facecolor="#F7F7F7",
        fontsize=8.6,
        linestyle="--",
    )

    add_arrow(ax, (0.21, 0.69), (0.51, 0.60), "measured future state")
    add_arrow(ax, (0.21, 0.44), (0.28, 0.44), "features")
    add_arrow(ax, (0.46, 0.44), (0.51, 0.57), "prediction")
    add_arrow(ax, (0.60, 0.53), (0.60, 0.42), "compare")
    add_arrow(ax, (0.69, 0.60), (0.74, 0.60), "$|r| > p99\\_abs$")
    add_arrow(ax, (0.83, 0.53), (0.83, 0.415), "window result")
    add_arrow(ax, (0.83, 0.285), (0.83, 0.23), "flag")
    add_arrow(ax, (0.79, 0.17), (0.75, 0.125), "yes")
    add_arrow(ax, (0.87, 0.17), (0.86, 0.125), "no")
    add_arrow(ax, (0.67, 0.17), (0.76, 0.80), connectionstyle="arc3,rad=-0.25")
    add_arrow(ax, (0.92, 0.17), (0.88, 0.78), connectionstyle="arc3,rad=0.25")

    ax.text(
        0.50,
        0.02,
        "Recovery rule: if attack_detected, use $\\hat{x}_{V9}[k+1]$; otherwise use $x_{measured}[k+1]$.",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    save_figure(fig, "figure1_detection_recovery_pipeline")


# =============================================================================
# Figure 2
# =============================================================================

def make_figure2():
    fig, ax = setup_canvas(
        title="Figure 2. Training and Deployment Workflow"
    )

    add_group_label(ax, 0.04, 0.87, "Offline training and calibration")
    add_group_label(ax, 0.04, 0.36, "Deployment on attacked logs")

    y1 = 0.64
    w = 0.15
    h = 0.11

    blocks1 = [
        (0.04, "Clean baseline logs\nATT / IMU\nSITL flights"),
        (0.23, "Feature extraction\nand alignment\navailable at time $k$"),
        (0.42, "Train/evaluate\ncandidate predictors\nnaive, CV, ARX, hybrid"),
        (0.61, "Select final predictor\nV9 trained hybrid"),
        (0.80, "Generate clean residuals\nand compute\n$p99\\_abs$ threshold"),
    ]

    for x, text in blocks1:
        add_box(
            ax,
            (x, y1),
            w,
            h,
            text,
            facecolor="#EAF2FF" if x < 0.61 else "#ECFFEC",
            fontweight="bold" if x >= 0.61 else "normal",
            fontsize=8.6,
        )

    for i in range(len(blocks1) - 1):
        add_arrow(
            ax,
            (blocks1[i][0] + w, y1 + h / 2),
            (blocks1[i + 1][0], y1 + h / 2),
        )

    y2 = 0.17

    blocks2 = [
        (0.04, "Attacked logs\nmeasurement stream"),
        (0.24, "Deploy V9 predictor\nand $p99\\_abs$ threshold"),
        (0.44, "Detect attack\nW10_N3_mean1x"),
        (0.64, "Recover attacked signal\nusing V9 prediction"),
        (0.84, "Evaluate\nRMSE / MAE\nimprovement"),
    ]

    for x, text in blocks2:
        add_box(
            ax,
            (x, y2),
            0.14,
            0.12,
            text,
            facecolor="#FFECEC" if x < 0.64 else "#EFFFFA",
            fontweight="bold" if x >= 0.44 else "normal",
            fontsize=8.6,
        )

    for i in range(len(blocks2) - 1):
        add_arrow(
            ax,
            (blocks2[i][0] + 0.14, y2 + 0.06),
            (blocks2[i + 1][0], y2 + 0.06),
        )

    add_arrow(
        ax,
        (0.685, y1),
        (0.31, y2 + 0.12),
        "selected V9 model",
        connectionstyle="arc3,rad=-0.25",
    )

    add_arrow(
        ax,
        (0.875, y1),
        (0.32, y2 + 0.12),
        "$p99\\_abs$ threshold",
        connectionstyle="arc3,rad=-0.18",
    )

    ax.text(
        0.5,
        0.05,
        "Clean logs are used to learn normal behavior and calibrate the detector; attacked logs are used for detection, recovery, and final evaluation.",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
    )

    save_figure(fig, "figure2_training_deployment_workflow")


# =============================================================================
# Figure 3
# =============================================================================

def make_figure3():
    fig, ax = setup_canvas(
        title="Figure 3. V9 Hybrid Software Sensor Internal Structure"
    )

    add_group_label(ax, 0.04, 0.87, "Candidate predictors")
    add_group_label(ax, 0.62, 0.87, "Validation-weighted fusion")

    add_box(
        ax,
        (0.05, 0.56),
        0.19,
        0.15,
        "Inputs available\nat time $k$\n\nATT/IMU history\nprevious states\naligned features",
        facecolor="#EAF2FF",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.34, 0.72),
        0.18,
        0.10,
        "Naive predictor\n$\\hat{x}_{naive}[k+1]$",
        facecolor="#F7F7F7",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.34, 0.55),
        0.18,
        0.10,
        "Constant-velocity predictor\n$\\hat{x}_{cv}[k+1]$",
        facecolor="#F7F7F7",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.34, 0.38),
        0.18,
        0.10,
        "ARX-style predictor\n$\\hat{x}_{arx}[k+1]$",
        facecolor="#F7F7F7",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.64, 0.57),
        0.18,
        0.13,
        "Validation-based\nweights\n$w_{naive}, w_{cv}, w_{arx}$",
        facecolor="#FFF7E6",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.58, 0.29),
        0.30,
        0.18,
        "Weighted fusion\n\n$\\hat{x}_{V9}[k+1] =$\n$w_{naive}\\hat{x}_{naive}[k+1]$\n$+ w_{cv}\\hat{x}_{cv}[k+1]$\n$+ w_{arx}\\hat{x}_{arx}[k+1]$",
        facecolor="#ECFFEC",
        fontweight="bold",
        fontsize=9.0,
    )

    add_box(
        ax,
        (0.90, 0.43),
        0.08,
        0.15,
        "Future-state\nprediction\n$\\hat{x}[k+1]$",
        facecolor="#EFFFFA",
        fontweight="bold",
        fontsize=8.6,
    )

    add_arrow(ax, (0.24, 0.64), (0.34, 0.77), "state history")
    add_arrow(ax, (0.24, 0.64), (0.34, 0.60), "trend")
    add_arrow(ax, (0.24, 0.64), (0.34, 0.43), "features")

    add_arrow(ax, (0.52, 0.77), (0.64, 0.64), "$\\hat{x}_{naive}$")
    add_arrow(ax, (0.52, 0.60), (0.64, 0.64), "$\\hat{x}_{cv}$")
    add_arrow(ax, (0.52, 0.43), (0.64, 0.64), "$\\hat{x}_{arx}$")

    add_arrow(ax, (0.73, 0.57), (0.73, 0.47), "weights")
    add_arrow(ax, (0.88, 0.38), (0.90, 0.50), "fused output")

    ax.text(
        0.5,
        0.10,
        "V9 is a hybrid future-state software sensor, not only a previous-value predictor.",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    save_figure(fig, "figure3_v9_hybrid_internal_structure")


# =============================================================================
# Figure 4
# =============================================================================

def make_figure4():
    fig, ax = setup_canvas(
        title="Figure 4. Residual and Window Detection Logic"
    )

    add_group_label(ax, 0.04, 0.87, "Residual generation")
    add_group_label(ax, 0.58, 0.87, "Window-based attack decision")

    add_box(
        ax,
        (0.06, 0.63),
        0.18,
        0.12,
        "Predicted future state\n$\\hat{x}_{V9}[k+1]$",
        facecolor="#ECFFEC",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.06, 0.39),
        0.18,
        0.12,
        "Measured / attacked\nfuture state\n$x_{measured}[k+1]$",
        facecolor="#EAF2FF",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.33, 0.51),
        0.19,
        0.14,
        "Residual\n\n$r[k+1] = x_{measured}[k+1]\n- \\hat{x}_{V9}[k+1]$",
        facecolor="#FFF7E6",
        fontweight="bold",
        fontsize=8.8,
    )

    add_box(
        ax,
        (0.33, 0.29),
        0.19,
        0.10,
        "Absolute residual\n$|r[k+1]|$",
        facecolor="#FFF7E6",
        fontweight="bold",
    )

    add_decision(
        ax,
        center=(0.64, 0.45),
        w=0.18,
        h=0.15,
        text="$|r[k+1]| > p99\\_abs$?",
    )

    add_box(
        ax,
        (0.76, 0.58),
        0.19,
        0.15,
        "10-sample window\nW10\n\nrecord recent\nthreshold violations",
        facecolor="#FFECEC",
        fontweight="bold",
        fontsize=8.8,
    )

    add_decision(
        ax,
        center=(0.855, 0.37),
        w=0.17,
        h=0.14,
        text="At least 3\nviolations?\nN3",
    )

    add_box(
        ax,
        (0.76, 0.12),
        0.19,
        0.11,
        "Attack detected\nflag = 1",
        facecolor="#EFFFFA",
        fontweight="bold",
    )

    x0 = 0.55
    y0 = 0.18
    cell_w = 0.026
    gap = 0.006

    for i in range(10):
        violation = i in [2, 5, 7, 8]
        add_box(
            ax,
            (x0 + i * (cell_w + gap), y0),
            cell_w,
            0.04,
            "1" if violation else "0",
            fontsize=8,
            facecolor="#FFDDDD" if violation else "#EEEEEE",
            radius=0.004,
        )

    ax.text(
        x0 + 5 * (cell_w + gap),
        y0 - 0.025,
        "Example W10 vector: 4 violations $\\geq$ 3",
        ha="center",
        va="top",
        fontsize=8.5,
    )

    add_arrow(ax, (0.24, 0.69), (0.33, 0.59), "prediction")
    add_arrow(ax, (0.24, 0.45), (0.33, 0.57), "measurement")
    add_arrow(ax, (0.425, 0.51), (0.425, 0.39), "absolute value")
    add_arrow(ax, (0.52, 0.34), (0.58, 0.43), "compare")
    add_arrow(ax, (0.72, 0.47), (0.77, 0.58), "violation bit")
    add_arrow(ax, (0.855, 0.58), (0.855, 0.44), "count")
    add_arrow(ax, (0.855, 0.30), (0.855, 0.23), "yes")

    ax.text(
        0.5,
        0.045,
        "W10_N3_mean1x declares an attack only when repeated threshold violations appear inside the detection window.",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
    )

    save_figure(fig, "figure4_residual_window_detection_logic")


# =============================================================================
# Figure 5
# =============================================================================

def make_figure5():
    fig, ax = setup_canvas(
        title="Figure 5. Recovery Decision Logic"
    )

    add_group_label(ax, 0.04, 0.87, "Recovery switch")

    add_box(
        ax,
        (0.08, 0.63),
        0.20,
        0.12,
        "Attacked measurement\n$x_{attacked}[k+1]$",
        facecolor="#EAF2FF",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.08, 0.42),
        0.20,
        0.12,
        "V9 prediction\n$\\hat{x}_{V9}[k+1]$",
        facecolor="#ECFFEC",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.08, 0.21),
        0.20,
        0.12,
        "Attack detection flag\nfrom W10_N3_mean1x",
        facecolor="#FFECEC",
        fontweight="bold",
    )

    add_decision(
        ax,
        center=(0.50, 0.48),
        w=0.20,
        h=0.16,
        text="attack_detected?",
    )

    add_box(
        ax,
        (0.64, 0.61),
        0.21,
        0.11,
        "YES\nuse V9 prediction\n$\\hat{x}_{V9}[k+1]$",
        facecolor="#ECFFEC",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.64, 0.27),
        0.21,
        0.11,
        "NO\nkeep measurement\n$x_{attacked}[k+1]$",
        facecolor="#EAF2FF",
        fontweight="bold",
    )

    add_box(
        ax,
        (0.89, 0.43),
        0.09,
        0.14,
        "Recovered\nsignal\n$x_{recovered}[k+1]$",
        facecolor="#EFFFFA",
        fontweight="bold",
        fontsize=8.5,
    )

    add_box(
        ax,
        (0.31, 0.08),
        0.48,
        0.13,
        "$x_{recovered}[k+1] =\n"
        "\\begin{cases}\n"
        "\\hat{x}_{V9}[k+1], & \\text{if attack\\_detected} \\\\\n"
        "x_{attacked}[k+1], & \\text{otherwise}\n"
        "\\end{cases}$",
        facecolor="#F7F7F7",
        fontweight="bold",
        fontsize=11,
    )

    add_arrow(ax, (0.28, 0.69), (0.43, 0.52), "measurement candidate")
    add_arrow(ax, (0.28, 0.48), (0.43, 0.49), "prediction candidate")
    add_arrow(ax, (0.28, 0.27), (0.43, 0.45), "flag")
    add_arrow(ax, (0.56, 0.53), (0.64, 0.66), "yes")
    add_arrow(ax, (0.56, 0.43), (0.64, 0.32), "no")
    add_arrow(ax, (0.85, 0.66), (0.89, 0.52))
    add_arrow(ax, (0.85, 0.32), (0.89, 0.48))

    ax.text(
        0.5,
        0.025,
        "During detected attacks, the corrupted sensor value is replaced by the V9 software-sensor prediction.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

    save_figure(fig, "figure5_recovery_decision_logic")


# =============================================================================
# Documentation files
# =============================================================================

def write_readme():
    text = """# Phase 7 Methodology Figures

This folder contains paper-style methodology diagrams for the ArduPilot SITL software-sensor attack detection and recovery project.

These figures are architecture and logic diagrams, not result plots. They are intended to complement the Phase 6 result figures.

## Final configuration represented

- Software sensor: V9 trained hybrid
- Prediction target: future state x_hat[k+1] using information available at time k
- Roll/Pitch residual: attacked_future[k+1] - V9_prediction[k+1]
- Yaw residual: circular angular difference
- Threshold: p99_abs
- Detector: W10_N3_mean1x
- Recovery rule:

    if attack_detected:
        recovered[k+1] = V9_prediction[k+1]
    else:
        recovered[k+1] = attacked_measurement[k+1]

## Figure 1: Software-Sensor-Based Attack Detection and Recovery Pipeline

Files:

- figure1_detection_recovery_pipeline.png
- figure1_detection_recovery_pipeline.pdf

This is the main Section 4.2.2-style methodology figure. It shows the runtime pipeline from sensor measurement to residual generation, threshold comparison, window detection, attack decision, recovery switch, and recovered output.

## Figure 2: Training and Deployment Workflow

Files:

- figure2_training_deployment_workflow.png
- figure2_training_deployment_workflow.pdf

This figure shows the full workflow. Clean baseline logs are used for feature extraction, candidate predictor evaluation, V9 selection, clean residual generation, and p99_abs threshold computation. Attacked logs are then used for detection, recovery, and RMSE/MAE evaluation.

## Figure 3: V9 Hybrid Software Sensor Internal Structure

Files:

- figure3_v9_hybrid_internal_structure.png
- figure3_v9_hybrid_internal_structure.pdf

This figure explains the internal V9 predictor. It combines naive, constant-velocity, and ARX-style predictors using validation-based weights.

## Figure 4: Residual and Window Detection Logic

Files:

- figure4_residual_window_detection_logic.png
- figure4_residual_window_detection_logic.pdf

This figure explains how residuals become attack decisions. The absolute residual is compared with p99_abs, and the W10_N3_mean1x detector checks whether at least three threshold violations occur inside a ten-sample window.

## Figure 5: Recovery Decision Logic

Files:

- figure5_recovery_decision_logic.png
- figure5_recovery_decision_logic.pdf

This figure explains the recovery switch. If an attack is detected, the recovered signal uses the V9 prediction. Otherwise, the measured signal is preserved.

## Recommended use

Use Figure 1 as the main Section 4.2.2-style architecture diagram. Use Figures 2-5 as supporting methodology diagrams.
"""
    path = OUT_DIR / "README.md"
    path.write_text(text)
    print(f"[OK] Saved {path}")


def write_latex():
    text = r"""\section{Software-Sensor-Based Attack Detection and Recovery Methodology}

Figure~\ref{fig:phase7_pipeline} summarizes the proposed software-sensor-based attack detection and recovery pipeline. The V9 trained hybrid software sensor predicts the future state $\hat{x}_{V9}[k+1]$ using information available at time $k$. The predicted future state is compared with the measured future state $x_{measured}[k+1]$ to compute the residual. For Roll and Pitch, the residual is computed using direct subtraction. For Yaw, circular angular difference is used to handle angular wrap-around.

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_methodology_figures/figure1_detection_recovery_pipeline.pdf}
    \caption{Software-sensor-based attack detection and recovery pipeline. The V9 trained hybrid predictor estimates the future state, the residual is compared against the $p99\_abs$ threshold, and the W10\_N3\_mean1x detector determines whether the recovery switch should replace the attacked measurement with the software-sensor prediction.}
    \label{fig:phase7_pipeline}
\end{figure*}

Figure~\ref{fig:phase7_workflow} shows the complete training and deployment workflow. Clean baseline logs are used to train and evaluate candidate predictors, select the V9 trained hybrid predictor, generate clean residuals, and compute the $p99\_abs$ threshold. The selected predictor and threshold are then deployed on attacked logs for detection, recovery, and RMSE/MAE evaluation.

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_methodology_figures/figure2_training_deployment_workflow.pdf}
    \caption{Training and deployment workflow for the software-sensor recovery system. Clean logs are used for predictor selection and threshold calibration, while attacked logs are used for detection, recovery, and quantitative evaluation.}
    \label{fig:phase7_workflow}
\end{figure*}

Figure~\ref{fig:phase7_v9_internal} illustrates the internal structure of the V9 trained hybrid software sensor. V9 combines a naive predictor, a constant-velocity predictor, and an ARX-style predictor using validation-based weights:

\[
\hat{x}_{V9}[k+1] =
w_{naive}\hat{x}_{naive}[k+1]
+ w_{cv}\hat{x}_{cv}[k+1]
+ w_{arx}\hat{x}_{arx}[k+1].
\]

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_methodology_figures/figure3_v9_hybrid_internal_structure.pdf}
    \caption{Internal structure of the V9 trained hybrid software sensor. The final future-state estimate is produced by validation-weighted fusion of naive, constant-velocity, and ARX-style predictions.}
    \label{fig:phase7_v9_internal}
\end{figure*}

Figure~\ref{fig:phase7_window_detector} presents the residual and window-based detection logic. The absolute residual is compared with the $p99\_abs$ threshold. The W10\_N3\_mean1x detector declares an attack when at least three threshold violations occur within a ten-sample window.

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_methodology_figures/figure4_residual_window_detection_logic.pdf}
    \caption{Residual and window-based detection logic. The detector uses the $p99\_abs$ threshold and declares an attack when repeated threshold violations occur inside a ten-sample window.}
    \label{fig:phase7_window_detector}
\end{figure*}

Figure~\ref{fig:phase7_recovery_switch} shows the recovery decision logic. When an attack is detected, the recovered signal is replaced by the V9 prediction. Otherwise, the original measured signal is preserved:

\[
x_{recovered}[k+1] =
\begin{cases}
\hat{x}_{V9}[k+1], & \text{if attack\_detected},\\
x_{attacked}[k+1], & \text{otherwise}.
\end{cases}
\]

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_methodology_figures/figure5_recovery_decision_logic.pdf}
    \caption{Recovery decision logic. The recovery switch substitutes the attacked measurement with the V9 software-sensor prediction only when the detector flags an attack.}
    \label{fig:phase7_recovery_switch}
\end{figure*}
"""
    path = OUT_DIR / "phase7_latex_snippets.tex"
    path.write_text(text)
    print(f"[OK] Saved {path}")


def write_summary():
    text = """# Phase 7 Section 4.2.2 Methodology Summary

The original paper's Section 4.2.2 methodology figure explains the software-sensor-based recovery architecture. Your Phase 7 figures serve the same purpose for your ArduPilot SITL reproduction.

Your Phase 6 package already provides result figures: RMSE/MAE plots, histograms, scatter plots, per-target improvement plots, and representative recovery time-series plots. Phase 7 adds the missing methodology diagrams that explain how the system works.

## Connection to the original methodology

The core idea is that a software sensor predicts the expected future state of the system. The measured sensor value is compared against this prediction. If the residual becomes abnormal according to a clean-data threshold and a window detector, the system treats the measurement as attacked. During detected attack periods, the attacked measurement is replaced by the software-sensor prediction.

Your implementation follows this structure:

1. Clean logs define normal behavior.
2. V9 predicts the future state x_hat[k+1] using information available at time k.
3. The measured future state is compared with the V9 prediction.
4. The residual is checked against the p99_abs threshold.
5. The W10_N3_mean1x window detector makes the attack decision.
6. During detected attacks, recovery replaces the attacked signal with the V9 prediction.
7. Recovery quality is evaluated with RMSE and MAE.

## Project-specific realization

Your project uses ArduPilot SITL data, not the original paper's dataset. Therefore, the numerical results and plots do not need to match the original paper exactly. The important point is that the methodology is aligned with the paper's software-sensor recovery concept.

Your final configuration is:

- Software sensor: V9 trained hybrid
- Threshold: p99_abs
- Detector: W10_N3_mean1x
- Recovery rule: use V9 prediction when attack_detected is true; otherwise keep the measured value

## How to describe the figures

Figure 1 is the main Section 4.2.2-style diagram. It shows the complete runtime detection and recovery pipeline.

Figure 2 explains how clean baseline data is used for training and threshold calibration, and how the trained system is deployed on attacked logs.

Figure 3 explains the V9 hybrid predictor. This is important because V9 is not just a previous-value predictor. It fuses naive, constant-velocity, and ARX-style predictions using validation-based weights.

Figure 4 explains the residual and window detector. It shows that the attack decision is based on repeated threshold violations inside a ten-sample window.

Figure 5 explains the final recovery switch. It shows exactly when the system uses the V9 prediction and when it keeps the measured value.

## Suggested professor-facing explanation

These figures are methodology diagrams. They explain the system architecture and logic behind the quantitative results. The Phase 6 figures show that recovery improved the attacked signals. The Phase 7 figures explain how that recovery was performed: future-state prediction, residual generation, clean thresholding, window-based attack detection, and conditional signal replacement.
"""
    path = OUT_DIR / "phase7_section_4_2_2_methodology_summary.md"
    path.write_text(text)
    print(f"[OK] Saved {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("============================================================")
    print("Phase 7 Methodology Figure Generation")
    print(f"Output directory: {OUT_DIR}")
    print("============================================================")

    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure5()

    write_readme()
    write_latex()
    write_summary()

    print("============================================================")
    print("Phase 7 completed successfully.")
    print("Generated PNG, PDF, README, LaTeX snippet, and Markdown summary.")
    print("============================================================")


if __name__ == "__main__":
    main()