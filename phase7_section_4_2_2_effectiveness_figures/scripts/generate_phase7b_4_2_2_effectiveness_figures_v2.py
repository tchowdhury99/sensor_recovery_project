#!/usr/bin/env python3

"""
Phase 7B v2: Corrected Section 4.2.2-Style Effectiveness Figure Generator

This version fixes the previous issues:

1. Figure B is generated directly from residual_results_v9/*.csv.
2. Detection examples are forced to use W10_N3_mean1x only.
3. Recovery examples are generated separately for Roll, Pitch, and Yaw
   using official Phase 5 recovered CSVs.
4. Summary plot uses the official Phase 5 W10_N3_mean1x recovery summary.
5. Output files use *_v2 names so the old draft figures are not silently confused.

Project configuration:
- Software sensor: V9 trained hybrid
- Threshold: p99_abs
- Detector: W10_N3_mean1x
- Recovery:
    if attack_detected:
        recovered[k+1] = V9_prediction[k+1]
    else:
        recovered[k+1] = attacked_measurement[k+1]
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path("/home/tchowdh4/sensor_recovery_project")
OUT_DIR = PROJECT_ROOT / "phase7_section_4_2_2_effectiveness_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / "phase7b_v2_generation_log.txt"

TARGETS = ["Roll", "Pitch", "Yaw"]
FINAL_DETECTOR = "W10_N3_mean1x"


# =============================================================================
# Logging
# =============================================================================

def log(msg):
    print(msg)
    with LOG_PATH.open("a") as f:
        f.write(str(msg) + "\n")


# =============================================================================
# Basic helpers
# =============================================================================

def read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        log(f"[WARN] Could not read {path}: {e}")
        return None


def rel_time(df):
    """
    Return a time vector.
    Priority:
    1. x
    2. time_sec
    3. timestamp
    4. TimeUS converted to seconds
    5. sample index
    """
    for col in ["x", "time_sec", "time_s", "timestamp", "TimeS", "time"]:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(x).sum() > 0:
                x = x - np.nanmin(x)
                return x

    if "TimeUS" in df.columns:
        x = pd.to_numeric(df["TimeUS"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).sum() > 0:
            x = (x - np.nanmin(x)) / 1_000_000.0
            return x

    return np.arange(len(df), dtype=float)


def numeric_series(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def save_fig(fig, stem):
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"

    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    log(f"[OK] Saved {png}")
    log(f"[OK] Saved {pdf}")


def find_first_existing(patterns):
    """
    Return first existing file from a list of glob patterns.
    """
    for pattern in patterns:
        matches = sorted(PROJECT_ROOT.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_all_existing(patterns):
    out = []
    for pattern in patterns:
        out.extend(sorted(PROJECT_ROOT.glob(pattern)))
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for p in out:
        if str(p) not in seen:
            unique.append(p)
            seen.add(str(p))
    return unique


def safe_col(df, candidates):
    """
    Return first column that exists.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_column_contains(df, required_tokens, forbidden_tokens=None):
    """
    Flexible column finder.
    """
    if forbidden_tokens is None:
        forbidden_tokens = []

    required_tokens = [t.lower() for t in required_tokens]
    forbidden_tokens = [t.lower() for t in forbidden_tokens]

    for col in df.columns:
        c = col.lower()
        if all(t in c for t in required_tokens) and not any(t in c for t in forbidden_tokens):
            return col

    return None


# =============================================================================
# Thresholds
# =============================================================================

def load_p99_thresholds():
    """
    Load p99_abs thresholds from the official V9 threshold table.
    """
    threshold_paths = [
        PROJECT_ROOT / "thresholds_v9" / "phase3_report_table_v9_hybrid_only.csv",
        PROJECT_ROOT / "thresholds_v9" / "phase3_clean_residual_thresholds_v9.csv",
    ]

    thresholds = {}

    for path in threshold_paths:
        if not path.exists():
            continue

        df = read_csv(path)
        if df is None:
            continue

        if "target" not in df.columns:
            continue

        value_col = None
        for c in ["p99_abs", "recommended_threshold", "threshold"]:
            if c in df.columns:
                value_col = c
                break

        if value_col is None:
            continue

        for target in TARGETS:
            sub = df[df["target"].astype(str).str.lower() == target.lower()]
            if len(sub) > 0:
                val = pd.to_numeric(sub[value_col], errors="coerce").dropna()
                if len(val) > 0:
                    thresholds[target] = float(val.iloc[0])

        if thresholds:
            log(f"[INFO] Loaded p99_abs thresholds from {path}")
            log(f"[INFO] Thresholds: {thresholds}")
            return thresholds

    log("[ERROR] Could not load p99_abs thresholds.")
    return thresholds


# =============================================================================
# Figure A: V9 prediction accuracy
# =============================================================================

def make_figure_a_prediction_accuracy():
    """
    Uses baseline_05_mixed_maneuver prediction output if available.
    Falls back to any V9 trained hybrid prediction output.
    """
    path = find_first_existing([
        "prediction_results_v9_trained_hybrid/baseline_05_mixed_maneuver_ATT_trained_hybrid_prediction_outputs.csv",
        "prediction_results_v9_trained_hybrid/*mixed*prediction_outputs.csv",
        "prediction_results_v9_trained_hybrid/*prediction_outputs.csv",
    ])

    if path is None:
        log("[SKIP] Figure A: no V9 prediction output file found.")
        return False

    df = read_csv(path)
    if df is None:
        return False

    log(f"[INFO] Figure A source: {path}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)

    plotted = 0

    for ax, target in zip(axes, TARGETS):
        x = rel_time(df)

        # Common prediction output column patterns.
        true_col = safe_col(df, [
            f"True_{target}",
            f"CleanFuture_{target}",
            f"{target}_true",
            f"true_{target}",
            target,
        ])

        pred_col = safe_col(df, [
            f"V9PredictedFuture_{target}",
            f"V9_prediction_{target}",
            f"{target}_V9_prediction",
            f"hybrid_prediction_{target}",
            f"{target}_hybrid_prediction",
            f"PredictedFuture_{target}",
        ])

        # More flexible fallback.
        if pred_col is None:
            pred_col = find_column_contains(df, [target, "pred"])

        if true_col is None:
            true_col = find_column_contains(df, [target, "true"])

        if true_col is None:
            true_col = find_column_contains(df, [target, "clean"])

        # In prediction outputs, raw Roll/Pitch/Yaw often represent clean measured signal.
        if true_col is None and target in df.columns:
            true_col = target

        if true_col is None or pred_col is None:
            ax.text(0.5, 0.5, f"Missing clean or V9 prediction for {target}", ha="center", va="center")
            ax.axis("off")
            continue

        y_true = numeric_series(df, true_col)
        y_pred = numeric_series(df, pred_col)

        n = min(len(x), len(y_true), len(y_pred))
        x_use = x[:n]
        y_true = y_true[:n]
        y_pred = y_pred[:n]

        ax.plot(x_use, y_true, linewidth=1.1, label="Measured / clean")
        ax.plot(x_use, y_pred, linewidth=1.1, linestyle="--", label="V9 prediction")
        ax.set_ylabel(target)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        plotted += 1

    axes[-1].set_xlabel("Time (s) or sample index")
    fig.suptitle("Section 4.2.2-Style Figure A: V9 Software Sensor Prediction Accuracy", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if plotted == 0:
        plt.close(fig)
        log("[SKIP] Figure A: no targets plotted.")
        return False

    save_fig(fig, "figure_4_2_2_a_v9_prediction_accuracy_v2")
    return True


# =============================================================================
# Figure B: Clean residuals + p99_abs thresholds
# =============================================================================

def find_clean_residual_file(target):
    """
    Prefer baseline_05_mixed_maneuver for Roll/Pitch/Yaw.
    Otherwise use first residual file for that target.
    """
    patterns = [
        f"residual_results_v9/baseline_05_mixed_maneuver_{target}_clean_residuals_v9.csv",
        f"residual_results_v9/*_{target}_clean_residuals_v9.csv",
    ]
    return find_first_existing(patterns)


def make_figure_b_clean_residual_threshold(thresholds):
    fig, axes = plt.subplots(3, 2, figsize=(13, 8))

    plotted = 0

    for row, target in enumerate(TARGETS):
        path = find_clean_residual_file(target)
        ax_ts = axes[row, 0]
        ax_hist = axes[row, 1]

        if path is None:
            ax_ts.text(0.5, 0.5, f"No clean residual file for {target}", ha="center", va="center")
            ax_hist.axis("off")
            continue

        df = read_csv(path)
        if df is None:
            ax_ts.text(0.5, 0.5, f"Could not read residual file for {target}", ha="center", va="center")
            ax_hist.axis("off")
            continue

        log(f"[INFO] Figure B {target} source: {path}")

        x = rel_time(df)

        abs_col = safe_col(df, [
            "hybrid_abs_residual",
            f"AbsResidual_{target}",
            f"{target}_abs_residual",
        ])

        res_col = safe_col(df, [
            "hybrid_residual",
            f"Residual_{target}",
            f"{target}_residual",
        ])

        if abs_col is not None:
            abs_r = numeric_series(df, abs_col)
        elif res_col is not None:
            abs_r = np.abs(numeric_series(df, res_col))
        else:
            ax_ts.text(0.5, 0.5, f"No residual column for {target}", ha="center", va="center")
            ax_hist.axis("off")
            continue

        threshold = thresholds.get(target, float(np.nanquantile(abs_r, 0.99)))

        ax_ts.plot(x[:len(abs_r)], abs_r, linewidth=1.0, label="|clean residual|")
        ax_ts.axhline(threshold, linestyle="--", linewidth=1.2, label=f"p99_abs = {threshold:.5g}")
        ax_ts.set_ylabel(f"{target}\n|residual|")
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend(fontsize=8, loc="best")

        clean_abs = abs_r[np.isfinite(abs_r)]
        ax_hist.hist(clean_abs, bins=50, alpha=0.8)
        ax_hist.axvline(threshold, linestyle="--", linewidth=1.2, label="p99_abs")
        ax_hist.set_ylabel(target)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend(fontsize=8, loc="best")

        plotted += 1

    axes[-1, 0].set_xlabel("Time (s) or sample index")
    axes[-1, 1].set_xlabel("|clean residual|")
    fig.suptitle("Section 4.2.2-Style Figure B: Clean Residuals and p99_abs Thresholds", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if plotted == 0:
        plt.close(fig)
        log("[SKIP] Figure B: no targets plotted.")
        return False

    save_fig(fig, "figure_4_2_2_b_clean_residual_threshold_v2")
    return True


# =============================================================================
# Figure C: W10_N3_mean1x detection examples
# =============================================================================

def find_detection_file(target):
    """
    Force the final detector W10_N3_mean1x.
    Prefer pulse attacks because they show clear detection windows.
    Search corrected first, then original detection_v9.
    """
    patterns = [
        f"attack_detection_v9_corrected/detection_csvs/*{target}*pulse*{FINAL_DETECTOR}*.csv",
        f"attack_detection_v9_corrected/detection_csvs/*{target}*bias*{FINAL_DETECTOR}*.csv",
        f"attack_detection_v9_corrected/detection_csvs/*{target}*ramp*{FINAL_DETECTOR}*.csv",
        f"attack_detection_v9/detection_csvs/*{target}*pulse*{FINAL_DETECTOR}*.csv",
        f"attack_detection_v9/detection_csvs/*{target}*bias*{FINAL_DETECTOR}*.csv",
        f"attack_detection_v9/detection_csvs/*{target}*ramp*{FINAL_DETECTOR}*.csv",
    ]

    candidates = find_all_existing(patterns)

    # Extra safety: reject non-final detector names.
    candidates = [p for p in candidates if FINAL_DETECTOR in p.name]

    if not candidates:
        return None

    # Prefer baseline_01_hover if possible.
    for p in candidates:
        if "baseline_01_hover" in p.name:
            return p

    return candidates[0]


def shade_regions(ax, x, flag, label):
    flag = np.asarray(flag).astype(bool)
    x = np.asarray(x)

    n = min(len(x), len(flag))
    x = x[:n]
    flag = flag[:n]

    in_region = False
    start = None
    used_label = False

    for i, active in enumerate(flag):
        if active and not in_region:
            start = x[i]
            in_region = True
        elif not active and in_region:
            end = x[i]
            ax.axvspan(start, end, alpha=0.20, label=label if not used_label else None)
            used_label = True
            in_region = False

    if in_region:
        ax.axvspan(start, x[-1], alpha=0.20, label=label if not used_label else None)


def make_figure_c_detection_examples():
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)

    plotted = 0

    for ax, target in zip(axes, TARGETS):
        path = find_detection_file(target)

        if path is None:
            ax.text(0.5, 0.5, f"No {FINAL_DETECTOR} detection file for {target}", ha="center", va="center")
            ax.axis("off")
            continue

        df = read_csv(path)
        if df is None:
            ax.text(0.5, 0.5, f"Could not read detection file for {target}", ha="center", va="center")
            ax.axis("off")
            continue

        log(f"[INFO] Figure C {target} source: {path}")

        x = rel_time(df)

        abs_col = safe_col(df, [
            f"AbsResidual_{target}",
            "AbsResidual",
            "abs_residual",
        ])

        threshold_col = safe_col(df, [
            "Threshold",
            "p99_abs",
            "threshold",
        ])

        flag_col = safe_col(df, [
            "WindowCombinedFlag",
            "WindowCountFlag",
            "PointDetectionFlag",
            "attack_detected",
            "Detected",
        ])

        attack_col = safe_col(df, [
            "AttackActive",
            "attack_active",
        ])

        if abs_col is None:
            ax.text(0.5, 0.5, f"No AbsResidual column for {target}", ha="center", va="center")
            ax.axis("off")
            continue

        abs_r = numeric_series(df, abs_col)

        if threshold_col is not None:
            threshold_vals = numeric_series(df, threshold_col)
            threshold = float(pd.Series(threshold_vals).dropna().iloc[0])
        else:
            threshold = float(np.nanquantile(abs_r, 0.99))

        ax.plot(x[:len(abs_r)], abs_r, linewidth=1.0, label="|attack residual|")
        ax.axhline(threshold, linestyle="--", linewidth=1.2, label=f"p99_abs = {threshold:.5g}")

        if attack_col is not None:
            attack_flag = numeric_series(df, attack_col) > 0
            shade_regions(ax, x, attack_flag, "attack active")

        if flag_col is not None:
            det_flag = numeric_series(df, flag_col) > 0
            shade_regions(ax, x, det_flag, "detected region")

        ax.set_ylabel(f"{target}\n|residual|")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        plotted += 1

    axes[-1].set_xlabel("Time (s) or sample index")
    fig.suptitle(f"Section 4.2.2-Style Figure C: Final {FINAL_DETECTOR} Detection Examples", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if plotted == 0:
        plt.close(fig)
        log("[SKIP] Figure C: no targets plotted.")
        return False

    save_fig(fig, "figure_4_2_2_c_W10_N3_detection_examples_v2")
    return True


# =============================================================================
# Figure D: W10_N3_mean1x official recovery examples
# =============================================================================

def find_recovery_file(target):
    """
    Force official Phase 5 W10_N3_mean1x recovery.
    Prefer pulse, then bias, then ramp.
    """
    patterns = [
        f"recovery_v9_corrected_official/recovered_csvs/*{target}*pulse*{FINAL_DETECTOR}*OFFICIAL_PHASE5_RECOVERED.csv",
        f"recovery_v9_corrected_official/recovered_csvs/*{target}*bias*{FINAL_DETECTOR}*OFFICIAL_PHASE5_RECOVERED.csv",
        f"recovery_v9_corrected_official/recovered_csvs/*{target}*ramp*{FINAL_DETECTOR}*OFFICIAL_PHASE5_RECOVERED.csv",
        f"recovery_v9_corrected_official/recovered_csvs/*{target}*{FINAL_DETECTOR}*OFFICIAL_PHASE5_RECOVERED.csv",
    ]

    candidates = find_all_existing(patterns)
    candidates = [p for p in candidates if FINAL_DETECTOR in p.name]

    if not candidates:
        return None

    for p in candidates:
        if "baseline_01_hover" in p.name:
            return p

    return candidates[0]


def make_figure_d_recovery_examples():
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)

    plotted = 0

    for ax, target in zip(axes, TARGETS):
        path = find_recovery_file(target)

        if path is None:
            ax.text(0.5, 0.5, f"No official {FINAL_DETECTOR} recovery file for {target}", ha="center", va="center")
            ax.axis("off")
            continue

        df = read_csv(path)
        if df is None:
            ax.text(0.5, 0.5, f"Could not read recovery file for {target}", ha="center", va="center")
            ax.axis("off")
            continue

        log(f"[INFO] Figure D {target} source: {path}")

        x = rel_time(df)

        clean_col = safe_col(df, [
            f"CleanFuture_{target}",
            f"TrueFuture_{target}",
            f"True_{target}",
        ])

        attacked_col = safe_col(df, [
            f"AttackedFuture_{target}",
            f"Attacked_{target}",
        ])

        pred_col = safe_col(df, [
            f"V9PredictedFuture_{target}",
            f"V9Prediction_{target}",
        ])

        recovered_col = safe_col(df, [
            f"{target}_RecoveredFuture_V9_{FINAL_DETECTOR}_Phase5",
            f"{target}_RecoveredFuture_V9_W10_N3_mean1x_Phase5",
        ])

        if recovered_col is None:
            recovered_col = find_column_contains(df, [target, "recovered"])

        flag_col = safe_col(df, [
            "WindowCombinedFlag",
            "WindowCountFlag",
            "PointDetectionFlag",
            "AttackActive",
        ])

        if recovered_col is None:
            ax.text(0.5, 0.5, f"No recovered column for {target}", ha="center", va="center")
            ax.axis("off")
            continue

        if clean_col is not None:
            ax.plot(x, numeric_series(df, clean_col), linewidth=1.0, label="clean/reference")

        if attacked_col is not None:
            ax.plot(x, numeric_series(df, attacked_col), linewidth=1.0, linestyle="--", label="attacked")

        if pred_col is not None:
            ax.plot(x, numeric_series(df, pred_col), linewidth=1.0, linestyle=":", label="V9 prediction")

        ax.plot(x, numeric_series(df, recovered_col), linewidth=1.2, label="recovered")

        if flag_col is not None:
            flag = numeric_series(df, flag_col) > 0
            shade_regions(ax, x, flag, "detected/recovery region")

        ax.set_ylabel(target)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        plotted += 1

    axes[-1].set_xlabel("Time (s) or sample index")
    fig.suptitle(f"Section 4.2.2-Style Figure D: Official {FINAL_DETECTOR} Recovery Examples", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if plotted == 0:
        plt.close(fig)
        log("[SKIP] Figure D: no targets plotted.")
        return False

    save_fig(fig, "figure_4_2_2_d_W10_N3_recovery_examples_v2")
    return True


# =============================================================================
# Figure E: Recovery improvement summary
# =============================================================================

def find_summary_col(df, required_tokens):
    return find_column_contains(df, required_tokens)


def make_figure_e_recovery_summary():
    path = PROJECT_ROOT / "recovery_v9_corrected_official" / "recovery_reports" / "phase5_OFFICIAL_W10_N3_mean1x_recovery_summary.csv"

    if not path.exists():
        path = find_first_existing([
            "recovery_v9_corrected_official/recovery_reports/*W10_N3_mean1x*summary*.csv",
            "phase6_final_package/reports/*summary*.csv",
        ])

    if path is None or not path.exists():
        log("[SKIP] Figure E: no official recovery summary CSV found.")
        return False

    df = read_csv(path)
    if df is None:
        return False

    log(f"[INFO] Figure E source: {path}")

    attacked_rmse_col = find_summary_col(df, ["attacked", "rmse"])
    recovered_rmse_col = find_summary_col(df, ["recovered", "rmse"])
    attacked_mae_col = find_summary_col(df, ["attacked", "mae"])
    recovered_mae_col = find_summary_col(df, ["recovered", "mae"])

    metrics = []

    if attacked_rmse_col and recovered_rmse_col:
        a = pd.to_numeric(df[attacked_rmse_col], errors="coerce").dropna()
        r = pd.to_numeric(df[recovered_rmse_col], errors="coerce").dropna()
        if len(a) > 0 and len(r) > 0:
            metrics.append(("RMSE", float(a.mean()), float(r.mean())))

    if attacked_mae_col and recovered_mae_col:
        a = pd.to_numeric(df[attacked_mae_col], errors="coerce").dropna()
        r = pd.to_numeric(df[recovered_mae_col], errors="coerce").dropna()
        if len(a) > 0 and len(r) > 0:
            metrics.append(("MAE", float(a.mean()), float(r.mean())))

    if not metrics:
        log("[SKIP] Figure E: could not infer attacked/recovered RMSE/MAE columns.")
        log(f"[INFO] Columns: {list(df.columns)}")
        return False

    labels = [m[0] for m in metrics]
    attacked_vals = [m[1] for m in metrics]
    recovered_vals = [m[2] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - width / 2, attacked_vals, width, label="Attacked")
    ax.bar(x + width / 2, recovered_vals, width, label="Recovered")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean error")
    ax.set_title("Section 4.2.2-Style Figure E: Official Recovery Improvement Summary", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for i, (label, a, r) in enumerate(metrics):
        if a > 0:
            reduction = 100.0 * (a - r) / a
            ax.text(i, max(a, r) * 1.03, f"{reduction:.1f}% mean-error reduction", ha="center", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "figure_4_2_2_e_recovery_improvement_summary_v2")
    return True


# =============================================================================
# Documentation
# =============================================================================

def write_v2_readme(statuses):
    status_text = "\n".join([f"- {name}: {'generated' if ok else 'skipped'}" for name, ok in statuses])

    text = f"""# Phase 7B v2: Corrected Section 4.2.2-Style Effectiveness Figures

This folder contains corrected Section 4.2.2-style effectiveness figures for the ArduPilot SITL software-sensor recovery project.

## Generated status

{status_text}

## Corrections in v2

- Detection examples are forced to use the final detector: W10_N3_mean1x.
- Clean residual threshold figure is generated directly from residual_results_v9 files.
- Recovery examples are generated separately for Roll, Pitch, and Yaw using official Phase 5 recovered CSVs.
- RMSE/MAE improvement summary uses the official W10_N3_mean1x Phase 5 recovery summary.
- Output filenames end with _v2 to avoid confusion with the earlier draft figures.

## Mapping to Section 4.2.2

- Figure A: software sensor prediction accuracy.
- Figure B: clean residual/error behavior and p99_abs threshold.
- Figure C: final window detector behavior under attack.
- Figure D: official recovery behavior under attack.
- Figure E: quantitative RMSE/MAE recovery improvement.

## Final configuration

- Software sensor: V9 trained hybrid
- Threshold: p99_abs
- Detector: W10_N3_mean1x
- Recovery rule: use V9 prediction during detected attacks, otherwise keep the measured signal.

## Recommended paper wording

These are Section 4.2.2-style effectiveness figures generated using my ArduPilot SITL dataset. They do not duplicate the original paper's data; instead, they reproduce the same evaluation logic with my final V9/p99_abs/W10_N3_mean1x configuration.
"""
    path = OUT_DIR / "README_PHASE7B_V2.md"
    path.write_text(text)
    log(f"[OK] Saved {path}")


def write_v2_latex():
    text = r"""\subsection{Section 4.2.2-Style Effectiveness Evaluation}

The following figures reproduce the effectiveness-evaluation style of Section 4.2.2 using the ArduPilot SITL dataset and the final V9/p99\_abs/W10\_N3\_mean1x recovery configuration.

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_section_4_2_2_effectiveness_figures/figure_4_2_2_a_v9_prediction_accuracy_v2.pdf}
    \caption{V9 software-sensor prediction accuracy. The V9 trained hybrid predictor estimates the future attitude state using information available at time $k$ and closely follows the measured clean Roll, Pitch, and Yaw signals.}
    \label{fig:v9_prediction_accuracy_v2}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_section_4_2_2_effectiveness_figures/figure_4_2_2_b_clean_residual_threshold_v2.pdf}
    \caption{Clean residual behavior and $p99\_abs$ thresholds. The thresholds are computed from clean residual distributions and are used as the abnormal-deviation boundary during attack detection.}
    \label{fig:clean_residual_threshold_v2}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_section_4_2_2_effectiveness_figures/figure_4_2_2_c_W10_N3_detection_examples_v2.pdf}
    \caption{Final W10\_N3\_mean1x window-detection examples. The detector declares an attack when at least three residual threshold violations occur inside a ten-sample window.}
    \label{fig:w10n3_detection_examples_v2}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_section_4_2_2_effectiveness_figures/figure_4_2_2_d_W10_N3_recovery_examples_v2.pdf}
    \caption{Official W10\_N3\_mean1x recovery examples. During detected attack periods, the corrupted measurement is replaced by the V9 software-sensor prediction.}
    \label{fig:w10n3_recovery_examples_v2}
\end{figure*}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.48\textwidth]{phase7_section_4_2_2_effectiveness_figures/figure_4_2_2_e_recovery_improvement_summary_v2.pdf}
    \caption{Official recovery improvement summary. The recovered signal substantially reduces mean RMSE and MAE compared with the attacked signal.}
    \label{fig:recovery_improvement_summary_v2}
\end{figure}
"""
    path = OUT_DIR / "phase7b_v2_4_2_2_latex_snippets.tex"
    path.write_text(text)
    log(f"[OK] Saved {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    log("============================================================")
    log("Phase 7B v2: Corrected Section 4.2.2-Style Figure Generation")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Output directory: {OUT_DIR}")
    log(f"Final detector forced: {FINAL_DETECTOR}")
    log("============================================================")

    thresholds = load_p99_thresholds()

    statuses = []

    ok = make_figure_a_prediction_accuracy()
    statuses.append(("Figure A prediction accuracy", ok))

    ok = make_figure_b_clean_residual_threshold(thresholds)
    statuses.append(("Figure B clean residual threshold", ok))

    ok = make_figure_c_detection_examples()
    statuses.append(("Figure C W10_N3 detection examples", ok))

    ok = make_figure_d_recovery_examples()
    statuses.append(("Figure D W10_N3 recovery examples", ok))

    ok = make_figure_e_recovery_summary()
    statuses.append(("Figure E recovery improvement summary", ok))

    write_v2_readme(statuses)
    write_v2_latex()

    log("============================================================")
    log("Phase 7B v2 completed.")
    for name, ok in statuses:
        log(f"{name}: {'generated' if ok else 'skipped'}")
    log("============================================================")


if __name__ == "__main__":
    main()
