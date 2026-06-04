#!/usr/bin/env python3

"""
PHASE 5B: XKF1-DRIVEN ADAPTIVE-WEIGHT V9 RECOVERY EXPERIMENT

This is separate from official Phase 5.

Official Phase 5:
    - Uses original V9 trained hybrid prediction.
    - Uses W10_N3_mean1x detection.
    - Already completed successfully.

Phase 5B:
    - Uses the same official W10_N3_mean1x detection files.
    - Loads V9 component predictions from PredictionFile:
        naive component
        constant-velocity component
        ARX component
    - Loads XKF1 flight context:
        horizontal speed
        acceleration proxy
        gyro proxy
    - Computes smoothed adaptive weights.
    - Builds adaptive V9 prediction.
    - Compares adaptive recovery against original V9 recovery.

Important:
    This script does NOT overwrite official Phase 5 results.
"""

import os
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASE = "/home/tchowdh4/sensor_recovery_project"

DETECTION_DIR = f"{BASE}/attack_detection_v9_corrected/detection_csvs"

XKF1_DIR = f"{BASE}/logs/extracted_csv"

OUT_CSV_DIR = f"{BASE}/recovery_results_v9_adaptive_weights"
OUT_FIG_DIR = f"{BASE}/figures/recovery_v9_adaptive_weights"
OUT_REPORT_DIR = f"{BASE}/recovery_reports_v9_adaptive_weights"

os.makedirs(OUT_CSV_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_REPORT_DIR, exist_ok=True)


# ============================================================
# Experiment settings
# ============================================================

OFFICIAL_DETECTOR_TAG = "W10_N3_mean1x_detection"

# Phase 5B tests Roll and Pitch first.
TARGETS = ["Roll", "Pitch"]

# Smoothed adaptive-weight profiles.
PROFILE_WEIGHTS = {
    "hover": {
        "w_naive": 0.80,
        "w_cv": 0.10,
        "w_arx": 0.10,
    },
    "translation": {
        "w_naive": 0.20,
        "w_cv": 0.60,
        "w_arx": 0.20,
    },
    "maneuver": {
        "w_naive": 0.05,
        "w_cv": 0.25,
        "w_arx": 0.70,
    },
}

SMOOTHING_PREVIOUS = 0.90
SMOOTHING_TARGET = 0.10

# Flight-context thresholds.
HOVER_SPEED_THRESHOLD = 0.5
ACCEL_LOW_THRESHOLD = 0.75
ACCEL_HIGH_THRESHOLD = 2.0
GYRO_HIGH_THRESHOLD = 0.20


# ============================================================
# Basic utilities
# ============================================================

def normalize_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def rmse(err):
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return np.nan
    return math.sqrt(np.mean(err ** 2))


def mae(err):
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return np.nan
    return np.mean(np.abs(err))


def max_abs(err):
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return np.nan
    return np.max(np.abs(err))


def percent_improvement(old, new):
    if not np.isfinite(old) or old == 0:
        return np.nan
    return 100.0 * (old - new) / old


def to_bool_series(s):
    if s.dtype == bool:
        return s.fillna(False)

    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(float) != 0

    lowered = s.astype(str).str.strip().str.lower()
    return lowered.isin([
        "1",
        "true",
        "yes",
        "y",
        "detected",
        "attack",
        "attacked",
        "active",
    ])


def compute_error(target, measured, reference):
    """
    Phase 5B is Roll/Pitch first, so normal subtraction is enough.
    Kept as function for consistency.
    """
    measured = np.asarray(measured, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return measured - reference


def get_target_from_file(path):
    name = os.path.basename(path)

    for target in TARGETS:
        if f"_{target}_" in name:
            return target

    return None


def parse_baseline_from_detection_file(path):
    """
    Example:
        baseline_05_mixed_maneuver_Roll_pulse_W10_N3_mean1x_detection.csv

    Returns:
        baseline_05_mixed_maneuver
    """
    name = os.path.basename(path).replace(".csv", "")

    name = re.sub(
        r"_(Roll|Pitch|Yaw)_(bias|pulse|ramp)_W10_N3_mean1x_detection$",
        "",
        name,
    )

    return name


def find_time_column(df):
    candidates = [
        "x",
        "t",
        "time",
        "time_sec",
        "timestamp",
        "TimeUS",
        "timeus",
    ]

    norm = {normalize_name(c): c for c in df.columns}

    for c in candidates:
        n = normalize_name(c)
        if n in norm:
            return norm[n]

    return None


def find_detection_column(df):
    candidates = [
        "PointDetectionFlag",
        "point_detection_flag",
        "WindowCombinedFlag",
        "detection",
        "detected",
    ]

    norm = {normalize_name(c): c for c in df.columns}

    for c in candidates:
        n = normalize_name(c)
        if n in norm:
            return norm[n]

    for c in df.columns:
        n = normalize_name(c)
        if "detect" in n:
            return c

    raise ValueError("Could not find detection column.")


# ============================================================
# Load and align V9 component predictions
# ============================================================

def load_prediction_components(det_df, target):
    """
    The detection CSV contains a PredictionFile column.
    That file contains:
        Roll_naive_component
        Roll_constant_velocity_component
        Roll_arx_component
        Pitch_naive_component
        Pitch_constant_velocity_component
        Pitch_arx_component
    """

    if "PredictionFile" not in det_df.columns:
        raise ValueError("Detection CSV does not contain PredictionFile column.")

    prediction_file = str(det_df["PredictionFile"].iloc[0])

    if not os.path.exists(prediction_file):
        raise FileNotFoundError(f"PredictionFile not found: {prediction_file}")

    pred_df = pd.read_csv(prediction_file)

    required_cols = {
        "true_future": f"{target}_true_future",
        "hybrid_prediction": f"{target}_hybrid_prediction",
        "arx": f"{target}_arx_component",
        "naive": f"{target}_naive_component",
        "cv": f"{target}_constant_velocity_component",
    }

    for label, col in required_cols.items():
        if col not in pred_df.columns:
            raise ValueError(
                f"Missing {label} column in prediction file:\n"
                f"  {prediction_file}\n"
                f"Expected column:\n"
                f"  {col}"
            )

    n = len(det_df)

    if len(pred_df) < n:
        raise ValueError(
            f"Prediction file has fewer rows than detection file.\n"
            f"Detection rows: {n}\n"
            f"Prediction rows: {len(pred_df)}\n"
            f"Prediction file: {prediction_file}"
        )

    # Phase 4 already used AlignmentOffset and aligned future predictions.
    # The detection CSV rows correspond to the first n rows of the Phase 2 prediction output.
    out = pd.DataFrame(index=det_df.index)

    out[f"{target}_component_true_future"] = pred_df[required_cols["true_future"]].iloc[:n].to_numpy()
    out[f"{target}_component_original_hybrid"] = pred_df[required_cols["hybrid_prediction"]].iloc[:n].to_numpy()
    out[f"{target}_component_arx"] = pred_df[required_cols["arx"]].iloc[:n].to_numpy()
    out[f"{target}_component_naive"] = pred_df[required_cols["naive"]].iloc[:n].to_numpy()
    out[f"{target}_component_cv"] = pred_df[required_cols["cv"]].iloc[:n].to_numpy()
    out["Phase5B_PredictionFile"] = prediction_file

    return out, prediction_file


# ============================================================
# Load and align XKF1
# ============================================================

def find_xkf1_file_for_detection(det_path):
    baseline = parse_baseline_from_detection_file(det_path)

    expected = f"{XKF1_DIR}/{baseline}_XKF1.csv"

    if os.path.exists(expected):
        return expected

    candidates = sorted(glob.glob(f"{BASE}/**/{baseline}_XKF1.csv", recursive=True))

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find XKF1 file for baseline:\n"
        f"  {baseline}\n"
        f"Expected:\n"
        f"  {expected}"
    )


def load_and_align_xkf1(det_df, det_path):
    """
    Align XKF1 to detection CSV length.

    Since detection CSV has x/sample index and XKF1 has TimeUS, we use normalized row/time interpolation.
    This is sufficient for context features, not for exact sensor replacement.
    """

    xkf1_file = find_xkf1_file_for_detection(det_path)

    xkf = pd.read_csv(xkf1_file)

    required = ["VN", "VE"]
    for c in required:
        if c not in xkf.columns:
            raise ValueError(f"XKF1 file missing required column {c}: {xkf1_file}")

    n = len(det_df)

    # Use normalized interpolation for robust alignment.
    src_index = np.linspace(0.0, 1.0, len(xkf))
    dst_index = np.linspace(0.0, 1.0, n)

    aligned = pd.DataFrame(index=det_df.index)

    for col in ["TimeUS", "VN", "VE", "VD", "GX", "GY", "GZ", "Roll", "Pitch", "Yaw"]:
        if col in xkf.columns:
            aligned[f"XKF1_{col}"] = np.interp(
                dst_index,
                src_index,
                xkf[col].astype(float).to_numpy(),
            )

    aligned["Phase5B_XKF1File"] = xkf1_file

    return aligned, xkf1_file


# ============================================================
# XKF1 context and adaptive weights
# ============================================================

def compute_xkf1_context(xkf_df):
    out = xkf_df.copy()

    vn = out["XKF1_VN"].astype(float).to_numpy()
    ve = out["XKF1_VE"].astype(float).to_numpy()

    out["horizontal_speed"] = np.sqrt(vn ** 2 + ve ** 2)

    # Acceleration proxy.
    dvn = np.diff(vn, prepend=vn[0])
    dve = np.diff(ve, prepend=ve[0])
    out["accel_proxy"] = np.sqrt(dvn ** 2 + dve ** 2)

    gyro_cols = [c for c in ["XKF1_GX", "XKF1_GY", "XKF1_GZ"] if c in out.columns]

    if gyro_cols:
        gyro_data = out[gyro_cols].astype(float).to_numpy()
        out["gyro_proxy"] = np.linalg.norm(gyro_data, axis=1)
    else:
        out["gyro_proxy"] = 0.0

    labels = []

    for _, row in out.iterrows():
        speed = row["horizontal_speed"]
        accel = row["accel_proxy"]
        gyro = row["gyro_proxy"]

        if accel >= ACCEL_HIGH_THRESHOLD or gyro >= GYRO_HIGH_THRESHOLD:
            labels.append("maneuver")
        elif speed < HOVER_SPEED_THRESHOLD and accel < ACCEL_LOW_THRESHOLD:
            labels.append("hover")
        else:
            labels.append("translation")

    out["flight_context"] = labels

    return out


def compute_smoothed_weights(context_df):
    """
    Apply:
        W_current = 0.9 * W_previous + 0.1 * W_target
    """

    w_naive = []
    w_cv = []
    w_arx = []

    previous = PROFILE_WEIGHTS["hover"].copy()

    for label in context_df["flight_context"].astype(str):
        target = PROFILE_WEIGHTS.get(label, PROFILE_WEIGHTS["translation"])

        current = {
            "w_naive": SMOOTHING_PREVIOUS * previous["w_naive"] + SMOOTHING_TARGET * target["w_naive"],
            "w_cv": SMOOTHING_PREVIOUS * previous["w_cv"] + SMOOTHING_TARGET * target["w_cv"],
            "w_arx": SMOOTHING_PREVIOUS * previous["w_arx"] + SMOOTHING_TARGET * target["w_arx"],
        }

        total = current["w_naive"] + current["w_cv"] + current["w_arx"]

        current["w_naive"] /= total
        current["w_cv"] /= total
        current["w_arx"] /= total

        w_naive.append(current["w_naive"])
        w_cv.append(current["w_cv"])
        w_arx.append(current["w_arx"])

        previous = current

    return pd.DataFrame({
        "w_naive_adaptive": w_naive,
        "w_cv_adaptive": w_cv,
        "w_arx_adaptive": w_arx,
    })


# ============================================================
# Plotting
# ============================================================

def make_recovery_comparison_plot(df, path, target):
    time_col = find_time_column(df)

    if time_col is not None:
        t = df[time_col].to_numpy()
        xlabel = time_col
    else:
        t = np.arange(len(df))
        xlabel = "sample index"

    clean_col = f"CleanFuture_{target}"
    attacked_col = f"AttackedFuture_{target}"
    original_col = f"OriginalV9RecoveredFuture_{target}_Phase5B"
    adaptive_col = f"AdaptiveV9RecoveredFuture_{target}_Phase5B"

    plt.figure(figsize=(14, 6))

    plt.plot(t, df[clean_col], linewidth=1.6, label="Clean/reference future")
    plt.plot(t, df[attacked_col], linewidth=1.0, alpha=0.7, label="Attacked future")
    plt.plot(t, df[original_col], linewidth=1.4, label="Original V9 recovered")
    plt.plot(t, df[adaptive_col], linewidth=1.4, label="Adaptive-weight recovered")

    plt.title(f"Phase 5B Recovery Comparison: {target}")
    plt.xlabel(xlabel)
    plt.ylabel(target)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_error_comparison_plot(df, path, target):
    time_col = find_time_column(df)

    if time_col is not None:
        t = df[time_col].to_numpy()
        xlabel = time_col
    else:
        t = np.arange(len(df))
        xlabel = "sample index"

    plt.figure(figsize=(14, 5))

    plt.plot(
        t,
        df[f"AttackedError_{target}_Phase5B"],
        linewidth=1.0,
        alpha=0.7,
        label="Attacked error",
    )

    plt.plot(
        t,
        df[f"OriginalV9RecoveredError_{target}_Phase5B"],
        linewidth=1.3,
        label="Original V9 recovered error",
    )

    plt.plot(
        t,
        df[f"AdaptiveV9RecoveredError_{target}_Phase5B"],
        linewidth=1.3,
        label="Adaptive recovered error",
    )

    plt.axhline(0, linewidth=1.0)
    plt.title(f"Phase 5B Error Comparison: {target}")
    plt.xlabel(xlabel)
    plt.ylabel("error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_weight_trace_plot(df, path, target):
    time_col = find_time_column(df)

    if time_col is not None:
        t = df[time_col].to_numpy()
        xlabel = time_col
    else:
        t = np.arange(len(df))
        xlabel = "sample index"

    plt.figure(figsize=(14, 5))

    plt.plot(t, df["w_naive_adaptive"], linewidth=1.4, label="w_naive")
    plt.plot(t, df["w_cv_adaptive"], linewidth=1.4, label="w_cv")
    plt.plot(t, df["w_arx_adaptive"], linewidth=1.4, label="w_arx")

    plt.title(f"Phase 5B Adaptive Weight Trace: {target}")
    plt.xlabel(xlabel)
    plt.ylabel("weight")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_context_plot(df, path, target):
    time_col = find_time_column(df)

    if time_col is not None:
        t = df[time_col].to_numpy()
        xlabel = time_col
    else:
        t = np.arange(len(df))
        xlabel = "sample index"

    label_map = {
        "hover": 0,
        "translation": 1,
        "maneuver": 2,
    }

    y = df["flight_context"].map(label_map).fillna(-1).to_numpy()

    plt.figure(figsize=(14, 4))
    plt.plot(t, y, linewidth=1.3)

    plt.yticks([0, 1, 2], ["hover", "translation", "maneuver"])
    plt.title(f"Phase 5B XKF1 Flight Context: {target}")
    plt.xlabel(xlabel)
    plt.ylabel("context")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# Metrics
# ============================================================

def build_metrics(prefix, err):
    return {
        f"{prefix}_rmse": rmse(err),
        f"{prefix}_mae": mae(err),
        f"{prefix}_max_abs_error": max_abs(err),
    }


# ============================================================
# Process one detection file
# ============================================================

def process_one_file(det_path):
    df = pd.read_csv(det_path)

    target = get_target_from_file(det_path)

    if target not in TARGETS:
        return None

    det_col = find_detection_column(df)

    clean_col = f"CleanFuture_{target}"
    attacked_col = f"AttackedFuture_{target}"
    original_v9_col = f"V9PredictedFuture_{target}"

    for col in [clean_col, attacked_col, original_v9_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required detection column {col} in {det_path}")

    det_bool = to_bool_series(df[det_col]).to_numpy()

    # Load V9 components from PredictionFile.
    comp_df, prediction_file = load_prediction_components(df, target)

    # Load and compute XKF1 context.
    xkf_df, xkf1_file = load_and_align_xkf1(df, det_path)
    context_df = compute_xkf1_context(xkf_df)
    weights_df = compute_smoothed_weights(context_df)

    # Merge extra columns.
    for c in comp_df.columns:
        df[c] = comp_df[c].values

    for c in context_df.columns:
        df[c] = context_df[c].values

    for c in weights_df.columns:
        df[c] = weights_df[c].values

    naive_col = f"{target}_component_naive"
    cv_col = f"{target}_component_cv"
    arx_col = f"{target}_component_arx"

    adaptive_pred_col = f"AdaptiveV9PredictedFuture_{target}_Phase5B"

    df[adaptive_pred_col] = (
        df["w_naive_adaptive"].astype(float) * df[naive_col].astype(float)
        +
        df["w_cv_adaptive"].astype(float) * df[cv_col].astype(float)
        +
        df["w_arx_adaptive"].astype(float) * df[arx_col].astype(float)
    )

    original_recovered_col = f"OriginalV9RecoveredFuture_{target}_Phase5B"
    adaptive_recovered_col = f"AdaptiveV9RecoveredFuture_{target}_Phase5B"

    # Original V9 recovery, same as official Phase 5 rule.
    df[original_recovered_col] = np.where(
        det_bool,
        df[original_v9_col].astype(float).to_numpy(),
        df[attacked_col].astype(float).to_numpy(),
    )

    # Adaptive recovery.
    df[adaptive_recovered_col] = np.where(
        det_bool,
        df[adaptive_pred_col].astype(float).to_numpy(),
        df[attacked_col].astype(float).to_numpy(),
    )

    # Errors.
    attacked_err = compute_error(target, df[attacked_col], df[clean_col])
    original_err = compute_error(target, df[original_recovered_col], df[clean_col])
    adaptive_err = compute_error(target, df[adaptive_recovered_col], df[clean_col])

    df[f"AttackedError_{target}_Phase5B"] = attacked_err
    df[f"OriginalV9RecoveredError_{target}_Phase5B"] = original_err
    df[f"AdaptiveV9RecoveredError_{target}_Phase5B"] = adaptive_err

    attack_active_count = np.nan
    if "AttackActive" in df.columns:
        attack_active_count = int(np.sum(to_bool_series(df["AttackActive"]).to_numpy()))

    base_name = os.path.splitext(os.path.basename(det_path))[0]

    out_csv = f"{OUT_CSV_DIR}/{base_name}_PHASE5B_ADAPTIVE_WEIGHTS.csv"
    df.to_csv(out_csv, index=False)

    comparison_plot = f"{OUT_FIG_DIR}/{base_name}_phase5b_recovery_comparison_{target}.png"
    error_plot = f"{OUT_FIG_DIR}/{base_name}_phase5b_error_comparison_{target}.png"
    weight_plot = f"{OUT_FIG_DIR}/{base_name}_phase5b_weight_trace_{target}.png"
    context_plot = f"{OUT_FIG_DIR}/{base_name}_phase5b_context_{target}.png"

    make_recovery_comparison_plot(df, comparison_plot, target)
    make_error_comparison_plot(df, error_plot, target)
    make_weight_trace_plot(df, weight_plot, target)
    make_context_plot(df, context_plot, target)

    summary = {
        "input_detection_file": det_path,
        "prediction_file": prediction_file,
        "xkf1_file": xkf1_file,
        "output_csv": out_csv,
        "target": target,
        "baseline": parse_baseline_from_detection_file(det_path),
        "n_rows": len(df),
        "detected_count": int(np.sum(det_bool)),
        "attack_active_count": attack_active_count,
        "clean_col": clean_col,
        "attacked_col": attacked_col,
        "original_v9_col": original_v9_col,
        "adaptive_prediction_col": adaptive_pred_col,
        "original_recovered_col": original_recovered_col,
        "adaptive_recovered_col": adaptive_recovered_col,
        "comparison_plot": comparison_plot,
        "error_plot": error_plot,
        "weight_plot": weight_plot,
        "context_plot": context_plot,
    }

    summary.update(build_metrics("attacked", attacked_err))
    summary.update(build_metrics("original_v9_recovered", original_err))
    summary.update(build_metrics("adaptive_v9_recovered", adaptive_err))

    summary["original_improvement_over_attacked_rmse_percent"] = percent_improvement(
        summary["attacked_rmse"],
        summary["original_v9_recovered_rmse"],
    )

    summary["adaptive_improvement_over_attacked_rmse_percent"] = percent_improvement(
        summary["attacked_rmse"],
        summary["adaptive_v9_recovered_rmse"],
    )

    summary["adaptive_improvement_over_original_rmse_percent"] = percent_improvement(
        summary["original_v9_recovered_rmse"],
        summary["adaptive_v9_recovered_rmse"],
    )

    summary["adaptive_better_than_original_rmse"] = (
        summary["adaptive_v9_recovered_rmse"] < summary["original_v9_recovered_rmse"]
    )

    return summary


# ============================================================
# Write report
# ============================================================

def write_report(summary_df):
    report_path = f"{OUT_REPORT_DIR}/phase5b_adaptive_weight_interpretation.txt"

    lines = []
    lines.append("PHASE 5B XKF1-DRIVEN ADAPTIVE-WEIGHT RECOVERY INTERPRETATION")
    lines.append("=" * 90)
    lines.append("")
    lines.append("This is a separate experiment from official Phase 5.")
    lines.append("")
    lines.append("Official Phase 5 remains:")
    lines.append("  Original V9 trained hybrid recovery using W10_N3_mean1x detection.")
    lines.append("")
    lines.append("Phase 5B tests:")
    lines.append("  XKF1-driven adaptive blending of naive, constant-velocity, and ARX V9 components.")
    lines.append("")
    lines.append("Adaptive prediction:")
    lines.append("  x_adaptive[k+1] = w_naive*x_naive[k+1] + w_cv*x_cv[k+1] + w_arx*x_arx[k+1]")
    lines.append("")
    lines.append("Weight smoothing:")
    lines.append("  W_current = 0.9*W_previous + 0.1*W_target")
    lines.append("")
    lines.append("Decision rule:")
    lines.append("  Recommend adaptive weighting only if it improves recovery metrics over original V9 recovery.")
    lines.append("")

    if len(summary_df) == 0:
        lines.append("No files were processed.")
    else:
        n = len(summary_df)
        better = int(summary_df["adaptive_better_than_original_rmse"].sum())

        lines.append("Overall Phase 5B result:")
        lines.append(f"  Files evaluated: {n}")
        lines.append(f"  Adaptive better than original V9 by RMSE: {better}/{n}")
        lines.append("")
        lines.append("Mean metrics:")
        lines.append(f"  Mean attacked RMSE: {summary_df['attacked_rmse'].mean()}")
        lines.append(f"  Mean original V9 recovered RMSE: {summary_df['original_v9_recovered_rmse'].mean()}")
        lines.append(f"  Mean adaptive V9 recovered RMSE: {summary_df['adaptive_v9_recovered_rmse'].mean()}")
        lines.append("")
        lines.append(f"  Mean original improvement over attacked RMSE (%): {summary_df['original_improvement_over_attacked_rmse_percent'].mean()}")
        lines.append(f"  Mean adaptive improvement over attacked RMSE (%): {summary_df['adaptive_improvement_over_attacked_rmse_percent'].mean()}")
        lines.append(f"  Mean adaptive improvement over original RMSE (%): {summary_df['adaptive_improvement_over_original_rmse_percent'].mean()}")
        lines.append("")
        lines.append(f"  Mean attacked MAE: {summary_df['attacked_mae'].mean()}")
        lines.append(f"  Mean original V9 recovered MAE: {summary_df['original_v9_recovered_mae'].mean()}")
        lines.append(f"  Mean adaptive V9 recovered MAE: {summary_df['adaptive_v9_recovered_mae'].mean()}")
        lines.append("")

        if better == n:
            lines.append("Interpretation:")
            lines.append("  Adaptive weighting improved every evaluated Roll/Pitch case.")
            lines.append("  It may be reported as a useful experimental extension.")
        elif better > 0:
            lines.append("Interpretation:")
            lines.append("  Adaptive weighting improved some cases but not all.")
            lines.append("  It should remain an experimental extension, not the official method.")
        else:
            lines.append("Interpretation:")
            lines.append("  Adaptive weighting did not outperform original V9 recovery.")
            lines.append("  Keep original V9 recovery as the official method.")

        lines.append("")
        lines.append("Per-file results:")
        lines.append("-" * 90)

        for _, r in summary_df.iterrows():
            lines.append(f"Target: {r['target']}")
            lines.append(f"  Input detection file: {r['input_detection_file']}")
            lines.append(f"  Prediction file: {r['prediction_file']}")
            lines.append(f"  XKF1 file: {r['xkf1_file']}")
            lines.append(f"  Attacked RMSE: {r['attacked_rmse']}")
            lines.append(f"  Original V9 recovered RMSE: {r['original_v9_recovered_rmse']}")
            lines.append(f"  Adaptive V9 recovered RMSE: {r['adaptive_v9_recovered_rmse']}")
            lines.append(f"  Adaptive improvement over original RMSE (%): {r['adaptive_improvement_over_original_rmse_percent']}")
            lines.append(f"  Adaptive better than original: {r['adaptive_better_than_original_rmse']}")
            lines.append(f"  Output CSV: {r['output_csv']}")
            lines.append(f"  Comparison plot: {r['comparison_plot']}")
            lines.append(f"  Error plot: {r['error_plot']}")
            lines.append(f"  Weight plot: {r['weight_plot']}")
            lines.append(f"  Context plot: {r['context_plot']}")
            lines.append("-" * 90)

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 90)
    print("PHASE 5B: XKF1-DRIVEN ADAPTIVE-WEIGHT RECOVERY EXPERIMENT")
    print("=" * 90)
    print("")
    print("This script does NOT overwrite official Phase 5.")
    print("Outputs:")
    print(f"  CSVs:    {OUT_CSV_DIR}")
    print(f"  Figures: {OUT_FIG_DIR}")
    print(f"  Reports: {OUT_REPORT_DIR}")
    print("")

    files = sorted(glob.glob(f"{DETECTION_DIR}/*{OFFICIAL_DETECTOR_TAG}.csv"))

    files = [
        f for f in files
        if "_Roll_" in os.path.basename(f) or "_Pitch_" in os.path.basename(f)
    ]

    print(f"Official W10_N3 Roll/Pitch files selected: {len(files)}")
    print("")

    if len(files) == 0:
        raise FileNotFoundError("No official W10_N3 Roll/Pitch files found.")

    summaries = []

    for f in files:
        print("-" * 90)
        print(f"Processing:")
        print(f"  {f}")

        summary = process_one_file(f)
        summaries.append(summary)

        print(f"  Target: {summary['target']}")
        print(f"  Attacked RMSE: {summary['attacked_rmse']}")
        print(f"  Original V9 recovered RMSE: {summary['original_v9_recovered_rmse']}")
        print(f"  Adaptive V9 recovered RMSE: {summary['adaptive_v9_recovered_rmse']}")
        print(f"  Adaptive improvement over original RMSE (%): {summary['adaptive_improvement_over_original_rmse_percent']}")
        print(f"  Adaptive better than original: {summary['adaptive_better_than_original_rmse']}")
        print("")

    summary_df = pd.DataFrame(summaries)

    summary_path = f"{OUT_REPORT_DIR}/phase5b_adaptive_weight_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    report_path = write_report(summary_df)

    print("=" * 90)
    print("PHASE 5B COMPLETE")
    print("=" * 90)
    print("")
    print(f"Summary CSV:")
    print(f"  {summary_path}")
    print("")
    print(f"Interpretation report:")
    print(f"  {report_path}")
    print("")

    if len(summary_df) > 0:
        better = int(summary_df["adaptive_better_than_original_rmse"].sum())
        n = len(summary_df)

        print("Overall:")
        print(f"  Files evaluated: {n}")
        print(f"  Adaptive better than original V9: {better}/{n}")
        print(f"  Mean attacked RMSE: {summary_df['attacked_rmse'].mean()}")
        print(f"  Mean original V9 recovered RMSE: {summary_df['original_v9_recovered_rmse'].mean()}")
        print(f"  Mean adaptive V9 recovered RMSE: {summary_df['adaptive_v9_recovered_rmse'].mean()}")
        print(f"  Mean adaptive improvement over original RMSE (%): {summary_df['adaptive_improvement_over_original_rmse_percent'].mean()}")
        print("")

        if better == n:
            print("Decision: Adaptive weighting improved all cases. It may be considered as an extension.")
        elif better > 0:
            print("Decision: Adaptive weighting is mixed. Keep original V9 as official; report adaptive as experimental.")
        else:
            print("Decision: Adaptive weighting did not beat original V9. Keep original V9 as official.")


if __name__ == "__main__":
    main()