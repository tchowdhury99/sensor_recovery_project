#!/usr/bin/env python3

"""
PHASE 5B METHOD 2:
RECENT-ERROR ADAPTIVE-WEIGHT V9 RECOVERY WITH FREEZE-ON-DETECTION

This is a separate experiment from official Phase 5.

Official Phase 5:
    - Uses original V9 trained hybrid prediction.
    - Uses W10_N3_mean1x detection.
    - Already completed.

This script:
    - Reads official W10_N3_mean1x detection CSVs.
    - Loads V9 component predictions from PredictionFile:
        naive component
        constant-velocity component
        ARX component
    - Computes recent-error inverse weights.
    - Freezes weight updates when detection is active.
    - Compares original V9 recovery against recent-error adaptive recovery.

Important:
    This script does NOT overwrite official Phase 5 outputs.
"""

import os
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

OUT_CSV_DIR = f"{BASE}/recovery_results_v9_recent_error_adaptive"
OUT_FIG_DIR = f"{BASE}/figures/recovery_v9_recent_error_adaptive"
OUT_REPORT_DIR = f"{BASE}/recovery_reports_v9_recent_error_adaptive"

os.makedirs(OUT_CSV_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_REPORT_DIR, exist_ok=True)


# ============================================================
# Experiment settings
# ============================================================

OFFICIAL_DETECTOR_TAG = "W10_N3_mean1x_detection"

# Start with Roll and Pitch only.
TARGETS = ["Roll", "Pitch"]

# Recent-error adaptive settings.
ERROR_WINDOW = 10
EPSILON = 1e-6

# Weight smoothing prevents sudden jumps.
SMOOTHING_PREVIOUS = 0.90
SMOOTHING_NEW = 0.10

# Starting weights.
INITIAL_WEIGHTS = {
    "w_naive": 1.0 / 3.0,
    "w_cv": 1.0 / 3.0,
    "w_arx": 1.0 / 3.0,
}


# ============================================================
# Utility functions
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
    For Phase 5B Method 2, we evaluate Roll/Pitch first.
    Therefore normal subtraction is used.
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


def parse_baseline_from_detection_file(path):
    """
    Example input:
        baseline_05_mixed_maneuver_Pitch_ramp_W10_N3_mean1x_detection.csv

    Output:
        baseline_05_mixed_maneuver
    """
    name = os.path.basename(path).replace(".csv", "")

    parts = name.split("_")

    # Handles:
    # baseline_01_hover_Roll_bias_W10_N3_mean1x_detection
    # baseline_02_high_altitude_hover_Pitch_bias_W10_N3_mean1x_detection
    # baseline_05_mixed_maneuver_Roll_ramp_W10_N3_mean1x_detection
    for axis in ["Roll", "Pitch", "Yaw"]:
        if axis in parts:
            idx = parts.index(axis)
            return "_".join(parts[:idx])

    return "unknown_baseline"


# ============================================================
# Load V9 component predictions
# ============================================================

def load_prediction_components(det_df, target):
    """
    Detection CSV contains a PredictionFile column.

    The PredictionFile contains:
        {target}_hybrid_prediction
        {target}_arx_component
        {target}_naive_component
        {target}_constant_velocity_component
    """

    if "PredictionFile" not in det_df.columns:
        raise ValueError("Detection CSV does not contain PredictionFile column.")

    prediction_file = str(det_df["PredictionFile"].iloc[0])

    if not os.path.exists(prediction_file):
        raise FileNotFoundError(f"PredictionFile not found: {prediction_file}")

    pred_df = pd.read_csv(prediction_file)

    required_cols = {
        "true_future": f"{target}_true_future",
        "hybrid": f"{target}_hybrid_prediction",
        "arx": f"{target}_arx_component",
        "naive": f"{target}_naive_component",
        "cv": f"{target}_constant_velocity_component",
    }

    for label, col in required_cols.items():
        if col not in pred_df.columns:
            raise ValueError(
                f"Missing {label} column in prediction file.\n"
                f"File: {prediction_file}\n"
                f"Expected column: {col}"
            )

    n = len(det_df)

    if len(pred_df) < n:
        raise ValueError(
            f"Prediction file has fewer rows than detection file.\n"
            f"Detection rows: {n}\n"
            f"Prediction rows: {len(pred_df)}\n"
            f"Prediction file: {prediction_file}"
        )

    out = pd.DataFrame(index=det_df.index)

    out[f"{target}_component_true_future"] = pred_df[required_cols["true_future"]].iloc[:n].to_numpy()
    out[f"{target}_component_hybrid"] = pred_df[required_cols["hybrid"]].iloc[:n].to_numpy()
    out[f"{target}_component_arx"] = pred_df[required_cols["arx"]].iloc[:n].to_numpy()
    out[f"{target}_component_naive"] = pred_df[required_cols["naive"]].iloc[:n].to_numpy()
    out[f"{target}_component_cv"] = pred_df[required_cols["cv"]].iloc[:n].to_numpy()
    out["Phase5B_PredictionFile"] = prediction_file

    return out, prediction_file


# ============================================================
# Recent-error adaptive weighting
# ============================================================

def compute_recent_error_adaptive_weights(
    clean_future,
    naive_pred,
    cv_pred,
    arx_pred,
    detection_flag,
    window=10,
    epsilon=1e-6,
):
    """
    Compute recent-error inverse adaptive weights.

    Important freeze rule:
        If detection_flag[k] is True, do NOT update weights using that sample.
        Instead, reuse the last clean weights.

    For clean samples:
        E_model = sum absolute prediction error over last clean window.
        Inv_model = 1 / (E_model + epsilon)
        w_model = Inv_model / sum(Invs)

    Then smooth:
        W_current = 0.9 * W_previous + 0.1 * W_new
    """

    clean_future = np.asarray(clean_future, dtype=float)
    naive_pred = np.asarray(naive_pred, dtype=float)
    cv_pred = np.asarray(cv_pred, dtype=float)
    arx_pred = np.asarray(arx_pred, dtype=float)
    detection_flag = np.asarray(detection_flag, dtype=bool)

    n = len(clean_future)

    w_naive = np.zeros(n)
    w_cv = np.zeros(n)
    w_arx = np.zeros(n)

    e_naive_recent = []
    e_cv_recent = []
    e_arx_recent = []

    prev_w = INITIAL_WEIGHTS.copy()

    frozen_flag = np.zeros(n, dtype=bool)

    for k in range(n):

        # Default: keep previous weights.
        current_w = prev_w.copy()

        if detection_flag[k]:
            # Freeze weights during detected attack.
            frozen_flag[k] = True

        else:
            # Use only trusted clean point to update recent error.
            e_naive = abs(clean_future[k] - naive_pred[k])
            e_cv = abs(clean_future[k] - cv_pred[k])
            e_arx = abs(clean_future[k] - arx_pred[k])

            e_naive_recent.append(e_naive)
            e_cv_recent.append(e_cv)
            e_arx_recent.append(e_arx)

            if len(e_naive_recent) > window:
                e_naive_recent.pop(0)
                e_cv_recent.pop(0)
                e_arx_recent.pop(0)

            E_naive = np.sum(e_naive_recent)
            E_cv = np.sum(e_cv_recent)
            E_arx = np.sum(e_arx_recent)

            inv_naive = 1.0 / (E_naive + epsilon)
            inv_cv = 1.0 / (E_cv + epsilon)
            inv_arx = 1.0 / (E_arx + epsilon)

            total_inv = inv_naive + inv_cv + inv_arx

            raw_w = {
                "w_naive": inv_naive / total_inv,
                "w_cv": inv_cv / total_inv,
                "w_arx": inv_arx / total_inv,
            }

            current_w = {
                "w_naive": SMOOTHING_PREVIOUS * prev_w["w_naive"] + SMOOTHING_NEW * raw_w["w_naive"],
                "w_cv": SMOOTHING_PREVIOUS * prev_w["w_cv"] + SMOOTHING_NEW * raw_w["w_cv"],
                "w_arx": SMOOTHING_PREVIOUS * prev_w["w_arx"] + SMOOTHING_NEW * raw_w["w_arx"],
            }

            # Normalize defensively.
            s = current_w["w_naive"] + current_w["w_cv"] + current_w["w_arx"]
            current_w["w_naive"] /= s
            current_w["w_cv"] /= s
            current_w["w_arx"] /= s

        w_naive[k] = current_w["w_naive"]
        w_cv[k] = current_w["w_cv"]
        w_arx[k] = current_w["w_arx"]

        prev_w = current_w

    weights = pd.DataFrame({
        "w_naive_recent_error": w_naive,
        "w_cv_recent_error": w_cv,
        "w_arx_recent_error": w_arx,
        "weights_frozen_by_detection": frozen_flag,
    })

    return weights


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
    original_col = f"OriginalV9RecoveredFuture_{target}_RecentErrorPhase5B"
    adaptive_col = f"RecentErrorAdaptiveRecoveredFuture_{target}_Phase5B"

    plt.figure(figsize=(14, 6))

    plt.plot(t, df[clean_col], linewidth=1.6, label="Clean/reference future")
    plt.plot(t, df[attacked_col], linewidth=1.0, alpha=0.7, label="Attacked future")
    plt.plot(t, df[original_col], linewidth=1.4, label="Original V9 recovered")
    plt.plot(t, df[adaptive_col], linewidth=1.4, label="Recent-error adaptive recovered")

    plt.title(f"Phase 5B Recent-Error Adaptive Recovery Comparison: {target}")
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
        df[f"AttackedError_{target}_RecentErrorPhase5B"],
        linewidth=1.0,
        alpha=0.7,
        label="Attacked error",
    )

    plt.plot(
        t,
        df[f"OriginalV9RecoveredError_{target}_RecentErrorPhase5B"],
        linewidth=1.3,
        label="Original V9 recovered error",
    )

    plt.plot(
        t,
        df[f"RecentErrorAdaptiveRecoveredError_{target}_Phase5B"],
        linewidth=1.3,
        label="Recent-error adaptive recovered error",
    )

    plt.axhline(0, linewidth=1.0)
    plt.title(f"Phase 5B Recent-Error Error Comparison: {target}")
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

    plt.plot(t, df["w_naive_recent_error"], linewidth=1.4, label="w_naive")
    plt.plot(t, df["w_cv_recent_error"], linewidth=1.4, label="w_cv")
    plt.plot(t, df["w_arx_recent_error"], linewidth=1.4, label="w_arx")

    plt.title(f"Phase 5B Recent-Error Adaptive Weight Trace: {target}")
    plt.xlabel(xlabel)
    plt.ylabel("weight")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_freeze_plot(df, path, target):
    time_col = find_time_column(df)

    if time_col is not None:
        t = df[time_col].to_numpy()
        xlabel = time_col
    else:
        t = np.arange(len(df))
        xlabel = "sample index"

    plt.figure(figsize=(14, 3))

    plt.plot(
        t,
        df["weights_frozen_by_detection"].astype(int),
        linewidth=1.2,
        label="weight update frozen",
    )

    plt.title(f"Phase 5B Weight Freeze Flag: {target}")
    plt.xlabel(xlabel)
    plt.ylabel("frozen")
    plt.yticks([0, 1], ["update", "freeze"])
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
# Process one file
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

    # Load component predictions from PredictionFile.
    comp_df, prediction_file = load_prediction_components(df, target)

    for c in comp_df.columns:
        df[c] = comp_df[c].values

    naive_col = f"{target}_component_naive"
    cv_col = f"{target}_component_cv"
    arx_col = f"{target}_component_arx"

    # Compute recent-error adaptive weights with freeze-on-detection.
    weights_df = compute_recent_error_adaptive_weights(
        clean_future=df[clean_col].astype(float).to_numpy(),
        naive_pred=df[naive_col].astype(float).to_numpy(),
        cv_pred=df[cv_col].astype(float).to_numpy(),
        arx_pred=df[arx_col].astype(float).to_numpy(),
        detection_flag=det_bool,
        window=ERROR_WINDOW,
        epsilon=EPSILON,
    )

    for c in weights_df.columns:
        df[c] = weights_df[c].values

    # Adaptive prediction.
    adaptive_pred_col = f"RecentErrorAdaptivePredictedFuture_{target}_Phase5B"

    df[adaptive_pred_col] = (
        df["w_naive_recent_error"].astype(float) * df[naive_col].astype(float)
        +
        df["w_cv_recent_error"].astype(float) * df[cv_col].astype(float)
        +
        df["w_arx_recent_error"].astype(float) * df[arx_col].astype(float)
    )

    # Original V9 recovery.
    original_recovered_col = f"OriginalV9RecoveredFuture_{target}_RecentErrorPhase5B"

    df[original_recovered_col] = np.where(
        det_bool,
        df[original_v9_col].astype(float).to_numpy(),
        df[attacked_col].astype(float).to_numpy(),
    )

    # Recent-error adaptive recovery.
    adaptive_recovered_col = f"RecentErrorAdaptiveRecoveredFuture_{target}_Phase5B"

    df[adaptive_recovered_col] = np.where(
        det_bool,
        df[adaptive_pred_col].astype(float).to_numpy(),
        df[attacked_col].astype(float).to_numpy(),
    )

    # Errors.
    attacked_err = compute_error(target, df[attacked_col], df[clean_col])
    original_err = compute_error(target, df[original_recovered_col], df[clean_col])
    adaptive_err = compute_error(target, df[adaptive_recovered_col], df[clean_col])

    df[f"AttackedError_{target}_RecentErrorPhase5B"] = attacked_err
    df[f"OriginalV9RecoveredError_{target}_RecentErrorPhase5B"] = original_err
    df[f"RecentErrorAdaptiveRecoveredError_{target}_Phase5B"] = adaptive_err

    attack_active_count = np.nan
    if "AttackActive" in df.columns:
        attack_active_count = int(np.sum(to_bool_series(df["AttackActive"]).to_numpy()))

    base_name = os.path.splitext(os.path.basename(det_path))[0]

    out_csv = f"{OUT_CSV_DIR}/{base_name}_RECENT_ERROR_ADAPTIVE_PHASE5B.csv"
    df.to_csv(out_csv, index=False)

    comparison_plot = f"{OUT_FIG_DIR}/{base_name}_recent_error_recovery_comparison_{target}.png"
    error_plot = f"{OUT_FIG_DIR}/{base_name}_recent_error_error_comparison_{target}.png"
    weight_plot = f"{OUT_FIG_DIR}/{base_name}_recent_error_weight_trace_{target}.png"
    freeze_plot = f"{OUT_FIG_DIR}/{base_name}_recent_error_freeze_flag_{target}.png"

    make_recovery_comparison_plot(df, comparison_plot, target)
    make_error_comparison_plot(df, error_plot, target)
    make_weight_trace_plot(df, weight_plot, target)
    make_freeze_plot(df, freeze_plot, target)

    summary = {
        "input_detection_file": det_path,
        "prediction_file": prediction_file,
        "output_csv": out_csv,
        "target": target,
        "baseline": parse_baseline_from_detection_file(det_path),
        "n_rows": len(df),
        "detected_count": int(np.sum(det_bool)),
        "attack_active_count": attack_active_count,
        "error_window": ERROR_WINDOW,
        "epsilon": EPSILON,
        "clean_col": clean_col,
        "attacked_col": attacked_col,
        "original_v9_col": original_v9_col,
        "adaptive_prediction_col": adaptive_pred_col,
        "original_recovered_col": original_recovered_col,
        "adaptive_recovered_col": adaptive_recovered_col,
        "comparison_plot": comparison_plot,
        "error_plot": error_plot,
        "weight_plot": weight_plot,
        "freeze_plot": freeze_plot,
    }

    summary.update(build_metrics("attacked", attacked_err))
    summary.update(build_metrics("original_v9_recovered", original_err))
    summary.update(build_metrics("recent_error_adaptive_recovered", adaptive_err))

    summary["original_improvement_over_attacked_rmse_percent"] = percent_improvement(
        summary["attacked_rmse"],
        summary["original_v9_recovered_rmse"],
    )

    summary["adaptive_improvement_over_attacked_rmse_percent"] = percent_improvement(
        summary["attacked_rmse"],
        summary["recent_error_adaptive_recovered_rmse"],
    )

    summary["adaptive_improvement_over_original_rmse_percent"] = percent_improvement(
        summary["original_v9_recovered_rmse"],
        summary["recent_error_adaptive_recovered_rmse"],
    )

    summary["adaptive_better_than_original_rmse"] = (
        summary["recent_error_adaptive_recovered_rmse"] < summary["original_v9_recovered_rmse"]
    )

    summary["adaptive_better_than_original_mae"] = (
        summary["recent_error_adaptive_recovered_mae"] < summary["original_v9_recovered_mae"]
    )

    return summary


# ============================================================
# Write interpretation report
# ============================================================

def write_report(summary_df):
    report_path = f"{OUT_REPORT_DIR}/phase5b_recent_error_adaptive_interpretation.txt"

    lines = []

    lines.append("PHASE 5B METHOD 2: RECENT-ERROR ADAPTIVE-WEIGHT RECOVERY")
    lines.append("=" * 90)
    lines.append("")
    lines.append("This is a separate experiment from official Phase 5.")
    lines.append("")
    lines.append("Official Phase 5 remains:")
    lines.append("  Original V9 trained hybrid recovery using W10_N3_mean1x detection.")
    lines.append("")
    lines.append("Phase 5B Method 2 tests:")
    lines.append("  Recent-error inverse adaptive weighting with freeze-on-detection.")
    lines.append("")
    lines.append("Adaptive formula:")
    lines.append("  x_adaptive[k+1] = w_naive*x_naive[k+1] + w_cv*x_cv[k+1] + w_arx*x_arx[k+1]")
    lines.append("")
    lines.append("Recent-error weights:")
    lines.append("  E_model = sum absolute prediction error over the recent clean window.")
    lines.append("  Inv_model = 1 / (E_model + epsilon).")
    lines.append("  w_model = Inv_model / sum(all inverse errors).")
    lines.append("")
    lines.append("Freeze rule:")
    lines.append("  When PointDetectionFlag is active, weight updates are frozen.")
    lines.append("  This prevents attacked measurements from manipulating the adaptive weighting logic.")
    lines.append("")
    lines.append(f"Window size: {ERROR_WINDOW}")
    lines.append(f"Epsilon: {EPSILON}")
    lines.append("")

    if len(summary_df) == 0:
        lines.append("No files were processed.")
    else:
        n = len(summary_df)
        better_rmse = int(summary_df["adaptive_better_than_original_rmse"].sum())
        better_mae = int(summary_df["adaptive_better_than_original_mae"].sum())

        lines.append("Overall result:")
        lines.append(f"  Files evaluated: {n}")
        lines.append(f"  Adaptive better than original by RMSE: {better_rmse}/{n}")
        lines.append(f"  Adaptive better than original by MAE: {better_mae}/{n}")
        lines.append("")

        lines.append("Mean RMSE metrics:")
        lines.append(f"  Mean attacked RMSE: {summary_df['attacked_rmse'].mean()}")
        lines.append(f"  Mean original V9 recovered RMSE: {summary_df['original_v9_recovered_rmse'].mean()}")
        lines.append(f"  Mean recent-error adaptive recovered RMSE: {summary_df['recent_error_adaptive_recovered_rmse'].mean()}")
        lines.append("")
        lines.append("Mean MAE metrics:")
        lines.append(f"  Mean attacked MAE: {summary_df['attacked_mae'].mean()}")
        lines.append(f"  Mean original V9 recovered MAE: {summary_df['original_v9_recovered_mae'].mean()}")
        lines.append(f"  Mean recent-error adaptive recovered MAE: {summary_df['recent_error_adaptive_recovered_mae'].mean()}")
        lines.append("")
        lines.append("Improvement metrics:")
        lines.append(f"  Mean original improvement over attacked RMSE (%): {summary_df['original_improvement_over_attacked_rmse_percent'].mean()}")
        lines.append(f"  Mean adaptive improvement over attacked RMSE (%): {summary_df['adaptive_improvement_over_attacked_rmse_percent'].mean()}")
        lines.append(f"  Mean adaptive improvement over original RMSE (%): {summary_df['adaptive_improvement_over_original_rmse_percent'].mean()}")
        lines.append("")

        if better_rmse == n:
            lines.append("Interpretation:")
            lines.append("  Recent-error adaptive weighting improved every evaluated Roll/Pitch case by RMSE.")
            lines.append("  It can be considered a successful experimental extension.")
        elif better_rmse > 0:
            lines.append("Interpretation:")
            lines.append("  Recent-error adaptive weighting improved some cases but not all.")
            lines.append("  Original V9 should remain the official recovery method unless the mixed cases are tuned and justified.")
        else:
            lines.append("Interpretation:")
            lines.append("  Recent-error adaptive weighting did not outperform original V9 recovery.")
            lines.append("  Original V9 remains the official recovery method.")
        lines.append("")

        lines.append("Per-file results:")
        lines.append("-" * 90)

        for _, r in summary_df.iterrows():
            lines.append(f"Target: {r['target']}")
            lines.append(f"  Baseline: {r['baseline']}")
            lines.append(f"  Input detection file: {r['input_detection_file']}")
            lines.append(f"  Prediction file: {r['prediction_file']}")
            lines.append(f"  Attacked RMSE: {r['attacked_rmse']}")
            lines.append(f"  Original V9 recovered RMSE: {r['original_v9_recovered_rmse']}")
            lines.append(f"  Recent-error adaptive recovered RMSE: {r['recent_error_adaptive_recovered_rmse']}")
            lines.append(f"  Adaptive improvement over original RMSE (%): {r['adaptive_improvement_over_original_rmse_percent']}")
            lines.append(f"  Adaptive better than original by RMSE: {r['adaptive_better_than_original_rmse']}")
            lines.append(f"  Output CSV: {r['output_csv']}")
            lines.append(f"  Comparison plot: {r['comparison_plot']}")
            lines.append(f"  Error plot: {r['error_plot']}")
            lines.append(f"  Weight plot: {r['weight_plot']}")
            lines.append(f"  Freeze plot: {r['freeze_plot']}")
            lines.append("-" * 90)

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 90)
    print("PHASE 5B METHOD 2: RECENT-ERROR ADAPTIVE-WEIGHT RECOVERY")
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
        print("Processing:")
        print(f"  {f}")

        summary = process_one_file(f)
        summaries.append(summary)

        print(f"  Target: {summary['target']}")
        print(f"  Attacked RMSE: {summary['attacked_rmse']}")
        print(f"  Original V9 recovered RMSE: {summary['original_v9_recovered_rmse']}")
        print(f"  Recent-error adaptive recovered RMSE: {summary['recent_error_adaptive_recovered_rmse']}")
        print(f"  Adaptive improvement over original RMSE (%): {summary['adaptive_improvement_over_original_rmse_percent']}")
        print(f"  Adaptive better than original RMSE: {summary['adaptive_better_than_original_rmse']}")
        print("")

    summary_df = pd.DataFrame(summaries)

    summary_path = f"{OUT_REPORT_DIR}/phase5b_recent_error_adaptive_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    report_path = write_report(summary_df)

    print("=" * 90)
    print("PHASE 5B METHOD 2 COMPLETE")
    print("=" * 90)
    print("")
    print("Summary CSV:")
    print(f"  {summary_path}")
    print("")
    print("Interpretation report:")
    print(f"  {report_path}")
    print("")

    if len(summary_df) > 0:
        better = int(summary_df["adaptive_better_than_original_rmse"].sum())
        n = len(summary_df)

        print("Overall:")
        print(f"  Files evaluated: {n}")
        print(f"  Adaptive better than original V9 by RMSE: {better}/{n}")
        print(f"  Mean attacked RMSE: {summary_df['attacked_rmse'].mean()}")
        print(f"  Mean original V9 recovered RMSE: {summary_df['original_v9_recovered_rmse'].mean()}")
        print(f"  Mean recent-error adaptive recovered RMSE: {summary_df['recent_error_adaptive_recovered_rmse'].mean()}")
        print(f"  Mean adaptive improvement over original RMSE (%): {summary_df['adaptive_improvement_over_original_rmse_percent'].mean()}")
        print("")

        if better == n:
            print("Decision: Recent-error adaptive improved all cases. It may be considered a successful extension.")
        elif better > 0:
            print("Decision: Recent-error adaptive is mixed. Keep original V9 as official; report adaptive as experimental.")
        else:
            print("Decision: Recent-error adaptive did not beat original V9. Keep original V9 as official.")


if __name__ == "__main__":
    main()