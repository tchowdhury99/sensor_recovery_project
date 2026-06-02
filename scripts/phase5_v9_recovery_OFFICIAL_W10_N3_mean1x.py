#!/usr/bin/env python3

"""
PHASE 5 OFFICIAL RECOVERY SCRIPT

Purpose:
    Perform recovery after corrected Phase 4 detection.

Important:
    This official Phase 5 script uses ONLY the selected final detector:

        W10_N3_mean1x

    It does NOT process W20_N5, W30_N8, or other detector variants.

Phase 4 final decision:
    Software sensor:
        V9 trained hybrid

    Threshold:
        p99_abs

    Detector:
        W10_N3_mean1x

    Roll/Pitch residual:
        attacked_future[k+1] - V9_prediction[k+1]

    Yaw residual:
        circular angular difference

Phase 5 recovery rule:
    if detection is active:
        recovered_future[k+1] = V9_prediction[k+1]
    else:
        recovered_future[k+1] = attacked_future[k+1]

Input:
    /home/tchowdh4/sensor_recovery_project/attack_detection_v9_corrected/detection_csvs

Output:
    Official recovered CSVs:
        /home/tchowdh4/sensor_recovery_project/recovery_v9_corrected_official/recovered_csvs

    Official reports:
        /home/tchowdh4/sensor_recovery_project/recovery_v9_corrected_official/recovery_reports

    Official figures:
        /home/tchowdh4/sensor_recovery_project/figures/recovery_v9_corrected_official
"""

import os
import glob
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASE = "/home/tchowdh4/sensor_recovery_project"

DETECTION_DIR = f"{BASE}/attack_detection_v9_corrected/detection_csvs"

OUT_CSV_DIR = f"{BASE}/recovery_v9_corrected_official/recovered_csvs"
OUT_REPORT_DIR = f"{BASE}/recovery_v9_corrected_official/recovery_reports"
OUT_FIG_DIR = f"{BASE}/figures/recovery_v9_corrected_official"

os.makedirs(OUT_CSV_DIR, exist_ok=True)
os.makedirs(OUT_REPORT_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


# ============================================================
# Official detector choice
# ============================================================

OFFICIAL_DETECTOR_TAG = "W10_N3_mean1x_detection"

TARGETS = ["Roll", "Pitch", "Yaw"]


# ============================================================
# Utility functions
# ============================================================

def normalize_name(name):
    """
    Normalize a column/file name so matching becomes easier.
    """
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def circular_diff_deg(a, b):
    """
    Signed shortest angular difference a - b in degrees.

    Output range:
        [-180, 180)
    """
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0


def detect_target_from_filename_and_columns(path, df):
    """
    Determine whether this file is Roll, Pitch, or Yaw.
    """
    name = os.path.basename(path).lower()

    for target in TARGETS:
        if target.lower() in name:
            return target

    cols = [c.lower() for c in df.columns]

    for target in TARGETS:
        t = target.lower()
        for c in cols:
            if t in c:
                return target

    raise ValueError(
        f"Could not determine target from file name or columns: {path}"
    )


def find_time_column(df):
    """
    Find time column if available.
    If no time column exists, plotting will use sample index.
    """
    candidates = [
        "t",
        "time",
        "timesec",
        "time_sec",
        "timestamp",
        "timeus",
        "time_us",
        "TimeUS",
    ]

    normalized = {normalize_name(c): c for c in df.columns}

    for cand in candidates:
        n = normalize_name(cand)
        if n in normalized:
            return normalized[n]

    return None


def find_detection_column(df):
    """
    Find the detector flag column.

    In your corrected Phase 4 files, this appears to be:
        PointDetectionFlag

    But this function is robust to other possible names.
    """
    cols = list(df.columns)
    norm_to_original = {normalize_name(c): c for c in cols}

    exact_candidates = [
        "pointdetectionflag",
        "point_detection_flag",
        "w10_n3_mean1x",
        "detected_w10_n3_mean1x",
        "detection_w10_n3_mean1x",
        "attack_detected_w10_n3_mean1x",
        "final_detection_w10_n3_mean1x",
    ]

    for cand in exact_candidates:
        n = normalize_name(cand)
        if n in norm_to_original:
            return norm_to_original[n]

    for c in cols:
        n = normalize_name(c)
        if "w10" in n and "n3" in n and "mean1x" in n:
            return c

    fallback_keywords = [
        "final_detection",
        "attack_detected",
        "detected",
        "detection",
        "is_attack",
    ]

    for key in fallback_keywords:
        for c in cols:
            n = normalize_name(c)
            if key in n:
                return c

    raise ValueError(
        "Could not find detection flag column. "
        "Expected PointDetectionFlag or a W10_N3_mean1x detection column."
    )


def find_attacked_future_column(df, target):
    """
    Find attacked future measurement column.

    Expected examples:
        AttackedFuture_Roll
        AttackedFuture_Pitch
        AttackedFuture_Yaw
    """
    cols = list(df.columns)
    t = target.lower()

    # Best expected pattern.
    expected = f"attackedfuture_{t}"
    for c in cols:
        if normalize_name(c) == expected:
            return c

    # Target-specific attacked future.
    for c in cols:
        n = normalize_name(c)
        if t in n and "attacked" in n and "future" in n:
            return c

    # Generic attacked future.
    for c in cols:
        n = normalize_name(c)
        if "attacked" in n and "future" in n:
            if "active" not in n and "detected" not in n and "detection" not in n:
                return c

    # Target-specific attacked.
    for c in cols:
        n = normalize_name(c)
        if t in n and "attacked" in n:
            if "active" not in n and "detected" not in n and "detection" not in n:
                return c

    raise ValueError(
        f"Could not find attacked future column for target={target}. "
        f"Expected something like AttackedFuture_{target}."
    )


def find_prediction_column(df, target):
    """
    Find V9 predicted future column.

    Expected examples:
        V9PredictedFuture_Roll
        V9PredictedFuture_Pitch
        V9PredictedFuture_Yaw
    """
    cols = list(df.columns)
    t = target.lower()

    # Best expected pattern.
    expected = f"v9predictedfuture_{t}"
    for c in cols:
        if normalize_name(c) == expected:
            return c

    # Target-specific V9 prediction.
    for c in cols:
        n = normalize_name(c)
        if t in n and "v9" in n and ("pred" in n or "prediction" in n):
            return c

    # Target-specific future prediction.
    for c in cols:
        n = normalize_name(c)
        if t in n and "future" in n and ("pred" in n or "prediction" in n):
            return c

    # General V9 prediction.
    for c in cols:
        n = normalize_name(c)
        if "v9" in n and ("pred" in n or "prediction" in n):
            return c

    # Generic prediction fallback.
    for c in cols:
        n = normalize_name(c)
        if "pred" in n or "prediction" in n or "software_sensor" in n:
            if "residual" not in n and "error" not in n:
                return c

    raise ValueError(
        f"Could not find V9 prediction column for target={target}. "
        f"Expected something like V9PredictedFuture_{target}."
    )


def find_clean_reference_column(df, target):
    """
    Find clean/reference future value.

    Expected examples:
        CleanFuture_Roll
        CleanFuture_Pitch
        CleanFuture_Yaw
    """
    cols = list(df.columns)
    t = target.lower()

    expected = f"cleanfuture_{t}"
    for c in cols:
        if normalize_name(c) == expected:
            return c

    strong_patterns = [
        "clean_future",
        "cleanfuture",
        "reference_future",
        "truth_future",
        "ground_truth_future",
        "actual_clean_future",
        "normal_future",
        "original_future",
    ]

    for c in cols:
        n = normalize_name(c)
        if t in n:
            for p in strong_patterns:
                if normalize_name(p) in n:
                    return c

    for c in cols:
        n = normalize_name(c)
        for p in strong_patterns:
            if normalize_name(p) in n:
                return c

    return None


def find_attack_active_column(df):
    """
    Find ground-truth attack-active column if available.

    Expected examples:
        AttackActive
        AttackActive_Roll
        AttackActive_Pitch
        AttackActive_Yaw
    """
    for c in df.columns:
        n = normalize_name(c)
        if "attackactive" in n or "attack_active" in n:
            return c
    return None


def to_bool_series(s):
    """
    Convert detection/attack-active column into boolean.
    Handles:
        0/1
        True/False
        yes/no
        detected/not detected
    """
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
    Compute error against clean/reference future.

    Roll/Pitch:
        measured - reference

    Yaw:
        circular angular difference
    """
    measured = np.asarray(measured, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if target.lower() == "yaw":
        return circular_diff_deg(measured, reference)

    return measured - reference


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


def detection_quality_metrics(det_bool, attack_bool):
    """
    Compute detection quality if AttackActive exists.
    """
    if attack_bool is None:
        return {
            "TP": np.nan,
            "FP": np.nan,
            "TN": np.nan,
            "FN": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "false_alarm_rate": np.nan,
        }

    det = np.asarray(det_bool, dtype=bool)
    atk = np.asarray(attack_bool, dtype=bool)

    tp = int(np.sum(det & atk))
    fp = int(np.sum(det & ~atk))
    tn = int(np.sum(~det & ~atk))
    fn = int(np.sum(~det & atk))

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm_rate,
    }


def safe_filename(name):
    """
    Make safe file name for figure/report outputs.
    """
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


# ============================================================
# Plot functions
# ============================================================

def make_recovery_plot(
    df,
    out_path,
    target,
    time_col,
    clean_col,
    attacked_col,
    pred_col,
    det_col,
    recovered_col,
):
    """
    Plot clean, attacked, V9 prediction, and recovered signal.
    """
    if time_col is not None:
        t = df[time_col].to_numpy()
        x_label = time_col
    else:
        t = np.arange(len(df))
        x_label = "sample index"

    detected = to_bool_series(df[det_col]).to_numpy()

    plt.figure(figsize=(14, 6))

    if clean_col is not None:
        plt.plot(
            t,
            df[clean_col],
            linewidth=1.5,
            label="Clean/reference future",
        )

    plt.plot(
        t,
        df[attacked_col],
        linewidth=1.0,
        alpha=0.7,
        label="Attacked future",
    )

    plt.plot(
        t,
        df[pred_col],
        linewidth=1.0,
        alpha=0.8,
        label="V9 predicted future",
    )

    plt.plot(
        t,
        df[recovered_col],
        linewidth=1.8,
        label="Recovered future",
    )

    # Mark detected samples.
    if np.any(detected):
        det_t = t[detected]

        # Limit vertical markers so figure stays readable.
        max_markers = 250
        if len(det_t) > max_markers:
            idx = np.linspace(0, len(det_t) - 1, max_markers).astype(int)
            det_t = det_t[idx]

        for dt in det_t:
            plt.axvline(dt, alpha=0.08, linewidth=0.8)

    plt.title(f"Official Phase 5 Recovery using V9 and W10_N3_mean1x: {target}")
    plt.xlabel(x_label)
    plt.ylabel(f"{target}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_error_plot(
    df,
    out_path,
    target,
    time_col,
    clean_col,
    attacked_col,
    recovered_col,
):
    """
    Plot attacked error and recovered error against clean/reference.
    """
    if clean_col is None:
        return

    if time_col is not None:
        t = df[time_col].to_numpy()
        x_label = time_col
    else:
        t = np.arange(len(df))
        x_label = "sample index"

    attacked_err = compute_error(target, df[attacked_col], df[clean_col])
    recovered_err = compute_error(target, df[recovered_col], df[clean_col])

    plt.figure(figsize=(14, 5))

    plt.plot(
        t,
        attacked_err,
        linewidth=1.0,
        alpha=0.8,
        label="Attacked error",
    )

    plt.plot(
        t,
        recovered_err,
        linewidth=1.4,
        label="Recovered error",
    )

    plt.axhline(0, linewidth=1.0)

    if target.lower() == "yaw":
        ylabel = "Circular error"
    else:
        ylabel = "Error"

    plt.title(f"Official Phase 5 Error Reduction: {target}")
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# Main recovery logic for one CSV file
# ============================================================

def process_one_file(path):
    """
    Process one official W10_N3_mean1x detection CSV.
    """
    df = pd.read_csv(path)

    target = detect_target_from_filename_and_columns(path, df)

    time_col = find_time_column(df)
    det_col = find_detection_column(df)
    attacked_col = find_attacked_future_column(df, target)
    pred_col = find_prediction_column(df, target)
    clean_col = find_clean_reference_column(df, target)
    attack_active_col = find_attack_active_column(df)

    det_bool = to_bool_series(df[det_col])

    recovered_col = f"{target}_RecoveredFuture_V9_W10_N3_mean1x_Phase5"
    source_col = f"{target}_RecoverySource_Phase5"

    # ========================================================
    # Core Phase 5 recovery rule
    # ========================================================
    df[recovered_col] = np.where(
        det_bool.to_numpy(),
        df[pred_col].astype(float).to_numpy(),
        df[attacked_col].astype(float).to_numpy(),
    )

    df[source_col] = np.where(
        det_bool.to_numpy(),
        "V9_prediction",
        "attacked_measurement",
    )

    # For Yaw, preserve circular convention.
    if target.lower() == "yaw":
        attacked_values = df[attacked_col].astype(float).to_numpy()
        recovered_values = df[recovered_col].astype(float).to_numpy()

        finite_attacked = attacked_values[np.isfinite(attacked_values)]

        if len(finite_attacked) > 0:
            # If yaw is represented as 0..360, keep recovered in 0..360.
            if np.nanmin(finite_attacked) >= -1 and np.nanmax(finite_attacked) > 180:
                df[recovered_col] = recovered_values % 360.0
            else:
                # Otherwise keep recovered around [-180, 180).
                df[recovered_col] = circular_diff_deg(recovered_values, 0.0)

    # ========================================================
    # Error metrics
    # ========================================================
    attacked_rmse = np.nan
    recovered_rmse = np.nan
    attacked_mae = np.nan
    recovered_mae = np.nan
    rmse_improvement_percent = np.nan
    mae_improvement_percent = np.nan

    attacked_attack_region_rmse = np.nan
    recovered_attack_region_rmse = np.nan
    attacked_clean_region_rmse = np.nan
    recovered_clean_region_rmse = np.nan

    if clean_col is not None:
        attacked_error_col = f"{target}_AttackedErrorVsClean_Phase5"
        recovered_error_col = f"{target}_RecoveredErrorVsClean_Phase5"

        df[attacked_error_col] = compute_error(
            target,
            df[attacked_col],
            df[clean_col],
        )

        df[recovered_error_col] = compute_error(
            target,
            df[recovered_col],
            df[clean_col],
        )

        attacked_rmse = rmse(df[attacked_error_col])
        recovered_rmse = rmse(df[recovered_error_col])

        attacked_mae = mae(df[attacked_error_col])
        recovered_mae = mae(df[recovered_error_col])

        if np.isfinite(attacked_rmse) and attacked_rmse > 0:
            rmse_improvement_percent = 100.0 * (
                attacked_rmse - recovered_rmse
            ) / attacked_rmse

        if np.isfinite(attacked_mae) and attacked_mae > 0:
            mae_improvement_percent = 100.0 * (
                attacked_mae - recovered_mae
            ) / attacked_mae

        if attack_active_col is not None:
            attack_bool = to_bool_series(df[attack_active_col]).to_numpy()

            if np.any(attack_bool):
                attacked_attack_region_rmse = rmse(
                    df.loc[attack_bool, attacked_error_col]
                )
                recovered_attack_region_rmse = rmse(
                    df.loc[attack_bool, recovered_error_col]
                )

            if np.any(~attack_bool):
                attacked_clean_region_rmse = rmse(
                    df.loc[~attack_bool, attacked_error_col]
                )
                recovered_clean_region_rmse = rmse(
                    df.loc[~attack_bool, recovered_error_col]
                )
    else:
        warnings.warn(
            f"No clean/reference column found for {os.path.basename(path)}. "
            "Recovered CSV will still be produced, but RMSE/MAE will be NaN."
        )

    # ========================================================
    # Detection quality metrics
    # ========================================================
    attack_bool_for_quality = None
    attack_active_count = np.nan

    if attack_active_col is not None:
        attack_bool_for_quality = to_bool_series(df[attack_active_col]).to_numpy()
        attack_active_count = int(np.sum(attack_bool_for_quality))

    quality = detection_quality_metrics(
        det_bool.to_numpy(),
        attack_bool_for_quality,
    )

    # ========================================================
    # Save recovered CSV
    # ========================================================
    base_name = os.path.splitext(os.path.basename(path))[0]
    out_csv = f"{OUT_CSV_DIR}/{base_name}_OFFICIAL_PHASE5_RECOVERED.csv"
    df.to_csv(out_csv, index=False)

    # ========================================================
    # Save figures
    # ========================================================
    recovery_plot = f"{OUT_FIG_DIR}/{base_name}_official_phase5_recovery_{target}.png"
    error_plot = f"{OUT_FIG_DIR}/{base_name}_official_phase5_error_reduction_{target}.png"

    make_recovery_plot(
        df=df,
        out_path=recovery_plot,
        target=target,
        time_col=time_col,
        clean_col=clean_col,
        attacked_col=attacked_col,
        pred_col=pred_col,
        det_col=det_col,
        recovered_col=recovered_col,
    )

    make_error_plot(
        df=df,
        out_path=error_plot,
        target=target,
        time_col=time_col,
        clean_col=clean_col,
        attacked_col=attacked_col,
        recovered_col=recovered_col,
    )

    # ========================================================
    # Summary row
    # ========================================================
    summary = {
        "input_file": path,
        "output_recovered_csv": out_csv,
        "target": target,
        "official_detector": "W10_N3_mean1x",
        "threshold_type": "p99_abs",
        "software_sensor": "V9_trained_hybrid",
        "time_col": time_col,
        "detection_col": det_col,
        "attacked_future_col": attacked_col,
        "v9_prediction_col": pred_col,
        "clean_reference_col": clean_col,
        "attack_active_col": attack_active_col,
        "n_rows": len(df),
        "detected_count": int(np.sum(det_bool.to_numpy())),
        "attack_active_count": attack_active_count,
        "recovered_by_v9_count": int(np.sum(det_bool.to_numpy())),
        "kept_attacked_measurement_count": int(len(df) - np.sum(det_bool.to_numpy())),
        "attacked_rmse": attacked_rmse,
        "recovered_rmse": recovered_rmse,
        "rmse_improvement_percent": rmse_improvement_percent,
        "attacked_mae": attacked_mae,
        "recovered_mae": recovered_mae,
        "mae_improvement_percent": mae_improvement_percent,
        "attacked_attack_region_rmse": attacked_attack_region_rmse,
        "recovered_attack_region_rmse": recovered_attack_region_rmse,
        "attacked_clean_region_rmse": attacked_clean_region_rmse,
        "recovered_clean_region_rmse": recovered_clean_region_rmse,
        "TP": quality["TP"],
        "FP": quality["FP"],
        "TN": quality["TN"],
        "FN": quality["FN"],
        "precision": quality["precision"],
        "recall": quality["recall"],
        "false_alarm_rate": quality["false_alarm_rate"],
        "recovery_plot": recovery_plot,
        "error_plot": error_plot if clean_col is not None else None,
    }

    return summary


# ============================================================
# Report writer
# ============================================================

def write_interpretation(summary_df):
    """
    Write official Phase 5 interpretation report.
    """
    report_path = f"{OUT_REPORT_DIR}/phase5_OFFICIAL_W10_N3_mean1x_recovery_interpretation.txt"

    lines = []

    lines.append("OFFICIAL PHASE 5 V9 RECOVERY INTERPRETATION")
    lines.append("=" * 90)
    lines.append("")
    lines.append("This is the official Phase 5 recovery report.")
    lines.append("")
    lines.append("This report uses ONLY the selected corrected Phase 4 detector:")
    lines.append("  Detector: W10_N3_mean1x")
    lines.append("  Threshold: p99_abs")
    lines.append("  Software sensor: V9 trained hybrid")
    lines.append("")
    lines.append("Recovery rule:")
    lines.append("  if W10_N3_mean1x detection is active:")
    lines.append("      recovered_future[k+1] = V9_prediction[k+1]")
    lines.append("  else:")
    lines.append("      recovered_future[k+1] = attacked_future[k+1]")
    lines.append("")
    lines.append("Error definition:")
    lines.append("  Roll/Pitch: normal subtraction")
    lines.append("  Yaw: circular angular difference")
    lines.append("")
    lines.append("Important:")
    lines.append("  Phase 4 detection is not redone here.")
    lines.append("  Phase 5 only consumes corrected Phase 4 official detection CSVs.")
    lines.append("")

    if len(summary_df) == 0:
        lines.append("No official W10_N3_mean1x files were processed.")
    else:
        lines.append("Processed official files:")
        lines.append("-" * 90)

        for _, r in summary_df.iterrows():
            lines.append(f"Target: {r['target']}")
            lines.append(f"  Input file: {r['input_file']}")
            lines.append(f"  Output recovered CSV: {r['output_recovered_csv']}")
            lines.append(f"  Detection column: {r['detection_col']}")
            lines.append(f"  Attacked future column: {r['attacked_future_col']}")
            lines.append(f"  V9 prediction column: {r['v9_prediction_col']}")
            lines.append(f"  Clean/reference column: {r['clean_reference_col']}")
            lines.append(f"  Rows: {r['n_rows']}")
            lines.append(f"  Detected/recovered-by-V9 rows: {r['recovered_by_v9_count']}")
            lines.append(f"  Kept attacked-measurement rows: {r['kept_attacked_measurement_count']}")
            lines.append(f"  Attack-active rows: {r['attack_active_count']}")
            lines.append("")

            lines.append("  Error metrics:")
            lines.append(f"    Attacked RMSE: {r['attacked_rmse']}")
            lines.append(f"    Recovered RMSE: {r['recovered_rmse']}")
            lines.append(f"    RMSE improvement (%): {r['rmse_improvement_percent']}")
            lines.append(f"    Attacked MAE: {r['attacked_mae']}")
            lines.append(f"    Recovered MAE: {r['recovered_mae']}")
            lines.append(f"    MAE improvement (%): {r['mae_improvement_percent']}")
            lines.append("")

            lines.append("  Region metrics:")
            lines.append(f"    Attacked RMSE during attack region: {r['attacked_attack_region_rmse']}")
            lines.append(f"    Recovered RMSE during attack region: {r['recovered_attack_region_rmse']}")
            lines.append(f"    Attacked RMSE during clean region: {r['attacked_clean_region_rmse']}")
            lines.append(f"    Recovered RMSE during clean region: {r['recovered_clean_region_rmse']}")
            lines.append("")

            lines.append("  Detection quality carried from Phase 4:")
            lines.append(f"    TP: {r['TP']}")
            lines.append(f"    FP: {r['FP']}")
            lines.append(f"    TN: {r['TN']}")
            lines.append(f"    FN: {r['FN']}")
            lines.append(f"    Precision: {r['precision']}")
            lines.append(f"    Recall: {r['recall']}")
            lines.append(f"    False alarm rate: {r['false_alarm_rate']}")
            lines.append("")
            lines.append(f"  Recovery plot: {r['recovery_plot']}")
            lines.append(f"  Error plot: {r['error_plot']}")
            lines.append("-" * 90)
            lines.append("")

        valid = summary_df[
            np.isfinite(summary_df["attacked_rmse"]) &
            np.isfinite(summary_df["recovered_rmse"])
        ].copy()

        lines.append("")
        lines.append("OFFICIAL OVERALL SUMMARY")
        lines.append("=" * 90)

        if len(valid) > 0:
            mean_attacked_rmse = valid["attacked_rmse"].mean()
            mean_recovered_rmse = valid["recovered_rmse"].mean()
            mean_rmse_improvement = valid["rmse_improvement_percent"].mean()

            mean_attacked_mae = valid["attacked_mae"].mean()
            mean_recovered_mae = valid["recovered_mae"].mean()
            mean_mae_improvement = valid["mae_improvement_percent"].mean()

            improved_count = int(np.sum(valid["recovered_rmse"] < valid["attacked_rmse"]))
            total_count = int(len(valid))

            lines.append(f"Official files evaluated: {total_count}")
            lines.append(f"Files improved by recovery: {improved_count}/{total_count}")
            lines.append("")
            lines.append(f"Mean attacked RMSE: {mean_attacked_rmse}")
            lines.append(f"Mean recovered RMSE: {mean_recovered_rmse}")
            lines.append(f"Mean RMSE improvement (%): {mean_rmse_improvement}")
            lines.append("")
            lines.append(f"Mean attacked MAE: {mean_attacked_mae}")
            lines.append(f"Mean recovered MAE: {mean_recovered_mae}")
            lines.append(f"Mean MAE improvement (%): {mean_mae_improvement}")
            lines.append("")

            if improved_count == total_count:
                lines.append("Interpretation:")
                lines.append("  Official Phase 5 recovery reduced RMSE for every evaluated W10_N3_mean1x file.")
                lines.append("  This supports the V9 trained hybrid software sensor as a recovery source after detection.")
            elif improved_count > 0:
                lines.append("Interpretation:")
                lines.append("  Official Phase 5 recovery reduced RMSE for some, but not all, evaluated files.")
                lines.append("  Files without improvement should be inspected for false positives, late detection, or prediction mismatch.")
            else:
                lines.append("Interpretation:")
                lines.append("  Official Phase 5 recovery did not reduce RMSE.")
                lines.append("  This would indicate a mismatch between detector flags and V9 replacement quality.")
        else:
            lines.append("No valid RMSE rows were available.")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 90)
    print("OFFICIAL PHASE 5: V9 RECOVERY USING W10_N3_mean1x ONLY")
    print("=" * 90)
    print(f"Input detection directory:")
    print(f"  {DETECTION_DIR}")
    print("")
    print(f"Official detector tag:")
    print(f"  {OFFICIAL_DETECTOR_TAG}")
    print("")

    all_files = sorted(glob.glob(f"{DETECTION_DIR}/*.csv"))

    files = [
        f for f in all_files
        if OFFICIAL_DETECTOR_TAG in os.path.basename(f)
    ]

    print(f"All detection CSV files found: {len(all_files)}")
    print(f"Official W10_N3_mean1x files selected: {len(files)}")
    print("")

    if len(files) == 0:
        print("ERROR:")
        print("  No official W10_N3_mean1x detection files were found.")
        print("")
        print("Run this to inspect available detection files:")
        print(f"  ls -lh {DETECTION_DIR}")
        raise FileNotFoundError(
            f"No files matching {OFFICIAL_DETECTOR_TAG} in {DETECTION_DIR}"
        )

    print("Official files that will be processed:")
    for f in files:
        print(f"  {os.path.basename(f)}")
    print("")

    summaries = []

    for path in files:
        print("-" * 90)
        print(f"Processing official file:")
        print(f"  {path}")

        summary = process_one_file(path)
        summaries.append(summary)

        print(f"  Target: {summary['target']}")
        print(f"  Detection column: {summary['detection_col']}")
        print(f"  Attacked future column: {summary['attacked_future_col']}")
        print(f"  V9 prediction column: {summary['v9_prediction_col']}")
        print(f"  Clean/reference column: {summary['clean_reference_col']}")
        print(f"  Detected rows: {summary['detected_count']}")
        print(f"  Attack-active rows: {summary['attack_active_count']}")
        print(f"  Attacked RMSE: {summary['attacked_rmse']}")
        print(f"  Recovered RMSE: {summary['recovered_rmse']}")
        print(f"  RMSE improvement (%): {summary['rmse_improvement_percent']}")
        print(f"  Output CSV: {summary['output_recovered_csv']}")
        print("")

    summary_df = pd.DataFrame(summaries)

    summary_path = f"{OUT_REPORT_DIR}/phase5_OFFICIAL_W10_N3_mean1x_recovery_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    interpretation_path = write_interpretation(summary_df)

    print("=" * 90)
    print("OFFICIAL PHASE 5 COMPLETE")
    print("=" * 90)
    print("")
    print("Official recovered CSVs:")
    print(f"  {OUT_CSV_DIR}")
    print("")
    print("Official figures:")
    print(f"  {OUT_FIG_DIR}")
    print("")
    print("Official summary CSV:")
    print(f"  {summary_path}")
    print("")
    print("Official interpretation report:")
    print(f"  {interpretation_path}")
    print("")

    valid = summary_df[
        np.isfinite(summary_df["attacked_rmse"]) &
        np.isfinite(summary_df["recovered_rmse"])
    ].copy()

    if len(valid) > 0:
        print("Official overall numeric summary:")
        print(f"  Files evaluated: {len(valid)}")
        print(f"  Files improved: {int(np.sum(valid['recovered_rmse'] < valid['attacked_rmse']))}/{len(valid)}")
        print(f"  Mean attacked RMSE: {valid['attacked_rmse'].mean()}")
        print(f"  Mean recovered RMSE: {valid['recovered_rmse'].mean()}")
        print(f"  Mean RMSE improvement (%): {valid['rmse_improvement_percent'].mean()}")
        print(f"  Mean attacked MAE: {valid['attacked_mae'].mean()}")
        print(f"  Mean recovered MAE: {valid['recovered_mae'].mean()}")
        print(f"  Mean MAE improvement (%): {valid['mae_improvement_percent'].mean()}")
        print("")


if __name__ == "__main__":
    main()