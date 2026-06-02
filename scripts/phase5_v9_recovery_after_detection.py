#!/usr/bin/env python3

"""
PHASE 5: V9 RECOVERY AFTER CORRECTED PHASE 4 DETECTION

This script uses the corrected Phase 4 detection CSVs.

It does NOT redo Phase 4 detection.

Recovery rule:
    If W10_N3_mean1x detector says attack/detection is active:
        recovered_future = V9_prediction_future
    Else:
        recovered_future = attacked_future

For Roll/Pitch:
    residual/error = normal subtraction

For Yaw:
    residual/error = circular angular difference

Expected input:
    /home/tchowdh4/sensor_recovery_project/attack_detection_v9_corrected/detection_csvs

Outputs:
    recovered CSVs:
        /home/tchowdh4/sensor_recovery_project/recovery_v9_corrected/recovered_csvs

    summary reports:
        /home/tchowdh4/sensor_recovery_project/recovery_v9_corrected/recovery_reports

    figures:
        /home/tchowdh4/sensor_recovery_project/figures/recovery_v9_corrected
"""

import os
import glob
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = "/home/tchowdh4/sensor_recovery_project"

DETECTION_DIR = f"{BASE}/attack_detection_v9_corrected/detection_csvs"

OUT_CSV_DIR = f"{BASE}/recovery_v9_corrected/recovered_csvs"
OUT_REPORT_DIR = f"{BASE}/recovery_v9_corrected/recovery_reports"
OUT_FIG_DIR = f"{BASE}/figures/recovery_v9_corrected"

os.makedirs(OUT_CSV_DIR, exist_ok=True)
os.makedirs(OUT_REPORT_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


TARGETS = ["Roll", "Pitch", "Yaw"]


# -----------------------------
# Utility functions
# -----------------------------

def normalize_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def circular_diff_deg(a, b):
    """
    Return signed shortest angular difference a - b in degrees.
    Output range: [-180, 180)
    """
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0


def detect_target_from_filename_and_columns(path, df):
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

    return "Unknown"


def find_time_column(df):
    candidates = [
        "t", "time", "timesec", "time_sec", "timestamp", "timeus", "time_us"
    ]

    normalized = {normalize_name(c): c for c in df.columns}

    for cand in candidates:
        if cand in normalized:
            return normalized[cand]

    # Fallback: use row index later.
    return None


def find_detection_column(df):
    """
    Prefer the exact Phase 4 final detector:
        W10_N3_mean1x

    But because different scripts sometimes create slightly different names,
    this function searches robustly.
    """

    cols = list(df.columns)
    norm_to_original = {normalize_name(c): c for c in cols}

    exact_candidates = [
        "w10_n3_mean1x",
        "detected_w10_n3_mean1x",
        "detection_w10_n3_mean1x",
        "attack_detected_w10_n3_mean1x",
        "final_detection_w10_n3_mean1x",
    ]

    for cand in exact_candidates:
        if cand in norm_to_original:
            return norm_to_original[cand]

    # Broader search: must contain all key terms.
    for c in cols:
        n = normalize_name(c)
        if "w10" in n and "n3" in n and "mean1x" in n:
            return c

    # Fallbacks if your corrected CSV stored only one detection flag.
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
        "Could not find the detection flag column. "
        "Expected a column like W10_N3_mean1x or detected_W10_N3_mean1x."
    )


def find_attacked_future_column(df, target):
    """
    Find attacked future measurement column.
    This should be the attacked sensor value aligned to k+1.
    """

    cols = list(df.columns)
    t = target.lower()

    strong_patterns = [
        "attacked_future",
        "attack_future",
        "measured_future_attacked",
        "measurement_future_attacked",
        "attacked",
    ]

    # Prefer columns containing target name and attacked/future.
    for c in cols:
        n = normalize_name(c)
        if t in n and "attacked" in n and "future" in n:
            return c

    # Then any attacked_future-style column.
    for c in cols:
        n = normalize_name(c)
        for p in strong_patterns:
            if p in n:
                # Avoid flags like AttackActive.
                if "active" not in n and "detected" not in n and "detection" not in n:
                    return c

    # If target-specific attacked column exists.
    for c in cols:
        n = normalize_name(c)
        if t in n and "attacked" in n:
            if "active" not in n and "detected" not in n and "detection" not in n:
                return c

    raise ValueError(
        f"Could not find attacked future column for target={target}. "
        "Expected a column like attacked_future, Roll_attacked_future, etc."
    )


def find_prediction_column(df, target):
    """
    Find V9 software-sensor prediction column.
    """

    cols = list(df.columns)
    t = target.lower()

    # Strong target-specific V9 prediction names.
    for c in cols:
        n = normalize_name(c)
        if t in n and ("v9" in n or "hybrid" in n) and ("pred" in n or "prediction" in n):
            return c

    # General V9 prediction.
    for c in cols:
        n = normalize_name(c)
        if ("v9" in n or "hybrid" in n) and ("pred" in n or "prediction" in n):
            return c

    # Generic prediction fallback.
    for c in cols:
        n = normalize_name(c)
        if "pred" in n or "prediction" in n or "software_sensor" in n:
            # Avoid residual columns.
            if "residual" not in n and "error" not in n:
                return c

    raise ValueError(
        f"Could not find V9 prediction column for target={target}. "
        "Expected a column like V9_prediction, v9_pred_future, etc."
    )


def find_clean_reference_column(df, target):
    """
    Find clean/reference future value if available.

    This is needed for RMSE/MAE evaluation.
    If not present, recovery CSVs can still be produced, but metrics will be limited.
    """

    cols = list(df.columns)
    t = target.lower()

    strong_patterns = [
        "clean_future",
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
                if p in n:
                    return c

    for c in cols:
        n = normalize_name(c)
        for p in strong_patterns:
            if p in n:
                return c

    # More generic fallback.
    for c in cols:
        n = normalize_name(c)
        if ("clean" in n or "reference" in n or "truth" in n or "normal" in n) and "future" in n:
            return c

    return None


def find_attack_active_column(df):
    for c in df.columns:
        n = normalize_name(c)
        if "attackactive" in n or "attack_active" in n:
            return c
    return None


def to_bool_series(s):
    """
    Convert different possible detector flag formats to boolean.
    Handles 0/1, True/False, strings.
    """

    if s.dtype == bool:
        return s.fillna(False)

    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(float) != 0

    lowered = s.astype(str).str.strip().str.lower()
    return lowered.isin(["1", "true", "yes", "y", "detected", "attack", "attacked"])


def compute_error(target, measured, reference):
    if reference is None:
        return None

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
    Optional if AttackActive column is present.
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
    far = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": far,
    }


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
    if time_col is not None:
        t = df[time_col].to_numpy()
        x_label = time_col
    else:
        t = np.arange(len(df))
        x_label = "sample index"

    detected = to_bool_series(df[det_col]).to_numpy()

    plt.figure(figsize=(14, 6))

    if clean_col is not None:
        plt.plot(t, df[clean_col], linewidth=1.5, label="Clean/reference future")

    plt.plot(t, df[attacked_col], linewidth=1.0, alpha=0.7, label="Attacked future")
    plt.plot(t, df[pred_col], linewidth=1.0, alpha=0.8, label="V9 prediction")
    plt.plot(t, df[recovered_col], linewidth=1.8, label="Recovered future")

    # Show detections as vertical markers.
    if np.any(detected):
        det_t = t[detected]
        ymin = np.nanmin(df[[attacked_col, pred_col, recovered_col]].to_numpy(dtype=float))
        ymax = np.nanmax(df[[attacked_col, pred_col, recovered_col]].to_numpy(dtype=float))

        # Limit markers if extremely many, to keep plot readable.
        max_markers = 300
        if len(det_t) > max_markers:
            idx = np.linspace(0, len(det_t) - 1, max_markers).astype(int)
            det_t = det_t[idx]

        for dt in det_t:
            plt.axvline(dt, alpha=0.08, linewidth=0.8)

    plt.title(f"Phase 5 Recovery using V9 Prediction after Detection: {target}")
    plt.xlabel(x_label)
    plt.ylabel(f"{target} value")
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
    plt.plot(t, attacked_err, linewidth=1.0, alpha=0.8, label="Attacked error")
    plt.plot(t, recovered_err, linewidth=1.4, label="Recovered error")
    plt.axhline(0, linewidth=1.0)
    plt.title(f"Phase 5 Error Reduction: {target}")
    plt.xlabel(x_label)
    plt.ylabel("Circular error deg" if target.lower() == "yaw" else "Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main processing
# -----------------------------

def process_one_file(path):
    df = pd.read_csv(path)

    target = detect_target_from_filename_and_columns(path, df)

    time_col = find_time_column(df)
    det_col = find_detection_column(df)
    attacked_col = find_attacked_future_column(df, target)
    pred_col = find_prediction_column(df, target)
    clean_col = find_clean_reference_column(df, target)
    attack_active_col = find_attack_active_column(df)

    det_bool = to_bool_series(df[det_col])

    recovered_col = f"{target}_recovered_future_v9_phase5"

    # Core recovery rule.
    df[recovered_col] = np.where(
        det_bool.to_numpy(),
        df[pred_col].astype(float).to_numpy(),
        df[attacked_col].astype(float).to_numpy(),
    )

    # For yaw, keep output in a natural range similar to the attacked signal.
    if target.lower() == "yaw":
        attacked_values = df[attacked_col].astype(float).to_numpy()
        recovered_values = df[recovered_col].astype(float).to_numpy()

        # If original yaw looks like 0..360, wrap recovered to 0..360.
        finite_attacked = attacked_values[np.isfinite(attacked_values)]
        if len(finite_attacked) > 0 and np.nanmin(finite_attacked) >= -1 and np.nanmax(finite_attacked) > 180:
            df[recovered_col] = recovered_values % 360.0
        else:
            # Otherwise keep around [-180, 180).
            df[recovered_col] = circular_diff_deg(recovered_values, 0.0)

    # Add explicit source column.
    source_col = f"{target}_recovery_source_phase5"
    df[source_col] = np.where(det_bool.to_numpy(), "V9_prediction", "attacked_measurement")

    # Add errors if clean/reference exists.
    attacked_rmse = recovered_rmse = attacked_mae = recovered_mae = np.nan
    improvement_rmse_percent = np.nan
    improvement_mae_percent = np.nan

    attacked_attack_region_rmse = recovered_attack_region_rmse = np.nan
    attacked_clean_region_rmse = recovered_clean_region_rmse = np.nan

    if clean_col is not None:
        attacked_error_col = f"{target}_attacked_error_vs_clean_phase5"
        recovered_error_col = f"{target}_recovered_error_vs_clean_phase5"

        df[attacked_error_col] = compute_error(target, df[attacked_col], df[clean_col])
        df[recovered_error_col] = compute_error(target, df[recovered_col], df[clean_col])

        attacked_rmse = rmse(df[attacked_error_col])
        recovered_rmse = rmse(df[recovered_error_col])
        attacked_mae = mae(df[attacked_error_col])
        recovered_mae = mae(df[recovered_error_col])

        if np.isfinite(attacked_rmse) and attacked_rmse > 0:
            improvement_rmse_percent = 100.0 * (attacked_rmse - recovered_rmse) / attacked_rmse

        if np.isfinite(attacked_mae) and attacked_mae > 0:
            improvement_mae_percent = 100.0 * (attacked_mae - recovered_mae) / attacked_mae

        if attack_active_col is not None:
            attack_bool = to_bool_series(df[attack_active_col]).to_numpy()
            if np.any(attack_bool):
                attacked_attack_region_rmse = rmse(df.loc[attack_bool, attacked_error_col])
                recovered_attack_region_rmse = rmse(df.loc[attack_bool, recovered_error_col])
            if np.any(~attack_bool):
                attacked_clean_region_rmse = rmse(df.loc[~attack_bool, attacked_error_col])
                recovered_clean_region_rmse = rmse(df.loc[~attack_bool, recovered_error_col])

    else:
        warnings.warn(
            f"No clean/reference future column found for {os.path.basename(path)}. "
            "Recovered CSV will be produced, but RMSE/MAE will be NaN."
        )

    attack_bool_for_quality = None
    attack_active_count = np.nan

    if attack_active_col is not None:
        attack_bool_for_quality = to_bool_series(df[attack_active_col]).to_numpy()
        attack_active_count = int(np.sum(attack_bool_for_quality))

    quality = detection_quality_metrics(det_bool.to_numpy(), attack_bool_for_quality)

    base_name = os.path.splitext(os.path.basename(path))[0]
    out_csv = f"{OUT_CSV_DIR}/{base_name}_PHASE5_RECOVERED.csv"
    df.to_csv(out_csv, index=False)

    recovery_plot = f"{OUT_FIG_DIR}/{base_name}_phase5_recovery_{target}.png"
    error_plot = f"{OUT_FIG_DIR}/{base_name}_phase5_error_reduction_{target}.png"

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

    summary = {
        "input_file": path,
        "output_recovered_csv": out_csv,
        "target": target,
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
        "rmse_improvement_percent": improvement_rmse_percent,
        "attacked_mae": attacked_mae,
        "recovered_mae": recovered_mae,
        "mae_improvement_percent": improvement_mae_percent,
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


def write_interpretation(summary_df):
    report_path = f"{OUT_REPORT_DIR}/phase5_recovery_interpretation_v9_corrected.txt"

    lines = []
    lines.append("PHASE 5 V9 RECOVERY INTERPRETATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("This report summarizes recovery after corrected Phase 4 detection.")
    lines.append("")
    lines.append("Phase 5 recovery rule:")
    lines.append("  if W10_N3_mean1x detection is active:")
    lines.append("      recovered_future = V9 trained hybrid software-sensor prediction")
    lines.append("  else:")
    lines.append("      recovered_future = attacked future measurement")
    lines.append("")
    lines.append("Important:")
    lines.append("  - Phase 4 detection was not redone.")
    lines.append("  - Phase 5 only consumes corrected Phase 4 detection CSVs.")
    lines.append("  - Roll/Pitch error is normal subtraction.")
    lines.append("  - Yaw error is circular angular difference.")
    lines.append("")

    if len(summary_df) == 0:
        lines.append("No files were processed.")
    else:
        lines.append("Processed files:")
        lines.append("-" * 80)

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

            lines.append("  Detection quality carried from Phase 4 if AttackActive exists:")
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
            lines.append("-" * 80)
            lines.append("")

        valid_rmse = summary_df[
            np.isfinite(summary_df["attacked_rmse"]) &
            np.isfinite(summary_df["recovered_rmse"])
        ].copy()

        if len(valid_rmse) > 0:
            mean_attacked_rmse = valid_rmse["attacked_rmse"].mean()
            mean_recovered_rmse = valid_rmse["recovered_rmse"].mean()
            mean_improvement = valid_rmse["rmse_improvement_percent"].mean()

            lines.append("")
            lines.append("Overall summary:")
            lines.append("-" * 80)
            lines.append(f"Mean attacked RMSE: {mean_attacked_rmse}")
            lines.append(f"Mean recovered RMSE: {mean_recovered_rmse}")
            lines.append(f"Mean RMSE improvement (%): {mean_improvement}")
            lines.append("")

            improved_count = int(np.sum(valid_rmse["recovered_rmse"] < valid_rmse["attacked_rmse"]))
            total_count = int(len(valid_rmse))

            lines.append(f"Files improved by recovery: {improved_count}/{total_count}")
            lines.append("")

            if improved_count == total_count:
                lines.append("Interpretation:")
                lines.append("  Recovery reduced error for all evaluated targets/files.")
            elif improved_count > 0:
                lines.append("Interpretation:")
                lines.append("  Recovery reduced error for some targets/files, but not all.")
                lines.append("  Check false positives and prediction error in the files where recovery did not improve RMSE.")
            else:
                lines.append("Interpretation:")
                lines.append("  Recovery did not reduce RMSE in the evaluated files.")
                lines.append("  This usually means detection is firing in clean regions, the V9 prediction is worse than the attacked signal,")
                lines.append("  or the clean/reference alignment column is not the correct comparison column.")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path


def main():
    files = sorted(glob.glob(f"{DETECTION_DIR}/*.csv"))

    print("=" * 80)
    print("PHASE 5: V9 RECOVERY AFTER CORRECTED PHASE 4 DETECTION")
    print("=" * 80)
    print(f"Input detection directory: {DETECTION_DIR}")
    print(f"Found CSV files: {len(files)}")
    print("")

    if len(files) == 0:
        raise FileNotFoundError(
            f"No CSV files found in {DETECTION_DIR}. "
            "Check that corrected Phase 4 detection CSVs exist."
        )

    summaries = []

    for path in files:
        print(f"Processing: {path}")
        try:
            summary = process_one_file(path)
            summaries.append(summary)
            print(f"  Target: {summary['target']}")
            print(f"  Detection column: {summary['detection_col']}")
            print(f"  Attacked column: {summary['attacked_future_col']}")
            print(f"  Prediction column: {summary['v9_prediction_col']}")
            print(f"  Clean/reference column: {summary['clean_reference_col']}")
            print(f"  Output: {summary['output_recovered_csv']}")
            print("")
        except Exception as e:
            print("")
            print(f"ERROR while processing {path}")
            print(str(e))
            print("")
            raise

    summary_df = pd.DataFrame(summaries)

    summary_path = f"{OUT_REPORT_DIR}/phase5_recovery_summary_v9_corrected.csv"
    summary_df.to_csv(summary_path, index=False)

    interpretation_path = write_interpretation(summary_df)

    print("=" * 80)
    print("PHASE 5 COMPLETE")
    print("=" * 80)
    print(f"Recovered CSVs:")
    print(f"  {OUT_CSV_DIR}")
    print("")
    print(f"Figures:")
    print(f"  {OUT_FIG_DIR}")
    print("")
    print(f"Summary CSV:")
    print(f"  {summary_path}")
    print("")
    print(f"Interpretation report:")
    print(f"  {interpretation_path}")
    print("")


if __name__ == "__main__":
    main()