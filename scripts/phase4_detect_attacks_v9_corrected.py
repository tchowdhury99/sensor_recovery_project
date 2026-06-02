#!/usr/bin/env python3

"""
Corrected Phase 4 V9 attack detection.

Why this corrected script exists:
-------------------------------
The previous Phase 4 detector used the first rows of the attacked ATT file
and compared them directly with the saved V9 prediction output.

That was wrong because the saved V9 prediction output may correspond to a
shorter validation/test segment inside the original ATT file, not necessarily
the beginning of the file.

This script fixes that by:
1. Loading the clean ATT file.
2. Loading the matching V9 prediction output.
3. Finding the best alignment offset between clean ATT and V9 prediction.
4. Injecting synthetic attacks inside the aligned prediction segment.
5. Computing residuals correctly.
6. Using circular residual for Yaw.
7. Running pointwise and window-based detection.
8. Saving corrected metrics, CSVs, and figures.

No recovery is performed here.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path("/home/tchowdh4/sensor_recovery_project")

CLEAN_DIR = PROJECT_ROOT / "logs" / "extracted_csv"
PREDICTION_DIR = PROJECT_ROOT / "prediction_results_v9_trained_hybrid"

OUT_ROOT = PROJECT_ROOT / "attack_detection_v9_corrected"
DETECTION_CSV_DIR = OUT_ROOT / "detection_csvs"
ALIGNED_ATTACK_DIR = OUT_ROOT / "aligned_attacked_segments"

FIG_DIR = PROJECT_ROOT / "figures" / "attack_detection_v9_corrected"
REPORT_DIR = PROJECT_ROOT / "detection_reports_v9_corrected"

for d in [DETECTION_CSV_DIR, ALIGNED_ATTACK_DIR, FIG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Phase 3 primary p99_abs thresholds
# ============================================================

THRESHOLDS_P99 = {
    "baseline_01_hover": {
        "Roll": 0.006865,
        "Pitch": 0.009161,
        "Yaw": 0.013596,
    },
    "baseline_02_high_altitude_hover": {
        "Roll": 0.006861,
        "Pitch": 0.007947,
        "Yaw": 0.010773,
    },
    "baseline_03_forward_motion": {
        "Roll": 0.006821,
        "Pitch": 0.016190,
        "Yaw": 0.003050,
    },
    "baseline_04_yaw_rotation": {
        "Yaw": 0.003797,
    },
    "baseline_05_mixed_maneuver": {
        "Roll": 0.010307,
        "Pitch": 0.009697,
        "Yaw": 0.004795,
    },
}

BASELINES = list(THRESHOLDS_P99.keys())
AXES = ["Roll", "Pitch", "Yaw"]


# ============================================================
# Window detection rules
# ============================================================

WINDOW_RULES = [
    {"name": "W10_N3", "W": 10, "N": 3},
    {"name": "W20_N5", "W": 20, "N": 5},
    {"name": "W30_N8", "W": 30, "N": 8},
]

ROLLING_MEAN_MULTIPLIERS = [1.0, 1.25, 1.5]


# ============================================================
# Prediction column candidates
# ============================================================

PREDICTION_COLUMN_CANDIDATES = {
    "Roll": [
        "Roll_hybrid_prediction",
        "Roll_hybrid_pred",
        "Roll_v9_prediction",
        "Roll_v9_pred",
        "Pred_Roll",
        "Roll_Pred",
        "Roll_pred",
        "prediction_Roll",
        "Roll_prediction",
        "Hybrid_Roll",
    ],
    "Pitch": [
        "Pitch_hybrid_prediction",
        "Pitch_hybrid_pred",
        "Pitch_v9_prediction",
        "Pitch_v9_pred",
        "Pred_Pitch",
        "Pitch_Pred",
        "Pitch_pred",
        "prediction_Pitch",
        "Pitch_prediction",
        "Hybrid_Pitch",
    ],
    "Yaw": [
        "Yaw_hybrid_prediction",
        "Yaw_hybrid_pred",
        "Yaw_v9_prediction",
        "Yaw_v9_pred",
        "Pred_Yaw",
        "Yaw_Pred",
        "Yaw_pred",
        "prediction_Yaw",
        "Yaw_prediction",
        "Hybrid_Yaw",
    ],
}


# ============================================================
# Utility functions
# ============================================================

def infer_baseline_name(path_or_name):
    text = str(path_or_name)
    for baseline in BASELINES:
        if baseline in text:
            return baseline
    return None


def find_clean_att_file(baseline):
    """
    Finds the clean ATT CSV for a baseline.
    """
    candidates = sorted(CLEAN_DIR.rglob(f"*{baseline}*ATT*.csv"))

    if not candidates:
        candidates = sorted(CLEAN_DIR.rglob(f"*{baseline}*.csv"))
        candidates = [p for p in candidates if "att" in p.name.lower()]

    if not candidates:
        raise FileNotFoundError(
            f"No clean ATT file found for {baseline} under {CLEAN_DIR}"
        )

    # Prefer exact-looking ATT file.
    candidates = sorted(candidates, key=lambda p: len(p.name))
    return candidates[0]


def find_prediction_file(baseline):
    """
    Finds the V9 prediction output file for a baseline.
    """
    candidates = sorted(PREDICTION_DIR.rglob(f"*{baseline}*prediction*outputs*.csv"))

    if not candidates:
        candidates = sorted(PREDICTION_DIR.rglob(f"*{baseline}*.csv"))
        candidates = [
            p for p in candidates
            if "summary" not in p.name.lower()
            and "threshold" not in p.name.lower()
            and ("pred" in p.name.lower() or "hybrid" in p.name.lower())
        ]

    if not candidates:
        raise FileNotFoundError(
            f"No V9 prediction output file found for {baseline} under {PREDICTION_DIR}"
        )

    return candidates[0]


def find_prediction_column(pred_df, axis):
    """
    Finds the correct V9 prediction column for Roll/Pitch/Yaw.
    """
    for c in PREDICTION_COLUMN_CANDIDATES[axis]:
        if c in pred_df.columns:
            return c

    axis_lower = axis.lower()

    for c in pred_df.columns:
        cl = c.lower()
        if axis_lower in cl and any(k in cl for k in ["hybrid", "v9", "pred", "prediction"]):
            return c

    raise KeyError(
        f"Could not find prediction column for {axis}. "
        f"Available columns: {list(pred_df.columns)}"
    )


def circular_difference_deg(a, b):
    """
    Circular difference a - b for degree angles.
    Result is in [-180, 180).
    """
    return ((a - b + 180.0) % 360.0) - 180.0


def wrap_yaw_deg(yaw):
    """
    Wrap yaw to [0, 360).
    """
    return yaw % 360.0


def residual_signal(measured, predicted, axis):
    """
    Computes residual.
    Roll/Pitch use normal subtraction.
    Yaw uses circular angular subtraction.
    """
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if axis == "Yaw":
        return circular_difference_deg(measured, predicted)

    return measured - predicted


def abs_error_for_alignment(clean_segment, pred_segment, axis):
    """
    Error used to align prediction segment to clean ATT.
    """
    r = residual_signal(clean_segment, pred_segment, axis)
    return np.abs(r)


def find_best_alignment_offset(clean_df, pred_df, axis, pred_col):
    """
    Finds the best offset where the prediction output aligns with the clean ATT file.

    V9 prediction is assumed to predict future state k+1.

    For each possible offset s:
        compare prediction[i] with clean_axis[s + 1 + i]

    The best offset minimizes median absolute error.
    """
    clean_values = pd.to_numeric(clean_df[axis], errors="coerce").to_numpy(dtype=float)
    pred_values = pd.to_numeric(pred_df[pred_col], errors="coerce").to_numpy(dtype=float)

    # Remove NaNs from prediction by keeping finite indices only.
    finite_pred = np.isfinite(pred_values)
    pred_values = pred_values[finite_pred]

    if len(pred_values) < 10:
        raise ValueError(f"Too few finite prediction values for {axis}")

    n_clean = len(clean_values)
    n_pred = len(pred_values)

    max_offset = n_clean - n_pred - 1

    if max_offset < 0:
        raise ValueError(
            f"Prediction length {n_pred} is too long for clean file length {n_clean}"
        )

    best_offset = None
    best_median_error = float("inf")
    best_mean_error = float("inf")

    # Full sliding alignment.
    for offset in range(max_offset + 1):
        clean_future = clean_values[offset + 1: offset + 1 + n_pred]

        if len(clean_future) != n_pred:
            continue

        finite = np.isfinite(clean_future) & np.isfinite(pred_values)
        if finite.sum() < 10:
            continue

        err = abs_error_for_alignment(clean_future[finite], pred_values[finite], axis)
        median_err = float(np.nanmedian(err))
        mean_err = float(np.nanmean(err))

        if median_err < best_median_error:
            best_median_error = median_err
            best_mean_error = mean_err
            best_offset = offset

    if best_offset is None:
        raise ValueError(f"Could not align prediction for {axis}")

    return {
        "offset": int(best_offset),
        "median_abs_error": best_median_error,
        "mean_abs_error": best_mean_error,
        "n_prediction": int(n_pred),
    }


def make_attack_vector(n, attack_type, magnitude):
    """
    Creates attack vector and ground-truth AttackActive array.

    The attack window is always inside the aligned prediction segment.
    """
    attack = np.zeros(n, dtype=float)
    active = np.zeros(n, dtype=int)

    start = int(0.30 * n)
    end = int(0.60 * n)

    if attack_type == "bias":
        active[start:end] = 1
        attack[start:end] = magnitude

    elif attack_type == "ramp":
        active[start:end] = 1
        attack[start:end] = np.linspace(0.0, magnitude, end - start)

    elif attack_type == "pulse":
        # Short pulse inside the middle of the aligned segment.
        pulse_width = max(5, int(0.05 * n))
        center = int(0.45 * n)
        p0 = max(0, center - pulse_width // 2)
        p1 = min(n, center + pulse_width // 2)

        active[p0:p1] = 1
        attack[p0:p1] = magnitude

    else:
        raise ValueError(f"Unknown attack_type: {attack_type}")

    return attack, active


def compute_detection_flags(residual, threshold, W, N, mean_multiplier):
    abs_residual = np.abs(residual)

    point_flag = (abs_residual > threshold).astype(int)

    point_series = pd.Series(point_flag)
    window_count = point_series.rolling(window=W, min_periods=1).sum().to_numpy()
    window_count_flag = (window_count >= N).astype(int)

    abs_series = pd.Series(abs_residual)
    rolling_mean_abs = abs_series.rolling(window=W, min_periods=1).mean().to_numpy()
    window_mean_flag = (rolling_mean_abs > mean_multiplier * threshold).astype(int)

    window_combined_flag = (
        (window_count_flag == 1) | (window_mean_flag == 1)
    ).astype(int)

    return {
        "abs_residual": abs_residual,
        "point_flag": point_flag,
        "window_count": window_count,
        "rolling_mean_abs": rolling_mean_abs,
        "window_count_flag": window_count_flag,
        "window_mean_flag": window_mean_flag,
        "window_combined_flag": window_combined_flag,
    }


def safe_div(num, den):
    if den == 0:
        return np.nan
    return float(num) / float(den)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    m = min(len(y_true), len(y_pred))
    y_true = y_true[:m]
    y_pred = y_pred[:m]

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)

    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = np.nan

    fpr = safe_div(FP, FP + TN)

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }


def compute_detection_delay(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    attack_indices = np.where(y_true == 1)[0]

    if len(attack_indices) == 0:
        return np.nan

    attack_start = attack_indices[0]

    detection_indices = np.where(
        (np.arange(len(y_pred)) >= attack_start) & (y_pred == 1)
    )[0]

    if len(detection_indices) == 0:
        return np.nan

    return int(detection_indices[0] - attack_start)


def build_x_axis(clean_df, offset, n):
    """
    Builds plot x-axis for the aligned prediction segment.
    Uses TimeUS if available.
    """
    future_start = offset + 1
    future_end = future_start + n

    if "TimeUS" in clean_df.columns:
        t = pd.to_numeric(clean_df["TimeUS"], errors="coerce").to_numpy(dtype=float)
        x = (t[future_start:future_end] - t[future_start]) / 1_000_000.0
        return x, "Time since aligned segment start (s)"

    if "TimeS" in clean_df.columns:
        t = pd.to_numeric(clean_df["TimeS"], errors="coerce").to_numpy(dtype=float)
        x = t[future_start:future_end] - t[future_start]
        return x, "Time since aligned segment start (s)"

    return np.arange(n), "Aligned sample index"


def plot_detection(
    fig_path,
    x,
    x_label,
    baseline,
    axis,
    attack_type,
    window_rule_name,
    threshold,
    clean_future,
    attacked_future,
    prediction,
    residual,
    flags,
    attack_active,
):
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(x, clean_future, label=f"Clean future {axis}")
    ax1.plot(x, attacked_future, label=f"Attacked future {axis}", alpha=0.85)
    ax1.plot(x, prediction, label=f"V9 predicted future {axis}", alpha=0.85)
    ax1.set_ylabel(axis)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(x, residual, label="Residual")
    ax2.axhline(threshold, linestyle="--", label="+threshold")
    ax2.axhline(-threshold, linestyle="--", label="-threshold")
    ax2.set_ylabel("Residual")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(x, flags["abs_residual"], label="|Residual|")
    ax3.axhline(threshold, linestyle="--", label="threshold")
    ax3.set_ylabel("|Residual|")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.step(x, attack_active, where="post", label="AttackActive")
    ax4.step(x, flags["point_flag"], where="post", label="PointDetectionFlag", alpha=0.75)
    ax4.step(x, flags["window_combined_flag"], where="post", label="WindowCombinedFlag", alpha=0.75)
    ax4.set_ylabel("Flag")
    ax4.set_xlabel(x_label)
    ax4.set_ylim(-0.1, 1.2)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best")

    fig.suptitle(
        f"{baseline} | {axis} | {attack_type} | {window_rule_name}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    summary_rows = []
    alignment_rows = []

    for baseline in BASELINES:
        clean_path = find_clean_att_file(baseline)
        pred_path = find_prediction_file(baseline)

        print("=" * 80)
        print(f"[BASELINE] {baseline}")
        print(f"[CLEAN]    {clean_path}")
        print(f"[PRED]     {pred_path}")

        clean_df = pd.read_csv(clean_path)
        pred_df = pd.read_csv(pred_path)

        available_axes = [
            axis for axis in AXES
            if axis in clean_df.columns
            and axis in THRESHOLDS_P99[baseline]
        ]

        for axis in available_axes:
            threshold = THRESHOLDS_P99[baseline][axis]
            pred_col = find_prediction_column(pred_df, axis)

            prediction_all = pd.to_numeric(
                pred_df[pred_col],
                errors="coerce",
            ).to_numpy(dtype=float)

            finite_pred = np.isfinite(prediction_all)
            prediction = prediction_all[finite_pred]

            align = find_best_alignment_offset(
                clean_df=clean_df,
                pred_df=pred_df,
                axis=axis,
                pred_col=pred_col,
            )

            offset = align["offset"]
            n = align["n_prediction"]

            # Rebuild prediction after finite filtering.
            prediction = prediction[:n]

            clean_values = pd.to_numeric(
                clean_df[axis],
                errors="coerce",
            ).to_numpy(dtype=float)

            clean_future = clean_values[offset + 1: offset + 1 + n]

            if len(clean_future) != len(prediction):
                m = min(len(clean_future), len(prediction))
                clean_future = clean_future[:m]
                prediction = prediction[:m]
                n = m

            x, x_label = build_x_axis(clean_df, offset, n)

            alignment_rows.append({
                "baseline": baseline,
                "axis": axis,
                "clean_file": str(clean_path),
                "prediction_file": str(pred_path),
                "prediction_column": pred_col,
                "alignment_offset": offset,
                "n_prediction": n,
                "alignment_median_abs_error": align["median_abs_error"],
                "alignment_mean_abs_error": align["mean_abs_error"],
                "threshold_p99_abs": threshold,
                "alignment_error_over_threshold": align["median_abs_error"] / threshold if threshold > 0 else np.nan,
            })

            print()
            print(f"[ALIGN] {axis}")
            print(f"        prediction column: {pred_col}")
            print(f"        offset: {offset}")
            print(f"        n: {n}")
            print(f"        median abs clean residual: {align['median_abs_error']:.8f}")
            print(f"        threshold p99_abs: {threshold:.8f}")
            print(f"        error / threshold: {align['median_abs_error'] / threshold:.3f}")

            # Attack magnitudes relative to Phase 3 threshold.
            attack_scenarios = [
                ("bias", 4.0 * threshold),
                ("ramp", 6.0 * threshold),
                ("pulse", 8.0 * threshold),
            ]

            for attack_type, magnitude in attack_scenarios:
                attack_vector, attack_active = make_attack_vector(
                    n=n,
                    attack_type=attack_type,
                    magnitude=magnitude,
                )

                attacked_future = clean_future.copy() + attack_vector

                if axis == "Yaw":
                    attacked_future = wrap_yaw_deg(attacked_future)

                residual = residual_signal(
                    measured=attacked_future,
                    predicted=prediction,
                    axis=axis,
                )

                # Save aligned attacked segment once per axis/attack.
                aligned_attack_df = pd.DataFrame({
                    "x": x,
                    "Baseline": baseline,
                    "Axis": axis,
                    "AttackType": attack_type,
                    "AttackMagnitude": magnitude,
                    "AttackActive": attack_active,
                    f"CleanFuture_{axis}": clean_future,
                    f"AttackedFuture_{axis}": attacked_future,
                    f"V9PredictedFuture_{axis}": prediction,
                    f"Residual_{axis}": residual,
                    f"AbsResidual_{axis}": np.abs(residual),
                    "Threshold_p99_abs": threshold,
                    "AlignmentOffset": offset,
                    "PredictionColumn": pred_col,
                    "CleanFile": str(clean_path),
                    "PredictionFile": str(pred_path),
                })

                aligned_attack_path = (
                    ALIGNED_ATTACK_DIR
                    / f"{baseline}_{axis}_{attack_type}_aligned_attacked_segment.csv"
                )
                aligned_attack_df.to_csv(aligned_attack_path, index=False)

                for rule in WINDOW_RULES:
                    for mean_multiplier in ROLLING_MEAN_MULTIPLIERS:
                        W = rule["W"]
                        N = rule["N"]
                        window_rule_name = f"{rule['name']}_mean{mean_multiplier:g}x"

                        flags = compute_detection_flags(
                            residual=residual,
                            threshold=threshold,
                            W=W,
                            N=N,
                            mean_multiplier=mean_multiplier,
                        )

                        detection_df = pd.DataFrame({
                            "x": x,
                            "Baseline": baseline,
                            "Axis": axis,
                            "AttackType": attack_type,
                            "AttackMagnitude": magnitude,
                            "AttackActive": attack_active,
                            f"CleanFuture_{axis}": clean_future,
                            f"AttackedFuture_{axis}": attacked_future,
                            f"V9PredictedFuture_{axis}": prediction,
                            f"Residual_{axis}": residual,
                            f"AbsResidual_{axis}": flags["abs_residual"],
                            "Threshold": threshold,
                            "PointDetectionFlag": flags["point_flag"],
                            "WindowCount": flags["window_count"],
                            "RollingMeanAbsResidual": flags["rolling_mean_abs"],
                            "WindowCountFlag": flags["window_count_flag"],
                            "WindowMeanFlag": flags["window_mean_flag"],
                            "WindowCombinedFlag": flags["window_combined_flag"],
                            "WindowRule": window_rule_name,
                            "W": W,
                            "N": N,
                            "RollingMeanMultiplier": mean_multiplier,
                            "AlignmentOffset": offset,
                            "PredictionColumn": pred_col,
                            "CleanFile": str(clean_path),
                            "PredictionFile": str(pred_path),
                        })

                        detection_csv_path = (
                            DETECTION_CSV_DIR
                            / f"{baseline}_{axis}_{attack_type}_{window_rule_name}_detection.csv"
                        )
                        detection_df.to_csv(detection_csv_path, index=False)

                        fig_path = (
                            FIG_DIR
                            / f"{baseline}_{axis}_{attack_type}_{window_rule_name}.png"
                        )

                        plot_detection(
                            fig_path=fig_path,
                            x=x,
                            x_label=x_label,
                            baseline=baseline,
                            axis=axis,
                            attack_type=attack_type,
                            window_rule_name=window_rule_name,
                            threshold=threshold,
                            clean_future=clean_future,
                            attacked_future=attacked_future,
                            prediction=prediction,
                            residual=residual,
                            flags=flags,
                            attack_active=attack_active,
                        )

                        detector_outputs = [
                            ("point", flags["point_flag"]),
                            ("window_count", flags["window_count_flag"]),
                            ("window_mean", flags["window_mean_flag"]),
                            ("window_combined", flags["window_combined_flag"]),
                        ]

                        for detector_name, detector_flag in detector_outputs:
                            metrics = compute_metrics(
                                y_true=attack_active,
                                y_pred=detector_flag,
                            )
                            delay = compute_detection_delay(
                                y_true=attack_active,
                                y_pred=detector_flag,
                            )

                            row = {
                                "baseline": baseline,
                                "axis": axis,
                                "attack_type": attack_type,
                                "attack_magnitude": magnitude,
                                "detector": detector_name,
                                "window_rule": window_rule_name,
                                "W": W,
                                "N": N,
                                "rolling_mean_multiplier": mean_multiplier,
                                "threshold_type": "p99_abs",
                                "threshold": threshold,
                                "TP": metrics["TP"],
                                "FP": metrics["FP"],
                                "TN": metrics["TN"],
                                "FN": metrics["FN"],
                                "precision": metrics["precision"],
                                "recall": metrics["recall"],
                                "f1": metrics["f1"],
                                "false_positive_rate": metrics["false_positive_rate"],
                                "detection_delay_samples": delay,
                                "n_eval_samples": n,
                                "alignment_offset": offset,
                                "alignment_median_abs_error": align["median_abs_error"],
                                "alignment_mean_abs_error": align["mean_abs_error"],
                                "alignment_error_over_threshold": align["median_abs_error"] / threshold if threshold > 0 else np.nan,
                                "clean_file": str(clean_path),
                                "prediction_file": str(pred_path),
                                "prediction_column": pred_col,
                                "detection_csv": str(detection_csv_path),
                                "figure": str(fig_path),
                            }

                            summary_rows.append(row)

                        print(
                            f"[OK] {baseline} | {axis} | {attack_type} | {window_rule_name}"
                        )

    summary_df = pd.DataFrame(summary_rows)
    alignment_df = pd.DataFrame(alignment_rows)

    summary_path = REPORT_DIR / "phase4_detection_summary_v9_corrected.csv"
    alignment_path = REPORT_DIR / "phase4_alignment_report_v9_corrected.csv"

    summary_df.to_csv(summary_path, index=False)
    alignment_df.to_csv(alignment_path, index=False)

    print()
    print("=" * 80)
    print("[DONE] Corrected Phase 4 detection complete.")
    print(f"[SUMMARY]   {summary_path}")
    print(f"[ALIGNMENT] {alignment_path}")
    print(f"[CSV DIR]   {DETECTION_CSV_DIR}")
    print(f"[FIG DIR]   {FIG_DIR}")

    if len(summary_df) > 0:
        print()
        print("[TOP WINDOW-COMBINED RESULTS]")
        view = summary_df[summary_df["detector"] == "window_combined"].copy()
        view = view.sort_values(
            ["f1", "false_positive_rate", "detection_delay_samples"],
            ascending=[False, True, True],
        )

        cols = [
            "baseline",
            "axis",
            "attack_type",
            "window_rule",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "detection_delay_samples",
            "alignment_error_over_threshold",
        ]

        print(view[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()