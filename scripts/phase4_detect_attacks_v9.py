
#!/usr/bin/env python3

import os
import glob
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/tchowdh4/sensor_recovery_project")

ATTACKED_DIR = PROJECT_ROOT / "attack_detection_v9" / "attacked_att"
PREDICTION_DIR = PROJECT_ROOT / "prediction_results_v9_trained_hybrid"

DETECTION_OUT_DIR = PROJECT_ROOT / "attack_detection_v9" / "detection_csvs"
FIG_OUT_DIR = PROJECT_ROOT / "figures" / "attack_detection_v9"
REPORT_OUT_DIR = PROJECT_ROOT / "detection_reports_v9"

DETECTION_OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)

AXES = ["Roll", "Pitch", "Yaw"]

# Phase 3 thresholds.
THRESHOLDS = {
    "baseline_01_hover": {
        "Roll": {"p99_abs": 0.006865, "p99_5_abs": None},
        "Pitch": {"p99_abs": 0.009161, "p99_5_abs": None},
        "Yaw": {"p99_abs": 0.013596, "p99_5_abs": None},
    },
    "baseline_02_high_altitude_hover": {
        "Roll": {"p99_abs": 0.006861, "p99_5_abs": None},
        "Pitch": {"p99_abs": 0.007947, "p99_5_abs": None},
        "Yaw": {"p99_abs": 0.010773, "p99_5_abs": None},
    },
    "baseline_03_forward_motion": {
        "Roll": {"p99_abs": 0.006821, "p99_5_abs": None},
        "Pitch": {"p99_abs": 0.016190, "p99_5_abs": None},
        "Yaw": {"p99_abs": 0.003050, "p99_5_abs": None},
    },
    "baseline_04_yaw_rotation": {
        "Yaw": {"p99_abs": 0.003797, "p99_5_abs": None},
    },
    "baseline_05_mixed_maneuver": {
        "Roll": {"p99_abs": 0.010307, "p99_5_abs": None},
        "Pitch": {"p99_abs": 0.009697, "p99_5_abs": None},
        "Yaw": {"p99_abs": 0.004795, "p99_5_abs": None},
    },
}

BASELINES = list(THRESHOLDS.keys())

# Window rules to test.
# W = rolling window size in samples.
# N = number of threshold exceedances inside W required to declare attack.
WINDOW_RULES = [
    {"name": "W10_N3", "W": 10, "N": 3},
    {"name": "W20_N5", "W": 20, "N": 5},
    {"name": "W30_N8", "W": 30, "N": 8},
]

# Rolling mean multiplier:
# WindowMeanFlag = rolling_mean_abs_residual > multiplier * threshold
ROLLING_MEAN_MULTIPLIERS = [1.0, 1.25, 1.5]

# Common prediction column naming patterns.
PREDICTION_COLUMN_CANDIDATES = {
    "Roll": [
        "Pred_Roll",
        "Roll_Pred",
        "Roll_pred",
        "pred_Roll",
        "prediction_Roll",
        "Roll_prediction",
        "V9_Roll",
        "Roll_V9",
        "Roll_v9_pred",
        "v9_pred_Roll",
        "Hybrid_Roll",
        "Roll_hybrid_pred",
    ],
    "Pitch": [
        "Pred_Pitch",
        "Pitch_Pred",
        "Pitch_pred",
        "pred_Pitch",
        "prediction_Pitch",
        "Pitch_prediction",
        "V9_Pitch",
        "Pitch_V9",
        "Pitch_v9_pred",
        "v9_pred_Pitch",
        "Hybrid_Pitch",
        "Pitch_hybrid_pred",
    ],
    "Yaw": [
        "Pred_Yaw",
        "Yaw_Pred",
        "Yaw_pred",
        "pred_Yaw",
        "prediction_Yaw",
        "Yaw_prediction",
        "V9_Yaw",
        "Yaw_V9",
        "Yaw_v9_pred",
        "v9_pred_Yaw",
        "Hybrid_Yaw",
        "Yaw_hybrid_pred",
    ],
}


def infer_baseline_name(path_or_name):
    name = str(path_or_name)
    for baseline in BASELINES:
        if baseline in name:
            return baseline
    return None


def safe_div(num, den):
    return float(num) / float(den) if den != 0 else np.nan


def find_time_column(df):
    for c in ["TimeUS", "TimeS", "time_s", "timestamp", "Time", "t"]:
        if c in df.columns:
            return c
    return None


def build_time_axis(df):
    """
    Returns an x-axis for plotting.
    If TimeUS exists, convert to seconds relative to first sample.
    If TimeS exists, use seconds relative to first sample.
    Otherwise use sample index.
    """
    if "TimeUS" in df.columns:
        t = pd.to_numeric(df["TimeUS"], errors="coerce").to_numpy(dtype=float)
        return (t - np.nanmin(t)) / 1e6, "time_s"
    if "TimeS" in df.columns:
        t = pd.to_numeric(df["TimeS"], errors="coerce").to_numpy(dtype=float)
        return t - np.nanmin(t), "time_s"

    time_col = find_time_column(df)
    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(t).sum() > 0:
            return t - np.nanmin(t), time_col

    return np.arange(len(df)), "sample"


def list_prediction_files():
    return sorted(PREDICTION_DIR.rglob("*.csv"))


def score_prediction_file(pred_path: Path, attacked_path: Path, baseline: str):
    """
    Heuristic score for matching a V9 prediction CSV to an attacked ATT file.
    """
    pred_name = pred_path.name.lower()
    attacked_name = attacked_path.name.lower()

    score = 0

    if baseline and baseline.lower() in pred_name:
        score += 100

    # Use tokens from source stem.
    for token in baseline.lower().split("_"):
        if token and token in pred_name:
            score += 5

    # Prefer files that look like prediction outputs.
    for token in ["pred", "prediction", "v9", "hybrid"]:
        if token in pred_name:
            score += 10

    # Penalize summaries.
    for token in ["summary", "metric", "report", "threshold"]:
        if token in pred_name:
            score -= 50

    return score


def find_best_prediction_file(attacked_path: Path, baseline: str):
    files = list_prediction_files()
    if not files:
        raise FileNotFoundError(
            f"No prediction CSV files found under {PREDICTION_DIR}"
        )

    scored = []
    for p in files:
        s = score_prediction_file(p, attacked_path, baseline)
        if s > 0:
            scored.append((s, p))

    if not scored:
        raise FileNotFoundError(
            f"Could not find matching V9 prediction CSV for {attacked_path.name}. "
            f"Looked under {PREDICTION_DIR}."
        )

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]


def find_prediction_column(pred_df: pd.DataFrame, axis: str):
    """
    Finds prediction column for the requested axis.
    """
    candidates = PREDICTION_COLUMN_CANDIDATES[axis]

    for c in candidates:
        if c in pred_df.columns:
            return c

    # Flexible fallback:
    # A usable column should contain the axis and a prediction-related keyword.
    axis_lower = axis.lower()
    keyword_candidates = ["pred", "prediction", "v9", "hybrid"]

    for c in pred_df.columns:
        cl = c.lower()
        if axis_lower in cl and any(k in cl for k in keyword_candidates):
            return c

    raise KeyError(
        f"No V9 prediction column found for axis {axis}.\n"
        f"Available prediction columns:\n{list(pred_df.columns)}\n\n"
        f"Edit PREDICTION_COLUMN_CANDIDATES in this script if needed."
    )


def align_prediction_to_attacked(attacked_df, pred_df, axis):
    """
    Produces a predicted future signal aligned to attacked_df rows.

    The residual definition used here is:
        residual[k] = attacked_future[k+1] - V9_prediction[k+1]

    Practically:
      attacked_future_at_k = Attacked_axis shifted by -1
      prediction_at_k = V9 prediction column at row k or k+1 depending on file length

    Because different Phase 2 scripts may save predictions with different indexing,
    this function uses the most direct length alignment:
      - If pred_df length == attacked_df length, use prediction row k.
      - If pred_df length == attacked_df length - 1, use prediction row k for rows 0..n-2.
      - Otherwise truncate both to the common minimum length.
    """
    pred_col = find_prediction_column(pred_df, axis)
    pred = pd.to_numeric(pred_df[pred_col], errors="coerce").to_numpy(dtype=float)

    attacked_col = f"Attacked_{axis}"
    if attacked_col not in attacked_df.columns:
        raise KeyError(f"Missing attacked column: {attacked_col}")

    attacked_signal = pd.to_numeric(attacked_df[attacked_col], errors="coerce").to_numpy(dtype=float)

    attacked_future = attacked_signal[1:]

    if len(pred) == len(attacked_signal):
        pred_aligned = pred[:-1]
    elif len(pred) == len(attacked_signal) - 1:
        pred_aligned = pred
    else:
        m = min(len(attacked_future), len(pred))
        attacked_future = attacked_future[:m]
        pred_aligned = pred[:m]

    residual = attacked_future - pred_aligned

    return attacked_future, pred_aligned, residual, pred_col


def compute_flags(residual, threshold, window_rule, mean_multiplier):
    abs_residual = np.abs(residual)

    point_flag = (abs_residual > threshold).astype(int)

    W = window_rule["W"]
    N = window_rule["N"]

    s = pd.Series(point_flag)
    window_count = s.rolling(window=W, min_periods=1).sum().to_numpy()
    window_count_flag = (window_count >= N).astype(int)

    abs_s = pd.Series(abs_residual)
    rolling_mean_abs = abs_s.rolling(window=W, min_periods=1).mean().to_numpy()
    window_mean_flag = (rolling_mean_abs > mean_multiplier * threshold).astype(int)

    # Conservative combined rule:
    # declare attack if either the count rule or mean rule fires.
    window_combined_flag = np.logical_or(window_count_flag, window_mean_flag).astype(int)

    return {
        "abs_residual": abs_residual,
        "point_flag": point_flag,
        "window_count": window_count,
        "rolling_mean_abs": rolling_mean_abs,
        "window_count_flag": window_count_flag,
        "window_mean_flag": window_mean_flag,
        "window_combined_flag": window_combined_flag,
    }


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
    f1 = safe_div(2 * precision * recall, precision + recall) if np.isfinite(precision) and np.isfinite(recall) else np.nan
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
    """
    Delay in samples from attack start to first detection.
    Returns NaN if attack exists but detector never fires after attack start.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    attack_indices = np.where(y_true == 1)[0]
    if len(attack_indices) == 0:
        return np.nan

    attack_start = attack_indices[0]

    detection_indices = np.where((np.arange(len(y_pred)) >= attack_start) & (y_pred == 1))[0]
    if len(detection_indices) == 0:
        return np.nan

    return int(detection_indices[0] - attack_start)


def plot_detection(
    out_png,
    x,
    axis,
    clean_signal,
    attacked_future,
    pred_aligned,
    residual,
    abs_residual,
    threshold,
    y_true,
    point_flag,
    window_flag,
    title,
):
    m = min(
        len(x),
        len(clean_signal),
        len(attacked_future),
        len(pred_aligned),
        len(residual),
        len(abs_residual),
        len(y_true),
        len(point_flag),
        len(window_flag),
    )

    x = x[:m]
    clean_signal = clean_signal[:m]
    attacked_future = attacked_future[:m]
    pred_aligned = pred_aligned[:m]
    residual = residual[:m]
    abs_residual = abs_residual[:m]
    y_true = y_true[:m]
    point_flag = point_flag[:m]
    window_flag = window_flag[:m]

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(x, clean_signal, label=f"Clean true {axis}")
    ax1.plot(x, attacked_future, label=f"Attacked future {axis}", alpha=0.85)
    ax1.plot(x, pred_aligned, label=f"V9 predicted future {axis}", alpha=0.85)
    ax1.set_ylabel(axis)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(x, residual, label="Residual")
    ax2.axhline(threshold, linestyle="--", label="+threshold")
    ax2.axhline(-threshold, linestyle="--", label="-threshold")
    ax2.set_ylabel("Residual")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(x, abs_residual, label="Absolute residual")
    ax3.axhline(threshold, linestyle="--", label="threshold")
    ax3.set_ylabel("|Residual|")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.step(x, y_true, where="post", label="AttackActive")
    ax4.step(x, point_flag, where="post", label="PointDetectionFlag", alpha=0.75)
    ax4.step(x, window_flag, where="post", label="WindowCombinedFlag", alpha=0.75)
    ax4.set_ylabel("Flag")
    ax4.set_xlabel("Time / sample")
    ax4.set_ylim(-0.1, 1.2)
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    attacked_files = sorted(ATTACKED_DIR.glob("*.csv"))
    attacked_files = [p for p in attacked_files if "manifest" not in p.name.lower()]

    if not attacked_files:
        raise FileNotFoundError(
            f"No attacked CSV files found in {ATTACKED_DIR}. "
            "Run phase4_create_attacked_att_v9.py first or copy your attacked ATT files there."
        )

    summary_rows = []

    print(f"[INFO] Found {len(attacked_files)} attacked CSV files.")

    for attacked_path in attacked_files:
        attacked_df = pd.read_csv(attacked_path)
        baseline = infer_baseline_name(attacked_path.name)

        if baseline is None:
            print(f"[WARN] Could not infer baseline for {attacked_path.name}; skipping.")
            continue

        if "AttackAxis" not in attacked_df.columns or "AttackType" not in attacked_df.columns:
            print(f"[WARN] Missing AttackAxis/AttackType in {attacked_path.name}; skipping.")
            continue

        attack_axis = str(attacked_df["AttackAxis"].iloc[0])
        attack_type = str(attacked_df["AttackType"].iloc[0])

        if attack_axis not in AXES:
            print(f"[WARN] Bad attack axis {attack_axis} in {attacked_path.name}; skipping.")
            continue

        if baseline not in THRESHOLDS or attack_axis not in THRESHOLDS[baseline]:
            print(f"[WARN] No threshold for {baseline} {attack_axis}; skipping.")
            continue

        threshold = THRESHOLDS[baseline][attack_axis]["p99_abs"]

        pred_path = find_best_prediction_file(attacked_path, baseline)
        pred_df = pd.read_csv(pred_path)

        try:
            attacked_future, pred_aligned, residual, pred_col = align_prediction_to_attacked(
                attacked_df, pred_df, attack_axis
            )
        except Exception as e:
            print()
            print(f"[ERROR] Failed on {attacked_path.name}")
            print(f"        Prediction file: {pred_path}")
            print(f"        Error: {e}")
            print()
            continue

        # AttackActive corresponds to row k+1 because residual uses future measurement.
        y_true_full = attacked_df["AttackActive"].astype(int).to_numpy()
        y_true = y_true_full[1:]
        m = min(len(y_true), len(residual))
        y_true = y_true[:m]
        residual = residual[:m]
        attacked_future = attacked_future[:m]
        pred_aligned = pred_aligned[:m]

        true_col = f"True_{attack_axis}"
        if true_col in attacked_df.columns:
            clean_signal_full = pd.to_numeric(attacked_df[true_col], errors="coerce").to_numpy(dtype=float)
            clean_signal = clean_signal_full[1:1 + m]
        else:
            clean_signal = np.full(m, np.nan)

        x_full, x_label = build_time_axis(attacked_df)
        x = x_full[1:1 + m]

        for window_rule in WINDOW_RULES:
            for mean_multiplier in ROLLING_MEAN_MULTIPLIERS:
                flags = compute_flags(
                    residual=residual,
                    threshold=threshold,
                    window_rule=window_rule,
                    mean_multiplier=mean_multiplier,
                )

                point_metrics = compute_metrics(y_true, flags["point_flag"])
                window_count_metrics = compute_metrics(y_true, flags["window_count_flag"])
                window_mean_metrics = compute_metrics(y_true, flags["window_mean_flag"])
                window_combined_metrics = compute_metrics(y_true, flags["window_combined_flag"])

                point_delay = compute_detection_delay(y_true, flags["point_flag"])
                count_delay = compute_detection_delay(y_true, flags["window_count_flag"])
                mean_delay = compute_detection_delay(y_true, flags["window_mean_flag"])
                combined_delay = compute_detection_delay(y_true, flags["window_combined_flag"])

                rule_name = f'{window_rule["name"]}_mean{mean_multiplier:g}x'

                out_df = pd.DataFrame({
                    "x": x[:m],
                    "AttackActive": y_true[:m],
                    f"CleanFuture_{attack_axis}": clean_signal[:m],
                    f"AttackedFuture_{attack_axis}": attacked_future[:m],
                    f"V9PredictedFuture_{attack_axis}": pred_aligned[:m],
                    f"Residual_{attack_axis}": residual[:m],
                    f"AbsResidual_{attack_axis}": flags["abs_residual"][:m],
                    "Threshold": threshold,
                    "PointDetectionFlag": flags["point_flag"][:m],
                    "WindowCount": flags["window_count"][:m],
                    "RollingMeanAbsResidual": flags["rolling_mean_abs"][:m],
                    "WindowCountFlag": flags["window_count_flag"][:m],
                    "WindowMeanFlag": flags["window_mean_flag"][:m],
                    "WindowCombinedFlag": flags["window_combined_flag"][:m],
                    "Baseline": baseline,
                    "AttackAxis": attack_axis,
                    "AttackType": attack_type,
                    "WindowRule": rule_name,
                    "PredictionFile": str(pred_path),
                    "PredictionColumn": pred_col,
                })

                det_name = attacked_path.stem + f"__detect_{rule_name}.csv"
                det_path = DETECTION_OUT_DIR / det_name
                out_df.to_csv(det_path, index=False)

                fig_name = attacked_path.stem + f"__detect_{rule_name}.png"
                fig_path = FIG_OUT_DIR / fig_name

                plot_detection(
                    out_png=fig_path,
                    x=x,
                    axis=attack_axis,
                    clean_signal=clean_signal,
                    attacked_future=attacked_future,
                    pred_aligned=pred_aligned,
                    residual=residual,
                    abs_residual=flags["abs_residual"],
                    threshold=threshold,
                    y_true=y_true,
                    point_flag=flags["point_flag"],
                    window_flag=flags["window_combined_flag"],
                    title=f"{baseline} | {attack_type} attack on {attack_axis} | {rule_name}",
                )

                base_row = {
                    "attacked_file": str(attacked_path),
                    "detection_csv": str(det_path),
                    "figure": str(fig_path),
                    "baseline": baseline,
                    "attack_axis": attack_axis,
                    "attack_type": attack_type,
                    "threshold_type": "p99_abs",
                    "threshold": threshold,
                    "window_rule": rule_name,
                    "W": window_rule["W"],
                    "N": window_rule["N"],
                    "rolling_mean_multiplier": mean_multiplier,
                    "prediction_file": str(pred_path),
                    "prediction_column": pred_col,
                    "n_eval_samples": m,
                }

                for detector_name, metrics, delay in [
                    ("point", point_metrics, point_delay),
                    ("window_count", window_count_metrics, count_delay),
                    ("window_mean", window_mean_metrics, mean_delay),
                    ("window_combined", window_combined_metrics, combined_delay),
                ]:
                    row = dict(base_row)
                    row["detector"] = detector_name
                    row.update(metrics)
                    row["detection_delay_samples"] = delay
                    summary_rows.append(row)

                print(f"[OK] {attacked_path.name} | {attack_axis} | {attack_type} | {rule_name}")

    summary = pd.DataFrame(summary_rows)

    summary_path = REPORT_OUT_DIR / "phase4_detection_summary_v9.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print(f"[DONE] Detection summary written to:")
    print(f"       {summary_path}")
    print(f"[DONE] Detection CSVs:")
    print(f"       {DETECTION_OUT_DIR}")
    print(f"[DONE] Figures:")
    print(f"       {FIG_OUT_DIR}")

    if len(summary) > 0:
        print()
        print("[TOP WINDOW-COMBINED RESULTS BY F1]")
        cols = [
            "baseline",
            "attack_axis",
            "attack_type",
            "detector",
            "window_rule",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "detection_delay_samples",
        ]
        view = summary[summary["detector"] == "window_combined"].copy()
        view = view.sort_values(["f1", "false_positive_rate"], ascending=[False, True])
        print(view[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
