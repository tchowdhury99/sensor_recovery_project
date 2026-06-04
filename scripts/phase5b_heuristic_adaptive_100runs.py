#!/usr/bin/env python3

"""
PHASE 5B HEURISTIC ADAPTIVE-WEIGHT ROBUSTNESS TEST: 100 RUNS

Purpose:
    Run the XKF1/kinematic heuristic adaptive-weight recovery method 100 times
    using randomized heuristic parameters.

Goal:
    Check whether heuristic adaptive weighting is consistently better than
    original V9 recovery or whether it is unstable.

This script compares:
    1. Attacked signal
    2. Original V9 recovery
    3. Heuristic adaptive-weight recovery

Important:
    - Does NOT overwrite official Phase 5.
    - Uses only official W10_N3_mean1x detection files.
    - Tests Roll/Pitch only.
    - Saves all results to separate 100-run folders.

Success definition:
    A file-level result succeeds if:
        heuristic_adaptive_rmse < original_v9_recovered_rmse

    A run-level result succeeds if:
        all evaluated files succeed.

Expected evaluated files:
    24 Roll/Pitch official W10_N3 files.
"""

import os
import re
import glob
import math
import random
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE = "/home/tchowdh4/sensor_recovery_project"

DETECTION_DIR = f"{BASE}/attack_detection_v9_corrected/detection_csvs"
XKF1_DIR = f"{BASE}/logs/extracted_csv"

OUT_REPORT_DIR = f"{BASE}/recovery_reports_v9_adaptive_weights_100runs"
OUT_RESULT_DIR = f"{BASE}/recovery_results_v9_adaptive_weights_100runs"
OUT_FIG_DIR = f"{BASE}/figures/recovery_v9_adaptive_weights_100runs"

os.makedirs(OUT_REPORT_DIR, exist_ok=True)
os.makedirs(OUT_RESULT_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


# ============================================================
# Experiment settings
# ============================================================

OFFICIAL_DETECTOR_TAG = "W10_N3_mean1x_detection"

TARGETS = ["Roll", "Pitch"]

N_RUNS = 100

# Make the result reproducible.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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
    Roll/Pitch only: normal subtraction.
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
        baseline_05_mixed_maneuver_Pitch_ramp_W10_N3_mean1x_detection.csv

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


def parse_attack_type(path):
    name = os.path.basename(path).lower()

    if "_bias_" in name:
        return "bias"
    if "_pulse_" in name:
        return "pulse"
    if "_ramp_" in name:
        return "ramp"

    return "unknown"


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
# Load V9 component predictions
# ============================================================

def load_prediction_components(det_df, target):
    """
    Detection CSV has PredictionFile.
    PredictionFile has:
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
        "hybrid": f"{target}_hybrid_prediction",
        "arx": f"{target}_arx_component",
        "naive": f"{target}_naive_component",
        "cv": f"{target}_constant_velocity_component",
    }

    for label, col in required_cols.items():
        if col not in pred_df.columns:
            raise ValueError(
                f"Missing {label} component column in prediction file.\n"
                f"File: {prediction_file}\n"
                f"Expected column: {col}"
            )

    n = len(det_df)

    if len(pred_df) < n:
        raise ValueError(
            f"Prediction file has fewer rows than detection file.\n"
            f"Detection rows: {n}\n"
            f"Prediction rows: {len(pred_df)}"
        )

    out = pd.DataFrame(index=det_df.index)

    out[f"{target}_component_hybrid"] = pred_df[required_cols["hybrid"]].iloc[:n].to_numpy()
    out[f"{target}_component_arx"] = pred_df[required_cols["arx"]].iloc[:n].to_numpy()
    out[f"{target}_component_naive"] = pred_df[required_cols["naive"]].iloc[:n].to_numpy()
    out[f"{target}_component_cv"] = pred_df[required_cols["cv"]].iloc[:n].to_numpy()
    out["Phase5B_PredictionFile"] = prediction_file

    return out, prediction_file


# ============================================================
# Load XKF1 context
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
        f"Could not find XKF1 file for baseline {baseline}. Expected {expected}"
    )


def load_and_align_xkf1(det_df, det_path):
    """
    Align XKF1 context to detection row count using normalized interpolation.
    This is enough for flight-context estimation.
    """

    xkf1_file = find_xkf1_file_for_detection(det_path)
    xkf = pd.read_csv(xkf1_file)

    for col in ["VN", "VE"]:
        if col not in xkf.columns:
            raise ValueError(f"XKF1 file missing required column {col}: {xkf1_file}")

    n = len(det_df)

    src = np.linspace(0.0, 1.0, len(xkf))
    dst = np.linspace(0.0, 1.0, n)

    aligned = pd.DataFrame(index=det_df.index)

    for col in ["VN", "VE", "VD", "GX", "GY", "GZ", "Roll", "Pitch", "Yaw", "TimeUS"]:
        if col in xkf.columns:
            aligned[f"XKF1_{col}"] = np.interp(
                dst,
                src,
                xkf[col].astype(float).to_numpy(),
            )

    return aligned, xkf1_file


# ============================================================
# Random heuristic parameter generation
# ============================================================

def normalize_weight_triplet(a, b, c):
    s = a + b + c
    return {
        "w_naive": a / s,
        "w_cv": b / s,
        "w_arx": c / s,
    }


def random_weight_profile(profile_name):
    """
    Generate one randomized but reasonable weight vector.

    The random ranges reflect the intended intuition:
        hover: naive dominant
        translation: CV dominant
        maneuver: ARX dominant
    """

    if profile_name == "hover":
        w_naive = random.uniform(0.55, 0.90)
        w_cv = random.uniform(0.05, 0.30)
        w_arx = random.uniform(0.05, 0.30)

    elif profile_name == "translation":
        w_naive = random.uniform(0.05, 0.35)
        w_cv = random.uniform(0.40, 0.80)
        w_arx = random.uniform(0.05, 0.35)

    elif profile_name == "maneuver":
        w_naive = random.uniform(0.00, 0.25)
        w_cv = random.uniform(0.10, 0.45)
        w_arx = random.uniform(0.40, 0.90)

    else:
        raise ValueError(f"Unknown profile {profile_name}")

    return normalize_weight_triplet(w_naive, w_cv, w_arx)


def random_run_config(run_id):
    """
    Create randomized heuristic configuration for one run.
    """

    config = {
        "run_id": run_id,

        # Context thresholds.
        "hover_speed_threshold": random.uniform(0.3, 0.8),
        "accel_low_threshold": random.uniform(0.25, 1.25),
        "accel_high_threshold": random.uniform(1.25, 3.50),
        "gyro_high_threshold": random.uniform(0.05, 0.50),

        # Weight smoothing.
        "smoothing_previous": random.uniform(0.80, 0.98),
    }

    config["smoothing_target"] = 1.0 - config["smoothing_previous"]

    # Weight profiles.
    for profile in ["hover", "translation", "maneuver"]:
        w = random_weight_profile(profile)
        for key, value in w.items():
            config[f"{profile}_{key}"] = value

    return config


# ============================================================
# Heuristic context and weights
# ============================================================

def compute_context(xkf_df, config):
    out = xkf_df.copy()

    vn = out["XKF1_VN"].astype(float).to_numpy()
    ve = out["XKF1_VE"].astype(float).to_numpy()

    out["horizontal_speed"] = np.sqrt(vn ** 2 + ve ** 2)

    dvn = np.diff(vn, prepend=vn[0])
    dve = np.diff(ve, prepend=ve[0])
    out["accel_proxy"] = np.sqrt(dvn ** 2 + dve ** 2)

    gyro_cols = [c for c in ["XKF1_GX", "XKF1_GY", "XKF1_GZ"] if c in out.columns]

    if gyro_cols:
        gyro_arr = out[gyro_cols].astype(float).to_numpy()
        out["gyro_proxy"] = np.linalg.norm(gyro_arr, axis=1)
    else:
        out["gyro_proxy"] = 0.0

    labels = []

    for _, row in out.iterrows():
        speed = row["horizontal_speed"]
        accel = row["accel_proxy"]
        gyro = row["gyro_proxy"]

        if accel >= config["accel_high_threshold"] or gyro >= config["gyro_high_threshold"]:
            labels.append("maneuver")
        elif speed < config["hover_speed_threshold"] and accel < config["accel_low_threshold"]:
            labels.append("hover")
        else:
            labels.append("translation")

    out["flight_context"] = labels

    return out


def profile_weights_from_config(config, label):
    return {
        "w_naive": config[f"{label}_w_naive"],
        "w_cv": config[f"{label}_w_cv"],
        "w_arx": config[f"{label}_w_arx"],
    }


def compute_smoothed_heuristic_weights(context_df, config):
    w_naive = []
    w_cv = []
    w_arx = []

    previous = profile_weights_from_config(config, "hover")

    for label in context_df["flight_context"].astype(str):
        if label not in ["hover", "translation", "maneuver"]:
            label = "translation"

        target = profile_weights_from_config(config, label)

        current = {
            "w_naive": config["smoothing_previous"] * previous["w_naive"] + config["smoothing_target"] * target["w_naive"],
            "w_cv": config["smoothing_previous"] * previous["w_cv"] + config["smoothing_target"] * target["w_cv"],
            "w_arx": config["smoothing_previous"] * previous["w_arx"] + config["smoothing_target"] * target["w_arx"],
        }

        # Normalize defensively.
        s = current["w_naive"] + current["w_cv"] + current["w_arx"]
        current["w_naive"] /= s
        current["w_cv"] /= s
        current["w_arx"] /= s

        w_naive.append(current["w_naive"])
        w_cv.append(current["w_cv"])
        w_arx.append(current["w_arx"])

        previous = current

    return pd.DataFrame({
        "w_naive_heuristic": w_naive,
        "w_cv_heuristic": w_cv,
        "w_arx_heuristic": w_arx,
    })


# ============================================================
# One file evaluation
# ============================================================

def evaluate_one_file(det_path, config):
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
            raise ValueError(f"Missing required column {col} in {det_path}")

    det_bool = to_bool_series(df[det_col]).to_numpy()

    comp_df, prediction_file = load_prediction_components(df, target)
    xkf_df, xkf1_file = load_and_align_xkf1(df, det_path)

    context_df = compute_context(xkf_df, config)
    weights_df = compute_smoothed_heuristic_weights(context_df, config)

    naive_col = f"{target}_component_naive"
    cv_col = f"{target}_component_cv"
    arx_col = f"{target}_component_arx"

    df[naive_col] = comp_df[naive_col].to_numpy()
    df[cv_col] = comp_df[cv_col].to_numpy()
    df[arx_col] = comp_df[arx_col].to_numpy()

    df["w_naive_heuristic"] = weights_df["w_naive_heuristic"].to_numpy()
    df["w_cv_heuristic"] = weights_df["w_cv_heuristic"].to_numpy()
    df["w_arx_heuristic"] = weights_df["w_arx_heuristic"].to_numpy()

    adaptive_pred = (
        df["w_naive_heuristic"].astype(float).to_numpy() * df[naive_col].astype(float).to_numpy()
        +
        df["w_cv_heuristic"].astype(float).to_numpy() * df[cv_col].astype(float).to_numpy()
        +
        df["w_arx_heuristic"].astype(float).to_numpy() * df[arx_col].astype(float).to_numpy()
    )

    original_recovered = np.where(
        det_bool,
        df[original_v9_col].astype(float).to_numpy(),
        df[attacked_col].astype(float).to_numpy(),
    )

    adaptive_recovered = np.where(
        det_bool,
        adaptive_pred,
        df[attacked_col].astype(float).to_numpy(),
    )

    attacked_err = compute_error(target, df[attacked_col], df[clean_col])
    original_err = compute_error(target, original_recovered, df[clean_col])
    adaptive_err = compute_error(target, adaptive_recovered, df[clean_col])

    attacked_rmse = rmse(attacked_err)
    original_rmse = rmse(original_err)
    adaptive_rmse = rmse(adaptive_err)

    attacked_mae = mae(attacked_err)
    original_mae = mae(original_err)
    adaptive_mae = mae(adaptive_err)

    success_rmse = adaptive_rmse < original_rmse
    success_mae = adaptive_mae < original_mae

    context_counts = context_df["flight_context"].value_counts().to_dict()

    result = {
        "run_id": config["run_id"],
        "input_detection_file": det_path,
        "file_name": os.path.basename(det_path),
        "baseline": parse_baseline_from_detection_file(det_path),
        "target": target,
        "attack_type": parse_attack_type(det_path),
        "prediction_file": prediction_file,
        "xkf1_file": xkf1_file,
        "n_rows": len(df),
        "detected_count": int(np.sum(det_bool)),
        "attack_active_count": int(np.sum(to_bool_series(df["AttackActive"]))) if "AttackActive" in df.columns else np.nan,

        "attacked_rmse": attacked_rmse,
        "original_v9_recovered_rmse": original_rmse,
        "heuristic_adaptive_rmse": adaptive_rmse,

        "attacked_mae": attacked_mae,
        "original_v9_recovered_mae": original_mae,
        "heuristic_adaptive_mae": adaptive_mae,

        "attacked_max_abs_error": max_abs(attacked_err),
        "original_v9_recovered_max_abs_error": max_abs(original_err),
        "heuristic_adaptive_max_abs_error": max_abs(adaptive_err),

        "original_improvement_over_attacked_rmse_percent": percent_improvement(attacked_rmse, original_rmse),
        "heuristic_improvement_over_attacked_rmse_percent": percent_improvement(attacked_rmse, adaptive_rmse),
        "heuristic_improvement_over_original_rmse_percent": percent_improvement(original_rmse, adaptive_rmse),

        "success_rmse": bool(success_rmse),
        "success_mae": bool(success_mae),

        "context_hover_count": int(context_counts.get("hover", 0)),
        "context_translation_count": int(context_counts.get("translation", 0)),
        "context_maneuver_count": int(context_counts.get("maneuver", 0)),
    }

    # Add config fields so each result row is reproducible.
    for k, v in config.items():
        result[f"config_{k}"] = v

    return result


# ============================================================
# Main experiment
# ============================================================

def main():
    print("=" * 100)
    print("PHASE 5B HEURISTIC ADAPTIVE-WEIGHT ROBUSTNESS TEST: 100 RUNS")
    print("=" * 100)
    print("")
    print("This script does NOT overwrite official Phase 5.")
    print("")
    print(f"Detection input folder:")
    print(f"  {DETECTION_DIR}")
    print("")
    print(f"Output report folder:")
    print(f"  {OUT_REPORT_DIR}")
    print("")

    files = sorted(glob.glob(f"{DETECTION_DIR}/*{OFFICIAL_DETECTOR_TAG}.csv"))

    files = [
        f for f in files
        if "_Roll_" in os.path.basename(f) or "_Pitch_" in os.path.basename(f)
    ]

    print(f"Official W10_N3 Roll/Pitch files selected: {len(files)}")
    print(f"Number of randomized heuristic runs: {N_RUNS}")
    print("")

    if len(files) == 0:
        raise FileNotFoundError("No official W10_N3 Roll/Pitch detection files found.")

    all_results = []
    run_summaries = []

    for run_id in range(1, N_RUNS + 1):
        config = random_run_config(run_id)

        print("-" * 100)
        print(f"Run {run_id}/{N_RUNS}")

        run_results = []

        for f in files:
            result = evaluate_one_file(f, config)
            if result is not None:
                run_results.append(result)
                all_results.append(result)

        run_df = pd.DataFrame(run_results)

        n_files = len(run_df)
        n_success_rmse = int(run_df["success_rmse"].sum())
        n_success_mae = int(run_df["success_mae"].sum())
        n_fail_rmse = n_files - n_success_rmse
        n_fail_mae = n_files - n_success_mae

        run_success_all_rmse = n_success_rmse == n_files
        run_success_all_mae = n_success_mae == n_files

        run_summary = {
            "run_id": run_id,
            "n_files": n_files,
            "success_count_rmse": n_success_rmse,
            "fail_count_rmse": n_fail_rmse,
            "success_rate_rmse_percent": 100.0 * n_success_rmse / n_files if n_files else np.nan,
            "success_count_mae": n_success_mae,
            "fail_count_mae": n_fail_mae,
            "success_rate_mae_percent": 100.0 * n_success_mae / n_files if n_files else np.nan,
            "run_success_all_rmse": run_success_all_rmse,
            "run_success_all_mae": run_success_all_mae,
            "mean_original_rmse": run_df["original_v9_recovered_rmse"].mean(),
            "mean_heuristic_rmse": run_df["heuristic_adaptive_rmse"].mean(),
            "mean_heuristic_improvement_over_original_rmse_percent": run_df["heuristic_improvement_over_original_rmse_percent"].mean(),
            "mean_original_mae": run_df["original_v9_recovered_mae"].mean(),
            "mean_heuristic_mae": run_df["heuristic_adaptive_mae"].mean(),
        }

        for k, v in config.items():
            run_summary[f"config_{k}"] = v

        run_summaries.append(run_summary)

        print(f"  RMSE success: {n_success_rmse}/{n_files} ({run_summary['success_rate_rmse_percent']:.2f}%)")
        print(f"  MAE success:  {n_success_mae}/{n_files} ({run_summary['success_rate_mae_percent']:.2f}%)")
        print(f"  Mean original RMSE:  {run_summary['mean_original_rmse']}")
        print(f"  Mean heuristic RMSE: {run_summary['mean_heuristic_rmse']}")
        print(f"  Mean improvement over original RMSE (%): {run_summary['mean_heuristic_improvement_over_original_rmse_percent']}")
        print("")

    all_df = pd.DataFrame(all_results)
    run_summary_df = pd.DataFrame(run_summaries)

    all_results_path = f"{OUT_REPORT_DIR}/phase5b_heuristic_100run_all_results.csv"
    run_summary_path = f"{OUT_REPORT_DIR}/phase5b_heuristic_100run_run_summary.csv"

    all_df.to_csv(all_results_path, index=False)
    run_summary_df.to_csv(run_summary_path, index=False)

    # ========================================================
    # File failure summary
    # ========================================================

    file_summary = (
        all_df
        .groupby(["file_name", "baseline", "target", "attack_type"], as_index=False)
        .agg(
            evaluated_runs=("run_id", "count"),
            success_count_rmse=("success_rmse", "sum"),
            success_count_mae=("success_mae", "sum"),
            mean_original_rmse=("original_v9_recovered_rmse", "mean"),
            mean_heuristic_rmse=("heuristic_adaptive_rmse", "mean"),
            mean_heuristic_improvement_over_original_rmse_percent=("heuristic_improvement_over_original_rmse_percent", "mean"),
        )
    )

    file_summary["fail_count_rmse"] = file_summary["evaluated_runs"] - file_summary["success_count_rmse"]
    file_summary["fail_count_mae"] = file_summary["evaluated_runs"] - file_summary["success_count_mae"]
    file_summary["success_rate_rmse_percent"] = 100.0 * file_summary["success_count_rmse"] / file_summary["evaluated_runs"]
    file_summary["success_rate_mae_percent"] = 100.0 * file_summary["success_count_mae"] / file_summary["evaluated_runs"]

    file_summary = file_summary.sort_values(
        by=["fail_count_rmse", "mean_heuristic_improvement_over_original_rmse_percent"],
        ascending=[False, True],
    )

    file_summary_path = f"{OUT_REPORT_DIR}/phase5b_heuristic_100run_file_failure_summary.csv"
    file_summary.to_csv(file_summary_path, index=False)

    # ========================================================
    # Axis / attack summaries
    # ========================================================

    axis_summary = (
        all_df
        .groupby(["target"], as_index=False)
        .agg(
            evaluated_cases=("success_rmse", "count"),
            success_count_rmse=("success_rmse", "sum"),
            mean_heuristic_improvement_over_original_rmse_percent=("heuristic_improvement_over_original_rmse_percent", "mean"),
        )
    )

    axis_summary["success_rate_rmse_percent"] = 100.0 * axis_summary["success_count_rmse"] / axis_summary["evaluated_cases"]

    axis_summary_path = f"{OUT_REPORT_DIR}/phase5b_heuristic_100run_axis_summary.csv"
    axis_summary.to_csv(axis_summary_path, index=False)

    attack_summary = (
        all_df
        .groupby(["attack_type"], as_index=False)
        .agg(
            evaluated_cases=("success_rmse", "count"),
            success_count_rmse=("success_rmse", "sum"),
            mean_heuristic_improvement_over_original_rmse_percent=("heuristic_improvement_over_original_rmse_percent", "mean"),
        )
    )

    attack_summary["success_rate_rmse_percent"] = 100.0 * attack_summary["success_count_rmse"] / attack_summary["evaluated_cases"]

    attack_summary_path = f"{OUT_REPORT_DIR}/phase5b_heuristic_100run_attack_summary.csv"
    attack_summary.to_csv(attack_summary_path, index=False)

    # ========================================================
    # Interpretation report
    # ========================================================

    total_cases = len(all_df)
    total_success = int(all_df["success_rmse"].sum())
    total_fail = total_cases - total_success
    overall_success_rate = 100.0 * total_success / total_cases if total_cases else np.nan

    run_all_success_count = int(run_summary_df["run_success_all_rmse"].sum())
    run_all_success_rate = 100.0 * run_all_success_count / len(run_summary_df) if len(run_summary_df) else np.nan

    best_run = run_summary_df.sort_values(
        by=["success_count_rmse", "mean_heuristic_improvement_over_original_rmse_percent"],
        ascending=[False, False],
    ).iloc[0]

    worst_run = run_summary_df.sort_values(
        by=["success_count_rmse", "mean_heuristic_improvement_over_original_rmse_percent"],
        ascending=[True, True],
    ).iloc[0]

    report_path = f"{OUT_REPORT_DIR}/phase5b_heuristic_100run_interpretation.txt"

    lines = []
    lines.append("PHASE 5B HEURISTIC ADAPTIVE-WEIGHT ROBUSTNESS TEST: 100 RUNS")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Purpose:")
    lines.append("  Test whether the XKF1/kinematic heuristic adaptive-weight method is robust")
    lines.append("  across randomized threshold and weight configurations.")
    lines.append("")
    lines.append("Success definition:")
    lines.append("  A file succeeds if heuristic adaptive RMSE is lower than original V9 recovered RMSE.")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"  Detection folder: {DETECTION_DIR}")
    lines.append(f"  XKF1 folder: {XKF1_DIR}")
    lines.append("")
    lines.append("Outputs:")
    lines.append(f"  All results CSV: {all_results_path}")
    lines.append(f"  Run summary CSV: {run_summary_path}")
    lines.append(f"  File failure summary CSV: {file_summary_path}")
    lines.append(f"  Axis summary CSV: {axis_summary_path}")
    lines.append(f"  Attack summary CSV: {attack_summary_path}")
    lines.append("")
    lines.append("Overall result:")
    lines.append(f"  Runs: {N_RUNS}")
    lines.append(f"  Files per run: {len(files)}")
    lines.append(f"  Total file-level evaluations: {total_cases}")
    lines.append(f"  Total file-level successes by RMSE: {total_success}")
    lines.append(f"  Total file-level failures by RMSE: {total_fail}")
    lines.append(f"  Overall file-level success rate by RMSE (%): {overall_success_rate}")
    lines.append("")
    lines.append(f"  Runs where all files succeeded: {run_all_success_count}/{N_RUNS}")
    lines.append(f"  All-file run success rate (%): {run_all_success_rate}")
    lines.append("")
    lines.append("Best run:")
    lines.append(f"  run_id: {best_run['run_id']}")
    lines.append(f"  success_count_rmse: {best_run['success_count_rmse']}/{best_run['n_files']}")
    lines.append(f"  success_rate_rmse_percent: {best_run['success_rate_rmse_percent']}")
    lines.append(f"  mean_heuristic_improvement_over_original_rmse_percent: {best_run['mean_heuristic_improvement_over_original_rmse_percent']}")
    lines.append("")
    lines.append("Worst run:")
    lines.append(f"  run_id: {worst_run['run_id']}")
    lines.append(f"  success_count_rmse: {worst_run['success_count_rmse']}/{worst_run['n_files']}")
    lines.append(f"  success_rate_rmse_percent: {worst_run['success_rate_rmse_percent']}")
    lines.append(f"  mean_heuristic_improvement_over_original_rmse_percent: {worst_run['mean_heuristic_improvement_over_original_rmse_percent']}")
    lines.append("")
    lines.append("Most frequently failed files:")
    lines.append("-" * 100)

    for _, r in file_summary.head(10).iterrows():
        lines.append(f"File: {r['file_name']}")
        lines.append(f"  Baseline: {r['baseline']}")
        lines.append(f"  Target: {r['target']}")
        lines.append(f"  Attack type: {r['attack_type']}")
        lines.append(f"  Fail count by RMSE: {r['fail_count_rmse']}/{r['evaluated_runs']}")
        lines.append(f"  Success rate by RMSE (%): {r['success_rate_rmse_percent']}")
        lines.append(f"  Mean original RMSE: {r['mean_original_rmse']}")
        lines.append(f"  Mean heuristic RMSE: {r['mean_heuristic_rmse']}")
        lines.append(f"  Mean heuristic improvement over original RMSE (%): {r['mean_heuristic_improvement_over_original_rmse_percent']}")
        lines.append("")

    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("  If all-file run success rate is high, the heuristic adaptive method is robust.")
    lines.append("  If many runs fail on the same files, those scenarios are unstable for heuristic weighting.")
    lines.append("  If success rate is mixed, keep original V9 as official and report heuristic weighting as experimental.")
    lines.append("")

    if run_all_success_rate >= 90:
        lines.append("Final interpretation:")
        lines.append("  The heuristic adaptive method appears robust across randomized configurations.")
    elif overall_success_rate >= 70:
        lines.append("Final interpretation:")
        lines.append("  The heuristic adaptive method is partially useful but not consistently robust.")
        lines.append("  Original V9 should remain the official method unless a tuned configuration is justified.")
    else:
        lines.append("Final interpretation:")
        lines.append("  The heuristic adaptive method is unstable compared with original V9.")
        lines.append("  Original V9 should remain the official recovery method.")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print("=" * 100)
    print("100-RUN HEURISTIC ROBUSTNESS TEST COMPLETE")
    print("=" * 100)
    print("")
    print(f"All results:")
    print(f"  {all_results_path}")
    print("")
    print(f"Run summary:")
    print(f"  {run_summary_path}")
    print("")
    print(f"File failure summary:")
    print(f"  {file_summary_path}")
    print("")
    print(f"Axis summary:")
    print(f"  {axis_summary_path}")
    print("")
    print(f"Attack summary:")
    print(f"  {attack_summary_path}")
    print("")
    print(f"Interpretation report:")
    print(f"  {report_path}")
    print("")
    print("Overall:")
    print(f"  Total file-level evaluations: {total_cases}")
    print(f"  File-level successes: {total_success}")
    print(f"  File-level failures: {total_fail}")
    print(f"  File-level success rate (%): {overall_success_rate}")
    print(f"  Runs with all files successful: {run_all_success_count}/{N_RUNS}")
    print(f"  All-file run success rate (%): {run_all_success_rate}")
    print("")


if __name__ == "__main__":
    main()