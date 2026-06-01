import os
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "/home/tchowdh4/sensor_recovery_project"

V9_DIR = f"{BASE}/prediction_results_v9_trained_hybrid"
OUT_DIR = f"{BASE}/residual_results_v9"
FIG_DIR = f"{BASE}/figures/clean_residuals_v9"
THRESH_DIR = f"{BASE}/thresholds_v9"

THRESH_CSV = f"{THRESH_DIR}/phase3_clean_residual_thresholds_v9.csv"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(THRESH_DIR, exist_ok=True)

TARGETS = ["Roll", "Pitch", "Yaw"]

BASELINES = [
    "baseline_01_hover",
    "baseline_02_high_altitude_hover",
    "baseline_03_forward_motion",
    "baseline_04_yaw_rotation",
    "baseline_05_mixed_maneuver",
]


def normalize_col(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).lower())


def find_first_column(df, aliases):
    """
    Find the first matching column using flexible name matching.
    """
    norm_map = {normalize_col(c): c for c in df.columns}

    for alias in aliases:
        a = normalize_col(alias)
        if a in norm_map:
            return norm_map[a]

    # Partial match fallback
    for alias in aliases:
        a = normalize_col(alias)
        for nc, original in norm_map.items():
            if a in nc or nc in a:
                return original

    return None


def infer_baseline_name(path, df):
    """
    Infer baseline file name from column values or filename.
    """
    possible_cols = [
        "att_file", "file", "filename", "source_file", "log_file",
        "baseline", "flight", "flight_file"
    ]

    for c in possible_cols:
        col = find_first_column(df, [c])
        if col is not None:
            vals = df[col].dropna().astype(str).unique()
            for v in vals:
                for b in BASELINES:
                    if b in v:
                        return b

    fname = os.path.basename(path)
    for b in BASELINES:
        if b in fname:
            return b

    # fallback: remove extension
    return os.path.splitext(fname)[0]


def infer_target_name(path, df):
    """
    Infer target from target column or filename.
    """
    col = find_first_column(df, ["target", "axis", "attitude_target"])
    if col is not None:
        vals = df[col].dropna().astype(str).unique()
        for v in vals:
            for t in TARGETS:
                if t.lower() == v.lower() or t.lower() in v.lower():
                    return t

    fname = os.path.basename(path).lower()
    for t in TARGETS:
        if t.lower() in fname:
            return t

    return None


def find_time_column(df):
    aliases = [
        "time_sec", "TimeSec", "t", "time", "timestamp", "TimeUS",
        "prediction_time", "future_time"
    ]
    return find_first_column(df, aliases)


def get_time_seconds(df):
    tcol = find_time_column(df)

    if tcol is None:
        return np.arange(len(df), dtype=float), "sample_index"

    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy()

    # If TimeUS, convert to relative seconds
    if normalize_col(tcol) == normalize_col("TimeUS"):
        t = (t - np.nanmin(t)) / 1_000_000.0
        return t, "TimeUS_relative_sec"

    # If timestamp looks too large, make relative
    if np.nanmax(t) > 1_000_000:
        t = (t - np.nanmin(t)) / 1_000_000.0
        return t, f"{tcol}_relative_sec"

    return t, tcol


def find_prediction_columns_for_target(df, target):
    """
    Finds true_future and prediction columns for one target.
    This supports both long-format and per-target CSVs.
    """
    t = target

    true_aliases = [
        f"{t}_true_future", f"true_future_{t}", f"{t}_future_true",
        f"future_true_{t}", f"{t}_actual_future", f"actual_future_{t}",
        f"{t}_y_true", f"y_true_{t}", "true_future", "actual_future",
        "y_true", "true", "future_true"
    ]

    hybrid_aliases = [
        f"{t}_hybrid_prediction", f"hybrid_prediction_{t}",
        f"{t}_v9_prediction", f"v9_prediction_{t}",
        f"{t}_pred_hybrid", f"pred_hybrid_{t}",
        f"{t}_y_pred", f"y_pred_{t}",
        "hybrid_prediction", "v9_prediction", "pred_hybrid",
        "hybrid_pred", "prediction", "y_pred"
    ]

    naive_aliases = [
        f"{t}_naive_prediction", f"naive_prediction_{t}",
        f"{t}_naive_pred", f"naive_pred_{t}",
        "naive_prediction", "naive_pred"
    ]

    cv_aliases = [
        f"{t}_constant_velocity_prediction",
        f"constant_velocity_prediction_{t}",
        f"{t}_cv_prediction", f"cv_prediction_{t}",
        f"{t}_constant_velocity_pred",
        "constant_velocity_prediction",
        "constant_velocity_pred",
        "cv_prediction",
        "cv_pred"
    ]

    arx_aliases = [
        f"{t}_arx_prediction", f"arx_prediction_{t}",
        f"{t}_arx_component", f"arx_component_{t}",
        f"{t}_trained_arx_prediction", f"trained_arx_prediction_{t}",
        "arx_prediction", "arx_pred", "arx_component",
        "trained_arx_prediction", "trained_arx_pred"
    ]

    return {
        "true_future": find_first_column(df, true_aliases),
        "hybrid_prediction": find_first_column(df, hybrid_aliases),
        "naive_prediction": find_first_column(df, naive_aliases),
        "constant_velocity_prediction": find_first_column(df, cv_aliases),
        "arx_prediction": find_first_column(df, arx_aliases),
    }


def compute_stats(values):
    """
    Compute clean residual statistics for one residual vector.
    """
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "max_abs": np.nan,
            "p95_abs": np.nan,
            "p99_abs": np.nan,
            "p995_abs": np.nan,
            "mean_abs": np.nan,
            "std_abs": np.nan,
            "mean_abs_plus_3std_abs": np.nan,
        }

    abs_x = np.abs(x)

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "mae": float(np.mean(abs_x)),
        "rmse": float(np.sqrt(np.mean(x ** 2))),
        "max_abs": float(np.max(abs_x)),
        "p95_abs": float(np.percentile(abs_x, 95)),
        "p99_abs": float(np.percentile(abs_x, 99)),
        "p995_abs": float(np.percentile(abs_x, 99.5)),
        "mean_abs": float(np.mean(abs_x)),
        "std_abs": float(np.std(abs_x, ddof=1)) if len(abs_x) > 1 else 0.0,
        "mean_abs_plus_3std_abs": float(np.mean(abs_x) + 3 * (np.std(abs_x, ddof=1) if len(abs_x) > 1 else 0.0)),
    }


def choose_recommended_threshold(stats):
    """
    For clean residual thresholding, percentile thresholds are usually more robust
    than mean_abs + 3*std_abs when the residual distribution is non-Gaussian.

    Recommended default:
    - use p99_abs for balanced detection
    - keep p995_abs as conservative option
    """
    p99 = stats["p99_abs"]
    p995 = stats["p995_abs"]
    mean3 = stats["mean_abs_plus_3std_abs"]

    if np.isnan(p99):
        return np.nan, "unavailable"

    # If mean+3std is much lower than p99, use p99 because residuals may have tails.
    # If mean+3std is higher, p99 is still easier to interpret and tied to false alarm rate.
    return p99, "p99_abs"


def safe_series(df, col):
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def process_single_prediction_file(path):
    df = pd.read_csv(path)

    if df.empty:
        return [], []

    fname = os.path.basename(path)

    # Skip pure summary files
    lower_name = fname.lower()
    if "summary" in lower_name and not any(t.lower() in lower_name for t in TARGETS):
        return [], []

    baseline = infer_baseline_name(path, df)
    target_from_file = infer_target_name(path, df)

    # If target column exists with multiple targets, process each group.
    target_col = find_first_column(df, ["target", "axis", "attitude_target"])

    records = []
    residual_frames = []

    if target_col is not None:
        unique_targets = []
        for val in df[target_col].dropna().astype(str).unique():
            for t in TARGETS:
                if t.lower() == val.lower() or t.lower() in val.lower():
                    unique_targets.append(t)
        unique_targets = sorted(set(unique_targets), key=TARGETS.index)

        if unique_targets:
            for target in unique_targets:
                mask = df[target_col].astype(str).str.lower().str.contains(target.lower(), na=False)
                sub = df[mask].copy()
                r_records, r_frames = process_target_dataframe(path, sub, baseline, target)
                records.extend(r_records)
                residual_frames.extend(r_frames)
            return records, residual_frames

    # Otherwise infer one target from filename or wide columns.
    if target_from_file is not None:
        r_records, r_frames = process_target_dataframe(path, df.copy(), baseline, target_from_file)
        records.extend(r_records)
        residual_frames.extend(r_frames)
    else:
        # Wide-format possibility: process all targets if matching columns exist.
        for target in TARGETS:
            cols = find_prediction_columns_for_target(df, target)
            if cols["true_future"] is not None and cols["hybrid_prediction"] is not None:
                r_records, r_frames = process_target_dataframe(path, df.copy(), baseline, target)
                records.extend(r_records)
                residual_frames.extend(r_frames)

    return records, residual_frames


def process_target_dataframe(path, df, baseline, target):
    cols = find_prediction_columns_for_target(df, target)

    true_col = cols["true_future"]
    hybrid_col = cols["hybrid_prediction"]

    if true_col is None or hybrid_col is None:
        print(f"[SKIP] Could not find required true/hybrid columns for {target} in {os.path.basename(path)}")
        print(f"       true_future column found: {true_col}")
        print(f"       hybrid_prediction column found: {hybrid_col}")
        return [], []

    time_sec, time_source = get_time_seconds(df)

    true_future = safe_series(df, true_col)
    hybrid_pred = safe_series(df, hybrid_col)
    naive_pred = safe_series(df, cols["naive_prediction"])
    cv_pred = safe_series(df, cols["constant_velocity_prediction"])
    arx_pred = safe_series(df, cols["arx_prediction"])

    out = pd.DataFrame()
    out["time_sec"] = time_sec
    out["baseline_file"] = baseline
    out["target"] = target
    out["true_future"] = true_future
    out["hybrid_prediction"] = hybrid_pred

    out["hybrid_residual"] = out["true_future"] - out["hybrid_prediction"]
    out["hybrid_abs_residual"] = np.abs(out["hybrid_residual"])

    if naive_pred is not None:
        out["naive_prediction"] = naive_pred
        out["naive_residual"] = out["true_future"] - out["naive_prediction"]
        out["naive_abs_residual"] = np.abs(out["naive_residual"])
    else:
        out["naive_prediction"] = np.nan
        out["naive_residual"] = np.nan
        out["naive_abs_residual"] = np.nan

    if cv_pred is not None:
        out["constant_velocity_prediction"] = cv_pred
        out["constant_velocity_residual"] = out["true_future"] - out["constant_velocity_prediction"]
        out["constant_velocity_abs_residual"] = np.abs(out["constant_velocity_residual"])
    else:
        out["constant_velocity_prediction"] = np.nan
        out["constant_velocity_residual"] = np.nan
        out["constant_velocity_abs_residual"] = np.nan

    if arx_pred is not None:
        out["arx_prediction"] = arx_pred
        out["arx_residual"] = out["true_future"] - out["arx_prediction"]
        out["arx_abs_residual"] = np.abs(out["arx_residual"])
    else:
        out["arx_prediction"] = np.nan
        out["arx_residual"] = np.nan
        out["arx_abs_residual"] = np.nan

    # Drop rows where true or hybrid is missing
    out = out.dropna(subset=["true_future", "hybrid_prediction"]).reset_index(drop=True)

    if out.empty:
        return [], []

    source_name = os.path.splitext(os.path.basename(path))[0]

    residual_csv = f"{OUT_DIR}/{baseline}_{target}_clean_residuals_v9.csv"
    out.to_csv(residual_csv, index=False)

    records = []

    residual_types = {
        "hybrid": "hybrid_residual",
        "naive": "naive_residual",
        "constant_velocity": "constant_velocity_residual",
        "arx": "arx_residual",
    }

    for model_name, residual_col in residual_types.items():
        stats = compute_stats(out[residual_col])
        recommended_threshold, recommended_method = choose_recommended_threshold(stats)

        record = {
            "baseline_file": baseline,
            "target": target,
            "model": model_name,
            "source_prediction_csv": path,
            "residual_csv": residual_csv,
            "time_source": time_source,
            **stats,
            "recommended_threshold": recommended_threshold,
            "recommended_threshold_method": recommended_method,
        }
        records.append(record)

    make_plots(out, baseline, target)

    print(f"[OK] Saved residuals: {residual_csv}")

    return records, [out]


def make_plots(out, baseline, target):
    """
    Make three clean residual plots:
    1. signed hybrid residual vs time
    2. absolute hybrid residual with p99 threshold
    3. comparison of abs residuals: hybrid vs naive vs CV vs ARX
    """
    time = out["time_sec"]

    stats = compute_stats(out["hybrid_residual"])
    threshold, threshold_method = choose_recommended_threshold(stats)

    # 1. Signed hybrid residual
    plt.figure(figsize=(12, 5))
    plt.plot(time, out["hybrid_residual"], linewidth=1.0)
    plt.axhline(0, linestyle="--", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel(f"{target} residual")
    plt.title(f"Clean signed V9 hybrid residual: {baseline} - {target}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{baseline}_{target}_signed_hybrid_residual_v9.png", dpi=200)
    plt.close()

    # 2. Absolute hybrid residual with threshold
    plt.figure(figsize=(12, 5))
    plt.plot(time, out["hybrid_abs_residual"], linewidth=1.0, label="|hybrid residual|")
    if not np.isnan(threshold):
        plt.axhline(threshold, linestyle="--", linewidth=1.2, label=f"{threshold_method} threshold = {threshold:.6g}")
    plt.xlabel("Time (s)")
    plt.ylabel(f"|{target} residual|")
    plt.title(f"Clean absolute V9 hybrid residual with threshold: {baseline} - {target}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{baseline}_{target}_abs_hybrid_residual_threshold_v9.png", dpi=200)
    plt.close()

    # 3. Compare hybrid / naive / CV / ARX residuals
    plt.figure(figsize=(12, 5))
    plt.plot(time, out["hybrid_abs_residual"], linewidth=1.0, label="Hybrid V9")

    if out["naive_abs_residual"].notna().any():
        plt.plot(time, out["naive_abs_residual"], linewidth=0.8, label="Naive")

    if out["constant_velocity_abs_residual"].notna().any():
        plt.plot(time, out["constant_velocity_abs_residual"], linewidth=0.8, label="Constant velocity")

    if out["arx_abs_residual"].notna().any():
        plt.plot(time, out["arx_abs_residual"], linewidth=0.8, label="ARX component")

    plt.xlabel("Time (s)")
    plt.ylabel(f"|{target} residual|")
    plt.title(f"Clean residual comparison: {baseline} - {target}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{baseline}_{target}_residual_model_comparison_v9.png", dpi=200)
    plt.close()


def make_combined_summary_plots(threshold_df):
    """
    Plot summary bars of p99 residuals and RMSE values for hybrid model only.
    """
    hybrid = threshold_df[threshold_df["model"] == "hybrid"].copy()

    if hybrid.empty:
        return

    hybrid["label"] = hybrid["baseline_file"] + " / " + hybrid["target"]

    # p99 abs residual summary
    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(hybrid)), hybrid["p99_abs"])
    plt.xticks(np.arange(len(hybrid)), hybrid["label"], rotation=80, ha="right")
    plt.ylabel("p99 absolute residual")
    plt.title("Phase 3 clean residual p99 thresholds - V9 hybrid")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/phase3_v9_hybrid_p99_threshold_summary.png", dpi=200)
    plt.close()

    # RMSE summary
    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(hybrid)), hybrid["rmse"])
    plt.xticks(np.arange(len(hybrid)), hybrid["label"], rotation=80, ha="right")
    plt.ylabel("RMSE")
    plt.title("Phase 3 clean residual RMSE summary - V9 hybrid")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/phase3_v9_hybrid_rmse_summary.png", dpi=200)
    plt.close()


def main():
    print("\n========== PHASE 3: CLEAN RESIDUAL GENERATION USING V9 ==========\n")

    print("Input V9 prediction directory:")
    print(V9_DIR)
    print("\nOutput residual directory:")
    print(OUT_DIR)
    print("\nOutput figure directory:")
    print(FIG_DIR)
    print("\nOutput threshold directory:")
    print(THRESH_DIR)

    csv_files = sorted(glob.glob(f"{V9_DIR}/**/*.csv", recursive=True))

    if not csv_files:
        raise FileNotFoundError(f"No V9 CSV files found under {V9_DIR}")

    print(f"\nFound {len(csv_files)} CSV files under V9 prediction output directory.")

    all_records = []
    processed_files = 0

    for path in csv_files:
        fname = os.path.basename(path)
        print(f"\n--- Checking: {fname}")

        try:
            records, _ = process_single_prediction_file(path)
        except Exception as e:
            print(f"[ERROR] Failed to process {path}")
            print(f"        {type(e).__name__}: {e}")
            continue

        if records:
            all_records.extend(records)
            processed_files += 1

    if not all_records:
        print("\nNo residual records were created.")
        print("This means the script could not identify the needed columns.")
        print("Run the inspection script and send me the printed column names.")
        raise SystemExit(1)

    threshold_df = pd.DataFrame(all_records)

    # Sort for readability
    threshold_df["baseline_order"] = threshold_df["baseline_file"].apply(
        lambda x: BASELINES.index(x) if x in BASELINES else 999
    )
    threshold_df["target_order"] = threshold_df["target"].apply(
        lambda x: TARGETS.index(x) if x in TARGETS else 999
    )
    threshold_df["model_order"] = threshold_df["model"].map(
        {"hybrid": 0, "naive": 1, "constant_velocity": 2, "arx": 3}
    ).fillna(999)

    threshold_df = threshold_df.sort_values(
        ["baseline_order", "target_order", "model_order"]
    ).drop(columns=["baseline_order", "target_order", "model_order"])

    threshold_df.to_csv(THRESH_CSV, index=False)

    make_combined_summary_plots(threshold_df)

    print("\n========== PHASE 3 OUTPUT SUMMARY ==========")
    print(f"Processed prediction CSV groups: {processed_files}")
    print(f"Residual/stat records created: {len(threshold_df)}")

    print("\nSaved threshold summary:")
    print(THRESH_CSV)

    print("\nSaved clean residual CSVs in:")
    print(OUT_DIR)

    print("\nSaved figures in:")
    print(FIG_DIR)

    print("\nHybrid threshold summary:")
    hybrid = threshold_df[threshold_df["model"] == "hybrid"]
    cols = [
        "baseline_file", "target", "n", "mae", "rmse",
        "max_abs", "p95_abs", "p99_abs", "p995_abs",
        "mean_abs_plus_3std_abs", "recommended_threshold",
        "recommended_threshold_method"
    ]
    print(hybrid[cols].to_string(index=False))

    print("\n========== DONE: PHASE 3 CLEAN RESIDUAL GENERATION COMPLETE ==========\n")


if __name__ == "__main__":
    main()