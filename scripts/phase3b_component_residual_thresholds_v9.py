#!/usr/bin/env python3

"""
Phase 3B: Component Residual and Threshold Audit for V9 Trained Hybrid Predictor

Purpose:
    This script extends Phase 3 without modifying official Phase 1/2/3/4/5 outputs.

    It reads the official V9 trained hybrid prediction-output CSVs and computes
    clean residuals for each axis and each predictor component:

        residual = true_future - prediction

    Components:
        1. hybrid
        2. naive
        3. cv
        4. arx

Outputs:
    - Component residual CSVs
    - Component-wise clean residual threshold summary
    - Hybrid-vs-official Phase 3 threshold comparison
    - Figures
    - Markdown report

Scientific rule:
    This script is read-only with respect to official project outputs.
    It creates Phase 3B extension outputs only.
"""

from pathlib import Path
import re
import math
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path("/home/tchowdh4/sensor_recovery_project")

PRED_DIR = PROJECT_ROOT / "prediction_results_v9_trained_hybrid"
OFFICIAL_THRESH_DIR = PROJECT_ROOT / "thresholds_v9"

OUT_RESID_DIR = PROJECT_ROOT / "phase3b_component_residuals_v9"
OUT_THRESH_DIR = PROJECT_ROOT / "phase3b_component_thresholds_v9"
OUT_REPORT_DIR = OUT_THRESH_DIR / "reports"
OUT_FIG_DIR = PROJECT_ROOT / "figures" / "phase3b_component_residuals_v9"

SUMMARY_CSV = OUT_THRESH_DIR / "phase3b_component_clean_residual_thresholds.csv"
OFFICIAL_COMPARE_CSV = OUT_THRESH_DIR / "phase3b_hybrid_vs_official_phase3_threshold_comparison.csv"
REPORT_MD = OUT_REPORT_DIR / "phase3b_component_residual_threshold_report.md"

AXES = ["Roll", "Pitch", "Yaw"]
COMPONENTS = ["hybrid", "naive", "cv", "arx"]


# =============================================================================
# Utility functions
# =============================================================================

def ensure_dirs():
    OUT_RESID_DIR.mkdir(parents=True, exist_ok=True)
    OUT_THRESH_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_col(name: str) -> str:
    """
    Normalize a column name to make matching robust.
    Example:
        'Roll_Hybrid_Prediction' -> 'rollhybridprediction'
    """
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())



def find_column(df: pd.DataFrame, axis: str, role: str):
    """
    Exact V9 Phase 3B column mapping.

    Your V9 prediction-output CSVs use these columns:

        Roll_true_future
        Roll_hybrid_prediction
        Roll_arx_component
        Roll_naive_component
        Roll_constant_velocity_component

    Same pattern for Pitch and Yaw.
    """

    exact_map = {
        "true_future": f"{axis}_true_future",
        "hybrid": f"{axis}_hybrid_prediction",
        "naive": f"{axis}_naive_component",
        "cv": f"{axis}_constant_velocity_component",
        "arx": f"{axis}_arx_component",
    }

    col = exact_map.get(role)
    if col in df.columns:
        return col

    return None


def infer_baseline_name(csv_path: Path) -> str:
    name = csv_path.name
    name = name.replace("_trained_hybrid_prediction_outputs.csv", "")
    name = name.replace(".csv", "")
    return name


def compute_stats(values: np.ndarray):
    """
    Compute clean residual statistics.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    abs_values = np.abs(values)

    mean_residual = float(np.mean(values))
    std_residual = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    mae = float(np.mean(abs_values))
    rmse = float(np.sqrt(np.mean(values ** 2)))
    max_abs = float(np.max(abs_values))
    p95_abs = float(np.percentile(abs_values, 95))
    p99_abs = float(np.percentile(abs_values, 99))
    p995_abs = float(np.percentile(abs_values, 99.5))
    mean_abs = float(np.mean(abs_values))
    std_abs = float(np.std(abs_values, ddof=1)) if values.size > 1 else 0.0
    mean_abs_plus_3std_abs = float(mean_abs + 3.0 * std_abs)

    # Conservative but not fully max-driven:
    # Use the larger of 99.5th percentile and mean_abs + 3*std_abs.
    recommended_threshold = float(max(p995_abs, mean_abs_plus_3std_abs))

    return {
        "n": int(values.size),
        "mean_residual": mean_residual,
        "std_residual": std_residual,
        "mae": mae,
        "rmse": rmse,
        "max_abs_residual": max_abs,
        "p95_abs_residual": p95_abs,
        "p99_abs_residual": p99_abs,
        "p995_abs_residual": p995_abs,
        "mean_abs_plus_3std_abs_threshold": mean_abs_plus_3std_abs,
        "recommended_threshold": recommended_threshold,
    }


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# =============================================================================
# Official Phase 3 threshold comparison
# =============================================================================

def load_official_thresholds():
    """
    Try to load official Phase 3 thresholds.

    Because the exact file name/column names may differ, this function scans
    all CSVs in thresholds_v9 and extracts axis/threshold-like values.

    Expected flexible possibilities:
        axis, threshold
        Axis, RecommendedThreshold
        Roll threshold stored in row/column names
    """

    rows = []

    if not OFFICIAL_THRESH_DIR.exists():
        return pd.DataFrame()

    for csv_path in sorted(OFFICIAL_THRESH_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            warnings.warn(f"Could not read official threshold file {csv_path}: {exc}")
            continue

        norm_cols = {col: normalize_col(col) for col in df.columns}

        axis_col = None
        threshold_col = None

        for col, ncol in norm_cols.items():
            if ncol in ["axis", "attitudeaxis", "targetaxis"]:
                axis_col = col
            if any(token in ncol for token in ["recommendedthreshold", "threshold", "finalthreshold"]):
                threshold_col = col

        if axis_col is not None and threshold_col is not None:
            for _, row in df.iterrows():
                axis_val = str(row[axis_col])
                matched_axis = None
                for axis in AXES:
                    if axis.lower() in axis_val.lower():
                        matched_axis = axis
                        break
                if matched_axis is None:
                    continue

                threshold_val = pd.to_numeric(row[threshold_col], errors="coerce")
                if pd.notna(threshold_val):
                    rows.append({
                        "official_file": str(csv_path),
                        "axis": matched_axis,
                        "official_threshold": float(threshold_val),
                    })
            continue

        # Fallback: look for columns that directly mention Roll/Pitch/Yaw and threshold.
        for col in df.columns:
            ncol = normalize_col(col)
            for axis in AXES:
                if axis.lower() in ncol and "threshold" in ncol:
                    numeric_vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(numeric_vals) > 0:
                        rows.append({
                            "official_file": str(csv_path),
                            "axis": axis,
                            "official_threshold": float(numeric_vals.iloc[0]),
                        })

    if not rows:
        return pd.DataFrame()

    official = pd.DataFrame(rows)

    # If multiple official values exist per axis, keep the median but preserve files in report.
    official_reduced = (
        official
        .groupby("axis", as_index=False)
        .agg(
            official_threshold=("official_threshold", "median"),
            official_sources=("official_file", lambda x: "; ".join(sorted(set(x))))
        )
    )

    return official_reduced


def compare_hybrid_to_official(summary_df: pd.DataFrame):
    official_df = load_official_thresholds()

    if official_df.empty:
        return pd.DataFrame()

    hybrid_df = summary_df[
        (summary_df["component"] == "hybrid") &
        (summary_df["aggregation_level"] == "all_baselines")
    ][["axis", "recommended_threshold"]].copy()

    comp = hybrid_df.merge(official_df, on="axis", how="left")
    comp["absolute_difference"] = comp["recommended_threshold"] - comp["official_threshold"]
    comp["relative_difference_percent"] = np.where(
        comp["official_threshold"].abs() > 1e-12,
        100.0 * comp["absolute_difference"] / comp["official_threshold"],
        np.nan
    )

    def classify(row):
        if pd.isna(row["official_threshold"]):
            return "NO_OFFICIAL_THRESHOLD_FOUND"
        rel = abs(row["relative_difference_percent"])
        if rel <= 5:
            return "MATCH_CLOSE"
        elif rel <= 15:
            return "MODERATE_DIFFERENCE_REVIEW"
        else:
            return "LARGE_DIFFERENCE_INVESTIGATE"

    comp["comparison_status"] = comp.apply(classify, axis=1)
    return comp


# =============================================================================
# Plotting
# =============================================================================

def plot_component_summary(summary_df: pd.DataFrame):
    """
    Generate one bar chart per axis comparing recommended thresholds for
    hybrid, naive, cv, and arx.
    """

    all_df = summary_df[summary_df["aggregation_level"] == "all_baselines"].copy()

    for axis in AXES:
        df_axis = all_df[all_df["axis"] == axis].copy()
        if df_axis.empty:
            continue

        order = ["hybrid", "naive", "cv", "arx"]
        df_axis["component"] = pd.Categorical(df_axis["component"], categories=order, ordered=True)
        df_axis = df_axis.sort_values("component")

        plt.figure(figsize=(8, 5))
        plt.bar(df_axis["component"].astype(str), df_axis["recommended_threshold"])
        plt.xlabel("Predictor Component")
        plt.ylabel("Recommended Clean Residual Threshold")
        plt.title(f"Phase 3B Recommended Thresholds by Component: {axis}")
        plt.tight_layout()

        out_path = OUT_FIG_DIR / f"phase3b_{axis.lower()}_component_recommended_thresholds.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_residual_histograms(residual_all_df: pd.DataFrame):
    """
    Generate histogram figures for each axis/component.
    """

    for axis in AXES:
        for component in COMPONENTS:
            col = f"{component}_residual"
            df = residual_all_df[residual_all_df["axis"] == axis]
            if col not in df.columns:
                continue

            vals = pd.to_numeric(df[col], errors="coerce").dropna().values
            if len(vals) == 0:
                continue

            plt.figure(figsize=(8, 5))
            plt.hist(vals, bins=80)
            plt.xlabel("Residual = true_future - prediction")
            plt.ylabel("Count")
            plt.title(f"Phase 3B Clean Residual Distribution: {axis} / {component}")
            plt.tight_layout()

            out_path = OUT_FIG_DIR / f"phase3b_{axis.lower()}_{component}_residual_histogram.png"
            plt.savefig(out_path, dpi=200)
            plt.close()


def plot_abs_residual_with_threshold(residual_all_df: pd.DataFrame, summary_df: pd.DataFrame):
    """
    Plot absolute residual traces with recommended threshold.
    Creates one plot per axis/component.
    """

    all_summary = summary_df[summary_df["aggregation_level"] == "all_baselines"].copy()

    for axis in AXES:
        df_axis = residual_all_df[residual_all_df["axis"] == axis].copy()

        for component in COMPONENTS:
            col = f"{component}_residual"
            if col not in df_axis.columns:
                continue

            vals = pd.to_numeric(df_axis[col], errors="coerce").dropna().values
            if len(vals) == 0:
                continue

            row = all_summary[
                (all_summary["axis"] == axis) &
                (all_summary["component"] == component)
            ]

            if row.empty:
                continue

            threshold = float(row["recommended_threshold"].iloc[0])

            plt.figure(figsize=(10, 5))
            plt.plot(np.abs(vals), linewidth=0.8)
            plt.axhline(threshold, linestyle="--", linewidth=1.5)
            plt.xlabel("Clean sample index")
            plt.ylabel("Absolute residual")
            plt.title(f"Phase 3B Absolute Residual vs Threshold: {axis} / {component}")
            plt.tight_layout()

            out_path = OUT_FIG_DIR / f"phase3b_{axis.lower()}_{component}_abs_residual_threshold.png"
            plt.savefig(out_path, dpi=200)
            plt.close()


# =============================================================================
# Main Phase 3B computation
# =============================================================================

def process_prediction_files():
    prediction_files = sorted(PRED_DIR.glob("*_trained_hybrid_prediction_outputs.csv"))

    if not prediction_files:
        raise FileNotFoundError(
            f"No V9 prediction output CSVs found in: {PRED_DIR}\n"
            f"Expected pattern: *_trained_hybrid_prediction_outputs.csv"
        )

    print(f"[INFO] Found {len(prediction_files)} V9 prediction output files.")

    stats_rows = []
    residual_frames_for_global = []
    skipped_records = []

    for csv_path in prediction_files:
        baseline_name = infer_baseline_name(csv_path)
        print(f"[INFO] Processing: {csv_path.name}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            skipped_records.append({
                "file": str(csv_path),
                "axis": "ALL",
                "reason": f"Could not read CSV: {exc}",
            })
            continue

        for axis in AXES:
            col_true = find_column(df, axis, "true_future")
            col_hybrid = find_column(df, axis, "hybrid")
            col_naive = find_column(df, axis, "naive")
            col_cv = find_column(df, axis, "cv")
            col_arx = find_column(df, axis, "arx")

            col_map = {
                "true_future": col_true,
                "hybrid": col_hybrid,
                "naive": col_naive,
                "cv": col_cv,
                "arx": col_arx,
            }

            missing = [k for k, v in col_map.items() if v is None]
            if missing:
                skipped_records.append({
                    "file": str(csv_path),
                    "axis": axis,
                    "reason": f"Missing required columns: {missing}; available columns: {list(df.columns)}",
                })
                continue

            out_df = pd.DataFrame()
            out_df["source_prediction_file"] = str(csv_path)
            out_df["baseline_name"] = baseline_name
            out_df["axis"] = axis

            if "timestamp" in df.columns:
                out_df["timestamp"] = df["timestamp"]
            elif "TimeUS" in df.columns:
                out_df["TimeUS"] = df["TimeUS"]
            else:
                out_df["row_index"] = np.arange(len(df))

            out_df["true_future"] = safe_numeric(df[col_true])
            out_df["hybrid_prediction"] = safe_numeric(df[col_hybrid])
            out_df["naive_component"] = safe_numeric(df[col_naive])
            out_df["cv_component"] = safe_numeric(df[col_cv])
            out_df["arx_component"] = safe_numeric(df[col_arx])

            out_df["hybrid_residual"] = out_df["true_future"] - out_df["hybrid_prediction"]
            out_df["naive_residual"] = out_df["true_future"] - out_df["naive_component"]
            out_df["cv_residual"] = out_df["true_future"] - out_df["cv_component"]
            out_df["arx_residual"] = out_df["true_future"] - out_df["arx_component"]

            out_df = out_df.replace([np.inf, -np.inf], np.nan)

            residual_csv = OUT_RESID_DIR / f"{baseline_name}_{axis.lower()}_component_residuals.csv"
            out_df.to_csv(residual_csv, index=False)

            residual_frames_for_global.append(out_df)

            for component in COMPONENTS:
                residual_col = f"{component}_residual"
                stats = compute_stats(out_df[residual_col].values)

                if stats is None:
                    skipped_records.append({
                        "file": str(csv_path),
                        "axis": axis,
                        "reason": f"No finite residuals for component {component}",
                    })
                    continue

                stats_rows.append({
                    "aggregation_level": "per_baseline",
                    "baseline_name": baseline_name,
                    "axis": axis,
                    "component": component,
                    "source_prediction_file": str(csv_path),
                    **stats,
                })

    if not residual_frames_for_global:
        raise RuntimeError(
            "No residuals were generated. This usually means the script could not identify "
            "the required true_future/hybrid/naive/cv/arx columns in the prediction-output CSVs."
        )

    residual_all_df = pd.concat(residual_frames_for_global, ignore_index=True)

    # Global/all-baseline stats
    for axis in AXES:
        df_axis = residual_all_df[residual_all_df["axis"] == axis]
        for component in COMPONENTS:
            residual_col = f"{component}_residual"
            stats = compute_stats(df_axis[residual_col].values)

            if stats is None:
                continue

            stats_rows.append({
                "aggregation_level": "all_baselines",
                "baseline_name": "ALL",
                "axis": axis,
                "component": component,
                "source_prediction_file": "ALL",
                **stats,
            })

    summary_df = pd.DataFrame(stats_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    skipped_df = pd.DataFrame(skipped_records)
    if not skipped_df.empty:
        skipped_path = OUT_THRESH_DIR / "phase3b_skipped_or_warning_records.csv"
        skipped_df.to_csv(skipped_path, index=False)
        print(f"[WARN] Some records were skipped or warned. See: {skipped_path}")

    return residual_all_df, summary_df, skipped_df


def write_report(summary_df: pd.DataFrame, compare_df: pd.DataFrame, skipped_df: pd.DataFrame):
    all_summary = summary_df[summary_df["aggregation_level"] == "all_baselines"].copy()

    lines = []
    lines.append("# Phase 3B Component Residual and Threshold Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "Phase 3B extends the official Phase 3 clean residual analysis by computing "
        "component-wise clean residual statistics for the V9 trained hybrid predictor."
    )
    lines.append("")
    lines.append("This phase is read-only with respect to official Phase 1/2/3/4/5 outputs.")
    lines.append("It does not retrain V9 and does not overwrite official thresholds or detection/recovery results.")
    lines.append("")

    lines.append("## Input folders")
    lines.append("")
    lines.append(f"- V9 prediction outputs: `{PRED_DIR}`")
    lines.append(f"- Official Phase 3 thresholds: `{OFFICIAL_THRESH_DIR}`")
    lines.append("")

    lines.append("## Output folders")
    lines.append("")
    lines.append(f"- Component residual CSVs: `{OUT_RESID_DIR}`")
    lines.append(f"- Component threshold summaries: `{OUT_THRESH_DIR}`")
    lines.append(f"- Figures: `{OUT_FIG_DIR}`")
    lines.append("")

    lines.append("## Residual definition")
    lines.append("")
    lines.append("For every clean baseline file, axis, and component:")
    lines.append("")
    lines.append("```text")
    lines.append("residual[k] = true_future[k] - prediction_component[k]")
    lines.append("```")
    lines.append("")
    lines.append("Components:")
    lines.append("")
    lines.append("- `hybrid`: V9 weighted hybrid prediction")
    lines.append("- `naive`: previous-value / naive component")
    lines.append("- `cv`: constant-velocity component")
    lines.append("- `arx`: ARX component")
    lines.append("")

    lines.append("## All-baseline recommended thresholds")
    lines.append("")
    if all_summary.empty:
        lines.append("No all-baseline summary rows were generated.")
    else:
        display_cols = [
            "axis",
            "component",
            "n",
            "mae",
            "rmse",
            "p95_abs_residual",
            "p99_abs_residual",
            "p995_abs_residual",
            "mean_abs_plus_3std_abs_threshold",
            "recommended_threshold",
        ]
        lines.append(all_summary[display_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Hybrid threshold comparison against official Phase 3")
    lines.append("")
    if compare_df.empty:
        lines.append(
            "No official Phase 3 threshold comparison was produced. "
            "This may mean no readable threshold CSV was found in the official threshold folder, "
            "or the official threshold column names were not recognized."
        )
    else:
        display_cols = [
            "axis",
            "recommended_threshold",
            "official_threshold",
            "absolute_difference",
            "relative_difference_percent",
            "comparison_status",
        ]
        lines.append(compare_df[display_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Skipped or warning records")
    lines.append("")
    if skipped_df.empty:
        lines.append("No skipped records.")
    else:
        lines.append(f"Skipped/warning records were saved separately in `{OUT_THRESH_DIR}`.")
        lines.append("")
        lines.append(skipped_df.head(20).to_markdown(index=False))
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("Use the all-baseline rows as the main Phase 5B component-threshold reference.")
    lines.append("")
    lines.append("- If the hybrid Phase 3B thresholds are close to official Phase 3 thresholds, Phase 3B is an extension only.")
    lines.append("- If the hybrid thresholds differ moderately, inspect column mapping, baseline inclusion, and threshold formula.")
    lines.append("- If the hybrid thresholds differ largely, do not overwrite official results automatically; investigate first.")
    lines.append("- Naive/CV/ARX thresholds should be used only for Phase 5B adaptive-weight analysis unless you explicitly decide to redesign Phase 4/5.")
    lines.append("")

    lines.append("## Final decision rule")
    lines.append("")
    lines.append("Phase 3B PASS if:")
    lines.append("")
    lines.append("1. Residual CSVs are generated for Roll, Pitch, and Yaw.")
    lines.append("2. Hybrid, naive, CV, and ARX component statistics exist for each axis.")
    lines.append("3. No required prediction columns are missing.")
    lines.append("4. Hybrid thresholds are close to official Phase 3 thresholds, or any difference is explainable.")
    lines.append("")
    lines.append("Phase 4/5 rerun is needed only if:")
    lines.append("")
    lines.append("1. The official Phase 3 hybrid threshold was computed from the wrong column.")
    lines.append("2. The official Phase 3 hybrid residual used present-state instead of future-state alignment.")
    lines.append("3. The official Phase 3 threshold differs from Phase 3B because of a confirmed bug, not merely because Phase 3B adds component-level statistics.")
    lines.append("")
    lines.append("Phase 5B may proceed if:")
    lines.append("")
    lines.append("1. Phase 2B component alignment already passed.")
    lines.append("2. Phase 3B component residual statistics were generated successfully.")
    lines.append("3. Component thresholds are used as diagnostic/scientific support for adaptive recovery, not as silent replacements for official Phase 4/5 thresholds.")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs()

    print("[INFO] Phase 3B component residual/threshold audit started.")
    print("[INFO] This script will not overwrite official Phase 1/2/3/4/5 outputs.")

    residual_all_df, summary_df, skipped_df = process_prediction_files()

    print(f"[INFO] Wrote component threshold summary: {SUMMARY_CSV}")

    compare_df = compare_hybrid_to_official(summary_df)
    if not compare_df.empty:
        compare_df.to_csv(OFFICIAL_COMPARE_CSV, index=False)
        print(f"[INFO] Wrote official Phase 3 comparison: {OFFICIAL_COMPARE_CSV}")
    else:
        print("[WARN] Official Phase 3 threshold comparison was not available.")

    print("[INFO] Generating figures...")
    plot_component_summary(summary_df)
    plot_residual_histograms(residual_all_df)
    plot_abs_residual_with_threshold(residual_all_df, summary_df)

    write_report(summary_df, compare_df, skipped_df)

    print(f"[INFO] Wrote report: {REPORT_MD}")
    print("[INFO] Phase 3B completed.")


if __name__ == "__main__":
    main()