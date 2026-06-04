#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/tchowdh4/sensor_recovery_project")

RESID_DIR = PROJECT_ROOT / "phase3b_component_residuals_v9"
THRESH_DIR = PROJECT_ROOT / "phase3b_component_thresholds_v9"
REPORT_DIR = THRESH_DIR / "reports"
FIG_DIR = PROJECT_ROOT / "figures" / "phase3b_component_residuals_v9"

SUMMARY_CSV = THRESH_DIR / "phase3b_component_clean_residual_thresholds.csv"
REPORT_MD = REPORT_DIR / "phase3b_component_residual_threshold_report.md"

AXES = ["Roll", "Pitch", "Yaw"]
COMPONENTS = ["hybrid", "naive", "cv", "arx"]


def infer_axis_from_filename(path: Path):
    name = path.name.lower()
    if "_roll_component_residuals.csv" in name:
        return "Roll"
    if "_pitch_component_residuals.csv" in name:
        return "Pitch"
    if "_yaw_component_residuals.csv" in name:
        return "Yaw"
    return None


def infer_baseline_from_filename(path: Path):
    name = path.name
    for axis in ["roll", "pitch", "yaw"]:
        suffix = f"_{axis}_component_residuals.csv"
        if name.endswith(suffix):
            return name.replace(suffix, "")
    return path.stem


def compute_stats(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    abs_values = np.abs(values)

    mean_residual = float(np.mean(values))
    std_residual = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    mae = float(np.mean(abs_values))
    rmse = float(np.sqrt(np.mean(values ** 2)))
    max_abs = float(np.max(abs_values))
    p95_abs = float(np.percentile(abs_values, 95))
    p99_abs = float(np.percentile(abs_values, 99))
    p995_abs = float(np.percentile(abs_values, 99.5))
    mean_abs = float(np.mean(abs_values))
    std_abs = float(np.std(abs_values, ddof=1)) if len(values) > 1 else 0.0
    mean_abs_plus_3std_abs = float(mean_abs + 3.0 * std_abs)
    recommended_threshold = float(max(p995_abs, mean_abs_plus_3std_abs))

    return {
        "n": int(len(values)),
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


def main():
    THRESH_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    residual_files = sorted(RESID_DIR.glob("*_component_residuals.csv"))
    print(f"[INFO] Found {len(residual_files)} residual CSV files.")

    if len(residual_files) == 0:
        raise FileNotFoundError(f"No residual CSV files found in {RESID_DIR}")

    frames = []

    for path in residual_files:
        df = pd.read_csv(path)

        axis = infer_axis_from_filename(path)
        baseline = infer_baseline_from_filename(path)

        if axis is None:
            raise RuntimeError(f"Could not infer axis from filename: {path.name}")

        df["axis"] = axis
        df["baseline_name"] = baseline
        df["source_residual_file"] = str(path)

        frames.append(df)

    all_residuals = pd.concat(frames, ignore_index=True)

    required_cols = [
        "baseline_name",
        "axis",
        "hybrid_residual",
        "naive_residual",
        "cv_residual",
        "arx_residual",
    ]

    missing = [c for c in required_cols if c not in all_residuals.columns]
    if missing:
        raise RuntimeError(f"Missing required residual columns: {missing}")

    print("[INFO] Residual rows loaded:", len(all_residuals))
    print("[INFO] Baselines:", sorted(all_residuals["baseline_name"].unique()))
    print("[INFO] Axes:", sorted(all_residuals["axis"].unique()))

    rows = []

    for baseline in sorted(all_residuals["baseline_name"].unique()):
        df_b = all_residuals[all_residuals["baseline_name"] == baseline]

        for axis in AXES:
            df_a = df_b[df_b["axis"] == axis]

            for component in COMPONENTS:
                col = f"{component}_residual"
                stats = compute_stats(df_a[col])

                if stats is None:
                    continue

                rows.append({
                    "aggregation_level": "per_baseline",
                    "baseline_name": baseline,
                    "axis": axis,
                    "component": component,
                    "source_prediction_file": "PHASE3B_RESIDUAL_CSV",
                    **stats,
                })

    for axis in AXES:
        df_a = all_residuals[all_residuals["axis"] == axis]

        for component in COMPONENTS:
            col = f"{component}_residual"
            stats = compute_stats(df_a[col])

            if stats is None:
                continue

            rows.append({
                "aggregation_level": "all_baselines",
                "baseline_name": "ALL",
                "axis": axis,
                "component": component,
                "source_prediction_file": "ALL_PHASE3B_RESIDUAL_CSVS",
                **stats,
            })

    summary = pd.DataFrame(rows)

    if summary.empty:
        raise RuntimeError("Summary table is empty. No thresholds were computed.")

    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[INFO] Wrote summary CSV: {SUMMARY_CSV}")

    all_summary = summary[summary["aggregation_level"] == "all_baselines"].copy()

    if len(all_summary) != 12:
        print(f"[WARN] Expected 12 all-baseline rows, found {len(all_summary)}.")
    else:
        print("[INFO] Correct: found 12 all-baseline rows.")

    for axis in AXES:
        df_axis = all_summary[all_summary["axis"] == axis].copy()
        if df_axis.empty:
            continue

        order = ["hybrid", "naive", "cv", "arx"]
        df_axis["component"] = pd.Categorical(df_axis["component"], categories=order, ordered=True)
        df_axis = df_axis.sort_values("component")

        plt.figure(figsize=(8, 5))
        plt.bar(df_axis["component"].astype(str), df_axis["recommended_threshold"])
        plt.xlabel("Component")
        plt.ylabel("Recommended threshold")
        plt.title(f"Phase 3B all-baseline component thresholds: {axis}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"phase3b_{axis.lower()}_all_baseline_component_thresholds.png", dpi=200)
        plt.close()

    show_cols = [
        "axis",
        "component",
        "n",
        "mae",
        "rmse",
        "p99_abs_residual",
        "p995_abs_residual",
        "mean_abs_plus_3std_abs_threshold",
        "recommended_threshold",
    ]

    report = []
    report.append("# Phase 3B Component Residual and Threshold Report")
    report.append("")
    report.append("## Status")
    report.append("")
    report.append("Phase 3B all-baseline threshold repair completed successfully.")
    report.append("")
    report.append("The repair inferred baseline name and axis from the residual CSV filenames.")
    report.append("")
    report.append("## All-baseline recommended thresholds")
    report.append("")
    report.append(all_summary[show_cols].to_markdown(index=False))
    report.append("")
    report.append("## Decision")
    report.append("")
    report.append("Phase 3B passes if this report contains 12 all-baseline rows: Roll/Pitch/Yaw × hybrid/naive/CV/ARX.")
    report.append("")
    report.append("Phase 4/5 should not be rerun from this repair alone.")
    report.append("Phase 5B can use these component thresholds as diagnostic support for adaptive-weight recovery.")

    REPORT_MD.write_text("\n".join(report), encoding="utf-8")
    print(f"[INFO] Wrote report: {REPORT_MD}")

    print("")
    print("[INFO] All-baseline rows:")
    print(all_summary[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()