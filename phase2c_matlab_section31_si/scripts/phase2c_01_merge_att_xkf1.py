#!/usr/bin/env python3
"""
Phase 2C-1: Merge ATT and XKF1 logs for MATLAB System Identification.

Purpose:
- Use ATT as input/reference source:
    u[k] = [DesRoll, DesPitch, DesYaw]

- Use XKF1 as output/state source:
    y[k] = [PN, PE, PD, Roll, Pitch, Yaw, VN, VE, VD, GX, GY, GZ]

- Filter XKF1 to C == 0 only.
- Align ATT and XKF1 using TimeUS with nearest-time merge.
- Output one MATLAB-ready merged CSV per maneuver.

Author: Phase 2C MATLAB Section 3.1 implementation
"""

from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path("/home/tchowdh4/sensor_recovery_project/phase2c_matlab_section31_si")
RAW_DIR = BASE_DIR / "raw_csv"
MERGED_DIR = BASE_DIR / "merged_csv"
REPORT_DIR = BASE_DIR / "reports"

MERGED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


MANEUVERS = [
    "baseline_01_hover",
    "baseline_02_high_altitude_hover",
    "baseline_03_forward_motion",
    "baseline_04_yaw_rotation",
    "baseline_05_mixed_maneuver",
]

ATT_INPUT_COLS = ["DesRoll", "DesPitch", "DesYaw"]

XKF1_OUTPUT_COLS = [
    "PN", "PE", "PD",
    "Roll", "Pitch", "Yaw",
    "VN", "VE", "VD",
    "GX", "GY", "GZ",
]

KEEP_COLS = ["t_sec", "TimeUS"] + ATT_INPUT_COLS + XKF1_OUTPUT_COLS


def require_columns(df: pd.DataFrame, required_cols, file_path: Path):
    """
    Verify that all required columns exist in a dataframe.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"\nMissing columns in {file_path}:\n"
            f"Missing: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
        )


def clean_timeus(df: pd.DataFrame, file_path: Path):
    """
    Ensure TimeUS exists, is numeric, sorted, and has no NaN values.
    """
    require_columns(df, ["TimeUS"], file_path)

    df = df.copy()
    df["TimeUS"] = pd.to_numeric(df["TimeUS"], errors="coerce")
    df = df.dropna(subset=["TimeUS"])
    df["TimeUS"] = df["TimeUS"].astype(np.int64)
    df = df.sort_values("TimeUS").drop_duplicates(subset=["TimeUS"], keep="first")
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{file_path} became empty after TimeUS cleaning.")

    return df


def merge_one_maneuver(maneuver: str):
    """
    Merge one ATT/XKF1 pair.
    """
    att_path = RAW_DIR / f"{maneuver}_ATT.csv"
    xkf1_path = RAW_DIR / f"{maneuver}_XKF1.csv"

    if not att_path.exists():
        raise FileNotFoundError(f"Missing ATT file: {att_path}")

    if not xkf1_path.exists():
        raise FileNotFoundError(f"Missing XKF1 file: {xkf1_path}")

    print(f"\n=== Processing {maneuver} ===")
    print(f"ATT : {att_path}")
    print(f"XKF1: {xkf1_path}")

    att = pd.read_csv(att_path)
    xkf1 = pd.read_csv(xkf1_path)

    att = clean_timeus(att, att_path)
    xkf1 = clean_timeus(xkf1, xkf1_path)

    require_columns(att, ["TimeUS"] + ATT_INPUT_COLS, att_path)
    require_columns(xkf1, ["TimeUS", "C"] + XKF1_OUTPUT_COLS, xkf1_path)

    # Filter XKF1 to estimator core C == 0 only.
    xkf1["C"] = pd.to_numeric(xkf1["C"], errors="coerce")
    xkf1_c0 = xkf1[xkf1["C"] == 0].copy()

    if xkf1_c0.empty:
        raise ValueError(f"No XKF1 rows with C == 0 found in {xkf1_path}")

    xkf1_c0 = xkf1_c0.sort_values("TimeUS").reset_index(drop=True)
    att = att.sort_values("TimeUS").reset_index(drop=True)

    # Keep only necessary columns before merge.
    att_small = att[["TimeUS"] + ATT_INPUT_COLS].copy()
    xkf1_small = xkf1_c0[["TimeUS"] + XKF1_OUTPUT_COLS].copy()

    # Nearest timestamp alignment.
    # Direction nearest means each XKF1 row receives nearest ATT command sample.
    # Tolerance is intentionally 150 ms because logs are approximately 10 Hz.
    tolerance_us = 150_000

    merged = pd.merge_asof(
        xkf1_small,
        att_small,
        on="TimeUS",
        direction="nearest",
        tolerance=tolerance_us,
        suffixes=("_xkf1", "_att"),
    )

    before_drop = len(merged)
    merged = merged.dropna(subset=ATT_INPUT_COLS + XKF1_OUTPUT_COLS).copy()
    after_drop = len(merged)

    if merged.empty:
        raise ValueError(
            f"Merged file for {maneuver} is empty. "
            f"Try increasing tolerance_us or inspect TimeUS ranges."
        )

    # Add time in seconds relative to first merged sample.
    t0 = merged["TimeUS"].iloc[0]
    merged["t_sec"] = (merged["TimeUS"] - t0) / 1_000_000.0

    # Reorder columns.
    merged = merged[KEEP_COLS].copy()

    # Sort again for safety.
    merged = merged.sort_values("TimeUS").reset_index(drop=True)

    output_path = MERGED_DIR / f"{maneuver}_merged_ATT_XKF1_C0.csv"
    merged.to_csv(output_path, index=False)

    duration_sec = merged["t_sec"].iloc[-1] - merged["t_sec"].iloc[0]
    median_dt = merged["t_sec"].diff().dropna().median()
    approx_hz = 1.0 / median_dt if median_dt and median_dt > 0 else np.nan

    summary = {
        "maneuver": maneuver,
        "att_rows": len(att),
        "xkf1_rows_total": len(xkf1),
        "xkf1_rows_C0": len(xkf1_c0),
        "merged_rows_before_drop": before_drop,
        "merged_rows_after_drop": after_drop,
        "dropped_rows_after_merge": before_drop - after_drop,
        "duration_sec": duration_sec,
        "median_dt_sec": median_dt,
        "approx_sample_rate_hz": approx_hz,
        "output_file": str(output_path),
    }

    print(f"Saved: {output_path}")
    print(f"Rows after merge: {after_drop}")
    print(f"Duration sec: {duration_sec:.3f}")
    print(f"Median dt sec: {median_dt:.4f}")
    print(f"Approx Hz: {approx_hz:.2f}")

    return summary


def main():
    all_summaries = []

    for maneuver in MANEUVERS:
        summary = merge_one_maneuver(maneuver)
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_path = REPORT_DIR / "phase2c_01_merge_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Phase 2C-1 merge complete ===")
    print(f"Summary report saved to: {summary_path}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()