#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path("/home/tchowdh4/sensor_recovery_project/phase2c_matlab_section31_si")
MERGED_DIR = BASE_DIR / "merged_csv"

REQUIRED_COLS = [
    "t_sec", "TimeUS",
    "DesRoll", "DesPitch", "DesYaw",
    "PN", "PE", "PD",
    "Roll", "Pitch", "Yaw",
    "VN", "VE", "VD",
    "GX", "GY", "GZ",
]

def main():
    files = sorted(MERGED_DIR.glob("*_merged_ATT_XKF1_C0.csv"))

    if not files:
        raise FileNotFoundError(f"No merged CSV files found in {MERGED_DIR}")

    print(f"Found {len(files)} merged CSV files.\n")

    for f in files:
        df = pd.read_csv(f)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        nan_count = df[REQUIRED_COLS].isna().sum().sum()
        inf_count = np.isinf(df[REQUIRED_COLS].select_dtypes(include=[np.number])).sum().sum()

        dt = df["t_sec"].diff().dropna()
        median_dt = dt.median() if len(dt) else np.nan
        approx_hz = 1.0 / median_dt if median_dt and median_dt > 0 else np.nan

        print("=" * 80)
        print(f"File: {f.name}")
        print(f"Rows: {len(df)}")
        print(f"Missing required columns: {missing}")
        print(f"NaN count in required columns: {nan_count}")
        print(f"Inf count in numeric required columns: {inf_count}")
        print(f"t_sec start: {df['t_sec'].iloc[0]:.6f}")
        print(f"t_sec end:   {df['t_sec'].iloc[-1]:.6f}")
        print(f"median dt:   {median_dt:.6f} sec")
        print(f"approx Hz:   {approx_hz:.3f}")

        if missing:
            print("STATUS: FAIL - missing columns")
        elif nan_count > 0:
            print("STATUS: WARNING - contains NaN")
        elif inf_count > 0:
            print("STATUS: WARNING - contains Inf")
        elif len(df) < 50:
            print("STATUS: WARNING - very few rows")
        else:
            print("STATUS: OK")

    print("\nCheck complete.")

if __name__ == "__main__":
    main()