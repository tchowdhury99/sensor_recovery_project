#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np

# ============================================================
# PHASE 6: SOFTWARE-SENSOR RECOVERY
# ============================================================
# This script:
# 1. Reads attacked CSV files produced in Phase 5
# 2. Finds the corresponding clean baseline CSV files
# 3. Builds software-sensor reference signals
# 4. Computes residuals
# 5. Detects attacks using thresholds
# 6. Replaces attacked values with software-sensor values
# 7. Saves recovered CSV files
# ============================================================

PROJECT_DIR = os.path.expanduser("~/sensor_recovery_project")

CLEAN_DIR = os.path.join(PROJECT_DIR, "logs", "extracted_csv")
ATTACKED_DIR = os.path.join(PROJECT_DIR, "logs", "attacked_csv")
RECOVERED_DIR = os.path.join(PROJECT_DIR, "logs", "recovered_csv")

os.makedirs(RECOVERED_DIR, exist_ok=True)


# ------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------
# These thresholds are intentionally simple and conservative.
# We may tune them later after reviewing the residual plots.
# ------------------------------------------------------------

THRESHOLDS = {
    "GPS_Lat": 0.000005,     # Latitude difference threshold
    "GPS_Lng": 0.000005,     # Longitude difference threshold
    "BARO_Alt": 0.75,        # Altitude difference threshold in meters
    "IMU_GyrZ": 0.08         # Gyro Z threshold in rad/s
}


def recover_signal(
    attacked_df,
    clean_df,
    attacked_col,
    clean_col,
    software_col_name,
    residual_col_name,
    flag_col_name,
    recovered_col_name,
    threshold
):
    """
    Recover one attacked sensor signal.

    attacked_df: attacked CSV dataframe
    clean_df: corresponding clean CSV dataframe
    attacked_col: column name in attacked data
    clean_col: column name in clean data
    software_col_name: new software-sensor estimate column
    residual_col_name: residual column
    flag_col_name: attack detection flag column
    recovered_col_name: recovered output column
    threshold: attack detection threshold
    """

    # Make sure both files are aligned to the shortest common length
    n = min(len(attacked_df), len(clean_df))
    attacked_df = attacked_df.iloc[:n].copy()
    clean_df = clean_df.iloc[:n].copy()

    # Software-sensor estimate:
    # For this offline reproduction we use the corresponding clean baseline
    # as a reference estimate of what the sensor should have looked like.
    attacked_df[software_col_name] = clean_df[clean_col].values

    # Residual = absolute difference between attacked physical reading
    # and software-sensor estimate
    attacked_df[residual_col_name] = np.abs(
        attacked_df[attacked_col] - attacked_df[software_col_name]
    )

    # Detection flag
    attacked_df[flag_col_name] = (
        attacked_df[residual_col_name] > threshold
    ).astype(int)

    # Recovered signal:
    # If attack is detected -> use software sensor
    # Else -> keep physical sensor reading
    attacked_df[recovered_col_name] = np.where(
        attacked_df[flag_col_name] == 1,
        attacked_df[software_col_name],
        attacked_df[attacked_col]
    )

    return attacked_df


def process_gps_file(attacked_file):
    """
    Recover GPS latitude and longitude attacks.
    """

    filename = os.path.basename(attacked_file)
    clean_filename = filename.replace("_ATTACKED_GPS", "")
    clean_file = os.path.join(CLEAN_DIR, clean_filename)

    if not os.path.exists(clean_file):
        print(f"[WARNING] Clean GPS file not found for: {filename}")
        return

    attacked_df = pd.read_csv(attacked_file)
    clean_df = pd.read_csv(clean_file)

    # Column checks
    required_cols = ["Lat", "Lng"]
    for col in required_cols:
        if col not in attacked_df.columns or col not in clean_df.columns:
            print(f"[WARNING] Missing GPS column '{col}' in {filename}")
            return

    # Recover Latitude
    attacked_df = recover_signal(
        attacked_df=attacked_df,
        clean_df=clean_df,
        attacked_col="Lat_attacked",
        clean_col="Lat",
        software_col_name="SoftwareSensor_Lat",
        residual_col_name="Residual_Lat",
        flag_col_name="DetectionFlag_Lat",
        recovered_col_name="Recovered_Lat",
        threshold=THRESHOLDS["GPS_Lat"]
    )

    # Recover Longitude
    attacked_df = recover_signal(
        attacked_df=attacked_df,
        clean_df=clean_df,
        attacked_col="Lng_attacked",
        clean_col="Lng",
        software_col_name="SoftwareSensor_Lng",
        residual_col_name="Residual_Lng",
        flag_col_name="DetectionFlag_Lng",
        recovered_col_name="Recovered_Lng",
        threshold=THRESHOLDS["GPS_Lng"]
    )

    output_name = filename.replace(".csv", "_recovered.csv")
    output_path = os.path.join(RECOVERED_DIR, output_name)
    attacked_df.to_csv(output_path, index=False)

    print(f"[OK] GPS recovered: {output_name}")


def process_baro_file(attacked_file):
    """
    Recover BARO altitude attacks.
    """

    filename = os.path.basename(attacked_file)
    clean_filename = filename.replace("_ATTACKED_BARO", "")
    clean_file = os.path.join(CLEAN_DIR, clean_filename)

    if not os.path.exists(clean_file):
        print(f"[WARNING] Clean BARO file not found for: {filename}")
        return

    attacked_df = pd.read_csv(attacked_file)
    clean_df = pd.read_csv(clean_file)

    # ArduPilot BARO CSV may use 'Alt' for altitude
    if "Alt" not in attacked_df.columns or "Alt" not in clean_df.columns:
        print(f"[WARNING] Missing BARO Alt column in {filename}")
        return

    attacked_df = recover_signal(
        attacked_df=attacked_df,
        clean_df=clean_df,
        attacked_col="Alt_attacked",
        clean_col="Alt",
        software_col_name="SoftwareSensor_Alt",
        residual_col_name="Residual_Alt",
        flag_col_name="DetectionFlag_Alt",
        recovered_col_name="Recovered_Alt",
        threshold=THRESHOLDS["BARO_Alt"]
    )

    output_name = filename.replace(".csv", "_recovered.csv")
    output_path = os.path.join(RECOVERED_DIR, output_name)
    attacked_df.to_csv(output_path, index=False)

    print(f"[OK] BARO recovered: {output_name}")


def process_imu_file(attacked_file):
    """
    Recover IMU Gyro Z attacks.
    """

    filename = os.path.basename(attacked_file)
    clean_filename = filename.replace("_ATTACKED_IMU", "")
    clean_file = os.path.join(CLEAN_DIR, clean_filename)

    if not os.path.exists(clean_file):
        print(f"[WARNING] Clean IMU file not found for: {filename}")
        return

    attacked_df = pd.read_csv(attacked_file)
    clean_df = pd.read_csv(clean_file)

    # In Phase 5 we attacked GyrZ
    if "GyrZ" not in attacked_df.columns or "GyrZ" not in clean_df.columns:
        print(f"[WARNING] Missing IMU GyrZ column in {filename}")
        return

    attacked_df = recover_signal(
        attacked_df=attacked_df,
        clean_df=clean_df,
        attacked_col="GyrZ_attacked",
        clean_col="GyrZ",
        software_col_name="SoftwareSensor_GyrZ",
        residual_col_name="Residual_GyrZ",
        flag_col_name="DetectionFlag_GyrZ",
        recovered_col_name="Recovered_GyrZ",
        threshold=THRESHOLDS["IMU_GyrZ"]
    )

    output_name = filename.replace(".csv", "_recovered.csv")
    output_path = os.path.join(RECOVERED_DIR, output_name)
    attacked_df.to_csv(output_path, index=False)

    print(f"[OK] IMU recovered: {output_name}")


def main():
    print("\n========== PHASE 6: SOFTWARE-SENSOR RECOVERY ==========\n")

    attacked_files = glob.glob(os.path.join(ATTACKED_DIR, "*.csv"))

    if not attacked_files:
        print("[ERROR] No attacked CSV files found.")
        print(f"Expected location: {ATTACKED_DIR}")
        return

    print(f"[INFO] Found {len(attacked_files)} attacked CSV files.\n")

    for attacked_file in attacked_files:
        filename = os.path.basename(attacked_file)

        if "_GPS_" in filename or filename.endswith("_GPS_attacked.csv"):
            process_gps_file(attacked_file)

        elif "_BARO_" in filename or filename.endswith("_BARO_attacked.csv"):
            process_baro_file(attacked_file)

        elif "_IMU_" in filename or filename.endswith("_IMU_attacked.csv"):
            process_imu_file(attacked_file)

        else:
            print(f"[SKIP] Unrecognized attacked file type: {filename}")

    print("\n========== PHASE 6 COMPLETE ==========")
    print(f"Recovered CSV files saved in:\n{RECOVERED_DIR}\n")


if __name__ == "__main__":
    main()