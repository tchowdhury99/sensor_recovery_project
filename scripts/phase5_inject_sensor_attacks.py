#!/usr/bin/env python3

"""
Phase 5:
Inject controlled offline sensor attacks into extracted baseline CSV files.

Inputs:
    ~/sensor_recovery_project/logs/extracted_csv/

Outputs:
    ~/sensor_recovery_project/logs/attacked_csv/
    ~/sensor_recovery_project/figures/phase5_attack_validation/
    ~/sensor_recovery_project/logs/attacked_csv/phase5_attack_manifest.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
PROJECT_DIR = Path.home() / "sensor_recovery_project"
INPUT_DIR = PROJECT_DIR / "logs" / "extracted_csv"
OUTPUT_DIR = PROJECT_DIR / "logs" / "attacked_csv"
FIGURE_DIR = PROJECT_DIR / "figures" / "phase5_attack_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Attack configuration
# ------------------------------------------------------------
ATTACK_START_SEC = 10.0
ATTACK_END_SEC = 20.0

# GPS position spoofing offsets
GPS_LAT_OFFSET = 0.00012
GPS_LNG_OFFSET = -0.00012

# Barometer altitude attack in meters
BARO_ALT_OFFSET = 5.0

# Gyroscope Z-axis corruption in rad/s
GYRO_Z_OFFSET = 0.8


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def find_column(df, candidate_names):
    """
    Return the first column that matches one of candidate_names
    using case-insensitive matching.
    """
    lower_to_original = {col.lower(): col for col in df.columns}

    for candidate in candidate_names:
        candidate_lower = candidate.lower()
        if candidate_lower in lower_to_original:
            return lower_to_original[candidate_lower]

    return None


def add_time_seconds(df):
    """
    Create a TimeSec column.
    Primary choice: TimeUS from ArduPilot logs.
    """
    time_col = find_column(df, ["TimeUS", "timeus", "Time", "time"])

    if time_col is None:
        df["TimeSec"] = np.arange(len(df))
        return df, "sample_index"

    if "us" in time_col.lower():
        df["TimeSec"] = (df[time_col] - df[time_col].iloc[0]) / 1_000_000.0
    else:
        df["TimeSec"] = df[time_col] - df[time_col].iloc[0]

    return df, time_col


def build_attack_mask(df):
    """
    Boolean mask showing which rows lie inside the attack window.
    """
    return (df["TimeSec"] >= ATTACK_START_SEC) & (df["TimeSec"] <= ATTACK_END_SEC)


def save_plot(time_sec, clean_signal, attacked_signal, title, ylabel, output_path):
    """
    Save comparison plot: clean vs attacked signal.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(time_sec, clean_signal, label="Clean signal")
    plt.plot(time_sec, attacked_signal, label="Attacked signal", linestyle="--")
    plt.axvspan(ATTACK_START_SEC, ATTACK_END_SEC, alpha=0.2, label="Attack window")
    plt.xlabel("Time (seconds)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ------------------------------------------------------------
# Individual attack functions
# ------------------------------------------------------------
def attack_gps_file(csv_path):
    """
    Inject GPS spoofing attack:
    - Latitude offset
    - Longitude offset
    """
    df = pd.read_csv(csv_path)
    df, _ = add_time_seconds(df)
    attack_mask = build_attack_mask(df)

    lat_col = find_column(df, ["Lat", "Latitude"])
    lng_col = find_column(df, ["Lng", "Lon", "Longitude"])

    if lat_col is None or lng_col is None:
        print(f"[SKIP GPS] Required columns missing: {csv_path.name}")
        print(f"Available columns: {list(df.columns)}")
        return None

    df["AttackActive_GPS"] = attack_mask.astype(int)

    df[f"{lat_col}_attacked"] = df[lat_col].copy()
    df[f"{lng_col}_attacked"] = df[lng_col].copy()

    df.loc[attack_mask, f"{lat_col}_attacked"] += GPS_LAT_OFFSET
    df.loc[attack_mask, f"{lng_col}_attacked"] += GPS_LNG_OFFSET

    output_csv = OUTPUT_DIR / csv_path.name.replace(".csv", "_ATTACKED_GPS.csv")
    df.to_csv(output_csv, index=False)

    plot_lat = FIGURE_DIR / csv_path.name.replace(".csv", "_GPS_LAT_attack.png")
    plot_lng = FIGURE_DIR / csv_path.name.replace(".csv", "_GPS_LNG_attack.png")

    save_plot(
        df["TimeSec"],
        df[lat_col],
        df[f"{lat_col}_attacked"],
        f"GPS Latitude Attack — {csv_path.stem}",
        "Latitude",
        plot_lat
    )

    save_plot(
        df["TimeSec"],
        df[lng_col],
        df[f"{lng_col}_attacked"],
        f"GPS Longitude Attack — {csv_path.stem}",
        "Longitude",
        plot_lng
    )

    return {
        "input_file": csv_path.name,
        "output_file": output_csv.name,
        "sensor": "GPS",
        "attacked_signals": f"{lat_col}, {lng_col}",
        "attack_type": "constant position spoofing offset",
        "attack_start_sec": ATTACK_START_SEC,
        "attack_end_sec": ATTACK_END_SEC
    }


def attack_baro_file(csv_path):
    """
    Inject barometer attack:
    - Altitude offset
    """
    df = pd.read_csv(csv_path)
    df, _ = add_time_seconds(df)
    attack_mask = build_attack_mask(df)

    alt_col = find_column(df, ["Alt", "Altitude"])

    if alt_col is None:
        print(f"[SKIP BARO] Altitude column missing: {csv_path.name}")
        print(f"Available columns: {list(df.columns)}")
        return None

    df["AttackActive_BARO"] = attack_mask.astype(int)
    df[f"{alt_col}_attacked"] = df[alt_col].copy()

    df.loc[attack_mask, f"{alt_col}_attacked"] += BARO_ALT_OFFSET

    output_csv = OUTPUT_DIR / csv_path.name.replace(".csv", "_ATTACKED_BARO.csv")
    df.to_csv(output_csv, index=False)

    plot_alt = FIGURE_DIR / csv_path.name.replace(".csv", "_BARO_ALT_attack.png")

    save_plot(
        df["TimeSec"],
        df[alt_col],
        df[f"{alt_col}_attacked"],
        f"Barometer Altitude Attack — {csv_path.stem}",
        "Altitude",
        plot_alt
    )

    return {
        "input_file": csv_path.name,
        "output_file": output_csv.name,
        "sensor": "BARO",
        "attacked_signals": alt_col,
        "attack_type": "constant altitude offset",
        "attack_start_sec": ATTACK_START_SEC,
        "attack_end_sec": ATTACK_END_SEC
    }


def attack_imu_file(csv_path):
    """
    Inject gyroscope attack:
    - Z-axis angular velocity offset
    """
    df = pd.read_csv(csv_path)
    df, _ = add_time_seconds(df)
    attack_mask = build_attack_mask(df)

    gyrz_col = find_column(df, ["GyrZ", "GyroZ", "GYRZ"])

    if gyrz_col is None:
        print(f"[SKIP IMU] Gyroscope Z column missing: {csv_path.name}")
        print(f"Available columns: {list(df.columns)}")
        return None

    df["AttackActive_IMU"] = attack_mask.astype(int)
    df[f"{gyrz_col}_attacked"] = df[gyrz_col].copy()

    df.loc[attack_mask, f"{gyrz_col}_attacked"] += GYRO_Z_OFFSET

    output_csv = OUTPUT_DIR / csv_path.name.replace(".csv", "_ATTACKED_IMU.csv")
    df.to_csv(output_csv, index=False)

    plot_gyrz = FIGURE_DIR / csv_path.name.replace(".csv", "_IMU_GYRZ_attack.png")

    save_plot(
        df["TimeSec"],
        df[gyrz_col],
        df[f"{gyrz_col}_attacked"],
        f"Gyroscope Z Attack — {csv_path.stem}",
        "Angular velocity",
        plot_gyrz
    )

    return {
        "input_file": csv_path.name,
        "output_file": output_csv.name,
        "sensor": "IMU",
        "attacked_signals": gyrz_col,
        "attack_type": "constant gyroscope-Z bias",
        "attack_start_sec": ATTACK_START_SEC,
        "attack_end_sec": ATTACK_END_SEC
    }


# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
def main():
    manifest_rows = []

    print("\n========== PHASE 5: SENSOR ATTACK INJECTION ==========\n")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figure directory: {FIGURE_DIR}")
    print(f"Attack window:    {ATTACK_START_SEC}s to {ATTACK_END_SEC}s\n")

    # GPS files
    gps_files = sorted(INPUT_DIR.glob("*_GPS.csv"))
    for csv_file in gps_files:
        result = attack_gps_file(csv_file)
        if result is not None:
            manifest_rows.append(result)
            print(f"[OK] GPS attack injected: {csv_file.name}")

    # BARO files
    baro_files = sorted(INPUT_DIR.glob("*_BARO.csv"))
    for csv_file in baro_files:
        result = attack_baro_file(csv_file)
        if result is not None:
            manifest_rows.append(result)
            print(f"[OK] BARO attack injected: {csv_file.name}")

    # IMU files
    imu_files = sorted(INPUT_DIR.glob("*_IMU.csv"))
    for csv_file in imu_files:
        result = attack_imu_file(csv_file)
        if result is not None:
            manifest_rows.append(result)
            print(f"[OK] IMU attack injected: {csv_file.name}")

    # Save manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = OUTPUT_DIR / "phase5_attack_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print("\n========== PHASE 5 COMPLETE ==========")
    print(f"Attacked CSV files saved in: {OUTPUT_DIR}")
    print(f"Validation figures saved in: {FIGURE_DIR}")
    print(f"Manifest file saved as: {manifest_path}")
    print(f"Total attack datasets created: {len(manifest_rows)}\n")


if __name__ == "__main__":
    main()