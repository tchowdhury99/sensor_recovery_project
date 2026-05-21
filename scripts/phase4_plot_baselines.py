from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
BASE_DIR = Path.home() / "sensor_recovery_project"
CSV_DIR = BASE_DIR / "logs" / "extracted_csv"
FIG_DIR = BASE_DIR / "figures" / "phase4_baseline_validation"

FIG_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Baseline flight names
# ------------------------------------------------------------
BASELINES = [
    "baseline_01_hover",
    "baseline_02_high_altitude_hover",
    "baseline_03_forward_motion",
    "baseline_04_yaw_rotation",
    "baseline_05_mixed_maneuver",
]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def load_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]
    return df


def make_time_seconds(df):
    """
    Convert TimeUS to relative seconds.
    """
    if "TimeUS" in df.columns:
        return (df["TimeUS"] - df["TimeUS"].iloc[0]) / 1_000_000.0
    else:
        return range(len(df))


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300)
    plt.close()


# ------------------------------------------------------------
# ATT plots: Roll, Pitch, Yaw
# ------------------------------------------------------------
def plot_attitude(baseline):
    file_path = CSV_DIR / f"{baseline}_ATT.csv"

    if not file_path.exists():
        print(f"[SKIP] Missing file: {file_path}")
        return

    df = load_csv(file_path)
    t = make_time_seconds(df)

    # Roll
    if "Roll" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(t, df["Roll"], label="Actual Roll")
        if "DesRoll" in df.columns:
            plt.plot(t, df["DesRoll"], label="Desired Roll")
        plt.xlabel("Time (s)")
        plt.ylabel("Roll Angle")
        plt.title(f"{baseline}: Roll Response")
        plt.legend()
        plt.grid(True)
        save_plot(f"{baseline}_ATT_roll.png")

    # Pitch
    if "Pitch" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(t, df["Pitch"], label="Actual Pitch")
        if "DesPitch" in df.columns:
            plt.plot(t, df["DesPitch"], label="Desired Pitch")
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch Angle")
        plt.title(f"{baseline}: Pitch Response")
        plt.legend()
        plt.grid(True)
        save_plot(f"{baseline}_ATT_pitch.png")

    # Yaw
    if "Yaw" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(t, df["Yaw"], label="Actual Yaw")
        if "DesYaw" in df.columns:
            plt.plot(t, df["DesYaw"], label="Desired Yaw")
        plt.xlabel("Time (s)")
        plt.ylabel("Yaw Angle")
        plt.title(f"{baseline}: Yaw Response")
        plt.legend()
        plt.grid(True)
        save_plot(f"{baseline}_ATT_yaw.png")

    print(f"[DONE] ATT plots: {baseline}")


# ------------------------------------------------------------
# BARO altitude plot
# ------------------------------------------------------------
def plot_baro(baseline):
    file_path = CSV_DIR / f"{baseline}_BARO.csv"

    if not file_path.exists():
        print(f"[SKIP] Missing file: {file_path}")
        return

    df = load_csv(file_path)
    t = make_time_seconds(df)

    altitude_column = None

    for candidate in ["Alt", "AltAMSL", "Altitude"]:
        if candidate in df.columns:
            altitude_column = candidate
            break

    if altitude_column:
        plt.figure(figsize=(10, 5))
        plt.plot(t, df[altitude_column], label=altitude_column)
        plt.xlabel("Time (s)")
        plt.ylabel("Altitude")
        plt.title(f"{baseline}: Barometer Altitude")
        plt.legend()
        plt.grid(True)
        save_plot(f"{baseline}_BARO_altitude.png")
        print(f"[DONE] BARO plot: {baseline}")
    else:
        print(f"[WARN] No altitude column found in {baseline}_BARO.csv")
        print("Available columns:", list(df.columns))


# ------------------------------------------------------------
# IMU gyroscope and accelerometer plots
# ------------------------------------------------------------
def plot_imu(baseline):
    file_path = CSV_DIR / f"{baseline}_IMU.csv"

    if not file_path.exists():
        print(f"[SKIP] Missing file: {file_path}")
        return

    df = load_csv(file_path)
    t = make_time_seconds(df)

    gyro_cols = [col for col in ["GyrX", "GyrY", "GyrZ"] if col in df.columns]
    acc_cols = [col for col in ["AccX", "AccY", "AccZ"] if col in df.columns]

    if gyro_cols:
        plt.figure(figsize=(10, 5))
        for col in gyro_cols:
            plt.plot(t, df[col], label=col)
        plt.xlabel("Time (s)")
        plt.ylabel("Gyroscope Reading")
        plt.title(f"{baseline}: IMU Gyroscope Signals")
        plt.legend()
        plt.grid(True)
        save_plot(f"{baseline}_IMU_gyro.png")

    if acc_cols:
        plt.figure(figsize=(10, 5))
        for col in acc_cols:
            plt.plot(t, df[col], label=col)
        plt.xlabel("Time (s)")
        plt.ylabel("Accelerometer Reading")
        plt.title(f"{baseline}: IMU Accelerometer Signals")
        plt.legend()
        plt.grid(True)
        save_plot(f"{baseline}_IMU_accel.png")

    print(f"[DONE] IMU plots: {baseline}")


# ------------------------------------------------------------
# GPS path plot
# ------------------------------------------------------------
def plot_gps(baseline):
    file_path = CSV_DIR / f"{baseline}_GPS.csv"

    if not file_path.exists():
        print(f"[SKIP] Missing file: {file_path}")
        return

    df = load_csv(file_path)

    lat_col = None
    lon_col = None

    for candidate in ["Lat", "Latitude"]:
        if candidate in df.columns:
            lat_col = candidate
            break

    for candidate in ["Lng", "Lon", "Longitude"]:
        if candidate in df.columns:
            lon_col = candidate
            break

    if lat_col and lon_col:
        plt.figure(figsize=(7, 6))
        plt.plot(df[lon_col], df[lat_col], marker=".", linewidth=1)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"{baseline}: GPS Trajectory")
        plt.grid(True)
        save_plot(f"{baseline}_GPS_trajectory.png")
        print(f"[DONE] GPS plot: {baseline}")
    else:
        print(f"[WARN] GPS latitude/longitude columns not found in {baseline}_GPS.csv")
        print("Available columns:", list(df.columns))


# ------------------------------------------------------------
# XKF1 selected state plots
# ------------------------------------------------------------
def plot_xkf1(baseline):
    file_path = CSV_DIR / f"{baseline}_XKF1.csv"

    if not file_path.exists():
        print(f"[SKIP] Missing file: {file_path}")
        return

    df = load_csv(file_path)
    t = make_time_seconds(df)

    candidate_groups = {
        "velocity": ["VN", "VE", "VD"],
        "position": ["PN", "PE", "PD"],
    }

    for group_name, columns in candidate_groups.items():
        available = [col for col in columns if col in df.columns]

        if available:
            plt.figure(figsize=(10, 5))
            for col in available:
                plt.plot(t, df[col], label=col)
            plt.xlabel("Time (s)")
            plt.ylabel(group_name.capitalize())
            plt.title(f"{baseline}: XKF1 {group_name.capitalize()} States")
            plt.legend()
            plt.grid(True)
            save_plot(f"{baseline}_XKF1_{group_name}.png")

    print(f"[DONE] XKF1 plots: {baseline}")


# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
def main():
    print("============================================")
    print("Phase 4: Baseline Visualization Starting")
    print("============================================")

    for baseline in BASELINES:
        print(f"\nProcessing {baseline}...")

        plot_attitude(baseline)
        plot_baro(baseline)
        plot_imu(baseline)
        plot_gps(baseline)
        plot_xkf1(baseline)

    print("\n============================================")
    print("Phase 4 plotting complete.")
    print(f"Figures saved to: {FIG_DIR}")
    print("============================================")


if __name__ == "__main__":
    main()