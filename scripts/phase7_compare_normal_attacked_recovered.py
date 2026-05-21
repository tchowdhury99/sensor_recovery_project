import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PHASE 7:
# Compare Normal vs Attacked vs Recovered signals
# ============================================================

BASE_DIR = os.path.expanduser("~/sensor_recovery_project")

RECOVERED_DIR = os.path.join(BASE_DIR, "logs", "recovered_csv")
FIGURE_DIR = os.path.join(
    BASE_DIR,
    "figures",
    "phase7_normal_attacked_recovered"
)

os.makedirs(FIGURE_DIR, exist_ok=True)


# ------------------------------------------------------------
# Utility function: save figure
# ------------------------------------------------------------
def save_plot(filename):
    output_path = os.path.join(FIGURE_DIR, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] {output_path}")


# ============================================================
# 1. GPS COMPARISON
# ============================================================

gps_file = os.path.join(
    RECOVERED_DIR,
    "baseline_01_hover_GPS_ATTACKED_GPS_recovered.csv"
)

gps_df = pd.read_csv(gps_file)

print("\n========== GPS COLUMNS ==========")
print(gps_df.columns.tolist())

# ------------------------------------------------------------
# GPS Latitude: Normal vs Attacked vs Recovered
# ------------------------------------------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    gps_df["TimeUS"],
    gps_df["Lat"],
    label="Normal Latitude",
    linewidth=2
)

plt.plot(
    gps_df["TimeUS"],
    gps_df["Lat_attacked"],
    label="Attacked Latitude",
    linestyle="--",
    linewidth=2
)

plt.plot(
    gps_df["TimeUS"],
    gps_df["Recovered_Lat"],
    label="Recovered Latitude",
    linewidth=2
)

plt.title("GPS Latitude: Normal vs Attacked vs Recovered")
plt.xlabel("TimeUS")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)

save_plot("phase7_gps_latitude_normal_attacked_recovered.png")


# ------------------------------------------------------------
# GPS Longitude: Normal vs Attacked vs Recovered
# ------------------------------------------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    gps_df["TimeUS"],
    gps_df["Lng"],
    label="Normal Longitude",
    linewidth=2
)

plt.plot(
    gps_df["TimeUS"],
    gps_df["Lng_attacked"],
    label="Attacked Longitude",
    linestyle="--",
    linewidth=2
)

plt.plot(
    gps_df["TimeUS"],
    gps_df["Recovered_Lng"],
    label="Recovered Longitude",
    linewidth=2
)

plt.title("GPS Longitude: Normal vs Attacked vs Recovered")
plt.xlabel("TimeUS")
plt.ylabel("Longitude")
plt.legend()
plt.grid(True)

save_plot("phase7_gps_longitude_normal_attacked_recovered.png")


# ------------------------------------------------------------
# GPS 2D Trajectory: Normal vs Attacked vs Recovered
# ------------------------------------------------------------
plt.figure(figsize=(8, 8))

plt.plot(
    gps_df["Lng"],
    gps_df["Lat"],
    label="Normal Trajectory",
    linewidth=2
)

plt.plot(
    gps_df["Lng_attacked"],
    gps_df["Lat_attacked"],
    label="Attacked Trajectory",
    linestyle="--",
    linewidth=2
)

plt.plot(
    gps_df["Recovered_Lng"],
    gps_df["Recovered_Lat"],
    label="Recovered Trajectory",
    linewidth=2
)

plt.title("GPS Trajectory: Normal vs Attacked vs Recovered")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)

save_plot("phase7_gps_trajectory_normal_attacked_recovered.png")


# ============================================================
# 2. BAROMETER COMPARISON
# ============================================================

baro_file = os.path.join(
    RECOVERED_DIR,
    "baseline_01_hover_BARO_ATTACKED_BARO_recovered.csv"
)

baro_df = pd.read_csv(baro_file)

print("\n========== BARO COLUMNS ==========")
print(baro_df.columns.tolist())

# ------------------------------------------------------------
# BARO Altitude: Normal vs Attacked vs Recovered
# ------------------------------------------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    baro_df["TimeUS"],
    baro_df["Alt"],
    label="Normal Altitude",
    linewidth=2
)

plt.plot(
    baro_df["TimeUS"],
    baro_df["Alt_attacked"],
    label="Attacked Altitude",
    linestyle="--",
    linewidth=2
)

plt.plot(
    baro_df["TimeUS"],
    baro_df["Recovered_Alt"],
    label="Recovered Altitude",
    linewidth=2
)

plt.title("BARO Altitude: Normal vs Attacked vs Recovered")
plt.xlabel("TimeUS")
plt.ylabel("Altitude")
plt.legend()
plt.grid(True)

save_plot("phase7_baro_altitude_normal_attacked_recovered.png")


# ============================================================
# 3. IMU COMPARISON
# ============================================================

imu_file = os.path.join(
    RECOVERED_DIR,
    "baseline_01_hover_IMU_ATTACKED_IMU_recovered.csv"
)

imu_df = pd.read_csv(imu_file)

print("\n========== IMU COLUMNS ==========")
print(imu_df.columns.tolist())

# ------------------------------------------------------------
# IMU GyrZ: Normal vs Attacked vs Recovered
# ------------------------------------------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    imu_df["TimeUS"],
    imu_df["GyrZ"],
    label="Normal GyrZ",
    linewidth=2
)

plt.plot(
    imu_df["TimeUS"],
    imu_df["GyrZ_attacked"],
    label="Attacked GyrZ",
    linestyle="--",
    linewidth=2
)

plt.plot(
    imu_df["TimeUS"],
    imu_df["Recovered_GyrZ"],
    label="Recovered GyrZ",
    linewidth=2
)

plt.title("IMU GyrZ: Normal vs Attacked vs Recovered")
plt.xlabel("TimeUS")
plt.ylabel("Gyroscope Z-axis")
plt.legend()
plt.grid(True)

save_plot("phase7_imu_gyrz_normal_attacked_recovered.png")


# ============================================================
# Final completion message
# ============================================================

print("\n====================================================")
print("PHASE 7 COMPLETE")
print("Normal vs Attacked vs Recovered plots generated.")
print(f"Figures saved in: {FIGURE_DIR}")
print("====================================================")