import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = "/home/tchowdh4/sensor_recovery_project"
CSV = f"{BASE}/csv"
FIG = f"{BASE}/figures"

os.makedirs(FIG, exist_ok=True)

# Load CSV files
att = pd.read_csv(f"{CSV}/baseline_ATT.csv")
imu = pd.read_csv(f"{CSV}/baseline_IMU.csv")
gps = pd.read_csv(f"{CSV}/baseline_GPS.csv")
baro = pd.read_csv(f"{CSV}/baseline_BARO.csv")

# Convert TimeUS to relative seconds
att["t"] = (att["TimeUS"] - att["TimeUS"].iloc[0]) / 1_000_000
imu["t"] = (imu["TimeUS"] - imu["TimeUS"].iloc[0]) / 1_000_000
gps["t"] = (gps["TimeUS"] - gps["TimeUS"].iloc[0]) / 1_000_000
baro["t"] = (baro["TimeUS"] - baro["TimeUS"].iloc[0]) / 1_000_000

# 1. Roll, Pitch, Yaw
plt.figure(figsize=(12, 6))
plt.plot(att["t"], att["Roll"], label="Roll")
plt.plot(att["t"], att["Pitch"], label="Pitch")
plt.plot(att["t"], att["Yaw"], label="Yaw")
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title("Baseline SITL Attitude: Roll, Pitch, Yaw")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIG}/baseline_attitude.png", dpi=300)
plt.show()

# 2. Desired vs actual roll
plt.figure(figsize=(12, 6))
plt.plot(att["t"], att["DesRoll"], label="Desired Roll")
plt.plot(att["t"], att["Roll"], label="Actual Roll")
plt.xlabel("Time (s)")
plt.ylabel("Roll angle (degrees)")
plt.title("Baseline SITL: Desired Roll vs Actual Roll")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIG}/baseline_desired_vs_actual_roll.png", dpi=300)
plt.show()

# 3. Gyroscope
plt.figure(figsize=(12, 6))
plt.plot(imu["t"], imu["GyrX"], label="GyrX")
plt.plot(imu["t"], imu["GyrY"], label="GyrY")
plt.plot(imu["t"], imu["GyrZ"], label="GyrZ")
plt.xlabel("Time (s)")
plt.ylabel("Angular velocity")
plt.title("Baseline SITL IMU Gyroscope")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIG}/baseline_gyro.png", dpi=300)
plt.show()

# 4. Accelerometer
plt.figure(figsize=(12, 6))
plt.plot(imu["t"], imu["AccX"], label="AccX")
plt.plot(imu["t"], imu["AccY"], label="AccY")
plt.plot(imu["t"], imu["AccZ"], label="AccZ")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")
plt.title("Baseline SITL IMU Accelerometer")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIG}/baseline_accel.png", dpi=300)
plt.show()

# 5. Barometer altitude
plt.figure(figsize=(12, 6))
plt.plot(baro["t"], baro["Alt"], label="BARO Altitude")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.title("Baseline SITL Barometer Altitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIG}/baseline_baro_altitude.png", dpi=300)
plt.show()

# 6. GPS altitude
plt.figure(figsize=(12, 6))
plt.plot(gps["t"], gps["Alt"], label="GPS Altitude")
plt.xlabel("Time (s)")
plt.ylabel("Altitude AMSL (m)")
plt.title("Baseline SITL GPS Altitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIG}/baseline_gps_altitude.png", dpi=300)
plt.show()

print("Finished. Figures saved in:")
print(FIG)