import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
csv_file = os.path.expanduser(
    "~/sensor_recovery_project/logs/extracted_csv/baseline_01_hover_IMU.csv"
)

output_file = os.path.expanduser(
    "~/sensor_recovery_project/figures/baseline_01_hover_imu.png"
)

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
df = pd.read_csv(csv_file)

# ---------------------------------------------------------
# Use primary IMU instance I = 0
# ---------------------------------------------------------
df = df[df["I"] == 0].copy()

# ---------------------------------------------------------
# Convert TimeUS to relative seconds
# ---------------------------------------------------------
df["Time_s"] = (df["TimeUS"] - df["TimeUS"].iloc[0]) / 1_000_000

# ---------------------------------------------------------
# Plot gyroscope and accelerometer signals
# ---------------------------------------------------------
plt.figure(figsize=(12, 9))

# Gyroscope
plt.subplot(2, 1, 1)
plt.plot(df["Time_s"], df["GyrX"], label="GyrX")
plt.plot(df["Time_s"], df["GyrY"], label="GyrY")
plt.plot(df["Time_s"], df["GyrZ"], label="GyrZ")
plt.title("Baseline 01 Hover — IMU Signals")
plt.ylabel("Angular Rate (rad/s)")
plt.grid(True)
plt.legend()

# Accelerometer
plt.subplot(2, 1, 2)
plt.plot(df["Time_s"], df["AccX"], label="AccX")
plt.plot(df["Time_s"], df["AccY"], label="AccY")
plt.plot(df["Time_s"], df["AccZ"], label="AccZ")
plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.grid(True)
plt.legend()

# ---------------------------------------------------------
# Save and show
# ---------------------------------------------------------
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.show()

print(f"Plot saved to: {output_file}")