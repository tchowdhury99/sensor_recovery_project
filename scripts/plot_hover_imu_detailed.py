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
    "~/sensor_recovery_project/figures/baseline_01_hover_imu_detailed.png"
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
# Create a clearer IMU figure
# ---------------------------------------------------------
plt.figure(figsize=(12, 12))

# Gyroscope signals
plt.subplot(3, 1, 1)
plt.plot(df["Time_s"], df["GyrX"], label="GyrX")
plt.plot(df["Time_s"], df["GyrY"], label="GyrY")
plt.plot(df["Time_s"], df["GyrZ"], label="GyrZ")
plt.title("Baseline 01 Hover — Detailed IMU Signals")
plt.ylabel("Angular Rate (rad/s)")
plt.grid(True)
plt.legend()

# Horizontal acceleration
plt.subplot(3, 1, 2)
plt.plot(df["Time_s"], df["AccX"], label="AccX")
plt.plot(df["Time_s"], df["AccY"], label="AccY")
plt.ylabel("Horizontal Acceleration (m/s²)")
plt.grid(True)
plt.legend()

# Vertical acceleration
plt.subplot(3, 1, 3)
plt.plot(df["Time_s"], df["AccZ"], label="AccZ")
plt.xlabel("Time (seconds)")
plt.ylabel("Vertical Acceleration (m/s²)")
plt.grid(True)
plt.legend()

# ---------------------------------------------------------
# Save and show
# ---------------------------------------------------------
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.show()

print(f"Plot saved to: {output_file}")