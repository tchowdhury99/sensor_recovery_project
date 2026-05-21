import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
csv_file = os.path.expanduser(
    "~/sensor_recovery_project/logs/extracted_csv/baseline_01_hover_XKF1.csv"
)

output_file = os.path.expanduser(
    "~/sensor_recovery_project/figures/baseline_01_hover_xkf1.png"
)

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
df = pd.read_csv(csv_file)

# ---------------------------------------------------------
# Use EKF core C = 0 only
# ---------------------------------------------------------
df = df[df["C"] == 0].copy()

# ---------------------------------------------------------
# Convert TimeUS to relative seconds
# ---------------------------------------------------------
df["Time_s"] = (df["TimeUS"] - df["TimeUS"].iloc[0]) / 1_000_000

# ---------------------------------------------------------
# Convert Down-positive position into intuitive altitude
# PD is Down position, so altitude above origin is approximately -PD
# ---------------------------------------------------------
df["Estimated_Altitude"] = -df["PD"]

# ---------------------------------------------------------
# Plot EKF altitude, vertical velocity, and local horizontal position
# ---------------------------------------------------------
plt.figure(figsize=(12, 9))

# EKF local altitude
plt.subplot(3, 1, 1)
plt.plot(df["Time_s"], df["Estimated_Altitude"], label="EKF Estimated Altitude (-PD)")
plt.ylabel("Altitude (meters)")
plt.title("Baseline 01 Hover — EKF State Estimate")
plt.grid(True)
plt.legend()

# EKF vertical velocity
plt.subplot(3, 1, 2)
plt.plot(df["Time_s"], df["VD"], label="EKF Vertical Velocity VD")
plt.ylabel("VD (m/s)")
plt.grid(True)
plt.legend()

# EKF horizontal local position
plt.subplot(3, 1, 3)
plt.plot(df["Time_s"], df["PN"], label="North Position PN")
plt.plot(df["Time_s"], df["PE"], label="East Position PE")
plt.xlabel("Time (seconds)")
plt.ylabel("Position (meters)")
plt.grid(True)
plt.legend()

# ---------------------------------------------------------
# Save and show
# ---------------------------------------------------------
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.show()

print(f"Plot saved to: {output_file}")