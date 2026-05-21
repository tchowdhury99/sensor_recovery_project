import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
csv_file = os.path.expanduser(
    "~/sensor_recovery_project/logs/extracted_csv/baseline_01_hover_ATT.csv"
)

output_file = os.path.expanduser(
    "~/sensor_recovery_project/figures/baseline_01_hover_attitude.png"
)

# ---------------------------------------------------------
# Load the CSV file
# ---------------------------------------------------------
df = pd.read_csv(csv_file)

# ---------------------------------------------------------
# Convert TimeUS from microseconds to seconds
# Set the first timestamp to 0 so the x-axis is easier to read
# ---------------------------------------------------------
df["Time_s"] = (df["TimeUS"] - df["TimeUS"].iloc[0]) / 1_000_000

# ---------------------------------------------------------
# Plot Roll, Pitch, and Yaw
# ---------------------------------------------------------
plt.figure(figsize=(12, 8))

# Roll
plt.subplot(3, 1, 1)
plt.plot(df["Time_s"], df["Roll"], label="Actual Roll")
plt.plot(df["Time_s"], df["DesRoll"], label="Desired Roll")
plt.ylabel("Roll (degrees)")
plt.title("Baseline 01 Hover — Attitude Response")
plt.legend()
plt.grid(True)

# Pitch
plt.subplot(3, 1, 2)
plt.plot(df["Time_s"], df["Pitch"], label="Actual Pitch")
plt.plot(df["Time_s"], df["DesPitch"], label="Desired Pitch")
plt.ylabel("Pitch (degrees)")
plt.legend()
plt.grid(True)

# Yaw
plt.subplot(3, 1, 3)
plt.plot(df["Time_s"], df["Yaw"], label="Actual Yaw")
plt.plot(df["Time_s"], df["DesYaw"], label="Desired Yaw")
plt.xlabel("Time (seconds)")
plt.ylabel("Yaw (degrees)")
plt.legend()
plt.grid(True)

# ---------------------------------------------------------
# Save and show the plot
# ---------------------------------------------------------
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.show()

print(f"Plot saved to: {output_file}")