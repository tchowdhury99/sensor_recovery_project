import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
csv_file = os.path.expanduser(
    "~/sensor_recovery_project/logs/extracted_csv/baseline_01_hover_GPS.csv"
)

output_file = os.path.expanduser(
    "~/sensor_recovery_project/figures/baseline_01_hover_gps.png"
)

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
df = pd.read_csv(csv_file)

# ---------------------------------------------------------
# Use primary GPS instance I = 0
# ---------------------------------------------------------
df = df[df["I"] == 0].copy()

# ---------------------------------------------------------
# Convert TimeUS to relative seconds
# ---------------------------------------------------------
df["Time_s"] = (df["TimeUS"] - df["TimeUS"].iloc[0]) / 1_000_000

# ---------------------------------------------------------
# Convert GPS absolute altitude into relative altitude
# This makes it comparable to the BARO Alt plot
# ---------------------------------------------------------
df["Relative_GPS_Alt"] = df["Alt"] - df["Alt"].iloc[0]

# ---------------------------------------------------------
# Plot GPS altitude, speed, and vertical velocity
# ---------------------------------------------------------
plt.figure(figsize=(12, 9))

# GPS relative altitude
plt.subplot(3, 1, 1)
plt.plot(df["Time_s"], df["Relative_GPS_Alt"], label="GPS Relative Altitude")
plt.ylabel("Altitude (meters)")
plt.title("Baseline 01 Hover — GPS Behavior")
plt.grid(True)
plt.legend()

# GPS ground speed
plt.subplot(3, 1, 2)
plt.plot(df["Time_s"], df["Spd"], label="GPS Ground Speed")
plt.ylabel("Speed (m/s)")
plt.grid(True)
plt.legend()

# GPS vertical velocity
plt.subplot(3, 1, 3)
plt.plot(df["Time_s"], df["VZ"], label="GPS Vertical Velocity")
plt.xlabel("Time (seconds)")
plt.ylabel("VZ (m/s)")
plt.grid(True)
plt.legend()

# ---------------------------------------------------------
# Save and show
# ---------------------------------------------------------
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.show()

print(f"Plot saved to: {output_file}")