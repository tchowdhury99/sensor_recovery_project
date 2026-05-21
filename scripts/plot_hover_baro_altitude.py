import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
csv_file = os.path.expanduser(
    "~/sensor_recovery_project/logs/extracted_csv/baseline_01_hover_BARO.csv"
)

output_file = os.path.expanduser(
    "~/sensor_recovery_project/figures/baseline_01_hover_baro_altitude.png"
)

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
df = pd.read_csv(csv_file)

# ---------------------------------------------------------
# Use only primary barometer instance I = 0
# ---------------------------------------------------------
df = df[df["I"] == 0].copy()

# ---------------------------------------------------------
# Convert TimeUS to relative time in seconds
# ---------------------------------------------------------
df["Time_s"] = (df["TimeUS"] - df["TimeUS"].iloc[0]) / 1_000_000

# ---------------------------------------------------------
# Plot relative barometric altitude
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df["Time_s"], df["Alt"], label="BARO Relative Altitude")

plt.title("Baseline 01 Hover — Barometric Altitude")
plt.xlabel("Time (seconds)")
plt.ylabel("Altitude (meters)")
plt.grid(True)
plt.legend()

# ---------------------------------------------------------
# Save and show
# ---------------------------------------------------------
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.show()

print(f"Plot saved to: {output_file}")