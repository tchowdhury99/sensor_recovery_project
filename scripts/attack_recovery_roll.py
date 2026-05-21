import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

BASE = "/home/tchowdh4/sensor_recovery_project"
CSV = f"{BASE}/csv"
FIG = f"{BASE}/figures"

os.makedirs(FIG, exist_ok=True)

# Load attitude data
att = pd.read_csv(f"{CSV}/baseline_ATT.csv")

# Convert TimeUS to relative seconds
att["t"] = (att["TimeUS"] - att["TimeUS"].iloc[0]) / 1_000_000

time = att["t"].to_numpy()
real_roll = att["Roll"].astype(float).to_numpy()

# ------------------------------------------------------------
# Step 1: Create artificial sensor attack
# Attack: add false roll bias between 25s and 45s
# ------------------------------------------------------------
attacked_roll = real_roll.copy()

attack_start = 25
attack_end = 45
attack_bias = 25.0

attack_mask = (time >= attack_start) & (time <= attack_end)
attacked_roll[attack_mask] = attacked_roll[attack_mask] + attack_bias

# ------------------------------------------------------------
# Step 2: Software sensor prediction
# Beginner version: predicted roll = previous real roll
# Later we can improve this using linear regression/state-space model
# ------------------------------------------------------------
predicted_roll = np.zeros_like(real_roll)
predicted_roll[0] = real_roll[0]

for i in range(1, len(real_roll)):
    predicted_roll[i] = real_roll[i - 1]

# ------------------------------------------------------------
# Step 3: Detection and recovery logic
# If attacked sensor differs too much from software sensor,
# replace attacked reading with software-sensor prediction.
# ------------------------------------------------------------
threshold = 5.0

residual = np.abs(attacked_roll - predicted_roll)
recovered_roll = attacked_roll.copy()
recovery_mode = np.zeros_like(real_roll)

for i in range(len(real_roll)):
    if residual[i] > threshold:
        recovered_roll[i] = predicted_roll[i]
        recovery_mode[i] = 1
    else:
        recovered_roll[i] = attacked_roll[i]
        recovery_mode[i] = 0

# ------------------------------------------------------------
# Step 4: Save result table
# ------------------------------------------------------------
result = pd.DataFrame({
    "time_sec": time,
    "real_roll": real_roll,
    "attacked_roll": attacked_roll,
    "software_sensor_roll": predicted_roll,
    "residual": residual,
    "recovered_roll": recovered_roll,
    "recovery_mode": recovery_mode
})

result.to_csv(f"{CSV}/roll_attack_recovery_result.csv", index=False)

# ------------------------------------------------------------
# Step 5: Plot real vs attacked vs software sensor vs recovered
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(time, real_roll, label="Original Roll")
plt.plot(time, attacked_roll, label="Attacked Roll")
plt.plot(time, predicted_roll, label="Software Sensor Prediction")
plt.plot(time, recovered_roll, label="Recovered Roll")
plt.xlabel("Time (s)")
plt.ylabel("Roll angle (degrees)")
plt.title("Roll Sensor Attack and Software-Sensor Recovery")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIG}/roll_attack_recovery.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Step 6: Plot residual
# ------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(time, residual, label="Residual = |Attacked - Software Sensor|")
plt.axhline(y=threshold, linestyle="--", label="Detection Threshold")
plt.xlabel("Time (s)")
plt.ylabel("Residual")
plt.title("Attack Detection Residual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIG}/roll_attack_residual.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Step 7: Plot recovery mode
# ------------------------------------------------------------
plt.figure(figsize=(12, 3))
plt.plot(time, recovery_mode, label="Recovery Mode")
plt.xlabel("Time (s)")
plt.ylabel("0 = OFF, 1 = ON")
plt.title("Recovery Switch Activation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIG}/roll_recovery_switch.png", dpi=300)
plt.show()

print("Finished attack-recovery experiment.")
print(f"CSV saved to: {CSV}/roll_attack_recovery_result.csv")
print(f"Figures saved to: {FIG}")