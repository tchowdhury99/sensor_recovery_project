import os
import glob
import pandas as pd

BASE = "/home/tchowdh4/sensor_recovery_project"
V9_DIR = f"{BASE}/prediction_results_v9_trained_hybrid"

print("\n========== PHASE 3: INSPECT V9 PREDICTION OUTPUTS ==========\n")
print("V9 prediction directory:")
print(V9_DIR)

csv_files = sorted(glob.glob(f"{V9_DIR}/**/*.csv", recursive=True))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found under {V9_DIR}")

print(f"\nFound {len(csv_files)} CSV files.\n")

for path in csv_files:
    name = os.path.basename(path)

    # Skip the summary first, but still show its columns
    print("=" * 100)
    print(f"FILE: {name}")
    print(f"PATH: {path}")

    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as e:
        print(f"Could not read file: {e}")
        continue

    print(f"Rows shown: {len(df)}")
    print(f"Columns ({len(df.columns)}):")
    for c in df.columns:
        print(f"  - {c}")

    print("\nFirst rows:")
    print(df.head().to_string(index=False))
    print()