import os
import pandas as pd

BASE = os.path.expanduser("~/sensor_recovery_project")
CSV_DIR = os.path.join(BASE, "logs", "extracted_csv")
OUT_DIR = os.path.join(BASE, "phase1_audit")

os.makedirs(OUT_DIR, exist_ok=True)

print("\n========== PHASE 1: CSV AUDIT ==========")
print(f"CSV directory: {CSV_DIR}")

if not os.path.isdir(CSV_DIR):
    raise FileNotFoundError(f"CSV directory not found: {CSV_DIR}")

csv_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith(".csv")])

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {CSV_DIR}")

summary_rows = []

for fname in csv_files:
    path = os.path.join(CSV_DIR, fname)

    try:
        df = pd.read_csv(path, nrows=5)
        full_df = pd.read_csv(path)
    except Exception as e:
        print(f"\n[ERROR] Could not read {fname}: {e}")
        continue

    rows = len(full_df)
    cols = list(full_df.columns)

    has_timeus = "TimeUS" in cols
    has_timesec = "TimeSec" in cols

    file_type = "UNKNOWN"
    upper_name = fname.upper()

    if "_ATT" in upper_name or upper_name.endswith("ATT.CSV"):
        file_type = "ATT"
    elif "_IMU" in upper_name:
        file_type = "IMU"
    elif "_GPS" in upper_name:
        file_type = "GPS"
    elif "_BARO" in upper_name:
        file_type = "BARO"
    elif "_XKF" in upper_name:
        file_type = "XKF"

    att_required = ["Roll", "Pitch", "Yaw"]
    att_optional = ["DesRoll", "DesPitch", "DesYaw", "ErrRP", "ErrYaw"]

    has_att_required = all(c in cols for c in att_required)
    available_att_optional = [c for c in att_optional if c in cols]

    min_time = None
    max_time = None
    duration_sec = None

    if has_timeus and rows > 0:
        try:
            min_time = full_df["TimeUS"].min()
            max_time = full_df["TimeUS"].max()
            duration_sec = (max_time - min_time) / 1_000_000.0
        except Exception:
            pass
    elif has_timesec and rows > 0:
        try:
            min_time = full_df["TimeSec"].min()
            max_time = full_df["TimeSec"].max()
            duration_sec = max_time - min_time
        except Exception:
            pass

    summary_rows.append({
        "file": fname,
        "type_guess": file_type,
        "rows": rows,
        "columns_count": len(cols),
        "has_TimeUS": has_timeus,
        "has_TimeSec": has_timesec,
        "duration_sec": duration_sec,
        "has_ATT_required_RollPitchYaw": has_att_required,
        "ATT_optional_available": ",".join(available_att_optional),
        "columns": ",".join(cols)
    })

    print("\n----------------------------------------")
    print(f"File: {fname}")
    print(f"Type guess: {file_type}")
    print(f"Rows: {rows}")
    print(f"Columns: {len(cols)}")
    print(f"Duration sec: {duration_sec}")
    print(f"Columns:")
    print(cols)

    if file_type == "ATT" or has_att_required:
        print("ATT required columns available:", has_att_required)
        print("ATT optional columns available:", available_att_optional)

summary = pd.DataFrame(summary_rows)

summary_path = os.path.join(OUT_DIR, "csv_audit_summary.csv")
summary.to_csv(summary_path, index=False)

att_summary = summary[summary["has_ATT_required_RollPitchYaw"] == True]
att_path = os.path.join(OUT_DIR, "att_files_found.csv")
att_summary.to_csv(att_path, index=False)

print("\n========== AUDIT COMPLETE ==========")
print(f"Full audit saved to: {summary_path}")
print(f"ATT file list saved to: {att_path}")

print("\n========== ATT FILES FOUND ==========")
if len(att_summary) == 0:
    print("No files with Roll, Pitch, Yaw found.")
else:
    print(att_summary[["file", "rows", "duration_sec", "ATT_optional_available"]].to_string(index=False))
