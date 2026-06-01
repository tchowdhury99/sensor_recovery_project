
#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("/home/tchowdh4/sensor_recovery_project")

CLEAN_DIR = PROJECT_ROOT / "logs" / "extracted_csv"
OUT_DIR = PROJECT_ROOT / "attack_detection_v9" / "attacked_att"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 3 primary thresholds: p99_abs
P99_THRESHOLDS = {
    "baseline_01_hover": {
        "Roll": 0.006865,
        "Pitch": 0.009161,
        "Yaw": 0.013596,
    },
    "baseline_02_high_altitude_hover": {
        "Roll": 0.006861,
        "Pitch": 0.007947,
        "Yaw": 0.010773,
    },
    "baseline_03_forward_motion": {
        "Roll": 0.006821,
        "Pitch": 0.016190,
        "Yaw": 0.003050,
    },
    "baseline_04_yaw_rotation": {
        "Yaw": 0.003797,
    },
    "baseline_05_mixed_maneuver": {
        "Roll": 0.010307,
        "Pitch": 0.009697,
        "Yaw": 0.004795,
    },
}

BASELINES = list(P99_THRESHOLDS.keys())
AXES = ["Roll", "Pitch", "Yaw"]


def infer_baseline_name(path: Path):
    name = path.name
    for baseline in BASELINES:
        if baseline in name:
            return baseline
    return None


def find_att_files():
    """
    Finds likely ATT CSV files under logs/extracted_csv.
    Expected filenames contain baseline name and ATT.
    """
    all_csvs = sorted(CLEAN_DIR.rglob("*.csv"))
    att_files = []

    for p in all_csvs:
        lower = p.name.lower()
        baseline = infer_baseline_name(p)
        if baseline is None:
            continue

        # Accept common ATT naming patterns.
        if "att" in lower:
            att_files.append(p)

    return att_files


def find_time_column(df: pd.DataFrame):
    candidates = ["TimeUS", "TimeS", "time_s", "timestamp", "Time", "t"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_attack_profile(n, attack_type, magnitude):
    """
    Returns:
      attack_vector: numeric attack added to target axis
      attack_active: binary ground-truth label

    Attack window:
      middle 30% to 60% of the file
    """
    attack = np.zeros(n, dtype=float)
    active = np.zeros(n, dtype=int)

    start = int(0.30 * n)
    end = int(0.60 * n)
    if end <= start:
        raise ValueError("Attack window is empty; file is too short.")

    active[start:end] = 1

    if attack_type == "bias":
        attack[start:end] = magnitude

    elif attack_type == "ramp":
        attack[start:end] = np.linspace(0.0, magnitude, end - start)

    elif attack_type == "pulse":
        # Short pulse centered inside the attack window.
        width = max(5, int(0.05 * n))
        center = (start + end) // 2
        p0 = max(start, center - width // 2)
        p1 = min(end, center + width // 2)

        active[:] = 0
        active[p0:p1] = 1
        attack[p0:p1] = magnitude

    else:
        raise ValueError(f"Unknown attack_type: {attack_type}")

    return attack, active


def main():
    att_files = find_att_files()

    if not att_files:
        raise FileNotFoundError(
            f"No ATT CSV files found under {CLEAN_DIR}. "
            "Check your extracted CSV filenames."
        )

    print(f"[INFO] Found {len(att_files)} ATT candidate files.")

    manifest_rows = []

    for att_path in att_files:
        baseline = infer_baseline_name(att_path)
        if baseline is None:
            continue

        df = pd.read_csv(att_path)

        available_axes = [axis for axis in AXES if axis in df.columns]
        available_axes = [axis for axis in available_axes if axis in P99_THRESHOLDS[baseline]]

        if not available_axes:
            print(f"[WARN] No usable Roll/Pitch/Yaw columns in {att_path}")
            continue

        n = len(df)
        if n < 50:
            print(f"[WARN] Skipping very short file: {att_path}")
            continue

        time_col = find_time_column(df)

        for axis in available_axes:
            threshold = P99_THRESHOLDS[baseline][axis]

            # Attack magnitudes are threshold-relative.
            # Bias and ramp should be clearly detectable.
            # Pulse is stronger because it is short.
            scenarios = [
                ("bias", 4.0 * threshold),
                ("ramp", 6.0 * threshold),
                ("pulse", 8.0 * threshold),
            ]

            for attack_type, magnitude in scenarios:
                out = df.copy()

                # Preserve true clean signal.
                for a in AXES:
                    if a in out.columns:
                        out[f"True_{a}"] = out[a].astype(float)
                        out[f"Attacked_{a}"] = out[a].astype(float)

                attack_vec, active = make_attack_profile(n, attack_type, magnitude)

                out[f"Attacked_{axis}"] = out[f"True_{axis}"].astype(float) + attack_vec
                out["AttackActive"] = active
                out["AttackAxis"] = axis
                out["AttackType"] = attack_type
                out["AttackMagnitude"] = magnitude
                out["SourceCleanFile"] = str(att_path)

                if time_col is not None:
                    out["Phase4TimeColumn"] = time_col

                stem = att_path.stem
                out_name = f"{baseline}__{stem}__attack_{attack_type}_{axis}.csv"
                out_path = OUT_DIR / out_name
                out.to_csv(out_path, index=False)

                manifest_rows.append({
                    "baseline": baseline,
                    "source_clean_file": str(att_path),
                    "attacked_file": str(out_path),
                    "axis": axis,
                    "attack_type": attack_type,
                    "magnitude": magnitude,
                    "threshold_p99_abs": threshold,
                    "n_samples": n,
                })

                print(f"[OK] Wrote {out_path}")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUT_DIR / "phase4_attacked_att_manifest_v9.csv"
    manifest.to_csv(manifest_path, index=False)

    print()
    print(f"[DONE] Attack manifest written to:")
    print(f"       {manifest_path}")
    print(f"[DONE] Number of attacked files: {len(manifest)}")


if __name__ == "__main__":
    main()
