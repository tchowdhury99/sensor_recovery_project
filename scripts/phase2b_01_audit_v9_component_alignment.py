import os
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path("/home/tchowdh4/sensor_recovery_project")

DETECTION_DIR = BASE / "attack_detection_v9_corrected" / "detection_csvs"
PREDICTION_DIR = BASE / "prediction_results_v9_trained_hybrid"

OUT_DIR = BASE / "phase2b_v9_component_audit"
REPORT_DIR = OUT_DIR / "reports"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

AXES = ["Roll", "Pitch", "Yaw"]

REQUIRED_PREDICTION_COLUMNS = {
    axis: [
        f"{axis}_true_future",
        f"{axis}_hybrid_prediction",
        f"{axis}_arx_component",
        f"{axis}_naive_component",
        f"{axis}_constant_velocity_component",
        f"{axis}_hybrid_error",
        f"{axis}_arx_error",
        f"{axis}_naive_error",
        f"{axis}_cv_error",
    ]
    for axis in AXES
}

# Detection CSV column candidates. The exact names may vary across your scripts.
DETECTION_CLEAN_FUTURE_CANDIDATES = {
    axis: [
        f"CleanFuture_{axis}",
        f"{axis}_CleanFuture",
        f"{axis}_clean_future",
        f"clean_future_{axis}",
        f"{axis}_true_future",
    ]
    for axis in AXES
}

DETECTION_V9_PRED_CANDIDATES = {
    axis: [
        f"V9PredictedFuture_{axis}",
        f"{axis}_V9PredictedFuture",
        f"V9Prediction_{axis}",
        f"{axis}_V9Prediction",
        f"{axis}_hybrid_prediction",
        f"PredictedFuture_{axis}",
    ]
    for axis in AXES
}

PREDICTION_INDEX_CANDIDATES = [
    "PredictionRowIndex",
    "PredictionIndex",
    "PredictionFileRow",
    "PredRowIndex",
    "PredIndex",
    "V9PredictionIndex",
    "V9PredictionRowIndex",
    "row_index",
    "RowIndex",
]


def safe_float_array(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def rmse(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    return float(np.sqrt(np.mean(values ** 2)))


def mae(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    return float(np.mean(np.abs(values)))


def max_abs(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    return float(np.max(np.abs(values)))


def find_first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def resolve_prediction_file_path(raw_path):
    """
    PredictionFile may be:
      1. absolute path
      2. relative path from project base
      3. only a filename
    """
    if pd.isna(raw_path):
        return None

    raw = str(raw_path).strip()

    if raw == "":
        return None

    p = Path(raw)

    if p.is_absolute() and p.exists():
        return p

    # Relative to project base.
    p_base = BASE / raw
    if p_base.exists():
        return p_base

    # Relative to V9 prediction-output folder.
    p_pred = PREDICTION_DIR / raw
    if p_pred.exists():
        return p_pred

    # Filename only search inside V9 prediction-output folder.
    p_name = PREDICTION_DIR / Path(raw).name
    if p_name.exists():
        return p_name

    return p


def get_prediction_indices(det_df, pred_df):
    """
    If a detection CSV has a prediction row index column, use it.
    Otherwise assume row-by-row alignment from the beginning.
    """
    index_col = find_first_existing_column(det_df, PREDICTION_INDEX_CANDIDATES)

    if index_col is not None:
        raw_idx = pd.to_numeric(det_df[index_col], errors="coerce")
        valid_mask = raw_idx.notna()
        idx = raw_idx[valid_mask].astype(int).to_numpy()

        # Keep only indices that are valid for prediction dataframe.
        valid_idx_mask = (idx >= 0) & (idx < len(pred_df))

        det_positions = np.where(valid_mask.to_numpy())[0][valid_idx_mask]
        pred_positions = idx[valid_idx_mask]

        return {
            "alignment_method": f"index_column:{index_col}",
            "det_positions": det_positions,
            "pred_positions": pred_positions,
            "index_column": index_col,
        }

    n = min(len(det_df), len(pred_df))
    return {
        "alignment_method": "row_order_min_length",
        "det_positions": np.arange(n),
        "pred_positions": np.arange(n),
        "index_column": None,
    }


def plot_mismatch(det_file_stem, axis, t, clean_mismatch, pred_mismatch, out_path):
    plt.figure(figsize=(14, 5))
    plt.plot(t, clean_mismatch, label=f"{axis}: CleanFuture - Prediction true_future", linewidth=1.2)
    plt.plot(t, pred_mismatch, label=f"{axis}: V9PredictedFuture - Prediction hybrid", linewidth=1.2)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Aligned row index")
    plt.ylabel("Mismatch")
    plt.title(f"Phase 2B alignment mismatch: {det_file_stem} — {axis}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    print("\n========== PHASE 2B: V9 COMPONENT ALIGNMENT AUDIT ==========")
    print(f"Detection CSV folder: {DETECTION_DIR}")
    print(f"V9 prediction folder : {PREDICTION_DIR}")
    print(f"Output folder        : {OUT_DIR}")
    print("Read-only audit. No V9 retraining. No detection/recovery modification.\n")

    if not DETECTION_DIR.exists():
        raise FileNotFoundError(f"Detection folder not found: {DETECTION_DIR}")

    if not PREDICTION_DIR.exists():
        raise FileNotFoundError(f"V9 prediction folder not found: {PREDICTION_DIR}")

    detection_files = sorted(DETECTION_DIR.glob("*.csv"))

    if not detection_files:
        raise FileNotFoundError(f"No detection CSVs found in {DETECTION_DIR}")

    summary_rows = []
    inventory_rows = []
    component_rows = []

    for det_path in detection_files:
        print("\n------------------------------------------------------------")
        print(f"Detection CSV: {det_path.name}")

        try:
            det_df = pd.read_csv(det_path)
        except Exception as e:
            summary_rows.append({
                "detection_file": det_path.name,
                "axis": "ALL",
                "status": "FAIL",
                "reason": f"Could not read detection CSV: {e}",
            })
            print(f"[FAIL] Could not read detection CSV: {e}")
            continue

        if "PredictionFile" not in det_df.columns:
            summary_rows.append({
                "detection_file": det_path.name,
                "axis": "ALL",
                "status": "FAIL",
                "reason": "Missing PredictionFile column",
            })
            print("[FAIL] Missing PredictionFile column")
            continue

        prediction_file_values = (
            det_df["PredictionFile"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        prediction_file_values = prediction_file_values[prediction_file_values != ""]
        unique_prediction_files = sorted(prediction_file_values.unique())

        if len(unique_prediction_files) == 0:
            summary_rows.append({
                "detection_file": det_path.name,
                "axis": "ALL",
                "status": "FAIL",
                "reason": "PredictionFile column exists but contains no valid values",
            })
            print("[FAIL] PredictionFile column empty")
            continue

        print(f"Unique PredictionFile values: {len(unique_prediction_files)}")

        for raw_prediction_file in unique_prediction_files:
            pred_path = resolve_prediction_file_path(raw_prediction_file)

            inventory_row = {
                "detection_file": det_path.name,
                "raw_prediction_file": raw_prediction_file,
                "resolved_prediction_file": str(pred_path) if pred_path is not None else "",
                "prediction_file_exists": bool(pred_path is not None and pred_path.exists()),
            }

            inventory_rows.append(inventory_row)

            if pred_path is None or not pred_path.exists():
                print(f"[FAIL] PredictionFile not found: {raw_prediction_file}")
                summary_rows.append({
                    "detection_file": det_path.name,
                    "prediction_file": raw_prediction_file,
                    "axis": "ALL",
                    "status": "FAIL",
                    "reason": "PredictionFile path could not be resolved or does not exist",
                })
                continue

            try:
                pred_df = pd.read_csv(pred_path)
            except Exception as e:
                print(f"[FAIL] Could not read PredictionFile: {pred_path}")
                summary_rows.append({
                    "detection_file": det_path.name,
                    "prediction_file": str(pred_path),
                    "axis": "ALL",
                    "status": "FAIL",
                    "reason": f"Could not read PredictionFile: {e}",
                })
                continue

            # Subset detection rows that point to this prediction file.
            det_sub = det_df[
                det_df["PredictionFile"].astype(str).str.strip() == raw_prediction_file
            ].copy()

            print(f"PredictionFile: {pred_path.name}")
            print(f"Detection rows using this PredictionFile: {len(det_sub)}")
            print(f"Prediction rows available: {len(pred_df)}")

            align = get_prediction_indices(det_sub, pred_df)
            det_positions = align["det_positions"]
            pred_positions = align["pred_positions"]

            aligned_rows = len(det_positions)

            print(f"Alignment method: {align['alignment_method']}")
            print(f"Aligned rows: {aligned_rows}")

            # Component-column existence audit.
            for axis in AXES:
                missing_cols = [
                    col for col in REQUIRED_PREDICTION_COLUMNS[axis]
                    if col not in pred_df.columns
                ]

                component_rows.append({
                    "detection_file": det_path.name,
                    "prediction_file": str(pred_path),
                    "axis": axis,
                    "all_required_component_columns_exist": len(missing_cols) == 0,
                    "missing_component_columns": ",".join(missing_cols),
                    "available_prediction_columns_count": len(pred_df.columns),
                })

            if aligned_rows == 0:
                for axis in AXES:
                    summary_rows.append({
                        "detection_file": det_path.name,
                        "prediction_file": str(pred_path),
                        "axis": axis,
                        "status": "FAIL",
                        "reason": "No aligned rows between detection CSV and prediction file",
                        "alignment_method": align["alignment_method"],
                        "aligned_rows": 0,
                    })
                continue

            det_aligned = det_sub.iloc[det_positions].reset_index(drop=True)
            pred_aligned = pred_df.iloc[pred_positions].reset_index(drop=True)

            row_index = np.arange(aligned_rows)

            for axis in AXES:
                clean_col = find_first_existing_column(
                    det_aligned,
                    DETECTION_CLEAN_FUTURE_CANDIDATES[axis]
                )

                v9_pred_col = find_first_existing_column(
                    det_aligned,
                    DETECTION_V9_PRED_CANDIDATES[axis]
                )

                pred_true_col = f"{axis}_true_future"
                pred_hybrid_col = f"{axis}_hybrid_prediction"

                missing_reasons = []

                if clean_col is None:
                    missing_reasons.append("Missing detection CleanFuture column")

                if v9_pred_col is None:
                    missing_reasons.append("Missing detection V9PredictedFuture column")

                if pred_true_col not in pred_aligned.columns:
                    missing_reasons.append(f"Missing prediction column {pred_true_col}")

                if pred_hybrid_col not in pred_aligned.columns:
                    missing_reasons.append(f"Missing prediction column {pred_hybrid_col}")

                required_component_missing = [
                    col for col in REQUIRED_PREDICTION_COLUMNS[axis]
                    if col not in pred_aligned.columns
                ]

                if missing_reasons:
                    summary_rows.append({
                        "detection_file": det_path.name,
                        "prediction_file": str(pred_path),
                        "axis": axis,
                        "status": "FAIL",
                        "reason": "; ".join(missing_reasons),
                        "alignment_method": align["alignment_method"],
                        "aligned_rows": aligned_rows,
                        "detection_clean_future_col": clean_col,
                        "detection_v9_pred_col": v9_pred_col,
                        "prediction_true_future_col": pred_true_col,
                        "prediction_hybrid_col": pred_hybrid_col,
                        "missing_component_columns": ",".join(required_component_missing),
                    })
                    print(f"[FAIL] {axis}: {'; '.join(missing_reasons)}")
                    continue

                det_clean = safe_float_array(det_aligned[clean_col])
                det_v9 = safe_float_array(det_aligned[v9_pred_col])

                pred_true = safe_float_array(pred_aligned[pred_true_col])
                pred_hybrid = safe_float_array(pred_aligned[pred_hybrid_col])

                clean_mismatch = det_clean - pred_true
                pred_mismatch = det_v9 - pred_hybrid

                clean_max_abs = max_abs(clean_mismatch)
                clean_rmse = rmse(clean_mismatch)
                clean_mae = mae(clean_mismatch)

                pred_max_abs = max_abs(pred_mismatch)
                pred_rmse = rmse(pred_mismatch)
                pred_mae = mae(pred_mismatch)

                tolerance = 1e-9

                clean_pass = (
                    np.isfinite(clean_max_abs)
                    and clean_max_abs <= tolerance
                )

                pred_pass = (
                    np.isfinite(pred_max_abs)
                    and pred_max_abs <= tolerance
                )

                component_pass = len(required_component_missing) == 0

                overall_pass = clean_pass and pred_pass and component_pass

                status = "PASS" if overall_pass else "FAIL"

                reason_parts = []

                if not clean_pass:
                    reason_parts.append("CleanFuture mismatch exceeds tolerance")

                if not pred_pass:
                    reason_parts.append("V9PredictedFuture mismatch exceeds tolerance")

                if not component_pass:
                    reason_parts.append("Missing V9 component columns")

                if not reason_parts:
                    reason_parts.append("Aligned and component columns available")

                summary_rows.append({
                    "detection_file": det_path.name,
                    "prediction_file": str(pred_path),
                    "axis": axis,
                    "status": status,
                    "reason": "; ".join(reason_parts),
                    "alignment_method": align["alignment_method"],
                    "aligned_rows": aligned_rows,
                    "detection_rows_for_prediction_file": len(det_sub),
                    "prediction_rows_available": len(pred_df),
                    "detection_clean_future_col": clean_col,
                    "detection_v9_pred_col": v9_pred_col,
                    "prediction_true_future_col": pred_true_col,
                    "prediction_hybrid_col": pred_hybrid_col,
                    "clean_future_max_abs_mismatch": clean_max_abs,
                    "clean_future_rmse_mismatch": clean_rmse,
                    "clean_future_mae_mismatch": clean_mae,
                    "v9_prediction_max_abs_mismatch": pred_max_abs,
                    "v9_prediction_rmse_mismatch": pred_rmse,
                    "v9_prediction_mae_mismatch": pred_mae,
                    "component_columns_pass": component_pass,
                    "missing_component_columns": ",".join(required_component_missing),
                })

                print(
                    f"[{status}] {axis}: "
                    f"Clean max mismatch={clean_max_abs:.3e}, "
                    f"V9 max mismatch={pred_max_abs:.3e}, "
                    f"components={'OK' if component_pass else 'MISSING'}"
                )

                # Generate mismatch figure only when mismatch is nonzero or failed.
                if not overall_pass or clean_max_abs > 0 or pred_max_abs > 0:
                    fig_name = (
                        f"{det_path.stem}__{pred_path.stem}__{axis}_mismatch.png"
                    )
                    fig_path = FIG_DIR / fig_name
                    plot_mismatch(
                        det_file_stem=det_path.stem,
                        axis=axis,
                        t=row_index,
                        clean_mismatch=clean_mismatch,
                        pred_mismatch=pred_mismatch,
                        out_path=fig_path,
                    )

    summary_df = pd.DataFrame(summary_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    component_df = pd.DataFrame(component_rows)

    summary_path = OUT_DIR / "phase2b_v9_component_alignment_audit_summary.csv"
    inventory_path = OUT_DIR / "phase2b_prediction_file_inventory.csv"
    component_path = OUT_DIR / "phase2b_component_column_audit.csv"

    summary_df.to_csv(summary_path, index=False)
    inventory_df.to_csv(inventory_path, index=False)
    component_df.to_csv(component_path, index=False)

    # Build markdown interpretation report.
    total_rows = len(summary_df)
    pass_rows = int((summary_df.get("status", pd.Series(dtype=str)) == "PASS").sum()) if total_rows else 0
    fail_rows = int((summary_df.get("status", pd.Series(dtype=str)) == "FAIL").sum()) if total_rows else 0

    missing_prediction_files = 0
    if len(inventory_df) > 0 and "prediction_file_exists" in inventory_df.columns:
        missing_prediction_files = int((inventory_df["prediction_file_exists"] == False).sum())

    component_fail_rows = 0
    if len(component_df) > 0 and "all_required_component_columns_exist" in component_df.columns:
        component_fail_rows = int((component_df["all_required_component_columns_exist"] == False).sum())

    if fail_rows == 0 and missing_prediction_files == 0 and component_fail_rows == 0:
        final_decision = (
            "PASS: Phase 5B can safely use the V9 component columns, because all audited "
            "detection rows align with their PredictionFile outputs and all required component "
            "columns are present."
        )
    else:
        final_decision = (
            "FAIL or PARTIAL PASS: Phase 5B should not use V9 component columns blindly. "
            "Review the failed rows in the summary CSV, especially missing PredictionFile paths, "
            "missing component columns, or nonzero alignment mismatches."
        )

    report_lines = []
    report_lines.append("# Phase 2B V9 Component Alignment Audit Report\n")
    report_lines.append("## Purpose\n")
    report_lines.append(
        "This audit verifies that official Phase 4/Phase 5 detection CSVs correctly reference "
        "the V9 trained-hybrid prediction output files through the `PredictionFile` column. "
        "It also checks whether the V9 component columns are present and whether detection-file "
        "clean future / predicted future values match the corresponding V9 prediction-output columns.\n"
    )

    report_lines.append("## Inputs\n")
    report_lines.append(f"- Detection CSV folder: `{DETECTION_DIR}`\n")
    report_lines.append(f"- V9 prediction-output folder: `{PREDICTION_DIR}`\n")

    report_lines.append("## Outputs\n")
    report_lines.append(f"- Alignment summary: `{summary_path}`\n")
    report_lines.append(f"- Prediction file inventory: `{inventory_path}`\n")
    report_lines.append(f"- Component column audit: `{component_path}`\n")
    report_lines.append(f"- Figures folder: `{FIG_DIR}`\n")

    report_lines.append("## Summary\n")
    report_lines.append(f"- Total axis-level audit rows: `{total_rows}`\n")
    report_lines.append(f"- PASS rows: `{pass_rows}`\n")
    report_lines.append(f"- FAIL rows: `{fail_rows}`\n")
    report_lines.append(f"- Missing/unresolved PredictionFile entries: `{missing_prediction_files}`\n")
    report_lines.append(f"- Component-column failures: `{component_fail_rows}`\n")

    report_lines.append("## Final Decision\n")
    report_lines.append(final_decision + "\n")

    if fail_rows > 0 and len(summary_df) > 0:
        report_lines.append("## Failed Rows Preview\n")
        fail_preview = summary_df[summary_df["status"] == "FAIL"].head(20)
        report_lines.append(fail_preview.to_markdown(index=False))
        report_lines.append("\n")

    report_path = REPORT_DIR / "phase2b_v9_component_alignment_audit_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print("\n========== PHASE 2B AUDIT COMPLETE ==========")
    print(f"Summary CSV     : {summary_path}")
    print(f"Inventory CSV   : {inventory_path}")
    print(f"Component CSV   : {component_path}")
    print(f"Markdown report : {report_path}")
    print(f"Figures folder  : {FIG_DIR}")
    print("\nFinal decision:")
    print(final_decision)


if __name__ == "__main__":
    main()