import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/home/tchowdh4/sensor_recovery_project")

DETECTION_DIR = BASE / "attack_detection_v9_corrected" / "detection_csvs"
PREDICTION_DIR = BASE / "prediction_results_v9_trained_hybrid"

OUT_DIR = BASE / "phase2b_v9_component_audit"
REPORT_DIR = OUT_DIR / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

AXES = ["Roll", "Pitch", "Yaw"]

STRICT_TOL = 1e-9
CLEAN_WARNING_TOL = 1e-3


def infer_axis_from_filename(name):
    for axis in AXES:
        if f"_{axis}_" in name:
            return axis
    return None


def resolve_prediction_file(raw):
    raw = str(raw).strip()
    p = Path(raw)

    if p.is_absolute() and p.exists():
        return p

    p1 = BASE / raw
    if p1.exists():
        return p1

    p2 = PREDICTION_DIR / raw
    if p2.exists():
        return p2

    p3 = PREDICTION_DIR / Path(raw).name
    if p3.exists():
        return p3

    return None


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def arr(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def metrics(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]

    if len(diff) == 0:
        return {
            "max_abs": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
        }

    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
    }


def main():
    rows = []

    detection_files = sorted(DETECTION_DIR.glob("*.csv"))

    if not detection_files:
        raise FileNotFoundError(f"No CSV files found in {DETECTION_DIR}")

    for det_path in detection_files:
        det_name = det_path.name
        target_axis = infer_axis_from_filename(det_name)

        if target_axis is None:
            rows.append({
                "detection_file": det_name,
                "target_axis": "",
                "status": "FAIL",
                "reason": "Could not infer target axis from filename",
            })
            continue

        det = pd.read_csv(det_path)

        if "PredictionFile" not in det.columns:
            rows.append({
                "detection_file": det_name,
                "target_axis": target_axis,
                "status": "FAIL",
                "reason": "Missing PredictionFile column",
            })
            continue

        unique_prediction_files = (
            det["PredictionFile"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        if len(unique_prediction_files) != 1:
            rows.append({
                "detection_file": det_name,
                "target_axis": target_axis,
                "status": "FAIL",
                "reason": f"Expected exactly one PredictionFile, found {len(unique_prediction_files)}",
            })
            continue

        raw_pred = unique_prediction_files[0]
        pred_path = resolve_prediction_file(raw_pred)

        if pred_path is None:
            rows.append({
                "detection_file": det_name,
                "target_axis": target_axis,
                "prediction_file_raw": raw_pred,
                "status": "FAIL",
                "reason": "PredictionFile could not be resolved",
            })
            continue

        pred = pd.read_csv(pred_path)

        n = min(len(det), len(pred))
        det_a = det.iloc[:n].reset_index(drop=True)
        pred_a = pred.iloc[:n].reset_index(drop=True)

        axis = target_axis

        clean_col = first_existing(det_a, [
            f"CleanFuture_{axis}",
            f"{axis}_CleanFuture",
            f"{axis}_clean_future",
            f"clean_future_{axis}",
        ])

        v9_col = first_existing(det_a, [
            f"V9PredictedFuture_{axis}",
            f"{axis}_V9PredictedFuture",
            f"V9Prediction_{axis}",
            f"{axis}_V9Prediction",
            f"PredictedFuture_{axis}",
        ])

        pred_true_col = f"{axis}_true_future"
        pred_hybrid_col = f"{axis}_hybrid_prediction"

        component_cols = [
            f"{axis}_naive_component",
            f"{axis}_constant_velocity_component",
            f"{axis}_arx_component",
            f"{axis}_hybrid_prediction",
            f"{axis}_true_future",
        ]

        missing_components = [c for c in component_cols if c not in pred_a.columns]

        reasons = []

        if v9_col is None:
            reasons.append("Missing target-axis V9PredictedFuture column in detection CSV")

        if pred_hybrid_col not in pred_a.columns:
            reasons.append(f"Missing {pred_hybrid_col} in prediction file")

        if missing_components:
            reasons.append("Missing required component columns")

        v9_m = {
            "max_abs": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
        }

        clean_m = {
            "max_abs": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
        }

        if v9_col is not None and pred_hybrid_col in pred_a.columns:
            v9_diff = arr(det_a, v9_col) - arr(pred_a, pred_hybrid_col)
            v9_m = metrics(v9_diff)

        if clean_col is not None and pred_true_col in pred_a.columns:
            clean_diff = arr(det_a, clean_col) - arr(pred_a, pred_true_col)
            clean_m = metrics(clean_diff)

        v9_pass = (
            np.isfinite(v9_m["max_abs"])
            and v9_m["max_abs"] <= STRICT_TOL
        )

        component_pass = len(missing_components) == 0

        if not v9_pass:
            reasons.append("Target-axis V9PredictedFuture does not exactly match V9 hybrid_prediction")

        if clean_col is None:
            clean_status = "NOT_CHECKED_MISSING_CLEAN_COLUMN"
        elif not np.isfinite(clean_m["max_abs"]):
            clean_status = "NOT_CHECKED_NONFINITE"
        elif clean_m["max_abs"] <= STRICT_TOL:
            clean_status = "EXACT_PASS"
        elif clean_m["max_abs"] <= CLEAN_WARNING_TOL:
            clean_status = "SMALL_WARNING"
        else:
            clean_status = "WARNING_MISMATCH"

        if clean_status.startswith("WARNING"):
            reasons.append("CleanFuture differs from V9 true_future; this is warning for clean-reference provenance, not component alignment")

        if v9_pass and component_pass:
            status = "PASS_COMPONENT_ALIGNMENT"
        else:
            status = "FAIL_COMPONENT_ALIGNMENT"

        rows.append({
            "detection_file": det_name,
            "target_axis": target_axis,
            "prediction_file": str(pred_path),
            "rows_detection": len(det),
            "rows_prediction": len(pred),
            "aligned_rows_checked": n,
            "clean_future_column": clean_col if clean_col else "",
            "v9_prediction_column": v9_col if v9_col else "",
            "prediction_true_future_column": pred_true_col,
            "prediction_hybrid_column": pred_hybrid_col,
            "component_columns_pass": component_pass,
            "missing_component_columns": ",".join(missing_components),
            "v9_max_abs_mismatch": v9_m["max_abs"],
            "v9_rmse_mismatch": v9_m["rmse"],
            "v9_mae_mismatch": v9_m["mae"],
            "v9_alignment_pass": v9_pass,
            "clean_max_abs_mismatch": clean_m["max_abs"],
            "clean_rmse_mismatch": clean_m["rmse"],
            "clean_mae_mismatch": clean_m["mae"],
            "clean_future_status": clean_status,
            "status": status,
            "reason": "; ".join(reasons) if reasons else "Target-axis component alignment passed",
        })

    out = pd.DataFrame(rows)

    out_path = OUT_DIR / "phase2b_02_refined_target_axis_component_audit.csv"
    out.to_csv(out_path, index=False)

    total = len(out)
    passed = int((out["status"] == "PASS_COMPONENT_ALIGNMENT").sum())
    failed = total - passed

    clean_warning = int(out["clean_future_status"].astype(str).str.contains("WARNING", na=False).sum())

    report_path = REPORT_DIR / "phase2b_02_refined_target_axis_component_audit_report.md"

    lines = []
    lines.append("# Phase 2B Refined Target-Axis V9 Component Audit Report\n")
    lines.append("## Purpose\n")
    lines.append("This refined audit checks only the attacked/target axis inferred from each detection CSV filename. Missing non-target-axis columns are not treated as failures.\n")
    lines.append("## Decision Logic\n")
    lines.append("- `PASS_COMPONENT_ALIGNMENT` means the target-axis V9 predicted future column matches the V9 prediction file's hybrid prediction and the required component columns exist.\n")
    lines.append("- `CleanFuture` mismatch is reported separately as a provenance warning. It does not automatically block Phase 5B component use if V9 prediction alignment passes.\n")
    lines.append("## Summary\n")
    lines.append(f"- Total detection CSVs audited: `{total}`\n")
    lines.append(f"- Component-alignment pass: `{passed}`\n")
    lines.append(f"- Component-alignment fail: `{failed}`\n")
    lines.append(f"- CleanFuture warning rows: `{clean_warning}`\n")

    if failed == 0:
        lines.append("## Final Decision\n")
        lines.append("PASS: Phase 5B may safely use target-axis V9 component columns through the `PredictionFile` mapping. Use the target axis only for each detection CSV.\n")
    else:
        lines.append("## Final Decision\n")
        lines.append("PARTIAL/FAIL: Some target-axis component alignments failed. Review failed rows before using Phase 5B adaptive weighting.\n")

    fail_df = out[out["status"] != "PASS_COMPONENT_ALIGNMENT"]
    if len(fail_df) > 0:
        lines.append("## Failed Rows Preview\n")
        cols = [
            "detection_file",
            "target_axis",
            "status",
            "v9_max_abs_mismatch",
            "component_columns_pass",
            "missing_component_columns",
            "reason",
        ]
        lines.append(fail_df[cols].head(30).to_markdown(index=False))
        lines.append("\n")

    warning_df = out[out["clean_future_status"].astype(str).str.contains("WARNING", na=False)]
    if len(warning_df) > 0:
        lines.append("## CleanFuture Warning Preview\n")
        cols = [
            "detection_file",
            "target_axis",
            "clean_max_abs_mismatch",
            "clean_rmse_mismatch",
            "clean_future_status",
        ]
        lines.append(warning_df[cols].head(30).to_markdown(index=False))
        lines.append("\n")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print("\n========== PHASE 2B REFINED AUDIT COMPLETE ==========")
    print(f"Output CSV : {out_path}")
    print(f"Report     : {report_path}")
    print(f"Total      : {total}")
    print(f"Passed     : {passed}")
    print(f"Failed     : {failed}")
    print(f"CleanFuture warnings: {clean_warning}")

    if failed == 0:
        print("\nFINAL DECISION: PASS for Phase 5B component alignment.")
    else:
        print("\nFINAL DECISION: PARTIAL/FAIL. Review failed component-alignment rows.")


if __name__ == "__main__":
    main()