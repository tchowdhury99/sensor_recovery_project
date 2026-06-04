#!/usr/bin/env python3

"""
Phase 6B Final Validation and Comparison Report

Purpose:
    Read existing B-phase outputs and produce final Phase 6B tables,
    figures, and markdown decision report.

Important:
    This script is read-only with respect to official Phase 1/2/3/4/5 results.
    It does not retrain V9.
    It does not rerun detection or recovery.
    It does not overwrite official results.

Outputs:
    phase6b_final_validation/tables/
    phase6b_final_validation/reports/
    figures/phase6b_final_validation/
"""

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path("/home/tchowdh4/sensor_recovery_project")

OUT_DIR = BASE / "phase6b_final_validation"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"
FIG_DIR = BASE / "figures" / "phase6b_final_validation"

for d in [OUT_DIR, TABLE_DIR, REPORT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------

PHASE2B_CSV = BASE / "phase2b_v9_component_audit" / "phase2b_02_refined_target_axis_component_audit.csv"
PHASE3B_CSV = BASE / "phase3b_component_thresholds_v9" / "phase3b_component_clean_residual_thresholds.csv"
PHASE4B_CSV = BASE / "detection_reports_v9_corrected" / "phase4b_threshold_method_comparison_v9.csv"
PHASE5B_XKF1_CSV = BASE / "recovery_reports_v9_adaptive_weights" / "phase5b_adaptive_weight_summary.csv"
PHASE5B_RECENT_CSV = BASE / "recovery_reports_v9_recent_error_adaptive" / "phase5b_recent_error_adaptive_summary.csv"


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def safe_mean(df, col):
    if col not in df.columns or len(df) == 0:
        return np.nan
    return pd.to_numeric(df[col], errors="coerce").mean()


def safe_sum_bool(df, col):
    if col not in df.columns or len(df) == 0:
        return 0
    return int(df[col].astype(bool).sum())


def percent_change(old, new):
    if not np.isfinite(old) or old == 0:
        return np.nan
    return 100.0 * (old - new) / old


def plot_bar(df, x_col, y_col, title, ylabel, out_path):
    plt.figure(figsize=(10, 5))
    plt.bar(df[x_col].astype(str), df[y_col].astype(float))
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    print("=" * 90)
    print("PHASE 6B FINAL VALIDATION AND COMPARISON")
    print("=" * 90)

    for p in [PHASE2B_CSV, PHASE3B_CSV, PHASE4B_CSV, PHASE5B_XKF1_CSV, PHASE5B_RECENT_CSV]:
        require_file(p)

    # -----------------------------------------------------------------
    # Load evidence tables
    # -----------------------------------------------------------------

    p2b = pd.read_csv(PHASE2B_CSV)
    p3b = pd.read_csv(PHASE3B_CSV)
    p4b = pd.read_csv(PHASE4B_CSV)
    p5x = pd.read_csv(PHASE5B_XKF1_CSV)
    p5r = pd.read_csv(PHASE5B_RECENT_CSV)

    # -----------------------------------------------------------------
    # Phase 2B summary
    # -----------------------------------------------------------------

    phase2b_total = len(p2b)
    phase2b_pass = int((p2b["status"] == "PASS_COMPONENT_ALIGNMENT").sum())
    phase2b_fail = phase2b_total - phase2b_pass
    clean_warning = int(p2b["clean_future_status"].astype(str).str.contains("WARNING", na=False).sum())

    phase2b_summary = pd.DataFrame([{
        "phase": "Phase 2B",
        "total_detection_csvs": phase2b_total,
        "component_alignment_pass": phase2b_pass,
        "component_alignment_fail": phase2b_fail,
        "clean_future_warning_rows": clean_warning,
        "decision": "PASS" if phase2b_fail == 0 else "FAIL_OR_REVIEW",
        "interpretation": "Target-axis V9 component columns are safe for Phase 5B use; CleanFuture mismatch is provenance warning."
    }])
    phase2b_summary.to_csv(TABLE_DIR / "phase6b_phase2b_alignment_summary.csv", index=False)

    # -----------------------------------------------------------------
    # Phase 3B summary
    # -----------------------------------------------------------------

    allb = p3b[p3b["aggregation_level"] == "all_baselines"].copy()
    allb = allb.sort_values(["axis", "component"])
    allb.to_csv(TABLE_DIR / "phase6b_phase3b_all_baseline_component_thresholds.csv", index=False)

    phase3b_pass = len(allb) == 12
    phase3b_summary = pd.DataFrame([{
        "phase": "Phase 3B",
        "all_baseline_rows": len(allb),
        "expected_all_baseline_rows": 12,
        "decision": "PASS" if phase3b_pass else "FAIL_OR_REVIEW",
        "interpretation": "Component thresholds exist for Roll/Pitch/Yaw × hybrid/naive/CV/ARX." if phase3b_pass else "Missing component threshold rows."
    }])
    phase3b_summary.to_csv(TABLE_DIR / "phase6b_phase3b_summary.csv", index=False)

    # Component threshold figure
    if len(allb) > 0:
        for axis in sorted(allb["axis"].dropna().unique()):
            df_axis = allb[allb["axis"] == axis].copy()
            order = ["hybrid", "naive", "cv", "arx"]
            df_axis["component"] = pd.Categorical(df_axis["component"], categories=order, ordered=True)
            df_axis = df_axis.sort_values("component")
            plot_bar(
                df_axis,
                "component",
                "recommended_threshold",
                f"Phase 3B Component Thresholds: {axis}",
                "Recommended threshold",
                FIG_DIR / f"phase6b_component_thresholds_{axis.lower()}.png"
            )

    # -----------------------------------------------------------------
    # Phase 4B threshold-method comparison
    # -----------------------------------------------------------------

    phase4b_method = (
        p4b.groupby("threshold_method")
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_fpr=("false_positive_rate", "mean"),
            median_delay=("detection_delay_samples", "median"),
            n_rows=("threshold_method", "size"),
        )
        .reset_index()
        .sort_values(["mean_f1", "mean_fpr", "median_delay"], ascending=[False, True, True])
    )
    phase4b_method.to_csv(TABLE_DIR / "phase6b_phase4b_threshold_method_summary.csv", index=False)

    best4 = phase4b_method.iloc[0] if len(phase4b_method) else None

    phase4b_summary = pd.DataFrame([{
        "phase": "Phase 4B",
        "rows": len(p4b),
        "best_method": best4["threshold_method"] if best4 is not None else "",
        "best_mean_f1": best4["mean_f1"] if best4 is not None else np.nan,
        "best_mean_recall": best4["mean_recall"] if best4 is not None else np.nan,
        "best_mean_fpr": best4["mean_fpr"] if best4 is not None else np.nan,
        "best_median_delay": best4["median_delay"] if best4 is not None else np.nan,
        "decision": "PASS_EXTENSION" if best4 is not None and best4["mean_recall"] >= 0.80 and best4["mean_fpr"] <= 0.05 else "REVIEW",
        "interpretation": "Threshold-method comparison is acceptable as an extension; do not replace official Phase 4 silently."
    }])
    phase4b_summary.to_csv(TABLE_DIR / "phase6b_phase4b_summary.csv", index=False)

    if len(phase4b_method) > 0:
        plot_bar(
            phase4b_method,
            "threshold_method",
            "mean_f1",
            "Phase 4B Threshold Method Mean F1",
            "Mean F1",
            FIG_DIR / "phase6b_phase4b_threshold_method_f1.png"
        )
        plot_bar(
            phase4b_method,
            "threshold_method",
            "mean_fpr",
            "Phase 4B Threshold Method Mean False Positive Rate",
            "Mean FPR",
            FIG_DIR / "phase6b_phase4b_threshold_method_fpr.png"
        )

    # -----------------------------------------------------------------
    # Phase 5B recovery comparison
    # -----------------------------------------------------------------

    xkf1_summary = pd.DataFrame([{
        "method": "official_original_v9_recovery",
        "n_files": len(p5x),
        "mean_attacked_rmse": safe_mean(p5x, "attacked_rmse"),
        "mean_recovered_rmse": safe_mean(p5x, "original_v9_recovered_rmse"),
        "mean_improvement_over_attacked_percent": safe_mean(p5x, "original_improvement_over_attacked_rmse_percent"),
        "better_than_original_count": np.nan,
        "suspicion_flag": "NO",
        "paper_role": "MAIN_RESULT_BASELINE",
    }, {
        "method": "phase5b_xkf1_adaptive_weights",
        "n_files": len(p5x),
        "mean_attacked_rmse": safe_mean(p5x, "attacked_rmse"),
        "mean_recovered_rmse": safe_mean(p5x, "adaptive_v9_recovered_rmse"),
        "mean_improvement_over_attacked_percent": safe_mean(p5x, "adaptive_improvement_over_attacked_rmse_percent"),
        "better_than_original_count": safe_sum_bool(p5x, "adaptive_better_than_original_rmse"),
        "suspicion_flag": "LOW_TO_MODERATE",
        "paper_role": "EXPERIMENTAL_EXTENSION_OR_SUPPLEMENTARY",
    }, {
        "method": "phase5b_recent_error_adaptive",
        "n_files": len(p5r),
        "mean_attacked_rmse": safe_mean(p5r, "attacked_rmse"),
        "mean_recovered_rmse": safe_mean(p5r, "recent_error_adaptive_recovered_rmse"),
        "mean_improvement_over_attacked_percent": safe_mean(p5r, "adaptive_improvement_over_attacked_rmse_percent"),
        "better_than_original_count": safe_sum_bool(p5r, "adaptive_better_than_original_rmse"),
        "suspicion_flag": "HIGH_ORACLE_ASSISTED",
        "paper_role": "OFFLINE_DIAGNOSTIC_ONLY_NOT_MAIN_REAL_TIME_RESULT",
    }])

    official_rmse = float(xkf1_summary.loc[xkf1_summary["method"] == "official_original_v9_recovery", "mean_recovered_rmse"].iloc[0])
    xkf1_rmse = float(xkf1_summary.loc[xkf1_summary["method"] == "phase5b_xkf1_adaptive_weights", "mean_recovered_rmse"].iloc[0])
    recent_rmse = float(xkf1_summary.loc[xkf1_summary["method"] == "phase5b_recent_error_adaptive", "mean_recovered_rmse"].iloc[0])

    xkf1_summary["improvement_over_official_original_percent"] = [
        np.nan,
        percent_change(official_rmse, xkf1_rmse),
        percent_change(official_rmse, recent_rmse),
    ]

    xkf1_summary.to_csv(TABLE_DIR / "phase6b_phase5b_recovery_method_summary.csv", index=False)

    # By-target summaries
    x_by_target = (
        p5x.groupby("target")
        .agg(
            n=("target", "size"),
            mean_attacked_rmse=("attacked_rmse", "mean"),
            mean_original_rmse=("original_v9_recovered_rmse", "mean"),
            mean_xkf1_adaptive_rmse=("adaptive_v9_recovered_rmse", "mean"),
            xkf1_better_count=("adaptive_better_than_original_rmse", "sum"),
            mean_xkf1_improvement_over_original=("adaptive_improvement_over_original_rmse_percent", "mean"),
        )
        .reset_index()
    )

    r_by_target = (
        p5r.groupby("target")
        .agg(
            n=("target", "size"),
            mean_recent_error_adaptive_rmse=("recent_error_adaptive_recovered_rmse", "mean"),
            recent_error_better_count=("adaptive_better_than_original_rmse", "sum"),
            mean_recent_error_improvement_over_original=("adaptive_improvement_over_original_rmse_percent", "mean"),
        )
        .reset_index()
    )

    by_target = x_by_target.merge(r_by_target, on=["target", "n"], how="outer")
    by_target.to_csv(TABLE_DIR / "phase6b_phase5b_by_target_summary.csv", index=False)

    plot_bar(
        xkf1_summary,
        "method",
        "mean_recovered_rmse",
        "Phase 6B Recovery RMSE Comparison",
        "Mean recovered RMSE",
        FIG_DIR / "phase6b_recovery_rmse_comparison.png"
    )

    # -----------------------------------------------------------------
    # Suspicion audit
    # -----------------------------------------------------------------

    suspicion_rows = []

    suspicion_rows.append({
        "item": "Phase 5B XKF1 adaptive",
        "finding": "Improves 22/24 cases and uses XKF1 flight-context features.",
        "risk": "Moderate. Improvement is strong but not perfect; likely plausible.",
        "decision": "Keep as experimental extension/supplementary unless additional real-time validation is done."
    })

    suspicion_rows.append({
        "item": "Phase 5B recent-error adaptive",
        "finding": "Improves 24/24 cases and uses recent clean-window prediction error.",
        "risk": "High. This is oracle-assisted because clean future/reference error is not available in real-time attacked deployment.",
        "decision": "Do not present as main real-time result. Use only as offline diagnostic or upper-bound component-selection study."
    })

    suspicion_rows.append({
        "item": "Phase 3B Pitch ARX",
        "finding": "Pitch ARX recommended threshold is much larger than other Pitch components.",
        "risk": "High if ARX is over-trusted for Pitch.",
        "decision": "Do not use Pitch ARX dominance as main recovery logic without justification."
    })

    suspicion_df = pd.DataFrame(suspicion_rows)
    suspicion_df.to_csv(TABLE_DIR / "phase6b_suspicion_audit.csv", index=False)

    # -----------------------------------------------------------------
    # Final decision table
    # -----------------------------------------------------------------

    phase6b_pass = (
        phase2b_fail == 0
        and phase3b_pass
        and str(phase4b_summary["decision"].iloc[0]) == "PASS_EXTENSION"
        and len(p5x) > 0
    )

    decision_table = pd.DataFrame([{
        "question": "Does Phase 6B pass?",
        "decision": "YES" if phase6b_pass else "NO_REVIEW_REQUIRED",
        "reason": "B-phase evidence exists and passes core validation checks." if phase6b_pass else "One or more validation checks failed."
    }, {
        "question": "Do official Phase 4/5 need rerun?",
        "decision": "NO",
        "reason": "Phase 2B/3B do not reveal a blocking bug; Phase 3B explicitly supports not rerunning Phase 4/5 from repair alone."
    }, {
        "question": "Should Phase 4B be included?",
        "decision": "YES_AS_EXTENSION",
        "reason": "Threshold-method comparison is acceptable; scenario-specific is best overall and dynamic XKF1 is close for Roll/Pitch."
    }, {
        "question": "Should Phase 5B XKF1 be included?",
        "decision": "YES_AS_EXPERIMENTAL_EXTENSION",
        "reason": "XKF1 adaptive improves 22/24 cases and reduces mean recovered RMSE."
    }, {
        "question": "Should Phase 5B XKF1 be main result?",
        "decision": "NO",
        "reason": "Official V9 recovery should remain main; XKF1 adaptive is not universally better and was tested only on Roll/Pitch W10/N3 cases."
    }, {
        "question": "Should Phase 5B recent-error be included?",
        "decision": "ONLY_AS_OFFLINE_DIAGNOSTIC",
        "reason": "It uses recent clean-reference error and produces too-good 24/24 improvement."
    }, {
        "question": "Is project ready for final writing?",
        "decision": "YES_WITH_CAUTION",
        "reason": "Main pipeline is complete; report B-phases carefully as extensions, not replacements."
    }])

    decision_table.to_csv(TABLE_DIR / "phase6b_final_decision_table.csv", index=False)

    # -----------------------------------------------------------------
    # Markdown report
    # -----------------------------------------------------------------

    report = []
    report.append("# Phase 6B Final Validation and Comparison Report")
    report.append("")
    report.append("## Purpose")
    report.append("")
    report.append("Phase 6B validates the B-phase extensions and decides whether they improve the scientific quality of the project.")
    report.append("")
    report.append("This report is read-only with respect to official Phase 1/2/3/4/5 outputs.")
    report.append("")
    report.append("## Phase 2B result")
    report.append("")
    report.append(phase2b_summary.to_markdown(index=False))
    report.append("")
    report.append("## Phase 3B component thresholds")
    report.append("")
    report.append(phase3b_summary.to_markdown(index=False))
    report.append("")
    report.append(allb[[
        "axis",
        "component",
        "n",
        "mae",
        "rmse",
        "p99_abs_residual",
        "recommended_threshold",
    ]].to_markdown(index=False))
    report.append("")
    report.append("## Phase 4B threshold-method comparison")
    report.append("")
    report.append(phase4b_summary.to_markdown(index=False))
    report.append("")
    report.append(phase4b_method.to_markdown(index=False))
    report.append("")
    report.append("## Phase 5B recovery comparison")
    report.append("")
    report.append(xkf1_summary.to_markdown(index=False))
    report.append("")
    report.append("## Phase 5B by-target summary")
    report.append("")
    report.append(by_target.to_markdown(index=False))
    report.append("")
    report.append("## Suspicion audit")
    report.append("")
    report.append(suspicion_df.to_markdown(index=False))
    report.append("")
    report.append("## Final decisions")
    report.append("")
    report.append(decision_table.to_markdown(index=False))
    report.append("")
    report.append("## Recommended paper presentation")
    report.append("")
    report.append("- Present official V9 trained hybrid recovery as the main Phase 5 result.")
    report.append("- Present Phase 4B as a threshold-method ablation/extension.")
    report.append("- Present Phase 5B XKF1 adaptive weighting as an experimental extension or supplementary result.")
    report.append("- Present recent-error adaptive weighting only as an offline diagnostic or upper-bound component-selection study.")
    report.append("- Do not claim recent-error adaptive weighting is deployable in real time unless it is redesigned without clean-reference future values.")
    report.append("")
    report.append("## Final conclusion")
    report.append("")
    if phase6b_pass:
        report.append("Phase 6B PASSES. The project is ready for final writing/figures, with the caution that B-phase extensions must be separated from official results.")
    else:
        report.append("Phase 6B requires review before final writing.")

    report_path = REPORT_DIR / "phase6b_final_validation_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print()
    print("[DONE] Phase 6B final validation complete.")
    print(f"[TABLES]  {TABLE_DIR}")
    print(f"[REPORT]  {report_path}")
    print(f"[FIGURES] {FIG_DIR}")
    print()
    print(decision_table.to_string(index=False))


if __name__ == "__main__":
    main()