#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/tchowdh4/sensor_recovery_project")

SUMMARY_PATH = PROJECT_ROOT / "detection_reports_v9_corrected" / "phase4_detection_summary_v9_corrected.csv"
ALIGNMENT_PATH = PROJECT_ROOT / "detection_reports_v9_corrected" / "phase4_alignment_report_v9_corrected.csv"
OUT_PATH = PROJECT_ROOT / "detection_reports_v9_corrected" / "phase4_detection_interpretation_v9_corrected.txt"


def fmt(x):
    if pd.isna(x):
        return "nan"
    return f"{x:.4f}"


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")

    if not ALIGNMENT_PATH.exists():
        raise FileNotFoundError(f"Missing alignment file: {ALIGNMENT_PATH}")

    df = pd.read_csv(SUMMARY_PATH)
    align = pd.read_csv(ALIGNMENT_PATH)

    lines = []

    lines.append("CORRECTED PHASE 4 V9 ATTACK DETECTION INTERPRETATION")
    lines.append("=" * 75)
    lines.append("")

    lines.append("Input files:")
    lines.append(f"- Detection summary: {SUMMARY_PATH}")
    lines.append(f"- Alignment report: {ALIGNMENT_PATH}")
    lines.append(f"- Detection rows: {len(df)}")
    lines.append(f"- Alignment rows: {len(align)}")
    lines.append("")

    # ------------------------------------------------------------
    # Alignment quality
    # ------------------------------------------------------------
    align_cols = [
        "baseline",
        "axis",
        "alignment_offset",
        "n_prediction",
        "alignment_median_abs_error",
        "threshold_p99_abs",
        "alignment_error_over_threshold",
    ]

    lines.append("Alignment quality:")
    lines.append(align[align_cols].to_string(index=False))
    lines.append("")

    suspicious = align[align["alignment_error_over_threshold"] > 5.0]

    if len(suspicious) > 0:
        lines.append("Alignment decision:")
        lines.append("- WARNING: Some axes have alignment_error_over_threshold > 5.0.")
        lines.append("- These axes may still have prediction/measurement mismatch.")
        lines.append("- Inspect figures before trusting detection results.")
        lines.append("")
        lines.append(suspicious[align_cols].to_string(index=False))
        lines.append("")
        alignment_ok = False
    else:
        lines.append("Alignment decision:")
        lines.append("- PASS: All axes have acceptable prediction-to-measurement alignment.")
        lines.append("- The earlier huge Yaw residual problem is fixed.")
        lines.append("")
        alignment_ok = True

    if len(df) == 0:
        lines.append("Detection decision:")
        lines.append("- FAIL: No detection rows found.")
        OUT_PATH.write_text("\n".join(lines))
        print("\n".join(lines))
        return

    wc = df[df["detector"] == "window_combined"].copy()

    if wc.empty:
        lines.append("Detection decision:")
        lines.append("- FAIL: No window_combined detector rows found.")
        OUT_PATH.write_text("\n".join(lines))
        print("\n".join(lines))
        return

    # ------------------------------------------------------------
    # Window rule comparison
    # ------------------------------------------------------------
    rule_comparison = (
        wc.groupby("window_rule")
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_fpr=("false_positive_rate", "mean"),
            median_delay=("detection_delay_samples", "median"),
            n_rows=("window_rule", "size"),
        )
        .reset_index()
    )

    rule_comparison = rule_comparison.sort_values(
        ["mean_f1", "mean_fpr", "median_delay"],
        ascending=[False, True, True],
    )

    lines.append("Window rule comparison:")
    lines.append(rule_comparison.to_string(index=False))
    lines.append("")

    best_rule = rule_comparison.iloc[0]

    lines.append("Recommended window rule:")
    lines.append(f"- {best_rule['window_rule']}")
    lines.append(f"- mean precision: {fmt(best_rule['mean_precision'])}")
    lines.append(f"- mean recall: {fmt(best_rule['mean_recall'])}")
    lines.append(f"- mean F1: {fmt(best_rule['mean_f1'])}")
    lines.append(f"- mean FPR: {fmt(best_rule['mean_fpr'])}")
    lines.append(f"- median detection delay: {best_rule['median_delay']}")
    lines.append("")

    # ------------------------------------------------------------
    # Attack-type comparison
    # ------------------------------------------------------------
    attack_comparison = (
        wc.groupby("attack_type")
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_fpr=("false_positive_rate", "mean"),
            median_delay=("detection_delay_samples", "median"),
            n_rows=("attack_type", "size"),
        )
        .reset_index()
        .sort_values(["mean_f1", "mean_fpr"], ascending=[False, True])
    )

    lines.append("Attack-type comparison:")
    lines.append(attack_comparison.to_string(index=False))
    lines.append("")

    # ------------------------------------------------------------
    # Axis comparison
    # ------------------------------------------------------------
    axis_comparison = (
        wc.groupby("axis")
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_fpr=("false_positive_rate", "mean"),
            median_delay=("detection_delay_samples", "median"),
            n_rows=("axis", "size"),
        )
        .reset_index()
        .sort_values(["mean_f1", "mean_fpr"], ascending=[False, True])
    )

    lines.append("Axis comparison:")
    lines.append(axis_comparison.to_string(index=False))
    lines.append("")

    # ------------------------------------------------------------
    # Best-rule metrics
    # ------------------------------------------------------------
    best_wc = wc[wc["window_rule"] == best_rule["window_rule"]].copy()

    mean_precision = best_wc["precision"].mean()
    mean_recall = best_wc["recall"].mean()
    mean_f1 = best_wc["f1"].mean()
    mean_fpr = best_wc["false_positive_rate"].mean()
    median_delay = best_wc["detection_delay_samples"].median()

    lines.append("Best-rule overall performance:")
    lines.append(f"- precision: {fmt(mean_precision)}")
    lines.append(f"- recall: {fmt(mean_recall)}")
    lines.append(f"- F1: {fmt(mean_f1)}")
    lines.append(f"- false positive rate: {fmt(mean_fpr)}")
    lines.append(f"- median detection delay: {median_delay}")
    lines.append("")

    # ------------------------------------------------------------
    # Threshold interpretation
    # ------------------------------------------------------------
    lines.append("Threshold interpretation:")

    if mean_fpr <= 0.02:
        lines.append("- p99_abs looks strong. FPR is low.")
        threshold_decision = "p99_abs"
    elif mean_fpr <= 0.05:
        lines.append("- p99_abs is acceptable. FPR is below 5%, but inspect plots.")
        threshold_decision = "p99_abs"
    else:
        lines.append("- p99_abs appears too sensitive. FPR is above 5%.")
        lines.append("- Recommendation: test p99.5_abs before recovery.")
        threshold_decision = "test_p99_5_abs"

    lines.append("")

    # ------------------------------------------------------------
    # Phase 4 completion decision
    # ------------------------------------------------------------
    detection_ok = (
        pd.notna(mean_recall)
        and pd.notna(mean_fpr)
        and mean_recall >= 0.80
        and mean_fpr <= 0.05
    )

    lines.append("Phase 4 completion decision:")

    if alignment_ok and detection_ok:
        lines.append("- COMPLETE: Phase 4 controlled ATT attack detection is complete.")
        lines.append("- You can proceed to Phase 5 recovery after saving this report and key figures.")
    elif not alignment_ok:
        lines.append("- NOT COMPLETE: Alignment issue remains.")
        lines.append("- Do not proceed to recovery.")
    else:
        lines.append("- NOT COMPLETE: Detection performance is not acceptable yet.")
        if pd.isna(mean_recall):
            lines.append("  Reason: recall is NaN.")
        elif mean_recall < 0.80:
            lines.append(f"  Reason: mean recall is below 0.80: {fmt(mean_recall)}")
        if pd.isna(mean_fpr):
            lines.append("  Reason: FPR is NaN.")
        elif mean_fpr > 0.05:
            lines.append(f"  Reason: mean FPR is above 0.05: {fmt(mean_fpr)}")
        lines.append("- Do not proceed to recovery yet.")

    lines.append("")

    # ------------------------------------------------------------
    # Carry-forward configuration
    # ------------------------------------------------------------
    lines.append("Recommended carry-forward configuration:")
    lines.append("- Software sensor: V9 trained hybrid")
    lines.append("- Residual for Roll/Pitch: attacked_future[k+1] - V9_prediction[k+1]")
    lines.append("- Residual for Yaw: circular angular difference")
    lines.append(f"- Threshold decision: {threshold_decision}")
    lines.append(f"- Window detector: {best_rule['window_rule']}")
    lines.append("- Recovery: not performed in Phase 4")
    lines.append("")

    # ------------------------------------------------------------
    # Files
    # ------------------------------------------------------------
    lines.append("Important output folders:")
    lines.append(f"- Detection CSVs: {PROJECT_ROOT / 'attack_detection_v9_corrected' / 'detection_csvs'}")
    lines.append(f"- Aligned attacked segments: {PROJECT_ROOT / 'attack_detection_v9_corrected' / 'aligned_attacked_segments'}")
    lines.append(f"- Figures: {PROJECT_ROOT / 'figures' / 'attack_detection_v9_corrected'}")
    lines.append(f"- Reports: {PROJECT_ROOT / 'detection_reports_v9_corrected'}")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines))

    print("\n".join(lines))
    print()
    print(f"[DONE] Wrote corrected interpretation report to:")
    print(f"{OUT_PATH}")


if __name__ == "__main__":
    main()