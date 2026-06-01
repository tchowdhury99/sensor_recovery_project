
#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/tchowdh4/sensor_recovery_project")
SUMMARY_PATH = PROJECT_ROOT / "detection_reports_v9" / "phase4_detection_summary_v9.csv"
OUT_PATH = PROJECT_ROOT / "detection_reports_v9" / "phase4_detection_interpretation_v9.txt"

def fmt(x):
    if pd.isna(x):
        return "nan"
    return f"{x:.4f}"

def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")

    df = pd.read_csv(SUMMARY_PATH)

    lines = []
    lines.append("PHASE 4 V9 ATTACK DETECTION INTERPRETATION")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Input summary:")
    lines.append(f"- Summary CSV: {SUMMARY_PATH}")
    lines.append(f"- Rows: {len(df)}")
    lines.append("")

    if len(df) == 0:
        lines.append("No detection results found. Phase 4 is NOT complete.")
        OUT_PATH.write_text("\n".join(lines))
        print("\n".join(lines))
        return

    # Prefer window_combined because it implements paper-aligned window detection.
    wc = df[df["detector"] == "window_combined"].copy()

    if wc.empty:
        lines.append("No window_combined detector results found.")
        lines.append("Phase 4 is NOT complete because window-based detection was required.")
        OUT_PATH.write_text("\n".join(lines))
        print("\n".join(lines))
        return

    grouped_rule = (
        wc.groupby("window_rule")
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_fpr=("false_positive_rate", "mean"),
            median_delay=("detection_delay_samples", "median"),
            n=("f1", "count"),
        )
        .reset_index()
    )

    # Sort by high F1, then low FPR, then low delay.
    grouped_rule = grouped_rule.sort_values(
        ["mean_f1", "mean_fpr", "median_delay"],
        ascending=[False, True, True],
    )

    best_rule = grouped_rule.iloc[0]

    lines.append("Window rule comparison:")
    lines.append(grouped_rule.to_string(index=False))
    lines.append("")

    lines.append("Recommended window rule:")
    lines.append(f"- {best_rule['window_rule']}")
    lines.append(f"- mean precision: {fmt(best_rule['mean_precision'])}")
    lines.append(f"- mean recall: {fmt(best_rule['mean_recall'])}")
    lines.append(f"- mean F1: {fmt(best_rule['mean_f1'])}")
    lines.append(f"- mean false positive rate: {fmt(best_rule['mean_fpr'])}")
    lines.append(f"- median detection delay, samples: {best_rule['median_delay']}")
    lines.append("")

    grouped_attack = (
        wc.groupby("attack_type")
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_fpr=("false_positive_rate", "mean"),
            median_delay=("detection_delay_samples", "median"),
            n=("f1", "count"),
        )
        .reset_index()
        .sort_values(["mean_f1", "mean_fpr"], ascending=[False, True])
    )

    lines.append("Attack-type comparison:")
    lines.append(grouped_attack.to_string(index=False))
    lines.append("")

    grouped_axis = (
        wc.groupby("attack_axis")
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_fpr=("false_positive_rate", "mean"),
            median_delay=("detection_delay_samples", "median"),
            n=("f1", "count"),
        )
        .reset_index()
        .sort_values(["mean_f1", "mean_fpr"], ascending=[False, True])
    )

    lines.append("Axis comparison:")
    lines.append(grouped_axis.to_string(index=False))
    lines.append("")

    # Sensitivity decision for p99_abs.
    best_wc = wc[wc["window_rule"] == best_rule["window_rule"]].copy()

    mean_fpr = best_wc["false_positive_rate"].mean()
    mean_recall = best_wc["recall"].mean()
    mean_f1 = best_wc["f1"].mean()

    lines.append("Threshold interpretation:")
    if mean_fpr > 0.05:
        lines.append("- p99_abs appears sensitive: mean FPR is above 5%.")
        lines.append("- Recommendation: test p99.5_abs as the conservative backup before Phase 5.")
    elif mean_fpr > 0.02:
        lines.append("- p99_abs is moderately sensitive: mean FPR is between 2% and 5%.")
        lines.append("- Recommendation: p99_abs may still be usable, but compare against p99.5_abs.")
    else:
        lines.append("- p99_abs is not obviously too sensitive: mean FPR is below or near 2%.")
        lines.append("- Recommendation: keep p99_abs as primary unless visual plots show nuisance detections.")

    lines.append(f"- Best-rule mean recall: {fmt(mean_recall)}")
    lines.append(f"- Best-rule mean F1: {fmt(mean_f1)}")
    lines.append(f"- Best-rule mean FPR: {fmt(mean_fpr)}")
    lines.append("")

    # Completion decision.
    lines.append("Phase 4 completion decision:")
    if mean_recall >= 0.80 and mean_fpr <= 0.05:
        lines.append("- Phase 4 can be considered COMPLETE for controlled ATT attack detection.")
        lines.append("- You may proceed to Phase 5 recovery after documenting the selected threshold and window rule.")
    else:
        lines.append("- Phase 4 is NOT fully complete yet.")
        lines.append("- Improve detection configuration before moving to recovery.")
        if mean_recall < 0.80:
            lines.append("  Reason: recall is below 0.80.")
        if mean_fpr > 0.05:
            lines.append("  Reason: false positive rate is above 0.05.")
    lines.append("")

    lines.append("Recommended carry-forward configuration:")
    lines.append("- Software sensor: V9 trained hybrid")
    lines.append("- Residual: attacked_future[k+1] - V9_prediction[k+1]")
    lines.append("- Threshold family: p99_abs first; p99.5_abs if FPR is too high")
    lines.append(f"- Window detector: {best_rule['window_rule']}")
    lines.append("- Recovery: not performed in Phase 4")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))
    print()
    print(f"[DONE] Wrote interpretation report to: {OUT_PATH}")

if __name__ == "__main__":
    main()
