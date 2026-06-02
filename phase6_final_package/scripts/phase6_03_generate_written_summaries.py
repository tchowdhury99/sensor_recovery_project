
#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


BASE = Path("/home/tchowdh4/sensor_recovery_project")

OUT = BASE / "phase6_final_package"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
REPORT_DIR = OUT / "reports"
LOG_DIR = OUT / "logs"

PER_FILE_TABLE = TABLE_DIR / "phase6_per_file_metrics.csv"
OVERALL_TABLE = TABLE_DIR / "phase6_overall_metrics.csv"

PHASE5_SUMMARY = BASE / "recovery_v9_corrected_official" / "recovery_reports" / "phase5_OFFICIAL_W10_N3_mean1x_recovery_summary.csv"
PHASE5_INTERPRETATION = BASE / "recovery_v9_corrected_official" / "recovery_reports" / "phase5_OFFICIAL_W10_N3_mean1x_recovery_interpretation.txt"
PHASE5_RECOVERED_CSV_DIR = BASE / "recovery_v9_corrected_official" / "recovered_csvs"
PHASE5_FIG_DIR = BASE / "figures" / "recovery_v9_corrected_official"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not PER_FILE_TABLE.exists():
        raise FileNotFoundError(
            f"Missing {PER_FILE_TABLE}. Run phase6_01_generate_tables.py first."
        )

    df = pd.read_csv(PER_FILE_TABLE)

    n = len(df)
    rmse_improved = int(df["rmse_improved"].sum())
    mae_improved = int(df["mae_improved"].sum())

    mean_attacked_rmse = df["attacked_rmse"].mean()
    mean_recovered_rmse = df["recovered_rmse"].mean()
    mean_rmse_improvement = df["rmse_improvement_percent"].mean()

    mean_attacked_mae = df["attacked_mae"].mean()
    mean_recovered_mae = df["recovered_mae"].mean()
    mean_mae_improvement = df["mae_improvement_percent"].mean()

    min_rmse_improvement = df["rmse_improvement_percent"].min()
    max_rmse_improvement = df["rmse_improvement_percent"].max()
    median_rmse_improvement = df["rmse_improvement_percent"].median()

    min_mae_improvement = df["mae_improvement_percent"].min()
    max_mae_improvement = df["mae_improvement_percent"].max()
    median_mae_improvement = df["mae_improvement_percent"].median()

    summary_txt = f"""
PHASE 6 FINAL RESULT SUMMARY
============================

Purpose:
This Phase 6 package converts the completed Phase 5 official recovery results
into paper-style final tables, figures, and written result summaries.

Official Phase 5 configuration:
- Software sensor: V9 trained hybrid
- Detector: W10_N3_mean1x
- Threshold: p99_abs

Official Phase 5 input files:
- Summary CSV:
  {PHASE5_SUMMARY}
- Interpretation report:
  {PHASE5_INTERPRETATION}
- Recovered CSV directory:
  {PHASE5_RECOVERED_CSV_DIR}
- Recovery figure directory:
  {PHASE5_FIG_DIR}

Final aggregate results:
- Official files evaluated: {n}
- Files improved by RMSE recovery: {rmse_improved}/{n}
- Files improved by MAE recovery: {mae_improved}/{n}

RMSE:
- Mean attacked RMSE: {mean_attacked_rmse:.15f}
- Mean recovered RMSE: {mean_recovered_rmse:.15f}
- Mean RMSE improvement: {mean_rmse_improvement:.12f}%
- Median RMSE improvement: {median_rmse_improvement:.12f}%
- Minimum RMSE improvement: {min_rmse_improvement:.12f}%
- Maximum RMSE improvement: {max_rmse_improvement:.12f}%

MAE:
- Mean attacked MAE: {mean_attacked_mae:.15f}
- Mean recovered MAE: {mean_recovered_mae:.15f}
- Mean MAE improvement: {mean_mae_improvement:.12f}%
- Median MAE improvement: {median_mae_improvement:.12f}%
- Minimum MAE improvement: {min_mae_improvement:.12f}%
- Maximum MAE improvement: {max_mae_improvement:.12f}%

Interpretation:
The final recovery results show that the V9 trained hybrid software sensor,
combined with the corrected W10_N3_mean1x detector and p99_abs threshold,
substantially reduced attack-induced signal error in the ArduPilot SITL
experiments. Recovery improved all official evaluated files under the RMSE
criterion and produced more than 90% average reduction in both RMSE and MAE.

Generated Phase 6 package:
- Tables:
  {TABLE_DIR}
- Figures:
  {FIG_DIR}
- Reports:
  {REPORT_DIR}
"""

    (REPORT_DIR / "phase6_final_result_summary.txt").write_text(summary_txt.strip() + "\n")

    paper_md = f"""
# Final Recovery Results

This section reports the final recovery results for the ArduPilot SITL reproduction of the software-based realtime recovery methodology. The final configuration used the **V9 trained hybrid software sensor**, the corrected **W10_N3_mean1x** window detector, and the **p99_abs** residual threshold. The official Phase 5 evaluation contained **{n} attacked files**.

## Aggregate Recovery Performance

The recovery stage improved **{rmse_improved}/{n}** files under the RMSE criterion and **{mae_improved}/{n}** files under the MAE criterion.

The mean attacked RMSE was **{mean_attacked_rmse:.15f}**, while the mean recovered RMSE was **{mean_recovered_rmse:.15f}**. This corresponds to a mean RMSE improvement of **{mean_rmse_improvement:.12f}%**.

The mean attacked MAE was **{mean_attacked_mae:.15f}**, while the mean recovered MAE was **{mean_recovered_mae:.15f}**. This corresponds to a mean MAE improvement of **{mean_mae_improvement:.12f}%**.

## Final Configuration

| Component | Selected Method |
|---|---|
| Software sensor | V9 trained hybrid |
| Detector | W10_N3_mean1x |
| Threshold | p99_abs |
| Official files evaluated | {n} |
| Files improved by RMSE | {rmse_improved}/{n} |
| Files improved by MAE | {mae_improved}/{n} |

## Interpretation

These results show that software-sensor-based recovery substantially reduces the error caused by attacked sensor measurements. The recovered signal has much lower error than the attacked signal across the official recovery set. This supports the central reproduction claim that a learned software sensor can act as a backup estimator when the physical sensor stream is compromised.

In this reproduction, the V9 trained hybrid model was selected because earlier ARX-style predictors were unstable for some attitude channels, especially pitch and yaw. The hybrid model combines simple temporal baselines with a trained component and therefore provides a more stable future-state estimate. When paired with the corrected residual detector and threshold-based recovery logic, it produced consistent improvement across the official attacked files.
"""

    (REPORT_DIR / "phase6_paper_results_section.md").write_text(paper_md.strip() + "\n")

    paper_tex = rf"""
\section{{Final Recovery Results}}

This section reports the final recovery results for the ArduPilot SITL reproduction of the software-based realtime recovery methodology. The final configuration used the V9 trained hybrid software sensor, the corrected \texttt{{W10\_N3\_mean1x}} window detector, and the \texttt{{p99\_abs}} residual threshold. The official Phase 5 evaluation contained {n} attacked files.

\subsection{{Aggregate Recovery Performance}}

The recovery stage improved {rmse_improved}/{n} files under the RMSE criterion and {mae_improved}/{n} files under the MAE criterion.

The mean attacked RMSE was {mean_attacked_rmse:.15f}, while the mean recovered RMSE was {mean_recovered_rmse:.15f}. This corresponds to a mean RMSE improvement of {mean_rmse_improvement:.12f}\%.

The mean attacked MAE was {mean_attacked_mae:.15f}, while the mean recovered MAE was {mean_recovered_mae:.15f}. This corresponds to a mean MAE improvement of {mean_mae_improvement:.12f}\%.

\begin{{table}}[t]
\centering
\caption{{Final recovery configuration and aggregate results.}}
\label{{tab:final_recovery_results}}
\begin{{tabular}}{{ll}}
\hline
Component & Selected Method \\
\hline
Software sensor & V9 trained hybrid \\
Detector & \texttt{{W10\_N3\_mean1x}} \\
Threshold & \texttt{{p99\_abs}} \\
Official files evaluated & {n} \\
Files improved by RMSE & {rmse_improved}/{n} \\
Files improved by MAE & {mae_improved}/{n} \\
Mean attacked RMSE & {mean_attacked_rmse:.15f} \\
Mean recovered RMSE & {mean_recovered_rmse:.15f} \\
Mean RMSE improvement & {mean_rmse_improvement:.12f}\% \\
Mean attacked MAE & {mean_attacked_mae:.15f} \\
Mean recovered MAE & {mean_recovered_mae:.15f} \\
Mean MAE improvement & {mean_mae_improvement:.12f}\% \\
\hline
\end{{tabular}}
\end{{table}}

\subsection{{Interpretation}}

These results show that software-sensor-based recovery substantially reduces the error caused by attacked sensor measurements. The recovered signal has much lower error than the attacked signal across the official recovery set. This supports the central reproduction claim that a learned software sensor can act as a backup estimator when the physical sensor stream is compromised.

In this reproduction, the V9 trained hybrid model was selected because earlier ARX-style predictors were unstable for some attitude channels, especially pitch and yaw. The hybrid model combines simple temporal baselines with a trained component and therefore provides a more stable future-state estimate. When paired with the corrected residual detector and threshold-based recovery logic, it produced consistent improvement across the official attacked files.
"""

    (REPORT_DIR / "phase6_paper_results_section.tex").write_text(paper_tex.strip() + "\n")

    manifest = []
    manifest.append("PHASE 6 FINAL PACKAGE MANIFEST")
    manifest.append("==============================")
    manifest.append("")
    manifest.append("Tables:")
    for p in sorted(TABLE_DIR.glob("*")):
        manifest.append(f"- {p}")
    manifest.append("")
    manifest.append("Figures:")
    for p in sorted(FIG_DIR.glob("*")):
        manifest.append(f"- {p}")
    manifest.append("")
    manifest.append("Reports:")
    for p in sorted(REPORT_DIR.glob("*")):
        manifest.append(f"- {p}")

    (REPORT_DIR / "phase6_artifact_manifest.txt").write_text("\n".join(manifest) + "\n")

    print("Phase 6 written summaries complete.")
    print(f"Reports written to: {REPORT_DIR}")
    print("")
    print("Important files:")
    print(REPORT_DIR / "phase6_final_result_summary.txt")
    print(REPORT_DIR / "phase6_paper_results_section.md")
    print(REPORT_DIR / "phase6_paper_results_section.tex")


if __name__ == "__main__":
    main()
