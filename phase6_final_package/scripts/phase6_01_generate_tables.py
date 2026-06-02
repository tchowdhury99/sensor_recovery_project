#!/usr/bin/env python3

import os
from pathlib import Path

import pandas as pd
import numpy as np


BASE = Path("/home/tchowdh4/sensor_recovery_project")

PHASE5_SUMMARY = BASE / "recovery_v9_corrected_official" / "recovery_reports" / "phase5_OFFICIAL_W10_N3_mean1x_recovery_summary.csv"

OUT = BASE / "phase6_final_package"
TABLE_DIR = OUT / "tables"
LOG_DIR = OUT / "logs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def normalize(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def find_column(df, required=None, optional=None, forbidden=None):
    required = required or []
    optional = optional or []
    forbidden = forbidden or []

    matches = []

    for col in df.columns:
        n = normalize(col)

        if any(x in n for x in forbidden):
            continue

        if not all(x in n for x in required):
            continue

        if optional:
            if not any(x in n for x in optional):
                continue

        matches.append(col)

    if not matches:
        return None

    return sorted(matches, key=lambda c: len(str(c)))[0]


def pct_improvement(attacked, recovered):
    attacked = pd.to_numeric(attacked, errors="coerce")
    recovered = pd.to_numeric(recovered, errors="coerce")
    return ((attacked - recovered) / attacked) * 100.0


def save_md(df, path):
    try:
        path.write_text(df.to_markdown(index=False) + "\n")
    except Exception:
        path.write_text(df.to_string(index=False) + "\n")


def main():
    if not PHASE5_SUMMARY.exists():
        raise FileNotFoundError(f"Missing Phase 5 summary CSV: {PHASE5_SUMMARY}")

    df = pd.read_csv(PHASE5_SUMMARY)

    audit_lines = []
    audit_lines.append("PHASE 6 TABLE GENERATION COLUMN AUDIT")
    audit_lines.append("=====================================")
    audit_lines.append("")
    audit_lines.append(f"Input CSV: {PHASE5_SUMMARY}")
    audit_lines.append(f"Rows: {len(df)}")
    audit_lines.append("")
    audit_lines.append("Columns:")
    for col in df.columns:
        audit_lines.append(f"- {col}")

    file_col = find_column(df, optional=["file", "csv", "log"], forbidden=["improvement"])
    target_col = find_column(df, optional=["target", "axis", "state", "signal"])

    attacked_rmse_col = find_column(df, required=["rmse"], optional=["attacked", "attack"], forbidden=["improvement", "percent", "pct"])
    recovered_rmse_col = find_column(df, required=["rmse"], optional=["recovered", "recovery"], forbidden=["improvement", "percent", "pct"])
    rmse_improvement_col = find_column(df, required=["rmse"], optional=["improvement", "percent", "pct"])

    attacked_mae_col = find_column(df, required=["mae"], optional=["attacked", "attack"], forbidden=["improvement", "percent", "pct"])
    recovered_mae_col = find_column(df, required=["mae"], optional=["recovered", "recovery"], forbidden=["improvement", "percent", "pct"])
    mae_improvement_col = find_column(df, required=["mae"], optional=["improvement", "percent", "pct"])

    mapping = {
        "file_col": file_col,
        "target_col": target_col,
        "attacked_rmse_col": attacked_rmse_col,
        "recovered_rmse_col": recovered_rmse_col,
        "rmse_improvement_col": rmse_improvement_col,
        "attacked_mae_col": attacked_mae_col,
        "recovered_mae_col": recovered_mae_col,
        "mae_improvement_col": mae_improvement_col,
    }

    audit_lines.append("")
    audit_lines.append("Detected mapping:")
    for k, v in mapping.items():
        audit_lines.append(f"- {k}: {v}")

    required_cols = [
        attacked_rmse_col,
        recovered_rmse_col,
        attacked_mae_col,
        recovered_mae_col,
    ]

    if any(c is None for c in required_cols):
        audit_lines.append("")
        audit_lines.append("ERROR: Could not detect all required metric columns.")
        (LOG_DIR / "phase6_01_table_column_audit.txt").write_text("\n".join(audit_lines) + "\n")
        raise RuntimeError("Could not detect required columns. Check phase6_01_table_column_audit.txt")

    work = pd.DataFrame()

    if file_col:
        work["file"] = df[file_col].astype(str)
    else:
        work["file"] = [f"case_{i+1:03d}" for i in range(len(df))]

    if target_col:
        work["target"] = df[target_col].astype(str)
    else:
        work["target"] = "unknown"

    work["attacked_rmse"] = pd.to_numeric(df[attacked_rmse_col], errors="coerce")
    work["recovered_rmse"] = pd.to_numeric(df[recovered_rmse_col], errors="coerce")

    if rmse_improvement_col:
        work["rmse_improvement_percent"] = pd.to_numeric(df[rmse_improvement_col], errors="coerce")
    else:
        work["rmse_improvement_percent"] = pct_improvement(work["attacked_rmse"], work["recovered_rmse"])

    work["attacked_mae"] = pd.to_numeric(df[attacked_mae_col], errors="coerce")
    work["recovered_mae"] = pd.to_numeric(df[recovered_mae_col], errors="coerce")

    if mae_improvement_col:
        work["mae_improvement_percent"] = pd.to_numeric(df[mae_improvement_col], errors="coerce")
    else:
        work["mae_improvement_percent"] = pct_improvement(work["attacked_mae"], work["recovered_mae"])

    work["rmse_improved"] = work["recovered_rmse"] < work["attacked_rmse"]
    work["mae_improved"] = work["recovered_mae"] < work["attacked_mae"]

    n = len(work)

    overall = pd.DataFrame([
        ["Software sensor", "V9 trained hybrid"],
        ["Official detector", "W10_N3_mean1x"],
        ["Threshold", "p99_abs"],
        ["Official files evaluated", n],
        ["Files improved by RMSE", f"{int(work['rmse_improved'].sum())}/{n}"],
        ["Files improved by MAE", f"{int(work['mae_improved'].sum())}/{n}"],
        ["Mean attacked RMSE", work["attacked_rmse"].mean()],
        ["Mean recovered RMSE", work["recovered_rmse"].mean()],
        ["Mean RMSE improvement (%)", work["rmse_improvement_percent"].mean()],
        ["Mean attacked MAE", work["attacked_mae"].mean()],
        ["Mean recovered MAE", work["recovered_mae"].mean()],
        ["Mean MAE improvement (%)", work["mae_improvement_percent"].mean()],
    ], columns=["Metric", "Value"])

    per_target = (
        work.groupby("target")
        .agg(
            files=("file", "count"),
            mean_attacked_rmse=("attacked_rmse", "mean"),
            mean_recovered_rmse=("recovered_rmse", "mean"),
            mean_rmse_improvement_percent=("rmse_improvement_percent", "mean"),
            mean_attacked_mae=("attacked_mae", "mean"),
            mean_recovered_mae=("recovered_mae", "mean"),
            mean_mae_improvement_percent=("mae_improvement_percent", "mean"),
            rmse_improved_files=("rmse_improved", "sum"),
            mae_improved_files=("mae_improved", "sum"),
        )
        .reset_index()
    )

    best_cases = work.sort_values("rmse_improvement_percent", ascending=False).head(10)
    worst_cases = work.sort_values("rmse_improvement_percent", ascending=True).head(10)

    overall.to_csv(TABLE_DIR / "phase6_overall_metrics.csv", index=False)
    work.to_csv(TABLE_DIR / "phase6_per_file_metrics.csv", index=False)
    per_target.to_csv(TABLE_DIR / "phase6_per_target_metrics.csv", index=False)
    best_cases.to_csv(TABLE_DIR / "phase6_best_cases.csv", index=False)
    worst_cases.to_csv(TABLE_DIR / "phase6_worst_cases.csv", index=False)

    save_md(overall, TABLE_DIR / "phase6_overall_metrics.md")
    save_md(work, TABLE_DIR / "phase6_per_file_metrics.md")
    save_md(per_target, TABLE_DIR / "phase6_per_target_metrics.md")
    save_md(best_cases, TABLE_DIR / "phase6_best_cases.md")
    save_md(worst_cases, TABLE_DIR / "phase6_worst_cases.md")

    def latex_escape(x):
        s = str(x)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for a, b in replacements.items():
            s = s.replace(a, b)
        return s

    def write_simple_latex_table(df, path, caption, label):
        cols = list(df.columns)
        col_spec = "l" * len(cols)

        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(rf"\caption{{{caption}}}")
        lines.append(rf"\label{{{label}}}")
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\hline")
        lines.append(" & ".join(latex_escape(c) for c in cols) + r" \\")
        lines.append(r"\hline")

        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(latex_escape(v))
            lines.append(" & ".join(vals) + r" \\")

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        path.write_text("\n".join(lines) + "\n")

    write_simple_latex_table(
        overall,
        TABLE_DIR / "phase6_overall_metrics.tex",
        "Final recovery performance summary.",
        "tab:phase6_overall_metrics"
    )

    write_simple_latex_table(
        per_target,
        TABLE_DIR / "phase6_per_target_metrics.tex",
        "Per-target recovery performance summary.",
        "tab:phase6_per_target_metrics"
    )

    audit_lines.append("")
    audit_lines.append("Generated files:")
    for p in sorted(TABLE_DIR.glob("*")):
        audit_lines.append(f"- {p}")

    (LOG_DIR / "phase6_01_table_column_audit.txt").write_text("\n".join(audit_lines) + "\n")

    print("Phase 6 table generation complete.")
    print(f"Tables written to: {TABLE_DIR}")
    print("")
    print("Key result check:")
    print(f"Files evaluated: {n}")
    print(f"RMSE improved: {int(work['rmse_improved'].sum())}/{n}")
    print(f"MAE improved: {int(work['mae_improved'].sum())}/{n}")
    print(f"Mean attacked RMSE: {work['attacked_rmse'].mean()}")
    print(f"Mean recovered RMSE: {work['recovered_rmse'].mean()}")
    print(f"Mean RMSE improvement: {work['rmse_improvement_percent'].mean()}%")
    print(f"Mean attacked MAE: {work['attacked_mae'].mean()}")
    print(f"Mean recovered MAE: {work['recovered_mae'].mean()}")
    print(f"Mean MAE improvement: {work['mae_improvement_percent'].mean()}%")


if __name__ == "__main__":
    main()