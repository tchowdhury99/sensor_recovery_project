
#!/usr/bin/env python3

from pathlib import Path
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path("/home/tchowdh4/sensor_recovery_project")

TABLE_DIR = BASE / "phase6_final_package" / "tables"
FIG_DIR = BASE / "phase6_final_package" / "figures"
LOG_DIR = BASE / "phase6_final_package" / "logs"

RECOVERED_CSV_DIR = BASE / "recovery_v9_corrected_official" / "recovered_csvs"

PER_FILE_TABLE = TABLE_DIR / "phase6_per_file_metrics.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name):
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"

    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()

    return png, pdf


def normalize(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def find_col(df, required=None, optional=None, forbidden=None):
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


def infer_time(df):
    for col in df.columns:
        n = normalize(col)
        if n in ["t", "time", "time_s", "time_sec", "timesec", "timestamp"]:
            s = pd.to_numeric(df[col], errors="coerce")
            return s, "Time (s)", col

        if n == "timeus":
            s = pd.to_numeric(df[col], errors="coerce")
            s = (s - s.iloc[0]) / 1_000_000.0
            return s, "Time (s)", col

        if n == "timems":
            s = pd.to_numeric(df[col], errors="coerce")
            s = (s - s.iloc[0]) / 1_000.0
            return s, "Time (s)", col

    return pd.Series(np.arange(len(df))), "Sample index", "sample_index"


def infer_target_from_name(path):
    name = Path(path).name.lower()

    if "roll" in name:
        return "roll"
    if "pitch" in name:
        return "pitch"
    if "yaw" in name:
        return "yaw"
    if "alt" in name:
        return "altitude"
    if "lat" in name:
        return "latitude"
    if "lng" in name or "lon" in name:
        return "longitude"

    return None


def infer_signal_cols(df, target_hint=None):
    clean_col = None
    attacked_col = None
    recovered_col = None

    terms = []
    if target_hint:
        terms.append(target_hint)

    if target_hint == "longitude":
        terms.extend(["lng", "lon"])
    if target_hint == "latitude":
        terms.extend(["lat"])
    if target_hint == "altitude":
        terms.extend(["alt"])

    for term in terms:
        clean_col = clean_col or find_col(
            df,
            required=[term],
            optional=["clean", "true", "truth", "original", "baseline", "actual"],
            forbidden=["attacked", "attack", "recovered", "recovery", "pred", "residual"]
        )

        attacked_col = attacked_col or find_col(
            df,
            required=[term],
            optional=["attacked", "attack"],
            forbidden=["active", "flag", "residual"]
        )

        recovered_col = recovered_col or find_col(
            df,
            required=[term],
            optional=["recovered", "recovery"],
            forbidden=["flag", "residual"]
        )

    if clean_col is None:
        clean_col = find_col(
            df,
            optional=["clean", "true", "truth", "original", "baseline", "actual"],
            forbidden=["attacked", "attack", "recovered", "recovery", "pred", "residual"]
        )

    if attacked_col is None:
        attacked_col = find_col(
            df,
            optional=["attacked", "attack"],
            forbidden=["active", "flag", "residual"]
        )

    if recovered_col is None:
        recovered_col = find_col(
            df,
            optional=["recovered", "recovery"],
            forbidden=["flag", "residual"]
        )

    return clean_col, attacked_col, recovered_col


def main():
    if not PER_FILE_TABLE.exists():
        raise FileNotFoundError(
            f"Missing {PER_FILE_TABLE}. Run phase6_01_generate_tables.py first."
        )

    df = pd.read_csv(PER_FILE_TABLE)

    plt.rcParams.update({
        "figure.figsize": (9, 5),
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    generated = []

    # Figure 1: Mean RMSE bar
    plt.figure()
    labels = ["Attacked", "Recovered"]
    values = [df["attacked_rmse"].mean(), df["recovered_rmse"].mean()]
    plt.bar(labels, values)
    plt.ylabel("Mean RMSE")
    plt.title("Mean RMSE Before and After Recovery")
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.6f}", ha="center", va="bottom")
    generated.extend(savefig("phase6_mean_rmse_bar"))

    # Figure 2: Mean MAE bar
    plt.figure()
    labels = ["Attacked", "Recovered"]
    values = [df["attacked_mae"].mean(), df["recovered_mae"].mean()]
    plt.bar(labels, values)
    plt.ylabel("Mean MAE")
    plt.title("Mean MAE Before and After Recovery")
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.6f}", ha="center", va="bottom")
    generated.extend(savefig("phase6_mean_mae_bar"))

    # Figure 3: Per-file RMSE improvement
    temp = df.sort_values("rmse_improvement_percent").reset_index(drop=True)
    plt.figure(figsize=(11, 5))
    plt.bar(np.arange(len(temp)), temp["rmse_improvement_percent"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Case index, sorted by RMSE improvement")
    plt.ylabel("RMSE improvement (%)")
    plt.title("Per-File RMSE Improvement After Recovery")
    generated.extend(savefig("phase6_per_file_rmse_improvement"))

    # Figure 4: Per-file MAE improvement
    temp = df.sort_values("mae_improvement_percent").reset_index(drop=True)
    plt.figure(figsize=(11, 5))
    plt.bar(np.arange(len(temp)), temp["mae_improvement_percent"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Case index, sorted by MAE improvement")
    plt.ylabel("MAE improvement (%)")
    plt.title("Per-File MAE Improvement After Recovery")
    generated.extend(savefig("phase6_per_file_mae_improvement"))

    # Figure 5: RMSE scatter
    plt.figure()
    plt.scatter(df["attacked_rmse"], df["recovered_rmse"])
    max_val = max(df["attacked_rmse"].max(), df["recovered_rmse"].max())
    plt.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)
    plt.xlabel("Attacked RMSE")
    plt.ylabel("Recovered RMSE")
    plt.title("Recovered RMSE vs. Attacked RMSE")
    generated.extend(savefig("phase6_rmse_scatter"))

    # Figure 6: MAE scatter
    plt.figure()
    plt.scatter(df["attacked_mae"], df["recovered_mae"])
    max_val = max(df["attacked_mae"].max(), df["recovered_mae"].max())
    plt.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)
    plt.xlabel("Attacked MAE")
    plt.ylabel("Recovered MAE")
    plt.title("Recovered MAE vs. Attacked MAE")
    generated.extend(savefig("phase6_mae_scatter"))

    # Figure 7: RMSE improvement histogram
    plt.figure()
    plt.hist(df["rmse_improvement_percent"].dropna(), bins=15)
    plt.xlabel("RMSE improvement (%)")
    plt.ylabel("Number of cases")
    plt.title("Distribution of RMSE Improvement")
    generated.extend(savefig("phase6_rmse_improvement_histogram"))

    # Figure 8: MAE improvement histogram
    plt.figure()
    plt.hist(df["mae_improvement_percent"].dropna(), bins=15)
    plt.xlabel("MAE improvement (%)")
    plt.ylabel("Number of cases")
    plt.title("Distribution of MAE Improvement")
    generated.extend(savefig("phase6_mae_improvement_histogram"))

    # Figure 9: Per-target boxplot if target labels exist
    if "target" in df.columns and df["target"].nunique() > 1:
        targets = sorted(df["target"].dropna().unique())

        data = [
            df.loc[df["target"] == t, "rmse_improvement_percent"].dropna().values
            for t in targets
        ]

        plt.figure(figsize=(10, 5))
        plt.boxplot(data, labels=targets)
        plt.ylabel("RMSE improvement (%)")
        plt.title("RMSE Improvement by Target")
        plt.xticks(rotation=30, ha="right")
        generated.extend(savefig("phase6_rmse_improvement_by_target"))

        data = [
            df.loc[df["target"] == t, "mae_improvement_percent"].dropna().values
            for t in targets
        ]

        plt.figure(figsize=(10, 5))
        plt.boxplot(data, labels=targets)
        plt.ylabel("MAE improvement (%)")
        plt.title("MAE Improvement by Target")
        plt.xticks(rotation=30, ha="right")
        generated.extend(savefig("phase6_mae_improvement_by_target"))

    # Representative time-series plots
    ts_log = []
    ts_log.append("PHASE 6 REPRESENTATIVE TIME-SERIES FIGURE LOG")
    ts_log.append("==============================================")
    ts_log.append(f"Recovered CSV directory: {RECOVERED_CSV_DIR}")
    ts_log.append("")

    recovered_files = sorted(glob.glob(str(RECOVERED_CSV_DIR / "*.csv")))
    ts_log.append(f"Recovered CSVs found: {len(recovered_files)}")

    made = 0
    max_plots = 12

    for f in recovered_files:
        if made >= max_plots:
            break

        try:
            rdf = pd.read_csv(f)
        except Exception as e:
            ts_log.append(f"SKIP {f}: cannot read CSV: {e}")
            continue

        target_hint = infer_target_from_name(f)
        t, xlabel, time_col = infer_time(rdf)
        clean_col, attacked_col, recovered_col = infer_signal_cols(rdf, target_hint)

        if clean_col is None or attacked_col is None or recovered_col is None:
            ts_log.append(
                f"SKIP {Path(f).name}: could not infer columns. "
                f"time={time_col}, clean={clean_col}, attacked={attacked_col}, recovered={recovered_col}"
            )
            continue

        n = len(rdf)
        if n > 5000:
            idx = np.linspace(0, n - 1, 5000).astype(int)
        else:
            idx = np.arange(n)

        plt.figure(figsize=(12, 5))
        plt.plot(t.iloc[idx], pd.to_numeric(rdf[clean_col], errors="coerce").iloc[idx], label="Clean/reference", linewidth=1.4)
        plt.plot(t.iloc[idx], pd.to_numeric(rdf[attacked_col], errors="coerce").iloc[idx], label="Attacked", linewidth=1.0)
        plt.plot(t.iloc[idx], pd.to_numeric(rdf[recovered_col], errors="coerce").iloc[idx], label="Recovered", linewidth=1.0)

        plt.xlabel(xlabel)
        plt.ylabel(target_hint if target_hint else "Signal value")
        plt.title(f"Representative Recovery Result: {Path(f).stem}")
        plt.legend()

        safe_stem = re.sub(r"[^A-Za-z0-9_\\-]+", "_", Path(f).stem)
        name = f"representative_recovery_{made+1:02d}_{safe_stem}"

        generated.extend(savefig(name))

        ts_log.append(
            f"MADE {name}: file={Path(f).name}, "
            f"time_col={time_col}, clean_col={clean_col}, "
            f"attacked_col={attacked_col}, recovered_col={recovered_col}"
        )

        made += 1

    ts_log.append("")
    ts_log.append(f"Representative plots made: {made}")

    (LOG_DIR / "phase6_02_timeseries_log.txt").write_text("\n".join(ts_log) + "\n")

    figure_manifest = []
    figure_manifest.append("PHASE 6 FIGURE MANIFEST")
    figure_manifest.append("=======================")
    figure_manifest.append("")
    for p in sorted(FIG_DIR.glob("*")):
        figure_manifest.append(f"- {p}")

    (LOG_DIR / "phase6_02_figure_manifest.txt").write_text("\n".join(figure_manifest) + "\n")

    print("Phase 6 figure generation complete.")
    print(f"Figures written to: {FIG_DIR}")
    print(f"Number of files generated: {len(generated)}")
    print("")
    print("Check representative plot log:")
    print(LOG_DIR / "phase6_02_timeseries_log.txt")


if __name__ == "__main__":
    main()
