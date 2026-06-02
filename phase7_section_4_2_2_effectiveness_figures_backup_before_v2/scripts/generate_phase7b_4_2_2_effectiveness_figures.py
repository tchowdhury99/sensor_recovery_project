#!/usr/bin/env python3

"""
Phase 7B: Section 4.2.2-Style Effectiveness Figure Generator

Purpose:
Generate effectiveness/result figures similar in purpose to Section 4.2.2
of "Software-based Realtime Recovery from Sensor Attacks on Robotic Vehicles",
but using this project's ArduPilot SITL V9/p99_abs/W10_N3_mean1x results.

This script is intentionally robust:
- It searches the project for CSV files.
- It creates an audit file showing which CSVs contain useful columns.
- It tries to infer useful columns for prediction, residual, detection, and recovery plots.
- It does not overwrite Phase 6 figures or Phase 7 architecture diagrams.

Output folder:
    /home/tchowdh4/sensor_recovery_project/phase7_section_4_2_2_effectiveness_figures

Generated figures:
    figure_4_2_2_a_v9_prediction_accuracy.png/.pdf
    figure_4_2_2_b_clean_residual_threshold.png/.pdf
    figure_4_2_2_c_window_detection_example.png/.pdf
    figure_4_2_2_d_recovery_timeseries_example.png/.pdf
    figure_4_2_2_e_recovery_improvement_summary.png/.pdf

Generated documentation:
    README.md
    phase7b_4_2_2_latex_snippets.tex
    phase7b_4_2_2_effectiveness_summary.md
    phase7b_csv_audit.csv
    phase7b_generation_log.txt

Important:
If some figure cannot be generated because the required CSV columns are not found,
the script will skip that figure and explain why in the log.
"""

from pathlib import Path
import argparse
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path("/home/tchowdh4/sensor_recovery_project")
OUT_DIR = PROJECT_ROOT / "phase7_section_4_2_2_effectiveness_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / "phase7b_generation_log.txt"


# =============================================================================
# Configuration
# =============================================================================

TARGETS = ["Roll", "Pitch", "Yaw"]

TIME_ALIASES = [
    "time", "time_s", "times", "timestamp", "t", "sec", "seconds",
    "relative_time_s", "relative_time", "TimeS", "TimeUS"
]

TARGET_ALIASES = [
    "target", "axis", "signal", "state", "attitude", "channel"
]

ROLE_ALIASES = {
    "true": [
        "true", "truth", "clean", "actual", "reference", "ground_truth",
        "groundtruth", "real", "baseline", "expected", "x_true", "y_true"
    ],
    "pred": [
        "prediction", "pred", "v9", "xhat", "x_hat", "yhat", "y_hat",
        "estimate", "estimated", "software", "software_sensor", "model"
    ],
    "attacked": [
        "attacked", "attack", "corrupted", "spoofed", "measured",
        "measurement", "sensor", "x_measured", "x_attacked"
    ],
    "recovered": [
        "recovered", "recovery", "corrected", "restored", "x_recovered"
    ],
    "residual": [
        "residual", "error", "abs_residual", "abs_error", "difference", "diff"
    ],
    "detected": [
        "attack_detected", "detected", "detection", "flag",
        "recovery_mode", "is_attack", "attack_flag"
    ],
    "threshold": [
        "threshold", "p99_abs", "p99", "thr", "ton", "limit"
    ],
    "rmse": [
        "rmse"
    ],
    "mae": [
        "mae"
    ],
    "improvement": [
        "improvement", "improve", "percent", "pct", "reduction"
    ],
}


# =============================================================================
# Logging
# =============================================================================

def log(msg):
    print(msg)
    with LOG_PATH.open("a") as f:
        f.write(str(msg) + "\n")


# =============================================================================
# Utility functions
# =============================================================================

def normalize_name(s):
    s = str(s)
    s = s.replace("\\", "")
    s = s.replace("/", "_")
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    s = s.replace(".", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s)
    return s.lower().strip("_")


def list_csv_files():
    skip_parts = {
        ".git",
        "venv-ardupilot",
        "__pycache__",
        "phase7_methodology_figures",
        "phase7_section_4_2_2_effectiveness_figures",
    }

    csvs = []
    for path in PROJECT_ROOT.rglob("*.csv"):
        if any(part in skip_parts for part in path.parts):
            continue

        name = str(path).lower()

        # Prefer relevant project result CSVs.
        relevant_tokens = [
            "v9", "phase5", "phase6", "recovery", "recover",
            "residual", "threshold", "detection", "prediction",
            "attack", "rmse", "mae", "summary"
        ]

        if any(tok in name for tok in relevant_tokens):
            csvs.append(path)

    return sorted(csvs)


def safe_read_csv(path, nrows=None):
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception as e:
        log(f"[WARN] Could not read CSV: {path} -> {e}")
        return None


def get_normalized_columns(df):
    return {normalize_name(c): c for c in df.columns}


def find_time_column(df):
    norm = get_normalized_columns(df)

    for alias in TIME_ALIASES:
        a = normalize_name(alias)
        if a in norm:
            return norm[a]

    for n, original in norm.items():
        if n in ["timeus", "time_usec", "time_microseconds"]:
            return original
        if "time" in n or n in ["t", "sec", "seconds"]:
            return original

    return None


def get_time_vector(df):
    time_col = find_time_column(df)

    if time_col is None:
        return np.arange(len(df), dtype=float)

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)

    # Convert TimeUS-style microseconds to relative seconds.
    if np.nanmax(t) > 1e6:
        t = (t - np.nanmin(t)) / 1_000_000.0
    else:
        t = t - np.nanmin(t)

    # If many NaNs, fallback.
    if np.isnan(t).sum() > 0.5 * len(t):
        return np.arange(len(df), dtype=float)

    return t


def find_target_column(df):
    norm = get_normalized_columns(df)

    for alias in TARGET_ALIASES:
        a = normalize_name(alias)
        if a in norm:
            return norm[a]

    for n, original in norm.items():
        if n in ["target", "axis", "signal", "state", "channel"]:
            return original

    return None


def col_matches_role(col_name, role):
    n = normalize_name(col_name)
    aliases = ROLE_ALIASES.get(role, [])

    for alias in aliases:
        a = normalize_name(alias)
        if a and a in n:
            return True

    return False


def col_matches_target(col_name, target):
    n = normalize_name(col_name)
    t = normalize_name(target)

    # Common abbreviations.
    target_patterns = {
        "roll": ["roll", "rll"],
        "pitch": ["pitch", "pit"],
        "yaw": ["yaw", "heading", "psi"],
    }

    allowed = target_patterns.get(t, [t])
    return any(p in n for p in allowed)


def filter_by_target(df, target):
    target_col = find_target_column(df)

    if target_col is None:
        return df

    vals = df[target_col].astype(str).str.lower()
    mask = vals.str.contains(target.lower(), na=False)

    if mask.sum() == 0:
        return df.iloc[0:0].copy()

    return df.loc[mask].copy()


def find_role_column(df, role, target=None, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = set()

    cols = list(df.columns)

    # First preference: column includes both target and role.
    if target is not None:
        candidates = [
            c for c in cols
            if c not in exclude_cols
            and col_matches_target(c, target)
            and col_matches_role(c, role)
        ]
        if candidates:
            return candidates[0]

    # Second preference: role-only match.
    candidates = [
        c for c in cols
        if c not in exclude_cols
        and col_matches_role(c, role)
    ]

    # Avoid selecting target/time columns accidentally.
    time_col = find_time_column(df)
    target_col = find_target_column(df)

    candidates = [
        c for c in candidates
        if c != time_col and c != target_col
    ]

    if candidates:
        return candidates[0]

    return None


def get_series(df, target, role):
    """
    Return x, y, column_name for a target and semantic role.
    Works for both:
    - long tables with a target/axis column
    - wide tables with Roll/Pitch/Yaw in the column names
    """
    time_col = find_time_column(df)
    target_col = find_target_column(df)

    if target_col is not None:
        sub = filter_by_target(df, target)
    else:
        sub = df

    if len(sub) == 0:
        return None, None, None

    exclude = set()
    if time_col:
        exclude.add(time_col)
    if target_col:
        exclude.add(target_col)

    col = find_role_column(sub, role, target=target if target_col is None else None, exclude_cols=exclude)

    if col is None:
        return None, None, None

    x = get_time_vector(sub)
    y = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(y) < 5:
        return None, None, None

    return x, y, col


def derive_residual(df, target):
    """
    Residual preference:
    1. existing residual column
    2. measured/attacked/true - prediction
    """
    x, r, col = get_series(df, target, "residual")
    if r is not None:
        return x, r, col

    x_pred, pred, pred_col = get_series(df, target, "pred")
    x_true, true, true_col = get_series(df, target, "true")

    if pred is not None and true is not None:
        n = min(len(pred), len(true), len(x_pred), len(x_true))
        x = x_pred[:n]
        r = true[:n] - pred[:n]
        return x, r, f"derived: {true_col} - {pred_col}"

    x_att, attacked, att_col = get_series(df, target, "attacked")

    if pred is not None and attacked is not None:
        n = min(len(pred), len(attacked), len(x_pred), len(x_att))
        x = x_pred[:n]
        r = attacked[:n] - pred[:n]
        return x, r, f"derived: {att_col} - {pred_col}"

    return None, None, None


def downsample_xy(x, y, max_points=2500):
    if x is None or y is None:
        return x, y

    n = len(y)
    if n <= max_points:
        return x, y

    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x[idx], y[idx]


def save_fig(fig, stem):
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"[OK] Saved {png}")
    log(f"[OK] Saved {pdf}")


def has_any_role(df, role):
    for c in df.columns:
        if col_matches_role(c, role):
            return True
    return False


def count_roles(df):
    return {
        role: int(has_any_role(df, role))
        for role in ROLE_ALIASES.keys()
    }


def audit_csvs(csvs):
    rows = []

    for path in csvs:
        df = safe_read_csv(path, nrows=100)
        if df is None:
            continue

        role_counts = count_roles(df)
        rows.append({
            "path": str(path),
            "file": path.name,
            "n_columns": len(df.columns),
            "columns": " | ".join(map(str, df.columns)),
            **role_counts,
        })

    audit = pd.DataFrame(rows)

    audit_path = OUT_DIR / "phase7b_csv_audit.csv"
    audit.to_csv(audit_path, index=False)
    log(f"[OK] Saved CSV audit: {audit_path}")

    return audit


def score_file_for_roles(row, required_roles):
    score = 0

    for role in required_roles:
        score += int(row.get(role, 0)) * 10

    path = str(row.get("path", "")).lower()

    bonus_tokens = [
        "v9", "phase5", "phase6", "recovery", "recover",
        "residual", "threshold", "detection", "prediction",
        "attack", "summary"
    ]

    for tok in bonus_tokens:
        if tok in path:
            score += 1

    return score


def choose_csv(audit, required_roles, preferred_keywords=None):
    if preferred_keywords is None:
        preferred_keywords = []

    if audit is None or len(audit) == 0:
        return None

    candidate = audit.copy()
    candidate["score"] = candidate.apply(lambda r: score_file_for_roles(r, required_roles), axis=1)

    for kw in preferred_keywords:
        candidate.loc[candidate["path"].str.lower().str.contains(kw.lower(), na=False), "score"] += 5

    candidate = candidate.sort_values("score", ascending=False)

    if candidate.iloc[0]["score"] <= 0:
        return None

    return Path(candidate.iloc[0]["path"])


def load_best_dataset(audit, roles, preferred_keywords=None):
    path = choose_csv(audit, roles, preferred_keywords=preferred_keywords)
    if path is None:
        return None, None

    df = safe_read_csv(path)

    if df is None:
        return None, None

    log(f"[INFO] Selected dataset for roles {roles}: {path}")
    return path, df


def find_thresholds(audit):
    """
    Try to find thresholds from threshold CSVs.
    Return dict: target -> threshold.
    """
    thresholds = {}

    if audit is None or len(audit) == 0:
        return thresholds

    candidates = audit.copy()
    candidates["score"] = 0
    candidates.loc[candidates["threshold"] > 0, "score"] += 10
    candidates.loc[candidates["path"].str.lower().str.contains("threshold", na=False), "score"] += 10
    candidates.loc[candidates["path"].str.lower().str.contains("p99", na=False), "score"] += 5
    candidates = candidates.sort_values("score", ascending=False)

    for _, row in candidates.iterrows():
        if row["score"] <= 0:
            continue

        path = Path(row["path"])
        df = safe_read_csv(path)

        if df is None or len(df) == 0:
            continue

        target_col = find_target_column(df)
        threshold_col = find_role_column(df, "threshold")

        if threshold_col is None:
            continue

        if target_col is not None:
            for target in TARGETS:
                sub = df[df[target_col].astype(str).str.lower().str.contains(target.lower(), na=False)]
                if len(sub) > 0:
                    val = pd.to_numeric(sub[threshold_col], errors="coerce").dropna()
                    if len(val) > 0:
                        thresholds[target] = float(val.iloc[0])
        else:
            # Wide threshold columns.
            for target in TARGETS:
                tcol = find_role_column(df, "threshold", target=target)
                if tcol:
                    val = pd.to_numeric(df[tcol], errors="coerce").dropna()
                    if len(val) > 0:
                        thresholds[target] = float(val.iloc[0])

        if thresholds:
            log(f"[INFO] Loaded thresholds from {path}: {thresholds}")
            return thresholds

    return thresholds


def infer_threshold_from_residual(x, r):
    if r is None or len(r) == 0:
        return None
    return float(np.nanquantile(np.abs(r), 0.99))


def shade_boolean_regions(ax, x, flag, color="0.9", label=None):
    """
    Shade contiguous True regions.
    """
    if x is None or flag is None:
        return

    flag = np.asarray(flag).astype(bool)
    if len(flag) != len(x):
        n = min(len(flag), len(x))
        flag = flag[:n]
        x = x[:n]

    in_region = False
    start = None
    added_label = False

    for i, f in enumerate(flag):
        if f and not in_region:
            start = x[i]
            in_region = True
        elif not f and in_region:
            end = x[i]
            ax.axvspan(start, end, alpha=0.25, color=color, label=label if not added_label else None)
            added_label = True
            in_region = False

    if in_region:
        ax.axvspan(start, x[-1], alpha=0.25, color=color, label=label if not added_label else None)


# =============================================================================
# Figure A: V9 prediction accuracy
# =============================================================================

def make_prediction_accuracy(audit, max_points):
    path, df = load_best_dataset(
        audit,
        roles=["pred"],
        preferred_keywords=["prediction", "v9", "phase2", "phase6"]
    )

    if df is None:
        log("[SKIP] Figure A: no prediction dataset found.")
        return False

    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(12, 8), sharex=True)
    if len(TARGETS) == 1:
        axes = [axes]

    plotted = 0

    for ax, target in zip(axes, TARGETS):
        x_true, y_true, true_col = get_series(df, target, "true")
        x_pred, y_pred, pred_col = get_series(df, target, "pred")

        if y_pred is None:
            ax.text(0.5, 0.5, f"No V9 prediction column found for {target}", ha="center", va="center")
            ax.axis("off")
            continue

        # If no true/clean column, try attacked/measured as comparison.
        if y_true is None:
            x_true, y_true, true_col = get_series(df, target, "attacked")

        if y_true is None:
            ax.text(0.5, 0.5, f"No measured/clean column found for {target}", ha="center", va="center")
            ax.axis("off")
            continue

        n = min(len(x_true), len(y_true), len(x_pred), len(y_pred))
        x = x_true[:n]
        y_true = y_true[:n]
        y_pred = y_pred[:n]

        x, y_true = downsample_xy(x, y_true, max_points=max_points)
        _, y_pred = downsample_xy(x_true[:n], y_pred, max_points=max_points)

        ax.plot(x, y_true, linewidth=1.2, label="Measured / clean")
        ax.plot(x, y_pred, linewidth=1.2, linestyle="--", label="V9 prediction")
        ax.set_ylabel(target)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        plotted += 1

    axes[-1].set_xlabel("Time (s) or sample index")
    fig.suptitle("Section 4.2.2-Style Figure A: V9 Software Sensor Prediction Accuracy", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if plotted == 0:
        plt.close(fig)
        log("[SKIP] Figure A: no target could be plotted.")
        return False

    save_fig(fig, "figure_4_2_2_a_v9_prediction_accuracy")
    return True


# =============================================================================
# Figure B: Clean residual threshold
# =============================================================================

def make_clean_residual_threshold(audit, thresholds, max_points):
    path, df = load_best_dataset(
        audit,
        roles=["residual"],
        preferred_keywords=["clean", "residual", "threshold", "v9", "phase3"]
    )

    if df is None:
        # Try prediction dataset and derive residual.
        path, df = load_best_dataset(
            audit,
            roles=["pred"],
            preferred_keywords=["prediction", "v9", "phase2", "phase3"]
        )

    if df is None:
        log("[SKIP] Figure B: no residual or prediction dataset found.")
        return False

    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(13, 8), sharex=False)

    plotted = 0

    for row, target in enumerate(TARGETS):
        ax_ts = axes[row, 0]
        ax_hist = axes[row, 1]

        x, r, source = derive_residual(df, target)

        if r is None:
            ax_ts.text(0.5, 0.5, f"No residual found for {target}", ha="center", va="center")
            ax_hist.axis("off")
            continue

        abs_r = np.abs(r)
        threshold = thresholds.get(target, None)
        if threshold is None:
            threshold = infer_threshold_from_residual(x, r)

        x_plot, abs_plot = downsample_xy(x, abs_r, max_points=max_points)

        ax_ts.plot(x_plot, abs_plot, linewidth=1.0, label="|clean residual|")
        ax_ts.axhline(threshold, linestyle="--", linewidth=1.2, label=f"p99_abs = {threshold:.4g}")
        ax_ts.set_ylabel(f"{target}\n|residual|")
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend(fontsize=8, loc="best")

        ax_hist.hist(abs_r[np.isfinite(abs_r)], bins=60, alpha=0.8)
        ax_hist.axvline(threshold, linestyle="--", linewidth=1.2, label="p99_abs")
        ax_hist.set_ylabel(target)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend(fontsize=8, loc="best")

        plotted += 1

    axes[-1, 0].set_xlabel("Time (s) or sample index")
    axes[-1, 1].set_xlabel("|residual|")

    fig.suptitle("Section 4.2.2-Style Figure B: Clean Residuals and p99_abs Threshold", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if plotted == 0:
        plt.close(fig)
        log("[SKIP] Figure B: no target could be plotted.")
        return False

    save_fig(fig, "figure_4_2_2_b_clean_residual_threshold")
    return True


# =============================================================================
# Figure C: Window detection example
# =============================================================================

def make_window_detection_example(audit, thresholds, max_points):
    path, df = load_best_dataset(
        audit,
        roles=["detected"],
        preferred_keywords=["detection", "attack", "phase4", "v9", "corrected"]
    )

    if df is None:
        # Try residual attack files.
        path, df = load_best_dataset(
            audit,
            roles=["residual"],
            preferred_keywords=["attack", "detection", "residual", "phase4"]
        )

    if df is None:
        log("[SKIP] Figure C: no detection/residual dataset found.")
        return False

    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(12, 8), sharex=True)
    if len(TARGETS) == 1:
        axes = [axes]

    plotted = 0

    for ax, target in zip(axes, TARGETS):
        x, r, source = derive_residual(df, target)

        if r is None:
            ax.text(0.5, 0.5, f"No residual found for {target}", ha="center", va="center")
            ax.axis("off")
            continue

        abs_r = np.abs(r)

        threshold = thresholds.get(target, None)
        if threshold is None:
            threshold = infer_threshold_from_residual(x, r)

        # Detection flag if available.
        x_flag, flag, flag_col = get_series(df, target, "detected")

        if flag is not None:
            n = min(len(x), len(abs_r), len(flag), len(x_flag))
            x_use = x[:n]
            abs_use = abs_r[:n]
            flag_use = flag[:n] > 0
        else:
            # Reconstruct simple W10_N3 flag from threshold.
            violation = abs_r > threshold
            window = 10
            needed = 3
            flag_use = np.zeros(len(violation), dtype=bool)
            for i in range(len(violation)):
                start = max(0, i - window + 1)
                flag_use[i] = violation[start:i+1].sum() >= needed
            x_use = x
            abs_use = abs_r

        x_plot, abs_plot = downsample_xy(x_use, abs_use, max_points=max_points)
        _, flag_plot = downsample_xy(x_use, flag_use.astype(float), max_points=max_points)
        flag_plot = flag_plot > 0.5

        ax.plot(x_plot, abs_plot, linewidth=1.0, label="|attack residual|")
        ax.axhline(threshold, linestyle="--", linewidth=1.2, label=f"p99_abs = {threshold:.4g}")
        shade_boolean_regions(ax, x_plot, flag_plot, color="red", label="detected region")
        ax.set_ylabel(f"{target}\n|residual|")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        plotted += 1

    axes[-1].set_xlabel("Time (s) or sample index")
    fig.suptitle("Section 4.2.2-Style Figure C: W10_N3_mean1x Window Detection Example", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if plotted == 0:
        plt.close(fig)
        log("[SKIP] Figure C: no target could be plotted.")
        return False

    save_fig(fig, "figure_4_2_2_c_window_detection_example")
    return True


# =============================================================================
# Figure D: Recovery time series
# =============================================================================

def make_recovery_timeseries(audit, max_points):
    path, df = load_best_dataset(
        audit,
        roles=["recovered"],
        preferred_keywords=["recovery", "recover", "phase5", "official", "corrected"]
    )

    if df is None:
        log("[SKIP] Figure D: no recovery dataset found.")
        return False

    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(12, 8), sharex=True)
    if len(TARGETS) == 1:
        axes = [axes]

    plotted = 0

    for ax, target in zip(axes, TARGETS):
        x_rec, rec, rec_col = get_series(df, target, "recovered")
        x_att, attacked, att_col = get_series(df, target, "attacked")
        x_true, true, true_col = get_series(df, target, "true")
        x_pred, pred, pred_col = get_series(df, target, "pred")
        x_flag, flag, flag_col = get_series(df, target, "detected")

        if rec is None:
            ax.text(0.5, 0.5, f"No recovered signal found for {target}", ha="center", va="center")
            ax.axis("off")
            continue

        # Use recovered x as base.
        n = len(rec)
        x = x_rec

        series = []

        if true is not None:
            n = min(n, len(true), len(x_true))
            series.append(("clean/reference", true[:n], "-"))

        if attacked is not None:
            n = min(n, len(attacked), len(x_att))
            series.append(("attacked", attacked[:n], "--"))

        if pred is not None:
            n = min(n, len(pred), len(x_pred))
            series.append(("V9 prediction", pred[:n], ":"))

        n = min(n, len(x))
        series.append(("recovered", rec[:n], "-"))

        x = x[:n]

        # Rebuild series after final n.
        for label, y, style in series:
            y = y[:n]
            x_plot, y_plot = downsample_xy(x, y, max_points=max_points)
            ax.plot(x_plot, y_plot, linestyle=style, linewidth=1.1, label=label)

        if flag is not None:
            n_flag = min(len(flag), len(x), n)
            x_flag_use = x[:n_flag]
            flag_use = flag[:n_flag] > 0
            x_flag_plot, flag_plot = downsample_xy(x_flag_use, flag_use.astype(float), max_points=max_points)
            shade_boolean_regions(ax, x_flag_plot, flag_plot > 0.5, color="red", label="detected/recovery region")

        ax.set_ylabel(target)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        plotted += 1

    axes[-1].set_xlabel("Time (s) or sample index")
    fig.suptitle("Section 4.2.2-Style Figure D: Representative Attack Recovery Time Series", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if plotted == 0:
        plt.close(fig)
        log("[SKIP] Figure D: no target could be plotted.")
        return False

    save_fig(fig, "figure_4_2_2_d_recovery_timeseries_example")
    return True


# =============================================================================
# Figure E: Recovery improvement summary
# =============================================================================

def make_recovery_improvement_summary(audit):
    path, df = load_best_dataset(
        audit,
        roles=["rmse"],
        preferred_keywords=["summary", "phase5", "phase6", "recovery", "rmse", "mae"]
    )

    if df is None:
        log("[SKIP] Figure E: no RMSE/MAE summary dataset found.")
        return False

    norm_cols = get_normalized_columns(df)

    def find_metric_col(metric, condition):
        cols = list(df.columns)
        metric = normalize_name(metric)
        condition = normalize_name(condition)

        # Prefer columns containing both.
        for c in cols:
            n = normalize_name(c)
            if metric in n and condition in n:
                return c

        return None

    attacked_rmse_col = find_metric_col("rmse", "attacked")
    recovered_rmse_col = find_metric_col("rmse", "recovered")
    attacked_mae_col = find_metric_col("mae", "attacked")
    recovered_mae_col = find_metric_col("mae", "recovered")

    metrics = []

    if attacked_rmse_col and recovered_rmse_col:
        attacked_rmse = pd.to_numeric(df[attacked_rmse_col], errors="coerce").dropna()
        recovered_rmse = pd.to_numeric(df[recovered_rmse_col], errors="coerce").dropna()
        if len(attacked_rmse) and len(recovered_rmse):
            metrics.append(("RMSE", float(attacked_rmse.mean()), float(recovered_rmse.mean())))

    if attacked_mae_col and recovered_mae_col:
        attacked_mae = pd.to_numeric(df[attacked_mae_col], errors="coerce").dropna()
        recovered_mae = pd.to_numeric(df[recovered_mae_col], errors="coerce").dropna()
        if len(attacked_mae) and len(recovered_mae):
            metrics.append(("MAE", float(attacked_mae.mean()), float(recovered_mae.mean())))

    if not metrics:
        log("[SKIP] Figure E: RMSE/MAE columns not recognized in selected summary file.")
        log(f"[INFO] Selected file was: {path}")
        log(f"[INFO] Columns were: {list(df.columns)}")
        return False

    labels = [m[0] for m in metrics]
    attacked_vals = [m[1] for m in metrics]
    recovered_vals = [m[2] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, attacked_vals, width, label="Attacked")
    ax.bar(x + width / 2, recovered_vals, width, label="Recovered")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean error")
    ax.set_title("Section 4.2.2-Style Figure E: Recovery Improvement Summary", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for i, (label, a, r) in enumerate(metrics):
        if a != 0:
            improvement = 100.0 * (a - r) / a
            ax.text(i, max(a, r) * 1.02, f"{improvement:.1f}% improvement", ha="center", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "figure_4_2_2_e_recovery_improvement_summary")
    return True


# =============================================================================
# Documentation
# =============================================================================

def write_readme(generated):
    text = f"""# Phase 7B: Section 4.2.2-Style Effectiveness Figures

This folder contains result/effectiveness figures modeled after Section 4.2.2 of the reference paper, but generated using this project's ArduPilot SITL V9 software-sensor recovery results.

## Generated status

{generated}

## Intended mapping to the paper

- Figure A corresponds to the paper's software sensor prediction figure.
- Figure B corresponds to clean residual and threshold behavior.
- Figure C corresponds to residual/window-based attack detection.
- Figure D corresponds to attack recovery time-series behavior.
- Figure E summarizes quantitative recovery improvement using RMSE/MAE.

## Project-specific configuration

- Software sensor: V9 trained hybrid
- Prediction target: future state x_hat[k+1] using information available at time k
- Threshold: p99_abs
- Detector: W10_N3_mean1x
- Recovery rule:

    if attack_detected:
        recovered[k+1] = V9_prediction[k+1]
    else:
        recovered[k+1] = attacked_measurement[k+1]

## Important note

These figures should be described as Section 4.2.2-style reproduction figures, not exact duplicates of the original paper. The original paper used its own robotic vehicle datasets and state-space software sensors. This project uses ArduPilot SITL logs and the selected V9 trained hybrid software sensor.
"""
    path = OUT_DIR / "README.md"
    path.write_text(text)
    log(f"[OK] Saved {path}")


def write_latex():
    text = r"""\subsection{Section 4.2.2-Style Effectiveness Evaluation}

The following figures reproduce the effectiveness-evaluation style of Section 4.2.2 using the ArduPilot SITL dataset and the final V9/p99\_abs/W10\_N3\_mean1x recovery configuration.

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_section_4_2_2_effectiveness_figures/figure_4_2_2_a_v9_prediction_accuracy.pdf}
    \caption{V9 software-sensor prediction accuracy under clean or representative logs. The V9 predictor estimates the future attitude state using information available at time $k$.}
    \label{fig:v9_prediction_accuracy}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_section_4_2_2_effectiveness_figures/figure_4_2_2_b_clean_residual_threshold.pdf}
    \caption{Clean residual behavior and $p99\_abs$ threshold. The threshold is estimated from clean residuals and later used to identify abnormal deviations under attack.}
    \label{fig:clean_residual_threshold}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_section_4_2_2_effectiveness_figures/figure_4_2_2_c_window_detection_example.pdf}
    \caption{Window-based detection using W10\_N3\_mean1x. The detector declares an attack when at least three threshold violations occur inside a ten-sample window.}
    \label{fig:window_detection_example}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{phase7_section_4_2_2_effectiveness_figures/figure_4_2_2_d_recovery_timeseries_example.pdf}
    \caption{Representative attack recovery time series. During detected attack periods, the corrupted measurement is replaced by the V9 software-sensor prediction.}
    \label{fig:recovery_timeseries_example}
\end{figure*}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.48\textwidth]{phase7_section_4_2_2_effectiveness_figures/figure_4_2_2_e_recovery_improvement_summary.pdf}
    \caption{Mean recovery improvement summary. The recovered signal reduces RMSE and MAE compared with the attacked measurement.}
    \label{fig:recovery_improvement_summary}
\end{figure}
"""
    path = OUT_DIR / "phase7b_4_2_2_latex_snippets.tex"
    path.write_text(text)
    log(f"[OK] Saved {path}")


def write_summary():
    text = """# Phase 7B Section 4.2.2-Style Effectiveness Summary

The original paper's Section 4.2.2 is an effectiveness evaluation section. It does not only present a block diagram. It shows whether the software sensor predicts real readings, whether error correction reduces prediction error, whether recovery parameters are selected appropriately, whether the system recovers from sensor attacks, and how the technique behaves under environmental and attack-scale variation.

This project's Phase 7B figures follow the same evaluation logic using the ArduPilot SITL reproduction results.

## Figure A: V9 prediction accuracy

This figure shows whether the V9 trained hybrid software sensor tracks the measured or clean attitude signal. It is the closest equivalent to the paper's software-sensor prediction figure.

## Figure B: Clean residual threshold

This figure shows normal residual behavior and the p99_abs threshold. It explains how the detection threshold is calibrated from clean behavior.

## Figure C: Window detection example

This figure shows how residual threshold violations become an attack decision under the W10_N3_mean1x detector.

## Figure D: Recovery time series

This figure shows the core recovery effect: attacked signal versus recovered signal, with detected/recovery regions if available.

## Figure E: Recovery improvement summary

This figure summarizes quantitative improvement using mean attacked and recovered RMSE/MAE values when the required summary columns are available.

## Professor-facing wording

These are Section 4.2.2-style effectiveness figures. They do not duplicate the original paper's data. Instead, they reproduce the same evaluation structure using my ArduPilot SITL dataset and my final configuration: V9 trained hybrid predictor, p99_abs threshold, W10_N3_mean1x detector, and conditional replacement recovery.
"""
    path = OUT_DIR / "phase7b_4_2_2_effectiveness_summary.md"
    path.write_text(text)
    log(f"[OK] Saved {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true", help="Only create CSV audit; do not generate figures.")
    parser.add_argument("--max-points", type=int, default=2500, help="Maximum plotted points per line.")
    args = parser.parse_args()

    if LOG_PATH.exists():
        LOG_PATH.unlink()

    warnings.filterwarnings("ignore", category=UserWarning)

    log("============================================================")
    log("Phase 7B: Section 4.2.2-Style Effectiveness Figure Generation")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Output directory: {OUT_DIR}")
    log("============================================================")

    csvs = list_csv_files()
    log(f"[INFO] Found {len(csvs)} relevant CSV files.")

    if len(csvs) == 0:
        log("[ERROR] No relevant CSV files found.")
        log("[HINT] Check whether Phase 3/4/5/6 result CSVs exist under the project directory.")
        sys.exit(1)

    audit = audit_csvs(csvs)

    if args.audit_only:
        log("[DONE] Audit-only mode complete.")
        return

    thresholds = find_thresholds(audit)
    if not thresholds:
        log("[WARN] No explicit p99_abs threshold CSV found. The script will infer p99 thresholds from residuals where needed.")

    generated_status = []

    status = make_prediction_accuracy(audit, max_points=args.max_points)
    generated_status.append(f"- Figure A V9 prediction accuracy: {'generated' if status else 'skipped'}")

    status = make_clean_residual_threshold(audit, thresholds, max_points=args.max_points)
    generated_status.append(f"- Figure B clean residual threshold: {'generated' if status else 'skipped'}")

    status = make_window_detection_example(audit, thresholds, max_points=args.max_points)
    generated_status.append(f"- Figure C window detection example: {'generated' if status else 'skipped'}")

    status = make_recovery_timeseries(audit, max_points=args.max_points)
    generated_status.append(f"- Figure D recovery time series: {'generated' if status else 'skipped'}")

    status = make_recovery_improvement_summary(audit)
    generated_status.append(f"- Figure E recovery improvement summary: {'generated' if status else 'skipped'}")

    generated_text = "\n".join(generated_status)

    write_readme(generated_text)
    write_latex()
    write_summary()

    log("============================================================")
    log("Phase 7B generation completed.")
    log("Generated status:")
    log(generated_text)
    log("============================================================")


if __name__ == "__main__":
    main()