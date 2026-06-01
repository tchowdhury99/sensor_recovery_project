import os
import json
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/sensor_recovery_project")
CSV_DIR = os.path.join(BASE, "logs", "extracted_csv")

MODEL_DIR = os.path.join(BASE, "models")
RESULT_DIR = os.path.join(BASE, "prediction_results")
FIG_DIR = os.path.join(BASE, "figures", "future_prediction")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TARGETS = ["Roll", "Pitch", "Yaw"]

STATE_COLS = ["Roll", "Pitch", "Yaw"]

CANDIDATE_INPUT_COLS = [
    "DesRoll",
    "DesPitch",
    "DesYaw",
    "ErrRP",
    "ErrYaw"
]

LAGS = 5
RIDGE_LAMBDA = 1e-4
TRAIN_RATIO = 0.70


def safe_name(name):
    return name.replace(".csv", "").replace("/", "_")


def unwrap_angle_degrees(series):
    """
    ArduPilot ATT angles are usually in degrees.
    Yaw can wrap near 0/360, so unwrap it for learning continuity.
    """
    rad = np.deg2rad(series.to_numpy(dtype=float))
    unwrapped_rad = np.unwrap(rad)
    return np.rad2deg(unwrapped_rad)


def wrap_360(angle_deg):
    """
    Wrap degrees to [0, 360).
    Used only for output visualization of yaw if needed.
    """
    return np.mod(angle_deg, 360.0)


def angle_error_deg(pred, true):
    """
    Smallest signed angular error in degrees.
    Useful for yaw.
    """
    return (pred - true + 180.0) % 360.0 - 180.0


def make_time_column(df):
    if "TimeUS" in df.columns:
        return (df["TimeUS"] - df["TimeUS"].iloc[0]) / 1_000_000.0
    elif "TimeSec" in df.columns:
        return df["TimeSec"] - df["TimeSec"].iloc[0]
    else:
        return pd.Series(np.arange(len(df)), name="sample_index")


def build_supervised_dataset(df, feature_cols, target_col, lags):
    """
    Build one-step-ahead dataset.

    X[k] contains:
      feature[k], feature[k-1], ..., feature[k-lags+1]

    y[k] contains:
      target[k+1]

    So the model learns future-state prediction, not present-state prediction.
    """
    X_rows = []
    y_rows = []
    time_rows = []

    values = df[feature_cols].to_numpy(dtype=float)
    target = df[target_col].to_numpy(dtype=float)
    time_values = df["t"].to_numpy(dtype=float)

    for k in range(lags - 1, len(df) - 1):
        row = []

        for lag in range(lags):
            idx = k - lag
            row.extend(values[idx, :])

        X_rows.append(row)
        y_rows.append(target[k + 1])
        time_rows.append(time_values[k + 1])

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    t = np.asarray(time_rows, dtype=float)

    feature_names = []
    for lag in range(lags):
        for col in feature_cols:
            feature_names.append(f"{col}_k-{lag}")

    return X, y, t, feature_names


def standardize_train_test(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0

    X_train_s = (X_train - mu) / sigma
    X_test_s = (X_test - mu) / sigma

    return X_train_s, X_test_s, mu, sigma


def fit_ridge_regression(X, y, ridge_lambda):
    """
    Closed-form ridge regression using numpy only.

    Model:
      y_hat = beta0 + beta1*x1 + ... + betan*xn
    """
    ones = np.ones((X.shape[0], 1))
    Xb = np.hstack([ones, X])

    I = np.eye(Xb.shape[1])
    I[0, 0] = 0.0  # do not regularize intercept

    beta = np.linalg.solve(Xb.T @ Xb + ridge_lambda * I, Xb.T @ y)
    return beta


def predict_linear(X, beta):
    ones = np.ones((X.shape[0], 1))
    Xb = np.hstack([ones, X])
    return Xb @ beta


def metrics(y_true, y_pred, target_name):
    err = y_pred - y_true

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum(err ** 2) / denom)

    return {
        "target": target_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def plot_prediction(t, y_true, y_pred, y_naive, title, ylabel, out_path):
    plt.figure(figsize=(14, 6))
    plt.plot(t, y_true, label="True future value", linewidth=2)
    plt.plot(t, y_pred, label="System-ID one-step prediction", linewidth=1.5)
    plt.plot(t, y_naive, label="Naive previous-value baseline", linewidth=1.0, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error(t, y_true, y_pred, y_naive, title, ylabel, out_path):
    model_err = y_pred - y_true
    naive_err = y_naive - y_true

    plt.figure(figsize=(14, 5))
    plt.plot(t, model_err, label="System-ID prediction error", linewidth=1.5)
    plt.plot(t, naive_err, label="Naive baseline error", linewidth=1.0, linestyle="--")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def process_att_file(csv_path):
    fname = os.path.basename(csv_path)
    base_name = safe_name(fname)

    print("\n============================================================")
    print(f"Processing ATT file: {fname}")
    print("============================================================")

    df = pd.read_csv(csv_path)

    required = ["Roll", "Pitch", "Yaw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[SKIP] Missing required ATT columns: {missing}")
        return []

    df = df.copy()
    df["t"] = make_time_column(df)

    # Keep only useful columns that exist.
    input_cols = [c for c in CANDIDATE_INPUT_COLS if c in df.columns]
    feature_cols = STATE_COLS + input_cols

    print(f"Feature columns used: {feature_cols}")
    print(f"Prediction targets: {TARGETS}")
    print(f"Lags used: {LAGS}")
    print("Prediction horizon: one sample ahead, y[k+1]")

    # Clean numeric data.
    keep_cols = ["t"] + feature_cols
    df = df[keep_cols].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if len(df) < 100:
        print("[SKIP] Not enough clean rows after filtering.")
        return []

    # Unwrap yaw internally for learning.
    # Roll and Pitch are usually continuous enough.
    if "Yaw" in df.columns:
        df["Yaw_raw_deg"] = df["Yaw"]
        df["Yaw"] = unwrap_angle_degrees(df["Yaw"])

    all_metrics = []
    prediction_output = pd.DataFrame()

    for target in TARGETS:
        print(f"\n--- Target: future {target}[k+1] ---")

        X, y, t, feature_names = build_supervised_dataset(
            df=df,
            feature_cols=feature_cols,
            target_col=target,
            lags=LAGS
        )

        # Naive baseline: y_hat[k+1] = y[k]
        # Because dataset starts at k = lags - 1, previous target is target[k].
        target_values = df[target].to_numpy(dtype=float)
        naive = []
        for k in range(LAGS - 1, len(df) - 1):
            naive.append(target_values[k])
        naive = np.asarray(naive, dtype=float)

        split = int(TRAIN_RATIO * len(X))

        X_train = X[:split]
        y_train = y[:split]

        X_test = X[split:]
        y_test = y[split:]
        t_test = t[split:]
        naive_test = naive[split:]

        X_train_s, X_test_s, mu, sigma = standardize_train_test(X_train, X_test)

        beta = fit_ridge_regression(X_train_s, y_train, RIDGE_LAMBDA)

        y_pred_train = predict_linear(X_train_s, beta)
        y_pred_test = predict_linear(X_test_s, beta)

        model_metrics = metrics(y_test, y_pred_test, target)
        naive_metrics = metrics(y_test, naive_test, target + "_naive_previous_value")

        print("System-ID model metrics:")
        print(model_metrics)

        print("Naive previous-value baseline metrics:")
        print(naive_metrics)

        improvement_rmse = naive_metrics["RMSE"] - model_metrics["RMSE"]
        improvement_percent = 100.0 * improvement_rmse / naive_metrics["RMSE"] if naive_metrics["RMSE"] > 1e-12 else float("nan")

        print(f"RMSE improvement over naive: {improvement_rmse:.6f} deg")
        print(f"RMSE improvement percent: {improvement_percent:.2f}%")

        result_row = {
            "file": fname,
            "target": target,
            "rows_total_after_cleaning": len(df),
            "supervised_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "lags": LAGS,
            "feature_columns": ",".join(feature_cols),
            "model_MAE": model_metrics["MAE"],
            "model_RMSE": model_metrics["RMSE"],
            "model_R2": model_metrics["R2"],
            "naive_MAE": naive_metrics["MAE"],
            "naive_RMSE": naive_metrics["RMSE"],
            "naive_R2": naive_metrics["R2"],
            "rmse_improvement_deg": improvement_rmse,
            "rmse_improvement_percent": improvement_percent
        }

        all_metrics.append(result_row)

        prediction_output[f"t_test_{target}"] = t_test
        prediction_output[f"{target}_true_future"] = y_test
        prediction_output[f"{target}_predicted_future_system_id"] = y_pred_test
        prediction_output[f"{target}_naive_previous_value"] = naive_test
        prediction_output[f"{target}_prediction_error"] = y_pred_test - y_test
        prediction_output[f"{target}_naive_error"] = naive_test - y_test

        model_object = {
            "file": fname,
            "target": target,
            "feature_cols": feature_cols,
            "feature_names": feature_names,
            "lags": LAGS,
            "ridge_lambda": RIDGE_LAMBDA,
            "train_ratio": TRAIN_RATIO,
            "mu": mu,
            "sigma": sigma,
            "beta": beta
        }

        model_path = os.path.join(MODEL_DIR, f"{base_name}_{target}_one_step_system_id_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_object, f)

        fig_path = os.path.join(FIG_DIR, f"{base_name}_{target}_future_prediction.png")
        plot_prediction(
            t=t_test,
            y_true=y_test,
            y_pred=y_pred_test,
            y_naive=naive_test,
            title=f"{fname}: one-step-ahead future {target} prediction",
            ylabel=f"{target} angle, degrees",
            out_path=fig_path
        )

        err_fig_path = os.path.join(FIG_DIR, f"{base_name}_{target}_prediction_error.png")
        plot_error(
            t=t_test,
            y_true=y_test,
            y_pred=y_pred_test,
            y_naive=naive_test,
            title=f"{fname}: one-step-ahead future {target} prediction error",
            ylabel=f"{target} error, degrees",
            out_path=err_fig_path
        )

    pred_path = os.path.join(RESULT_DIR, f"{base_name}_future_prediction_test_outputs.csv")
    prediction_output.to_csv(pred_path, index=False)

    print(f"\nSaved prediction output CSV: {pred_path}")
    print(f"Saved figures to: {FIG_DIR}")
    print(f"Saved models to: {MODEL_DIR}")

    return all_metrics


def main():
    print("\n========== PHASE 2: ATT FUTURE-STATE PREDICTION ==========")
    print("Method: system-identification-style one-step-ahead linear ARX model")
    print("Important: This predicts y[k+1], not y[k].")

    if not os.path.isdir(CSV_DIR):
        raise FileNotFoundError(f"CSV directory not found: {CSV_DIR}")

    csv_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith(".csv")])

    att_files = []
    for f in csv_files:
        path = os.path.join(CSV_DIR, f)
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue

        if all(c in header for c in ["Roll", "Pitch", "Yaw"]):
            att_files.append(path)

    if not att_files:
        raise FileNotFoundError("No ATT-style CSV files with Roll, Pitch, Yaw were found.")

    print("\nATT files selected:")
    for path in att_files:
        print(" -", os.path.basename(path))

    all_rows = []

    for path in att_files:
        rows = process_att_file(path)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No prediction results were produced.")

    summary = pd.DataFrame(all_rows)

    summary_path = os.path.join(RESULT_DIR, "phase2_att_future_prediction_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n========== PHASE 2 COMPLETE ==========")
    print(f"Summary saved to: {summary_path}")

    print("\n========== SUMMARY TABLE ==========")
    cols = [
        "file",
        "target",
        "model_RMSE",
        "naive_RMSE",
        "rmse_improvement_deg",
        "rmse_improvement_percent",
        "model_R2"
    ]
    print(summary[cols].to_string(index=False))

    print("\nNext step after this phase:")
    print("Only after these future predictions look reasonable, use the predictor inside attack detection/recovery.")


if __name__ == "__main__":
    main()
