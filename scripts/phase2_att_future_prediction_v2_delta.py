import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/sensor_recovery_project")
CSV_DIR = os.path.join(BASE, "logs", "extracted_csv")

MODEL_DIR = os.path.join(BASE, "models_v2_delta")
RESULT_DIR = os.path.join(BASE, "prediction_results_v2_delta")
FIG_DIR = os.path.join(BASE, "figures", "future_prediction_v2_delta")

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

LAGS = 8
RIDGE_LAMBDA = 1e-2
TRAIN_RATIO = 0.70


def safe_name(name):
    return name.replace(".csv", "")


def make_time_column(df):
    if "TimeUS" in df.columns:
        return (df["TimeUS"] - df["TimeUS"].iloc[0]) / 1_000_000.0
    if "TimeSec" in df.columns:
        return df["TimeSec"] - df["TimeSec"].iloc[0]
    return pd.Series(np.arange(len(df)), name="sample_index")


def angle_diff_deg(a, b):
    """
    Small signed angular difference a - b in degrees.
    Used for yaw delta if needed.
    """
    return (a - b + 180.0) % 360.0 - 180.0


def build_delta_dataset(df, feature_cols, target_col, lags):
    """
    Dataset for one-step-ahead delta prediction.

    Features at time k:
      feature[k], feature[k-1], ..., feature[k-lags+1]

    Target:
      delta_y[k+1] = y[k+1] - y[k]

    Final prediction:
      y_hat[k+1] = y[k] + predicted_delta
    """
    X_rows = []
    delta_rows = []
    y_now_rows = []
    y_future_rows = []
    t_rows = []

    feature_values = df[feature_cols].to_numpy(dtype=float)
    y_values = df[target_col].to_numpy(dtype=float)
    t_values = df["t"].to_numpy(dtype=float)

    for k in range(lags - 1, len(df) - 1):
        row = []

        for lag in range(lags):
            idx = k - lag
            row.extend(feature_values[idx, :])

        y_now = y_values[k]
        y_future = y_values[k + 1]

        if target_col == "Yaw":
            delta = angle_diff_deg(y_future, y_now)
        else:
            delta = y_future - y_now

        X_rows.append(row)
        delta_rows.append(delta)
        y_now_rows.append(y_now)
        y_future_rows.append(y_future)
        t_rows.append(t_values[k + 1])

    feature_names = []
    for lag in range(lags):
        for col in feature_cols:
            feature_names.append(f"{col}_k-{lag}")

    return (
        np.asarray(X_rows),
        np.asarray(delta_rows),
        np.asarray(y_now_rows),
        np.asarray(y_future_rows),
        np.asarray(t_rows),
        feature_names
    )


def standardize_train_test(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0.0] = 1.0

    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def fit_ridge(X, y, lam):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])

    I = np.eye(Xb.shape[1])
    I[0, 0] = 0.0

    beta = np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ y)
    return beta


def predict(X, beta):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return Xb @ beta


def compute_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum(err ** 2) / denom)

    return mae, rmse, r2


def plot_prediction(t, y_true, y_model, y_naive, y_cv, title, out_path):
    plt.figure(figsize=(14, 6))
    plt.plot(t, y_true, label="True future value", linewidth=2)
    plt.plot(t, y_model, label="Delta system-ID prediction", linewidth=1.5)
    plt.plot(t, y_naive, label="Naive y[k]", linestyle="--", linewidth=1.0)
    plt.plot(t, y_cv, label="Constant-velocity baseline", linestyle=":", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def process_file(path):
    fname = os.path.basename(path)
    base = safe_name(fname)

    print("\n============================================================")
    print(f"Processing: {fname}")
    print("============================================================")

    df = pd.read_csv(path)

    if not all(c in df.columns for c in TARGETS):
        print("[SKIP] Missing Roll/Pitch/Yaw")
        return []

    # ATT only. Do not process XKF1 here.
    if "_ATT" not in fname:
        print("[SKIP] Not an ATT file")
        return []

    df["t"] = make_time_column(df)

    input_cols = [c for c in CANDIDATE_INPUT_COLS if c in df.columns]
    feature_cols = STATE_COLS + input_cols

    keep_cols = ["t"] + feature_cols
    df = df[keep_cols].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    print(f"Rows after cleaning: {len(df)}")
    print(f"Features: {feature_cols}")
    print(f"Lags: {LAGS}")
    print("Model type: delta one-step-ahead system-ID")

    if len(df) < 200:
        print("[SKIP] Not enough data")
        return []

    rows = []
    pred_out = pd.DataFrame()

    for target in TARGETS:
        print(f"\n--- Target: {target}[k+1] ---")

        X, delta_y, y_now, y_future, t, feature_names = build_delta_dataset(
            df=df,
            feature_cols=feature_cols,
            target_col=target,
            lags=LAGS
        )

        split = int(TRAIN_RATIO * len(X))

        X_train = X[:split]
        X_test = X[split:]

        delta_train = delta_y[:split]
        delta_test = delta_y[split:]

        y_now_test = y_now[split:]
        y_future_test = y_future[split:]
        t_test = t[split:]

        # Baseline 1: y[k+1] = y[k]
        y_naive = y_now_test

        # Baseline 2: constant velocity:
        # y[k+1] = y[k] + (y[k] - y[k-1])
        y_values = df[target].to_numpy(dtype=float)
        cv_all = []
        for k in range(LAGS - 1, len(df) - 1):
            if target == "Yaw":
                prev_delta = angle_diff_deg(y_values[k], y_values[k - 1])
            else:
                prev_delta = y_values[k] - y_values[k - 1]
            cv_all.append(y_values[k] + prev_delta)
        y_cv = np.asarray(cv_all)[split:]

        X_train_s, X_test_s, mu, sigma = standardize_train_test(X_train, X_test)

        beta = fit_ridge(X_train_s, delta_train, RIDGE_LAMBDA)

        delta_pred = predict(X_test_s, beta)
        y_model = y_now_test + delta_pred

        model_mae, model_rmse, model_r2 = compute_metrics(y_future_test, y_model)
        naive_mae, naive_rmse, naive_r2 = compute_metrics(y_future_test, y_naive)
        cv_mae, cv_rmse, cv_r2 = compute_metrics(y_future_test, y_cv)

        print(f"Model RMSE: {model_rmse:.8f}")
        print(f"Naive RMSE: {naive_rmse:.8f}")
        print(f"CV RMSE   : {cv_rmse:.8f}")

        improvement_vs_naive = naive_rmse - model_rmse
        improvement_pct = 100.0 * improvement_vs_naive / naive_rmse if naive_rmse > 1e-12 else float("nan")

        print(f"Improvement vs naive: {improvement_pct:.2f}%")

        rows.append({
            "file": fname,
            "target": target,
            "rows_after_cleaning": len(df),
            "samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "lags": LAGS,
            "feature_columns": ",".join(feature_cols),
            "model_MAE": model_mae,
            "model_RMSE": model_rmse,
            "model_R2": model_r2,
            "naive_MAE": naive_mae,
            "naive_RMSE": naive_rmse,
            "naive_R2": naive_r2,
            "constant_velocity_MAE": cv_mae,
            "constant_velocity_RMSE": cv_rmse,
            "constant_velocity_R2": cv_r2,
            "rmse_improvement_over_naive": improvement_vs_naive,
            "rmse_improvement_percent_over_naive": improvement_pct
        })

        pred_out[f"t_{target}"] = t_test
        pred_out[f"{target}_true_future"] = y_future_test
        pred_out[f"{target}_predicted_future_delta_system_id"] = y_model
        pred_out[f"{target}_naive_yk"] = y_naive
        pred_out[f"{target}_constant_velocity"] = y_cv
        pred_out[f"{target}_model_error"] = y_model - y_future_test
        pred_out[f"{target}_naive_error"] = y_naive - y_future_test

        model = {
            "file": fname,
            "target": target,
            "model_type": "delta_one_step_ahead_ridge_arx",
            "feature_cols": feature_cols,
            "feature_names": feature_names,
            "lags": LAGS,
            "ridge_lambda": RIDGE_LAMBDA,
            "mu": mu,
            "sigma": sigma,
            "beta": beta
        }

        model_path = os.path.join(MODEL_DIR, f"{base}_{target}_delta_one_step_system_id_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        fig_path = os.path.join(FIG_DIR, f"{base}_{target}_v2_delta_future_prediction.png")
        plot_prediction(
            t=t_test,
            y_true=y_future_test,
            y_model=y_model,
            y_naive=y_naive,
            y_cv=y_cv,
            title=f"{fname}: {target}[k+1] delta system-ID prediction",
            out_path=fig_path
        )

    out_path = os.path.join(RESULT_DIR, f"{base}_v2_delta_prediction_outputs.csv")
    pred_out.to_csv(out_path, index=False)

    return rows


def main():
    print("\n========== PHASE 2 V2: ATT DELTA FUTURE PREDICTION ==========")
    print("Only ATT files are processed.")
    print("No attack detection. No recovery.")

    files = sorted([
        os.path.join(CSV_DIR, f)
        for f in os.listdir(CSV_DIR)
        if f.endswith(".csv") and "_ATT" in f
    ])

    if not files:
        raise FileNotFoundError("No ATT CSV files found.")

    all_rows = []

    for path in files:
        all_rows.extend(process_file(path))

    summary = pd.DataFrame(all_rows)
    summary_path = os.path.join(RESULT_DIR, "phase2_v2_att_delta_future_prediction_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n========== V2 SUMMARY ==========")
    show_cols = [
        "file",
        "target",
        "model_RMSE",
        "naive_RMSE",
        "constant_velocity_RMSE",
        "rmse_improvement_percent_over_naive",
        "model_R2"
    ]
    print(summary[show_cols].to_string(index=False))

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved models: {MODEL_DIR}")
    print(f"Saved figures: {FIG_DIR}")


if __name__ == "__main__":
    main()