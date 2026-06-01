import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/sensor_recovery_project")
CSV_DIR = os.path.join(BASE, "logs", "extracted_csv")

MODEL_DIR = os.path.join(BASE, "models_v5_trained_arx")
RESULT_DIR = os.path.join(BASE, "prediction_results_v5_trained_arx")
FIG_DIR = os.path.join(BASE, "figures", "future_prediction_v5_trained_arx")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TARGETS = ["Roll", "Pitch", "Yaw"]
STATE_COLS = ["Roll", "Pitch", "Yaw"]

INPUT_COLS_CANDIDATE = [
    "DesRoll",
    "DesPitch",
    "DesYaw"
]

# Use a small model first. Too many lags caused overfitting before.
DELTA_LAGS = 3
INPUT_LAGS = 2

TRAIN_RATIO = 0.70
RIDGE_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]


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
    Mainly useful for Yaw.
    """
    return (a - b + 180.0) % 360.0 - 180.0


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


def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    return mu, sigma


def standardize_apply(X, mu, sigma):
    return (X - mu) / sigma


def fit_ridge(X, y, lam):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])

    I = np.eye(Xb.shape[1])
    I[0, 0] = 0.0

    beta = np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ y)
    return beta


def predict_ridge(X, beta):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return Xb @ beta


def build_trained_arx_dataset(df, input_cols, target_col):
    """
    Training-based ARX/system-identification dataset.

    We do NOT use raw x[k] directly as the main learned quantity.
    We learn the one-step change:

        delta_x[k+1] = x[k+1] - x[k]

    Features:
        delta Roll[k], delta Roll[k-1], ...
        delta Pitch[k], delta Pitch[k-1], ...
        delta Yaw[k], delta Yaw[k-1], ...
        DesRoll[k], DesRoll[k-1], ...
        DesPitch[k], DesPitch[k-1], ...
        DesYaw[k], DesYaw[k-1], ...

    Prediction:
        x_hat[k+1] = x[k] + learned_delta
    """
    feature_names = []
    X_rows = []
    y_delta_rows = []
    y_now_rows = []
    y_future_rows = []
    y_naive_rows = []
    y_cv_rows = []
    t_rows = []

    state = {}
    delta_state = {}

    for col in STATE_COLS:
        vals = df[col].to_numpy(dtype=float)
        state[col] = vals

        d = np.zeros_like(vals)
        for i in range(1, len(vals)):
            if col == "Yaw":
                d[i] = angle_diff_deg(vals[i], vals[i - 1])
            else:
                d[i] = vals[i] - vals[i - 1]

        delta_state[col] = d

    input_values = {}
    for col in input_cols:
        input_values[col] = df[col].to_numpy(dtype=float)

    y_values = df[target_col].to_numpy(dtype=float)
    t_values = df["t"].to_numpy(dtype=float)

    start_k = max(DELTA_LAGS, INPUT_LAGS)

    for k in range(start_k, len(df) - 1):
        row = []

        # State-change history
        for lag in range(DELTA_LAGS):
            idx = k - lag
            for col in STATE_COLS:
                row.append(delta_state[col][idx])

        # Command/input history
        for lag in range(INPUT_LAGS):
            idx = k - lag
            for col in input_cols:
                row.append(input_values[col][idx])

        y_prev = y_values[k - 1]
        y_now = y_values[k]
        y_future = y_values[k + 1]

        if target_col == "Yaw":
            true_delta = angle_diff_deg(y_future, y_now)
            prev_delta = angle_diff_deg(y_now, y_prev)
        else:
            true_delta = y_future - y_now
            prev_delta = y_now - y_prev

        y_naive = y_now
        y_cv = y_now + prev_delta

        X_rows.append(row)
        y_delta_rows.append(true_delta)
        y_now_rows.append(y_now)
        y_future_rows.append(y_future)
        y_naive_rows.append(y_naive)
        y_cv_rows.append(y_cv)
        t_rows.append(t_values[k + 1])

    for lag in range(DELTA_LAGS):
        for col in STATE_COLS:
            feature_names.append(f"delta_{col}_k-{lag}")

    for lag in range(INPUT_LAGS):
        for col in input_cols:
            feature_names.append(f"{col}_k-{lag}")

    return {
        "X": np.asarray(X_rows),
        "y_delta": np.asarray(y_delta_rows),
        "y_now": np.asarray(y_now_rows),
        "y_future": np.asarray(y_future_rows),
        "y_naive": np.asarray(y_naive_rows),
        "y_cv": np.asarray(y_cv_rows),
        "t": np.asarray(t_rows),
        "feature_names": feature_names
    }


def choose_lambda_chronological(X_train, y_train):
    """
    Internal validation inside the training portion.
    We choose ridge lambda using chronological validation.
    """
    n = len(X_train)
    split = int(0.80 * n)

    X_subtrain = X_train[:split]
    y_subtrain = y_train[:split]

    X_val = X_train[split:]
    y_val = y_train[split:]

    mu, sigma = standardize_fit(X_subtrain)
    X_subtrain_s = standardize_apply(X_subtrain, mu, sigma)
    X_val_s = standardize_apply(X_val, mu, sigma)

    best_lam = None
    best_rmse = None

    for lam in RIDGE_LAMBDAS:
        beta = fit_ridge(X_subtrain_s, y_subtrain, lam)
        pred = predict_ridge(X_val_s, beta)
        _, rmse, _ = compute_metrics(y_val, pred)

        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_lam = lam

    return best_lam, best_rmse


def plot_prediction(t, y_true, y_trained, y_naive, y_cv, title, out_path):
    plt.figure(figsize=(14, 6))
    plt.plot(t, y_true, label="True future value", linewidth=2)
    plt.plot(t, y_trained, label="Trained ARX future prediction", linewidth=1.5)
    plt.plot(t, y_naive, label="Naive baseline x[k]", linestyle="--", linewidth=1.0)
    plt.plot(t, y_cv, label="Constant-velocity baseline", linestyle=":", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error(t, y_true, y_trained, y_naive, y_cv, title, out_path):
    plt.figure(figsize=(14, 5))
    plt.plot(t, y_trained - y_true, label="Trained ARX error", linewidth=1.5)
    plt.plot(t, y_naive - y_true, label="Naive error", linestyle="--", linewidth=1.0)
    plt.plot(t, y_cv - y_true, label="Constant-velocity error", linestyle=":", linewidth=1.0)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Prediction error (deg)")
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

    df["t"] = make_time_column(df)

    input_cols = [c for c in INPUT_COLS_CANDIDATE if c in df.columns]

    keep_cols = ["t"] + STATE_COLS + input_cols
    df = df[keep_cols].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    print(f"Rows after cleaning: {len(df)}")
    print(f"State columns: {STATE_COLS}")
    print(f"Input columns: {input_cols}")
    print(f"Delta lags: {DELTA_LAGS}")
    print(f"Input lags: {INPUT_LAGS}")
    print("Model: trained ARX / system-identification future-state predictor")

    if len(df) < 300:
        print("[SKIP] Not enough data")
        return []

    rows = []
    pred_out = pd.DataFrame()

    for target in TARGETS:
        print(f"\n--- Target: {target}[k+1] ---")

        data = build_trained_arx_dataset(df, input_cols, target)

        X = data["X"]
        y_delta = data["y_delta"]

        split = int(TRAIN_RATIO * len(X))

        X_train = X[:split]
        y_train = y_delta[:split]

        X_test = X[split:]
        y_delta_test = y_delta[split:]

        y_now_test = data["y_now"][split:]
        y_true_test = data["y_future"][split:]
        y_naive_test = data["y_naive"][split:]
        y_cv_test = data["y_cv"][split:]
        t_test = data["t"][split:]

        best_lam, internal_val_rmse = choose_lambda_chronological(X_train, y_train)

        mu, sigma = standardize_fit(X_train)
        X_train_s = standardize_apply(X_train, mu, sigma)
        X_test_s = standardize_apply(X_test, mu, sigma)

        beta = fit_ridge(X_train_s, y_train, best_lam)

        delta_pred_test = predict_ridge(X_test_s, beta)
        y_trained_test = y_now_test + delta_pred_test

        trained_mae, trained_rmse, trained_r2 = compute_metrics(y_true_test, y_trained_test)
        naive_mae, naive_rmse, naive_r2 = compute_metrics(y_true_test, y_naive_test)
        cv_mae, cv_rmse, cv_r2 = compute_metrics(y_true_test, y_cv_test)

        improvement_vs_naive = naive_rmse - trained_rmse
        improvement_pct_naive = 100.0 * improvement_vs_naive / naive_rmse if naive_rmse > 1e-12 else float("nan")

        improvement_vs_cv = cv_rmse - trained_rmse
        improvement_pct_cv = 100.0 * improvement_vs_cv / cv_rmse if cv_rmse > 1e-12 else float("nan")

        print(f"Best ridge lambda: {best_lam}")
        print(f"Internal validation delta-RMSE: {internal_val_rmse:.8f}")
        print(f"Trained ARX RMSE: {trained_rmse:.8f}")
        print(f"Naive RMSE      : {naive_rmse:.8f}")
        print(f"CV RMSE         : {cv_rmse:.8f}")
        print(f"Improvement over naive: {improvement_pct_naive:.2f}%")
        print(f"Improvement over CV   : {improvement_pct_cv:.2f}%")

        rows.append({
            "file": fname,
            "target": target,
            "rows_after_cleaning": len(df),
            "samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "delta_lags": DELTA_LAGS,
            "input_lags": INPUT_LAGS,
            "state_columns": ",".join(STATE_COLS),
            "input_columns": ",".join(input_cols),
            "feature_names": ",".join(data["feature_names"]),
            "best_ridge_lambda": best_lam,
            "internal_validation_delta_RMSE": internal_val_rmse,
            "trained_ARX_MAE": trained_mae,
            "trained_ARX_RMSE": trained_rmse,
            "trained_ARX_R2": trained_r2,
            "naive_MAE": naive_mae,
            "naive_RMSE": naive_rmse,
            "naive_R2": naive_r2,
            "constant_velocity_MAE": cv_mae,
            "constant_velocity_RMSE": cv_rmse,
            "constant_velocity_R2": cv_r2,
            "rmse_improvement_over_naive": improvement_vs_naive,
            "rmse_improvement_percent_over_naive": improvement_pct_naive,
            "rmse_improvement_over_cv": improvement_vs_cv,
            "rmse_improvement_percent_over_cv": improvement_pct_cv
        })

        pred_out[f"t_{target}"] = t_test
        pred_out[f"{target}_true_future"] = y_true_test
        pred_out[f"{target}_trained_ARX_prediction"] = y_trained_test
        pred_out[f"{target}_naive_prediction"] = y_naive_test
        pred_out[f"{target}_constant_velocity_prediction"] = y_cv_test
        pred_out[f"{target}_trained_ARX_error"] = y_trained_test - y_true_test
        pred_out[f"{target}_naive_error"] = y_naive_test - y_true_test
        pred_out[f"{target}_cv_error"] = y_cv_test - y_true_test

        model_obj = {
            "file": fname,
            "target": target,
            "model_type": "trained_delta_arx_one_step_future_state_predictor",
            "formula": "x_hat[k+1] = x[k] + W * phi[k]",
            "state_cols": STATE_COLS,
            "input_cols": input_cols,
            "feature_names": data["feature_names"],
            "delta_lags": DELTA_LAGS,
            "input_lags": INPUT_LAGS,
            "ridge_lambda": best_lam,
            "mu": mu,
            "sigma": sigma,
            "beta": beta
        }

        model_path = os.path.join(
            MODEL_DIR,
            f"{base}_{target}_trained_ARX_future_state_model.pkl"
        )

        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)

        fig_path = os.path.join(
            FIG_DIR,
            f"{base}_{target}_trained_ARX_future_prediction.png"
        )

        plot_prediction(
            t=t_test,
            y_true=y_true_test,
            y_trained=y_trained_test,
            y_naive=y_naive_test,
            y_cv=y_cv_test,
            title=f"{fname}: trained ARX one-step future {target} prediction",
            out_path=fig_path
        )

        err_path = os.path.join(
            FIG_DIR,
            f"{base}_{target}_trained_ARX_prediction_error.png"
        )

        plot_error(
            t=t_test,
            y_true=y_true_test,
            y_trained=y_trained_test,
            y_naive=y_naive_test,
            y_cv=y_cv_test,
            title=f"{fname}: trained ARX one-step future {target} prediction error",
            out_path=err_path
        )

    pred_path = os.path.join(
        RESULT_DIR,
        f"{base}_trained_ARX_prediction_outputs.csv"
    )

    pred_out.to_csv(pred_path, index=False)

    return rows


def main():
    print("\n========== PHASE 2 V5: TRAINED ARX FUTURE-STATE PREDICTION ==========")
    print("ATT only.")
    print("Training-based system-identification model.")
    print("One-step-ahead prediction: x_hat[k+1].")
    print("No attack detection.")
    print("No recovery.")

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

    summary_path = os.path.join(
        RESULT_DIR,
        "phase2_v5_att_trained_ARX_future_prediction_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\n========== V5 TRAINED ARX SUMMARY ==========")

    show_cols = [
        "file",
        "target",
        "best_ridge_lambda",
        "trained_ARX_RMSE",
        "naive_RMSE",
        "constant_velocity_RMSE",
        "rmse_improvement_percent_over_naive",
        "rmse_improvement_percent_over_cv",
        "trained_ARX_R2"
    ]

    print(summary[show_cols].to_string(index=False))

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved prediction CSVs: {RESULT_DIR}")
    print(f"Saved trained models: {MODEL_DIR}")
    print(f"Saved figures: {FIG_DIR}")

    print("\nPhase 2 status:")
    print("This is the training-based software sensor result.")
    print("Naive and constant-velocity are only baselines.")
    print("Do not move to attack recovery until you inspect the trained ARX results.")


if __name__ == "__main__":
    main()