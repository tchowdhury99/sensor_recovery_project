import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/sensor_recovery_project")
CSV_DIR = os.path.join(BASE, "logs", "extracted_csv")

MODEL_DIR = os.path.join(BASE, "models_v3_stable")
RESULT_DIR = os.path.join(BASE, "prediction_results_v3_stable")
FIG_DIR = os.path.join(BASE, "figures", "future_prediction_v3_stable")

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
RIDGE_LAMBDA = 1.0

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

GAMMA_GRID = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]


def safe_name(name):
    return name.replace(".csv", "")


def make_time_column(df):
    if "TimeUS" in df.columns:
        return (df["TimeUS"] - df["TimeUS"].iloc[0]) / 1_000_000.0
    if "TimeSec" in df.columns:
        return df["TimeSec"] - df["TimeSec"].iloc[0]
    return pd.Series(np.arange(len(df)), name="sample_index")


def angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def add_angle_deg(a, delta):
    return a + delta


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


def standardize_train_apply(X_train, X_apply):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0.0] = 1.0

    return (X_train - mu) / sigma, (X_apply - mu) / sigma, mu, sigma


def apply_standardize(X, mu, sigma):
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


def build_dataset(df, feature_cols, target_col, lags):
    """
    Build one-step-ahead prediction dataset.

    For each time k, the model sees:
      features from k, k-1, ..., k-lags+1

    It predicts:
      x[k+1]

    Baselines:
      naive: x[k]
      constant velocity: x[k] + (x[k] - x[k-1])
    """
    X_rows = []
    y_now_rows = []
    y_prev_rows = []
    y_future_rows = []
    y_naive_rows = []
    y_cv_rows = []
    residual_cv_rows = []
    t_rows = []

    feature_values = df[feature_cols].to_numpy(dtype=float)
    y_values = df[target_col].to_numpy(dtype=float)
    t_values = df["t"].to_numpy(dtype=float)

    for k in range(lags - 1, len(df) - 1):
        row = []

        for lag in range(lags):
            idx = k - lag
            row.extend(feature_values[idx, :])

        y_prev = y_values[k - 1]
        y_now = y_values[k]
        y_future = y_values[k + 1]

        if target_col == "Yaw":
            velocity = angle_diff_deg(y_now, y_prev)
            true_step = angle_diff_deg(y_future, y_now)
            residual_cv = true_step - velocity
            y_cv = add_angle_deg(y_now, velocity)
        else:
            velocity = y_now - y_prev
            y_cv = y_now + velocity
            residual_cv = y_future - y_cv

        X_rows.append(row)
        y_prev_rows.append(y_prev)
        y_now_rows.append(y_now)
        y_future_rows.append(y_future)
        y_naive_rows.append(y_now)
        y_cv_rows.append(y_cv)
        residual_cv_rows.append(residual_cv)
        t_rows.append(t_values[k + 1])

    feature_names = []
    for lag in range(lags):
        for col in feature_cols:
            feature_names.append(f"{col}_k-{lag}")

    return {
        "X": np.asarray(X_rows),
        "y_prev": np.asarray(y_prev_rows),
        "y_now": np.asarray(y_now_rows),
        "y_future": np.asarray(y_future_rows),
        "y_naive": np.asarray(y_naive_rows),
        "y_cv": np.asarray(y_cv_rows),
        "residual_cv": np.asarray(residual_cv_rows),
        "t": np.asarray(t_rows),
        "feature_names": feature_names
    }


def split_indices(n):
    train_end = int(TRAIN_RATIO * n)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

    idx_train = np.arange(0, train_end)
    idx_val = np.arange(train_end, val_end)
    idx_test = np.arange(val_end, n)

    return idx_train, idx_val, idx_test


def select_gamma(y_true_val, y_cv_val, residual_pred_val):
    best_gamma = None
    best_rmse = None

    for gamma in GAMMA_GRID:
        y_hat = y_cv_val + gamma * residual_pred_val
        _, rmse, _ = compute_metrics(y_true_val, y_hat)

        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_gamma = gamma

    return best_gamma, best_rmse


def plot_prediction(t, y_true, y_final, y_naive, y_cv, title, out_path):
    plt.figure(figsize=(14, 6))
    plt.plot(t, y_true, label="True future value", linewidth=2)
    plt.plot(t, y_final, label="Selected stable system-ID prediction", linewidth=1.5)
    plt.plot(t, y_naive, label="Naive y[k]", linestyle="--", linewidth=1.0)
    plt.plot(t, y_cv, label="Constant-velocity state predictor", linestyle=":", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error(t, y_true, y_final, y_naive, y_cv, title, out_path):
    plt.figure(figsize=(14, 5))
    plt.plot(t, y_final - y_true, label="Selected model error", linewidth=1.5)
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

    if "_ATT" not in fname:
        print("[SKIP] Not an ATT file.")
        return []

    df = pd.read_csv(path)

    if not all(c in df.columns for c in TARGETS):
        print("[SKIP] Missing Roll/Pitch/Yaw.")
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
    print("Model: stable one-step-ahead state-transition predictor")
    print("Base predictor: x[k+1] = 2*x[k] - x[k-1]")
    print("Optional learned residual correction selected by validation.")

    if len(df) < 300:
        print("[SKIP] Not enough data.")
        return []

    result_rows = []
    pred_out = pd.DataFrame()

    for target in TARGETS:
        print(f"\n--- Target: {target}[k+1] ---")

        data = build_dataset(df, feature_cols, target, LAGS)
        n = len(data["X"])

        idx_train, idx_val, idx_test = split_indices(n)

        X_train = data["X"][idx_train]
        X_val = data["X"][idx_val]
        X_test = data["X"][idx_test]

        residual_train = data["residual_cv"][idx_train]

        y_true_val = data["y_future"][idx_val]
        y_cv_val = data["y_cv"][idx_val]

        y_true_test = data["y_future"][idx_test]
        y_naive_test = data["y_naive"][idx_test]
        y_cv_test = data["y_cv"][idx_test]
        t_test = data["t"][idx_test]

        X_train_s, X_val_s, mu, sigma = standardize_train_apply(X_train, X_val)
        X_test_s = apply_standardize(X_test, mu, sigma)

        beta = fit_ridge(X_train_s, residual_train, RIDGE_LAMBDA)

        residual_pred_val = predict_ridge(X_val_s, beta)
        residual_pred_test = predict_ridge(X_test_s, beta)

        gamma, val_rmse = select_gamma(y_true_val, y_cv_val, residual_pred_val)

        y_final_test = y_cv_test + gamma * residual_pred_test

        model_mae, model_rmse, model_r2 = compute_metrics(y_true_test, y_final_test)
        naive_mae, naive_rmse, naive_r2 = compute_metrics(y_true_test, y_naive_test)
        cv_mae, cv_rmse, cv_r2 = compute_metrics(y_true_test, y_cv_test)

        improvement_vs_naive = naive_rmse - model_rmse
        improvement_pct_naive = 100.0 * improvement_vs_naive / naive_rmse if naive_rmse > 1e-12 else float("nan")

        improvement_vs_cv = cv_rmse - model_rmse
        improvement_pct_cv = 100.0 * improvement_vs_cv / cv_rmse if cv_rmse > 1e-12 else float("nan")

        print(f"Selected gamma: {gamma}")
        print(f"Validation RMSE after gamma selection: {val_rmse:.8f}")
        print(f"Final model RMSE: {model_rmse:.8f}")
        print(f"Naive RMSE      : {naive_rmse:.8f}")
        print(f"CV RMSE         : {cv_rmse:.8f}")
        print(f"Improvement over naive: {improvement_pct_naive:.2f}%")
        print(f"Improvement over CV   : {improvement_pct_cv:.2f}%")

        result_rows.append({
            "file": fname,
            "target": target,
            "rows_after_cleaning": len(df),
            "samples": n,
            "train_samples": len(idx_train),
            "val_samples": len(idx_val),
            "test_samples": len(idx_test),
            "lags": LAGS,
            "feature_columns": ",".join(feature_cols),
            "selected_gamma": gamma,
            "validation_RMSE_selected": val_rmse,
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
            "rmse_improvement_percent_over_naive": improvement_pct_naive,
            "rmse_improvement_over_cv": improvement_vs_cv,
            "rmse_improvement_percent_over_cv": improvement_pct_cv
        })

        pred_out[f"t_{target}"] = t_test
        pred_out[f"{target}_true_future"] = y_true_test
        pred_out[f"{target}_selected_model"] = y_final_test
        pred_out[f"{target}_naive_yk"] = y_naive_test
        pred_out[f"{target}_constant_velocity"] = y_cv_test
        pred_out[f"{target}_selected_model_error"] = y_final_test - y_true_test
        pred_out[f"{target}_naive_error"] = y_naive_test - y_true_test
        pred_out[f"{target}_cv_error"] = y_cv_test - y_true_test

        model_obj = {
            "file": fname,
            "target": target,
            "model_type": "stable_state_transition_with_optional_residual_correction",
            "base_model": "constant_velocity_x_kplus1_equals_2xk_minus_xkminus1",
            "feature_cols": feature_cols,
            "feature_names": data["feature_names"],
            "lags": LAGS,
            "ridge_lambda": RIDGE_LAMBDA,
            "selected_gamma": gamma,
            "mu": mu,
            "sigma": sigma,
            "beta_residual": beta
        }

        model_path = os.path.join(
            MODEL_DIR,
            f"{base}_{target}_stable_future_state_model.pkl"
        )

        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)

        fig_path = os.path.join(
            FIG_DIR,
            f"{base}_{target}_v3_stable_future_prediction.png"
        )

        plot_prediction(
            t=t_test,
            y_true=y_true_test,
            y_final=y_final_test,
            y_naive=y_naive_test,
            y_cv=y_cv_test,
            title=f"{fname}: {target}[k+1] stable future-state prediction",
            out_path=fig_path
        )

        err_path = os.path.join(
            FIG_DIR,
            f"{base}_{target}_v3_stable_prediction_error.png"
        )

        plot_error(
            t=t_test,
            y_true=y_true_test,
            y_final=y_final_test,
            y_naive=y_naive_test,
            y_cv=y_cv_test,
            title=f"{fname}: {target}[k+1] prediction error",
            out_path=err_path
        )

    pred_path = os.path.join(RESULT_DIR, f"{base}_v3_stable_prediction_outputs.csv")
    pred_out.to_csv(pred_path, index=False)

    return result_rows


def main():
    print("\n========== PHASE 2 V3: STABLE ATT FUTURE-STATE PREDICTION ==========")
    print("ATT only.")
    print("One-step-ahead future prediction only.")
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
        "phase2_v3_att_stable_future_prediction_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\n========== V3 SUMMARY ==========")

    show_cols = [
        "file",
        "target",
        "selected_gamma",
        "model_RMSE",
        "naive_RMSE",
        "constant_velocity_RMSE",
        "rmse_improvement_percent_over_naive",
        "rmse_improvement_percent_over_cv",
        "model_R2"
    ]

    print(summary[show_cols].to_string(index=False))

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved prediction CSVs: {RESULT_DIR}")
    print(f"Saved models: {MODEL_DIR}")
    print(f"Saved figures: {FIG_DIR}")

    print("\nInterpretation:")
    print("If selected_gamma = 0, the stable constant-velocity state predictor was best.")
    print("That is acceptable for one-step-ahead ATT prediction because the sampling interval is very small.")
    print("Do not move to attack recovery until these prediction figures look correct.")


if __name__ == "__main__":
    main()