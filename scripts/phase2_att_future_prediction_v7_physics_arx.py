import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/sensor_recovery_project")
CSV_DIR = os.path.join(BASE, "logs", "extracted_csv")

MODEL_DIR = os.path.join(BASE, "models_v7_physics_arx")
RESULT_DIR = os.path.join(BASE, "prediction_results_v7_physics_arx")
FIG_DIR = os.path.join(BASE, "figures", "future_prediction_v7_physics_arx")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TARGETS = ["Roll", "Pitch", "Yaw"]
ATT_COLS = ["Roll", "Pitch", "Yaw"]
DES_COLS = ["DesRoll", "DesPitch", "DesYaw"]
GYRO_COLS = ["GyrX", "GyrY", "GyrZ"]

TRAIN_RATIO = 0.70
LAGS = 3
RIDGE_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def make_time_column(df):
    if "TimeUS" in df.columns:
        return (df["TimeUS"] - df["TimeUS"].iloc[0]) / 1_000_000.0
    if "TimeSec" in df.columns:
        return df["TimeSec"] - df["TimeSec"].iloc[0]
    return pd.Series(np.arange(len(df)), name="sample_index")


def angle_diff_deg(a, b):
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


def find_matching_imu(att_file):
    prefix = att_file.replace("_ATT.csv", "")
    imu_file = prefix + "_IMU.csv"
    imu_path = os.path.join(CSV_DIR, imu_file)
    if os.path.exists(imu_path):
        return imu_path
    return None


def merge_att_imu(att_path, imu_path):
    att = pd.read_csv(att_path)
    imu = pd.read_csv(imu_path)

    att["t"] = make_time_column(att)
    imu["t"] = make_time_column(imu)

    required_att = ["t"] + ATT_COLS + [c for c in DES_COLS if c in att.columns]
    required_imu = ["t"] + GYRO_COLS

    missing_gyro = [c for c in GYRO_COLS if c not in imu.columns]
    if missing_gyro:
        raise ValueError(f"Missing gyro columns in {os.path.basename(imu_path)}: {missing_gyro}")

    att = att[required_att].apply(pd.to_numeric, errors="coerce")
    imu = imu[required_imu].apply(pd.to_numeric, errors="coerce")

    att = att.replace([np.inf, -np.inf], np.nan).dropna().sort_values("t").reset_index(drop=True)
    imu = imu.replace([np.inf, -np.inf], np.nan).dropna().sort_values("t").reset_index(drop=True)

    merged = pd.merge_asof(att, imu, on="t", direction="nearest")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # Time step for each ATT row.
    merged["dt"] = merged["t"].diff()
    merged["dt"] = merged["dt"].fillna(merged["dt"].median())
    merged.loc[merged["dt"] <= 0, "dt"] = merged["dt"].median()

    # Physics-informed gyro integration terms.
    # If gyro is rad/s, this approximates degrees per sample.
    merged["GyrX_dt_deg"] = merged["GyrX"] * merged["dt"] * 180.0 / np.pi
    merged["GyrY_dt_deg"] = merged["GyrY"] * merged["dt"] * 180.0 / np.pi
    merged["GyrZ_dt_deg"] = merged["GyrZ"] * merged["dt"] * 180.0 / np.pi

    # Desired attitude errors.
    if "DesRoll" in merged.columns:
        merged["ErrDesRoll"] = merged["DesRoll"] - merged["Roll"]
    else:
        merged["ErrDesRoll"] = 0.0

    if "DesPitch" in merged.columns:
        merged["ErrDesPitch"] = merged["DesPitch"] - merged["Pitch"]
    else:
        merged["ErrDesPitch"] = 0.0

    if "DesYaw" in merged.columns:
        merged["ErrDesYaw"] = [
            angle_diff_deg(d, y) for d, y in zip(merged["DesYaw"], merged["Yaw"])
        ]
    else:
        merged["ErrDesYaw"] = 0.0

    return merged


def add_attitude_deltas(df):
    for col in ATT_COLS:
        vals = df[col].to_numpy(dtype=float)
        d = np.zeros_like(vals)

        for i in range(1, len(vals)):
            if col == "Yaw":
                d[i] = angle_diff_deg(vals[i], vals[i - 1])
            else:
                d[i] = vals[i] - vals[i - 1]

        df[f"d{col}"] = d

    return df


def build_dataset(df, target):
    """
    Trained physics-informed ARX.

    Target:
        delta target[k+1] = target[k+1] - target[k]

    Features:
        previous attitude deltas
        gyro * dt in degrees
        desired attitude errors
        lagged for k, k-1, k-2
    """
    df = add_attitude_deltas(df.copy())

    base_features = [
        "dRoll", "dPitch", "dYaw",
        "GyrX_dt_deg", "GyrY_dt_deg", "GyrZ_dt_deg",
        "ErrDesRoll", "ErrDesPitch", "ErrDesYaw"
    ]

    X_rows = []
    y_delta_rows = []
    y_now_rows = []
    y_future_rows = []
    y_naive_rows = []
    y_cv_rows = []
    t_rows = []

    y_values = df[target].to_numpy(dtype=float)
    t_values = df["t"].to_numpy(dtype=float)

    start_k = LAGS

    for k in range(start_k, len(df) - 1):
        row = []

        for lag in range(LAGS):
            idx = k - lag
            for feature in base_features:
                row.append(float(df.iloc[idx][feature]))

        y_prev = y_values[k - 1]
        y_now = y_values[k]
        y_future = y_values[k + 1]

        if target == "Yaw":
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

    feature_names = []
    for lag in range(LAGS):
        for feature in base_features:
            feature_names.append(f"{feature}_k-{lag}")

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


def choose_lambda(X_train, y_train):
    n = len(X_train)
    split = int(0.80 * n)

    X_sub = X_train[:split]
    y_sub = y_train[:split]

    X_val = X_train[split:]
    y_val = y_train[split:]

    mu, sigma = standardize_fit(X_sub)
    X_sub_s = standardize_apply(X_sub, mu, sigma)
    X_val_s = standardize_apply(X_val, mu, sigma)

    best_lam = None
    best_rmse = None

    for lam in RIDGE_LAMBDAS:
        beta = fit_ridge(X_sub_s, y_sub, lam)
        pred = predict_ridge(X_val_s, beta)
        _, rmse, _ = compute_metrics(y_val, pred)

        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_lam = lam

    return best_lam, best_rmse


def plot_prediction(t, y_true, y_pred, y_naive, y_cv, title, out_path):
    plt.figure(figsize=(14, 6))
    plt.plot(t, y_true, label="True future value", linewidth=2)
    plt.plot(t, y_pred, label="Trained physics-informed ARX", linewidth=1.5)
    plt.plot(t, y_naive, label="Naive baseline", linestyle="--", linewidth=1.0)
    plt.plot(t, y_cv, label="Constant-velocity baseline", linestyle=":", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error(t, y_true, y_pred, y_naive, y_cv, title, out_path):
    plt.figure(figsize=(14, 5))
    plt.plot(t, y_pred - y_true, label="Trained ARX error", linewidth=1.5)
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


def process_pair(att_path, imu_path):
    att_name = os.path.basename(att_path)
    imu_name = os.path.basename(imu_path)
    base = att_name.replace(".csv", "")

    print("\n============================================================")
    print(f"ATT: {att_name}")
    print(f"IMU: {imu_name}")
    print("============================================================")

    df = merge_att_imu(att_path, imu_path)

    print(f"Merged rows: {len(df)}")
    print("Features include attitude deltas, gyro*dt terms, and desired-attitude errors.")

    rows = []
    pred_out = pd.DataFrame()

    for target in TARGETS:
        print(f"\n--- Target: {target}[k+1] ---")

        data = build_dataset(df, target)

        X = data["X"]
        y_delta = data["y_delta"]

        split = int(TRAIN_RATIO * len(X))

        X_train = X[:split]
        y_train = y_delta[:split]

        X_test = X[split:]
        y_now_test = data["y_now"][split:]
        y_true_test = data["y_future"][split:]
        y_naive_test = data["y_naive"][split:]
        y_cv_test = data["y_cv"][split:]
        t_test = data["t"][split:]

        best_lam, internal_val_rmse = choose_lambda(X_train, y_train)

        mu, sigma = standardize_fit(X_train)
        X_train_s = standardize_apply(X_train, mu, sigma)
        X_test_s = standardize_apply(X_test, mu, sigma)

        beta = fit_ridge(X_train_s, y_train, best_lam)

        delta_pred = predict_ridge(X_test_s, beta)
        y_pred = y_now_test + delta_pred

        model_mae, model_rmse, model_r2 = compute_metrics(y_true_test, y_pred)
        naive_mae, naive_rmse, naive_r2 = compute_metrics(y_true_test, y_naive_test)
        cv_mae, cv_rmse, cv_r2 = compute_metrics(y_true_test, y_cv_test)

        imp_naive = naive_rmse - model_rmse
        imp_naive_pct = 100.0 * imp_naive / naive_rmse if naive_rmse > 1e-12 else float("nan")

        imp_cv = cv_rmse - model_rmse
        imp_cv_pct = 100.0 * imp_cv / cv_rmse if cv_rmse > 1e-12 else float("nan")

        print(f"Best lambda: {best_lam}")
        print(f"Physics ARX RMSE: {model_rmse:.8f}")
        print(f"Naive RMSE      : {naive_rmse:.8f}")
        print(f"CV RMSE         : {cv_rmse:.8f}")
        print(f"Improvement over naive: {imp_naive_pct:.2f}%")
        print(f"Improvement over CV   : {imp_cv_pct:.2f}%")

        rows.append({
            "att_file": att_name,
            "imu_file": imu_name,
            "target": target,
            "merged_rows": len(df),
            "samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "lags": LAGS,
            "best_ridge_lambda": best_lam,
            "internal_validation_delta_RMSE": internal_val_rmse,
            "trained_physics_ARX_MAE": model_mae,
            "trained_physics_ARX_RMSE": model_rmse,
            "trained_physics_ARX_R2": model_r2,
            "naive_MAE": naive_mae,
            "naive_RMSE": naive_rmse,
            "naive_R2": naive_r2,
            "constant_velocity_MAE": cv_mae,
            "constant_velocity_RMSE": cv_rmse,
            "constant_velocity_R2": cv_r2,
            "rmse_improvement_over_naive": imp_naive,
            "rmse_improvement_percent_over_naive": imp_naive_pct,
            "rmse_improvement_over_cv": imp_cv,
            "rmse_improvement_percent_over_cv": imp_cv_pct,
            "feature_names": ",".join(data["feature_names"])
        })

        pred_out[f"t_{target}"] = t_test
        pred_out[f"{target}_true_future"] = y_true_test
        pred_out[f"{target}_trained_physics_ARX_prediction"] = y_pred
        pred_out[f"{target}_naive_prediction"] = y_naive_test
        pred_out[f"{target}_constant_velocity_prediction"] = y_cv_test
        pred_out[f"{target}_trained_physics_ARX_error"] = y_pred - y_true_test
        pred_out[f"{target}_naive_error"] = y_naive_test - y_true_test
        pred_out[f"{target}_cv_error"] = y_cv_test - y_true_test

        model_obj = {
            "att_file": att_name,
            "imu_file": imu_name,
            "target": target,
            "model_type": "trained_physics_informed_delta_arx",
            "formula": "x_hat[k+1] = x[k] + W * phi[k]",
            "feature_names": data["feature_names"],
            "lags": LAGS,
            "ridge_lambda": best_lam,
            "mu": mu,
            "sigma": sigma,
            "beta": beta
        }

        model_path = os.path.join(MODEL_DIR, f"{base}_{target}_physics_ARX_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)

        fig_path = os.path.join(FIG_DIR, f"{base}_{target}_physics_ARX_prediction.png")
        plot_prediction(
            t_test,
            y_true_test,
            y_pred,
            y_naive_test,
            y_cv_test,
            f"{att_name}: physics-informed trained ARX {target}[k+1]",
            fig_path
        )

        err_path = os.path.join(FIG_DIR, f"{base}_{target}_physics_ARX_error.png")
        plot_error(
            t_test,
            y_true_test,
            y_pred,
            y_naive_test,
            y_cv_test,
            f"{att_name}: physics-informed trained ARX {target}[k+1] error",
            err_path
        )

    pred_path = os.path.join(RESULT_DIR, f"{base}_physics_ARX_prediction_outputs.csv")
    pred_out.to_csv(pred_path, index=False)

    return rows


def main():
    print("\n========== PHASE 2 V7: PHYSICS-INFORMED TRAINED ARX ==========")
    print("Training-based system-identification model.")
    print("Features: attitude deltas + gyro*dt + desired attitude errors.")
    print("One-step-ahead prediction only.")
    print("No attack detection. No recovery.")

    att_files = sorted([
        f for f in os.listdir(CSV_DIR)
        if f.endswith(".csv") and "_ATT" in f
    ])

    all_rows = []

    for att_file in att_files:
        att_path = os.path.join(CSV_DIR, att_file)
        imu_path = find_matching_imu(att_file)

        if imu_path is None:
            print(f"[SKIP] No matching IMU file for {att_file}")
            continue

        all_rows.extend(process_pair(att_path, imu_path))

    if not all_rows:
        raise RuntimeError("No V7 results produced.")

    summary = pd.DataFrame(all_rows)
    summary_path = os.path.join(RESULT_DIR, "phase2_v7_physics_ARX_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n========== V7 PHYSICS-INFORMED ARX SUMMARY ==========")

    cols = [
        "att_file",
        "target",
        "best_ridge_lambda",
        "trained_physics_ARX_RMSE",
        "naive_RMSE",
        "constant_velocity_RMSE",
        "rmse_improvement_percent_over_naive",
        "rmse_improvement_percent_over_cv",
        "trained_physics_ARX_R2"
    ]

    print(summary[cols].to_string(index=False))

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved models: {MODEL_DIR}")
    print(f"Saved figures: {FIG_DIR}")


if __name__ == "__main__":
    main()