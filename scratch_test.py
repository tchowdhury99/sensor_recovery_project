import pandas as pd
import numpy as np

att = pd.read_csv("csv/baseline_ATT.csv")
att["t"] = (att["TimeUS"] - att["TimeUS"].iloc[0]) / 1_000_000
time = att["t"].to_numpy()
real_roll = att["Roll"].astype(float).to_numpy()
des_roll = att["DesRoll"].astype(float).to_numpy()

train_mask = time < 25
# Add intercept term
X_train = np.column_stack((real_roll[train_mask][:-1], des_roll[train_mask][:-1], np.ones(np.sum(train_mask)-1)))
y_train = real_roll[train_mask][1:]

coef, residuals, rank, s = np.linalg.lstsq(X_train, y_train, rcond=None)
print(f"Coefficients (A, B, C): {coef}")

predicted = X_train @ coef
mse = np.mean((predicted - y_train)**2)
print(f"MSE on train: {mse}")
