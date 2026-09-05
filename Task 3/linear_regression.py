"""
Task 3: Linear Regression (AI & ML Internship - Elevate Labs)
================================================================
Objective : Implement and understand simple & multiple linear regression.
Tools     : Pandas, Scikit-learn, Matplotlib

Steps followed (per the task's mini-guide):
1. Import and preprocess the dataset.
2. Split data into train-test sets.
3. Fit a Linear Regression model using sklearn.linear_model.
4. Evaluate model using MAE, MSE, R^2.
5. Plot regression line and interpret coefficients.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 120)

# ---------------------------------------------------------------
# 1. IMPORT AND PREPROCESS THE DATASET
# ---------------------------------------------------------------
df = pd.read_csv("data/Housing.csv")

print("=" * 70)
print("STEP 1: DATA OVERVIEW")
print("=" * 70)
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# Encode binary yes/no categorical columns as 1/0
binary_cols = ["mainroad", "guestroom", "basement",
                "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

# One-hot encode the multi-category column (furnishingstatus)
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

print("\nDataset after encoding categorical variables:")
print(df.head())

# ---------------------------------------------------------------
# 2. SPLIT DATA INTO TRAIN-TEST SETS
# ---------------------------------------------------------------
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n" + "=" * 70)
print("STEP 2: TRAIN-TEST SPLIT")
print("=" * 70)
print(f"Train size: {X_train.shape[0]} rows | Test size: {X_test.shape[0]} rows")

# =================================================================
# PART A: SIMPLE LINEAR REGRESSION  (single feature: area -> price)
# =================================================================
print("\n" + "=" * 70)
print("PART A: SIMPLE LINEAR REGRESSION (area -> price)")
print("=" * 70)

X_simple = df[["area"]]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

# ---- 3. Fit the model ----
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)
y_pred_s = simple_model.predict(X_test_s)

# ---- 4. Evaluate ----
mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
rmse_s = np.sqrt(mse_s)
r2_s = r2_score(y_test_s, y_pred_s)

print(f"Coefficient (slope):  {simple_model.coef_[0]:,.2f}")
print(f"Intercept:            {simple_model.intercept_:,.2f}")
print(f"MAE:                  {mae_s:,.2f}")
print(f"MSE:                  {mse_s:,.2f}")
print(f"RMSE:                 {rmse_s:,.2f}")
print(f"R^2 Score:            {r2_s:.4f}")

# ---- 5. Plot regression line ----
plt.figure(figsize=(8, 6))
plt.scatter(X_test_s, y_test_s, color="steelblue", alpha=0.6, label="Actual")
# sort for a clean line
order = np.argsort(X_test_s["area"].values)
plt.plot(X_test_s["area"].values[order], y_pred_s[order],
          color="crimson", linewidth=2, label="Regression line")
plt.xlabel("Area (sq. ft.)")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.tight_layout()
plt.savefig("plots/simple_regression_line.png", dpi=150)
plt.close()
print("Saved plot -> plots/simple_regression_line.png")

# =================================================================
# PART B: MULTIPLE LINEAR REGRESSION (all features -> price)
# =================================================================
print("\n" + "=" * 70)
print("PART B: MULTIPLE LINEAR REGRESSION (all features -> price)")
print("=" * 70)

# ---- 3. Fit the model ----
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
y_pred_m = multi_model.predict(X_test)

# ---- 4. Evaluate ----
mae_m = mean_absolute_error(y_test, y_pred_m)
mse_m = mean_squared_error(y_test, y_pred_m)
rmse_m = np.sqrt(mse_m)
r2_m = r2_score(y_test, y_pred_m)

print(f"MAE:                  {mae_m:,.2f}")
print(f"MSE:                  {mse_m:,.2f}")
print(f"RMSE:                 {rmse_m:,.2f}")
print(f"R^2 Score:            {r2_m:.4f}")

# ---- 5. Interpret coefficients ----
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": multi_model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)

print("\nCoefficients (effect of each feature on price, holding others constant):")
print(coef_df.to_string(index=False))
print(f"\nIntercept: {multi_model.intercept_:,.2f}")

# ---- Standardized coefficients, to fairly compare feature importance ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaled_model = LinearRegression()
scaled_model.fit(X_train_scaled, y_train)

coef_scaled_df = pd.DataFrame({
    "Feature": X.columns,
    "Standardized Coefficient": scaled_model.coef_
}).sort_values("Standardized Coefficient", key=abs, ascending=False)

print("\nStandardized coefficients (comparable feature importance):")
print(coef_scaled_df.to_string(index=False))

# ---- Plot: Actual vs Predicted (standard way to visualize multiple regression fit) ----
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_m, alpha=0.6, color="seagreen")
lims = [min(y_test.min(), y_pred_m.min()), max(y_test.max(), y_pred_m.max())]
plt.plot(lims, lims, color="crimson", linewidth=2, label="Perfect prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression: Actual vs Predicted Price")
plt.legend()
plt.tight_layout()
plt.savefig("plots/multiple_regression_actual_vs_predicted.png", dpi=150)
plt.close()
print("Saved plot -> plots/multiple_regression_actual_vs_predicted.png")

# ---- Plot: Feature importance bar chart ----
plt.figure(figsize=(8, 6))
plt.barh(coef_scaled_df["Feature"], coef_scaled_df["Standardized Coefficient"],
          color="darkorange")
plt.xlabel("Standardized Coefficient")
plt.title("Feature Importance (Standardized Coefficients)")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.close()
print("Saved plot -> plots/feature_importance.png")

# ---- Multicollinearity check (VIF) ----
print("\n" + "=" * 70)
print("BONUS: MULTICOLLINEARITY CHECK (Variance Inflation Factor)")
print("=" * 70)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_vif = X.astype(float)
    vif_df = pd.DataFrame({
        "Feature": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    }).sort_values("VIF", ascending=False)
    print(vif_df.to_string(index=False))
    print("\n(Rule of thumb: VIF > 5-10 indicates problematic multicollinearity.)")
except ImportError:
    print("statsmodels not installed - skipping VIF calculation.")

print("\nDone. All results and plots have been generated.")
