# Task 3: Linear Regression — AI & ML Internship (Elevate Labs)

**Objective:** Implement and understand simple & multiple linear regression.
**Tools:** Pandas, Scikit-learn, Matplotlib

## Dataset

`data/Housing.csv` — a House Price Prediction dataset with columns:
`price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
hotwaterheating, airconditioning, parking, prefarea, furnishingstatus`.

> Note: this environment has no internet access, so the file included here
> is a **synthetically generated dataset** built to match the structure and
> realistic relationships of the popular Kaggle "Housing Price Prediction"
> dataset used for this task. To use the real dataset instead, just download
> it from the link in the task PDF and replace `data/Housing.csv` — the
> script works unchanged since the column names match.

## What the script does (`linear_regression.py`)

1. **Import & preprocess** — loads the CSV, checks for missing values,
   encodes the `yes`/`no` columns as 1/0, and one-hot encodes
   `furnishingstatus`.
2. **Train-test split** — 80/20 split with `train_test_split`.
3. **Model fitting** —
   - **Part A (Simple Linear Regression):** `area` → `price`.
   - **Part B (Multiple Linear Regression):** all features → `price`.
4. **Evaluation** — MAE, MSE, RMSE, and R² for both models.
5. **Plots & interpretation** — regression line, actual-vs-predicted scatter,
   and a feature-importance bar chart from standardized coefficients.

## Results

### Simple Linear Regression (area → price)
| Metric | Value |
|---|---|
| Coefficient (slope) | ≈ 245.4 |
| Intercept | ≈ 2,681,991 |
| MAE | ≈ 486,826 |
| MSE | ≈ 3.82 × 10¹¹ |
| RMSE | ≈ 618,098 |
| R² | ≈ 0.73 |

*Interpretation:* every additional square foot of area is associated with
about ₹245 more in price, holding nothing else constant. Area alone explains
~73% of the variance in price.

### Multiple Linear Regression (all features → price)
| Metric | Value |
|---|---|
| MAE | ≈ 315,874 |
| MSE | ≈ 1.62 × 10¹¹ |
| RMSE | ≈ 402,764 |
| R² | ≈ 0.88 |

Adding the other features raises R² from 0.73 to 0.88 and lowers error
substantially — the extra features carry real predictive signal.
Largest standardized effects: `area`, `bathrooms`, `bedrooms`, `stories`,
`airconditioning`.

### Plots (in `plots/`)
- `simple_regression_line.png` — fitted line over the area/price scatter.
- `multiple_regression_actual_vs_predicted.png` — predicted vs actual price;
  points near the diagonal = accurate predictions.
- `feature_importance.png` — standardized coefficients, comparable across
  features regardless of original scale/units.

## Interview Questions

**1. What assumptions does linear regression make?**
Linearity (the relationship between X and y is linear), independence of
errors, homoscedasticity (constant variance of residuals), normality of
residuals, and no (or low) multicollinearity among predictors. It also
assumes features are measured without significant error.

**2. How do you interpret the coefficients?**
Each coefficient is the expected change in the target variable for a
one-unit increase in that feature, **holding all other features constant**.
In multiple regression this is a *partial* effect — it isolates that
feature's contribution net of the others. The sign shows direction
(positive/negative relationship) and the magnitude shows strength, though
magnitudes aren't directly comparable across features unless the features
are on the same scale (hence standardized coefficients).

**3. What is R² score and its significance?**
R² (coefficient of determination) measures the proportion of variance in
the target variable explained by the model, ranging from 0 to 1 (it can go
negative for a model worse than predicting the mean). An R² of 0.88 means
88% of the variation in price is explained by the features. It's useful for
judging overall fit, but it doesn't indicate whether the model is correctly
specified or whether individual predictors are significant — that's what
adjusted R² and p-values are for.

**4. When would you prefer MSE over MAE?**
MSE squares the errors, so it penalizes large errors much more heavily than
small ones — prefer it when large mistakes are especially costly and you
want the model to focus on avoiding them (it's also differentiable
everywhere, which is why it's the default loss for gradient-based fitting).
MAE treats all errors linearly and is more robust to outliers, so prefer it
when the data has outliers you don't want to dominate the loss, or when you
want an error metric in the same, easily-interpretable units as the target.

**5. How do you detect multicollinearity?**
- Compute a correlation matrix between predictors and look for high
  pairwise correlations (e.g., > 0.8).
- Compute the **Variance Inflation Factor (VIF)** for each predictor — a
  VIF above ~5–10 signals problematic multicollinearity.
- Watch for symptoms: coefficients with unstable signs or huge standard
  errors that swing a lot when a feature is added/removed, despite a good
  overall R².

**6. What is the difference between simple and multiple regression?**
Simple linear regression models the relationship between **one** predictor
and the target (`y = b0 + b1*x`). Multiple linear regression models the
relationship between **two or more** predictors and the target
(`y = b0 + b1*x1 + b2*x2 + ... + bn*xn`), letting you capture the combined
and isolated (partial) effects of several variables at once.

**7. Can linear regression be used for classification?**
Not directly and not well — it can output any real number, including
values outside [0, 1] or between classes, and it isn't optimized for
class boundaries or probabilities. **Logistic regression** is the correct
analogue for classification: it applies a sigmoid function to a linear
combination of features to output a bounded probability, and it's trained
with a loss function suited to classification.

**8. What happens if you violate regression assumptions?**
- *Non-linearity:* the model systematically under/over-predicts in certain
  ranges — biased, poor-fitting predictions.
- *Heteroscedasticity:* coefficient estimates stay unbiased, but standard
  errors (and thus confidence intervals / p-values / significance tests)
  become unreliable.
- *Non-normal residuals:* mainly affects the validity of hypothesis tests
  and confidence intervals, especially with small sample sizes.
- *Multicollinearity:* coefficients become unstable and hard to interpret,
  even though overall predictions may still be reasonable.
- *Correlated errors (non-independence):* standard errors are underestimated,
  leading to overconfident, misleading significance results.

In short: violating assumptions doesn't always break the predictions
themselves, but it commonly breaks the *inferences* you'd want to draw from
the coefficients (their reliability, significance, and interpretability).

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib
python generate_dataset.py   # creates data/Housing.csv (skip if you have the real dataset)
python linear_regression.py  # runs the full task and saves plots/
```

## Repo structure

```
task3_linear_regression/
├── data/
│   └── Housing.csv
├── plots/
│   ├── simple_regression_line.png
│   ├── multiple_regression_actual_vs_predicted.png
│   └── feature_importance.png
├── generate_dataset.py
├── linear_regression.py
└── README.md
```
