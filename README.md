#  Linear Regression Statistical View
A lightweight Python library for simple linear regression with full statistical output — F-statistics, R², confidence intervals, and model comparison — built from scratch using NumPy and SciPy.

## Overview
This project implements a clean object-oriented interface for fitting and interpreting simple linear regression models. It's designed for educational and analytical use cases where you need more than just predictions — you need to understand the statistics behind the model.

The library is validated through a Jupyter notebook using an Auto MPG-style dataset, demonstrating how to predict fuel efficiency (`mpg`) from vehicle features like `horsepower` and `weight`.
---
## Files

| File | Description |
|------|-------------|
| `linear_regression_view.py` | Core library — `LinearRegressionStatisticalView` class + `compare` function |
| `regression_notebook.ipynb` | Validation notebook with a worked example on the MPG dataset |
---
## What It Does
### `LinearRegressionStatisticalView`

Fits a simple linear regression model (one predictor, one response) and exposes full statistical diagnostics:

```python
from linear_regression_view import LinearRegressionStatisticalView
model = LinearRegressionStatisticalView(alpha=0.05)
model.fit(X, Y)
summary = model.summary()
```
**Summary output includes:**
- **F-statistic** — tests whether the model explains a significant portion of variance
- **F-critical** — threshold at the chosen significance level (α)
- **R²** — proportion of variance in Y explained by X
- **r (Pearson)** — signed correlation coefficient
- **Degrees of freedom** — for model, residuals, and total
- **Beta confidence intervals** — 95% CIs for the intercept (β₀) and slope (β₁)
### `compare()`
Fits two separate models and picks the better predictor based on F-statistic:

```python
from linear_regression_view import compare

result = compare(df, x_col_1="horsepower", x_col_2="weight", y_col="mpg", alpha=0.05)
# → best_predictor: "horsepower"
```
---
## Example Results (MPG Dataset, n=200)
Predicting `mpg` from `horsepower`:
```
F-statistic  : 1301.69
F-critical   : 3.89
R²           : 0.868
R (Pearson)  : -0.932
β₀ CI        : (39.15, 40.97)
β₁ CI        : (-0.120, -0.107)
```
The negative slope confirms that higher horsepower is associated with lower fuel efficiency. Both β coefficients are statistically significant (F >> F-critical).
**Model comparison:**
| Predictor | F-statistic |
|-----------|-------------|
| horsepower | 1301.69  |
| weight | 1031.24 |

`horsepower` is the stronger linear predictor of `mpg`.
---

## How the Math Works
The model estimates β₀ and β₁ using the closed-form OLS formulas:
```
β₁ = Sxy / Sxx
β₀ = ȳ - β₁ · x̄
```
Where `Sxx = Σ(xᵢ - x̄)²` and `Sxy = Σ(xᵢ - x̄)(yᵢ - ȳ)`.
The F-statistic is computed as:
F = MS_model / MS_residual = (SS_model / 1) / (SS_res / (n-2))
Confidence intervals use the t-distribution with `n - 2` degrees of freedom.
---
## Dependencies

numpy
scipy
pandas  (for the notebook and compare function)
Install with:
```bash
pip install numpy scipy pandas
```
---
## Usage
```python
import pandas as pd
from linear_regression_view import LinearRegressionStatisticalView, compare

df = pd.read_csv("your_data.csv")

# Fit a single model
model = LinearRegressionStatisticalView(alpha=0.05)
model.fit(df["horsepower"].values, df["mpg"].values)
print(model.summary())

# Compare two predictors
result = compare(df, x_col_1="horsepower", x_col_2="weight", y_col="mpg")
print(result["best_predictor"])
```
