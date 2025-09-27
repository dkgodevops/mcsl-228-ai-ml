"""
Polynomial Regression demo (synthetic dataset)

What this does:
- Creates a noisy quadratic dataset y = 1 + 2*x - 0.5*x^2 + noise
- Fits polynomial regressions of different degrees using sklearn
- Uses cross-validation to pick the best polynomial degree
- Plots the original data and the fitted curve for the chosen model

Requirements:
  numpy
  scikit-learn
  matplotlib
  (optional) pandas for nicer tables

Run:
  python polynomial_regression_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# -------------- 1) Generate synthetic dataset --------------
rng = np.random.RandomState(42)
n_samples = 100
X = rng.uniform(-3, 3, size=n_samples)            # 1D feature
# true function: y = 1 + 2*x - 0.5*x^2
y_true = 1 + 2 * X - 0.5 * X**2
y = y_true + rng.normal(scale=2.0, size=n_samples)  # add noise

# Reshape X for sklearn (n_samples, n_features)
X = X.reshape(-1, 1)

# Split into train/test so we can show generalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# -------------- 2) Fit polynomial regression models of several degrees --------------
degrees = [1, 2, 3, 5, 7]
cv_scores = {}
models = {}

for deg in degrees:
    # Create pipeline: PolynomialFeatures -> LinearRegression
    model = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False),
                          LinearRegression())
    # Use negative MSE (sklearn convention) as CV score; 5-fold CV on training set
    scores = cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
    mean_cv_mse = -scores.mean()
    cv_scores[deg] = mean_cv_mse

    # Fit on full training set and store
    model.fit(X_train, y_train)
    models[deg] = model

# -------------- 3) Choose best degree by lowest CV MSE --------------
best_deg = min(cv_scores, key=cv_scores.get)
best_model = models[best_deg]

# Evaluate on test set
y_pred = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

# -------------- 4) Print results --------------
print("Cross-validated MSE by degree (lower is better):")
for deg in degrees:
    print(f"  degree={deg:>2}  CV MSE = {cv_scores[deg]:.4f}")
print(f"\nSelected degree: {best_deg}")
print(f"Test MSE (degree {best_deg}): {test_mse:.4f}")
print(f"Test R^2  (degree {best_deg}): {test_r2:.4f}")

# -------------- 5) Plot data and fitted curves --------------
# Create a dense X axis for smooth curves
X_plot = np.linspace(X.min() - 0.5, X.max() + 0.5, 400).reshape(-1, 1)

plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, label="Train data", alpha=0.7)
plt.scatter(X_test, y_test, label="Test data", alpha=0.9, marker='x')

# Plot fitted curves for a couple of degrees (including best)
for deg in [1, 2, best_deg]:
    model = models[deg]
    y_plot = model.predict(X_plot)
    label = f"deg={deg}"
    if deg == best_deg:
        plt.plot(X_plot, y_plot, label=f"{label} (selected)", linewidth=3)
    else:
        plt.plot(X_plot, y_plot, label=label, linewidth=1.5, linestyle='--')

# Also plot the true underlying function (for reference)
y_true_plot = 1 + 2 * X_plot.ravel() - 0.5 * X_plot.ravel()**2
plt.plot(X_plot, y_true_plot, label="True function", linestyle=':', linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Regression (model selection via CV)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
