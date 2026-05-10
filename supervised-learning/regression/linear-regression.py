# =========================================
# LINEAR REGRESSION
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve


# =========================================
# DATA PREPARATION (X_train, y_train)
# =========================================

df = pd.read_csv("data.csv")
# Load dataset into DataFrame (table format)


# =========================================
# SELECT FEATURES (X) AND TARGET (y)
# =========================================
X = df.drop("target", axis=1)
# X contains input features (independent variables)

y = df["target"]
# y contains output/label (dependent variable)


# =========================================
# TRAIN-TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 80% → training data
# 20% → testing data
# random_state → same split every run


# =========================================
# EXPLANATION
# =========================================
# X → input features (like age, salary, etc.)
# y → output (like price, spam/not spam)

# train_test_split:
# Splits data into:
# - Training set (used to train model)
# - Testing set (used to evaluate model)

# test_size=0.2:
# 80% training, 20% testing

# random_state:
# Keeps split same every time (important for reproducibility)

# =========================================
# OPTIONAL (REAL PROJECTS)
# =========================================

1. Handling Categorical Data
X = pd.get_dummies(X)
# Converts categorical variables into numeric (one-hot encoding)
2. Feature Scaling (for KNN, SVM, PCA, etc.)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
# Fit on training data

X_test = scaler.transform(X_test)
# Apply same transformation to test data


# =========================================
# INTERVIEW READY ANSWER
# =========================================
# "First I load the dataset, separate features and target,
# then split into train and test sets using train_test_split.
# I also handle encoding and scaling depending on the model."


# =========================================
# BASE MODEL
# =========================================
model = LinearRegression()
# Fits straight-line relationship (y = mX + b)

model.fit(X_train, y_train)
# Learns slope and intercept from training data

y_pred = model.predict(X_test)
# Predicts values for unseen data

# Used for: price prediction, sales forecasting


# =========================================
# OTHER TYPES OF LINEAR REGRESSION
# =========================================

ridge = Ridge(alpha=1.0)
# L2 penalty → reduces large coefficients

lasso = Lasso(alpha=1.0)
# L1 penalty → sets some coefficients to zero

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
# Combines L1 + L2 regularization

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
elastic.fit(X_train, y_train)

# Ridge → keeps all features but reduces their impact.
# Lasso → removes some features completely.
# ElasticNet → balances both approaches.

# When to use:
# Ridge → when many correlated features exist.
# Lasso → when you want automatic feature selection.
# ElasticNet → when dataset is complex and mixed.

# =========================================
# MODEL EXPLANATION
# =========================================
# Equation: y = mX + b
# The model finds the best line that minimizes squared distance from all data points.
# It uses Least Squares method to reduce prediction error.

# Example:
# Predicting house price based on area.


# =========================================
# EVALUATION METRICS
# =========================================

mse = mean_squared_error(y_test, y_pred)
# MSE calculates the average of squared differences between actual and predicted values.
# Larger errors are penalized more due to squaring.

rmse = np.sqrt(mse)
# RMSE is the square root of MSE and gives error in the same unit as target.
# It is easier to interpret than MSE.

mae = mean_absolute_error(y_test, y_pred)
# MAE calculates average absolute difference.
# It treats all errors equally and is less sensitive to outliers.

r2 = r2_score(y_test, y_pred)
# R2 score shows how well the model explains variance in data.
# Value ranges from 0 to 1 (higher is better).

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

# Interpretation:
# Lower MSE, RMSE, MAE → better model
# Higher R2 → better model fit


# =========================================
# VISUALIZATION
# =========================================

plt.scatter(X_test, y_test)
# Actual data points

plt.plot(X_test, y_pred, color='red')
# Predicted line

plt.title("Linear Regression Fit")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

# If the red line closely follows points → good model.
# If far → poor fit.


# =========================================
# HYPERPARAMETER TUNING (RIDGE)
# =========================================

param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}
# alpha controls regularization strength.
# Higher alpha → simpler model, less overfitting but may underfit.

grid = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
# GridSearchCV tries all combinations of parameters.
# It uses cross-validation to find the best performing model.

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
# This gives the model with best hyperparameters.

print("Best Params:", grid.best_params_)

# Explanation:
# cv=5 → data split into 5 parts for validation.
# scoring → metric used to compare models.


# =========================================
# FINAL MODEL EVALUATION
# =========================================

y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Final RMSE:", rmse)

# This is the final performance after tuning.


# =========================================
# OVERFITTING / UNDERFITTING CHECK
# =========================================

y_train_pred = best_model.predict(X_train)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Interpretation:
# Train low, Test high → Overfitting (model memorized data)
# Both high → Underfitting (model too simple)
# Both low → Good model



# =========================================
# CROSS VALIDATION
# =========================================

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')

cv_rmse = np.sqrt(-cv_scores)

print("Mean CV RMSE:", cv_rmse.mean())
print("Std Dev:", cv_rmse.std())

# If standard deviation is high → model is unstable (high variance)
# If mean CV differs a lot from test → possible overfitting


# =========================================
# LEARNING CURVE
# =========================================

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, cv=5,
    scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_rmse = np.sqrt(-train_scores.mean(axis=1))
test_rmse = np.sqrt(-test_scores.mean(axis=1))

plt.plot(train_sizes, train_rmse, label="Train Error")
plt.plot(train_sizes, test_rmse, label="Validation Error")

plt.xlabel("Training Size")
plt.ylabel("RMSE")
plt.legend()
plt.title("Learning Curve")
plt.show()

# Interpretation:
# Large gap → Overfitting
# Both high → Underfitting
# Close and low → Good model

# =========================================
# NOISE
# =========================================
# Noise is random error or disturbance in data that cannot be learned.
# Example: wrong labels, measurement errors, randomness in real-world data.

# Even the best model cannot fully remove noise.


# =========================================
# FINAL INTERVIEW SUMMARY
# =========================================
# Linear Regression is simple, fast, and interpretable.
# Use Ridge/Lasso to handle overfitting.
# Evaluate using RMSE and R2 score.
# Always validate using cross-validation and learning curves.
