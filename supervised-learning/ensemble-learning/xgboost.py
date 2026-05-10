
# =========================================

# XGBOOST (EXTREME GRADIENT BOOSTING)

# =========================================


# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve


# =========================================
# 1. XGBOOST MODEL
# =========================================
model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
# XGBoost is an advanced version of Gradient Boosting.
# It is optimized for speed and performance with regularization.

model.fit(X_train, y_train)
# Learns patterns sequentially while correcting previous errors.

y_pred = model.predict(X_test)
# Final prediction combines all boosted trees.

# Used for: competitions, fraud detection, recommendation systems, tabular data


---

# =========================================

# MODEL EXPLANATION

# =========================================

# Key Idea:
# Same as Gradient Boosting but optimized

# Improvements over GBM:
# - Regularization (prevents overfitting)
# - Parallel processing (faster)
# - Handles missing values automatically

# Goal:
# Minimize loss function using gradient descent


---

# =========================================

# KEY CONCEPTS

# =========================================

# Boosting:
# Sequential learning from errors

# Regularization:
# Penalizes complex models to avoid overfitting

# Tree Pruning:
# Removes unnecessary branches

# Weighted learning:
# Focus more on difficult samples


---

# =========================================

# EVALUATION METRICS

# =========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


---

# =========================================

# CONFUSION MATRIX

# =========================================

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


---

# =========================================

# HYPERPARAMETER TUNING

# =========================================

param_grid = {
    'n_estimators': [100, 200],        # number of trees
    'learning_rate': [0.01, 0.1],      # step size
    'max_depth': [3, 6],               # depth of trees
    'subsample': [0.8, 1.0],           # row sampling
    'colsample_bytree': [0.8, 1.0]     # feature sampling
}

grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Params:", grid.best_params_)


---

# =========================================

# HYPERPARAMETER EXPLANATION

# =========================================

# n_estimators:
# Number of boosting rounds (trees)

# learning_rate:
# Controls how much each tree contributes

# max_depth:
# Controls complexity of trees

# subsample:
# Fraction of data used for each tree (reduces overfitting)

# colsample_bytree:
# Fraction of features used (adds randomness)


---

# =========================================

# FINAL MODEL EVALUATION

# =========================================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Final Accuracy:", accuracy)


---

# =========================================

# OVERFITTING / UNDERFITTING CHECK

# =========================================

y_train_pred = best_model.predict(X_train)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# XGBoost handles overfitting better due to regularization.


---

# =========================================

# CROSS VALIDATION

# =========================================

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print("CV Scores:", cv_scores)
print("Mean CV:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

# Low std → stable model


---

# =========================================

# LEARNING CURVE

# =========================================

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label="Train Accuracy")
plt.plot(train_sizes, test_mean, label="Validation Accuracy")

plt.legend()
plt.title("Learning Curve")
plt.show()


---

# =========================================

# NOISE

# =========================================

# XGBoost handles noise better than basic Gradient Boosting.
# Regularization helps ignore noisy patterns.

# Still requires proper tuning.


---

# =========================================

# FEATURE IMPORTANCE

# =========================================

import pandas as pd

importance = best_model.feature_importances_

pd.Series(importance).sort_values(ascending=False).plot(kind='bar')
plt.title("Feature Importance")
plt.show()


# Shows which features influence predictions most.
# Helps in feature selection and interpretation.


---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

# XGBoost is an optimized Gradient Boosting algorithm.

# Key strength:
# High accuracy + fast performance

# Key tuning:
# learning_rate, max_depth, n_estimators

# Advantage:
# Handles missing data, regularization, scalable

# Most used model in real-world ML


---
