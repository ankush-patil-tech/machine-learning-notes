# =========================================

# RANDOM FOREST

# =========================================

# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve


# =========================================
# 1. RANDOM FOREST MODEL
# =========================================
model = RandomForestClassifier(n_estimators=100)
# Random Forest is an ensemble model that combines multiple decision trees.
# Each tree is trained on random data samples and random features.

model.fit(X_train, y_train)
# The model builds many trees and aggregates their predictions.

y_pred = model.predict(X_test)
# Final prediction is based on majority voting (classification).

# Used for: fraud detection, recommendation systems, risk analysis


---

# =========================================

# MODEL EXPLANATION

# =========================================


# How it works:
# Step 1: Create multiple random samples of data (bootstrapping)
# Step 2: Train a decision tree on each sample
# Step 3: Each tree gives prediction
# Step 4: Final output = majority vote

# Key idea:
# Combining many weak models → strong model

# Advantage:
# Reduces overfitting compared to single decision tree


---

# =========================================

# KEY CONCEPTS

# =========================================


# Bagging (Bootstrap Aggregation):
# Random samples of data are used to train each tree.

# Feature Randomness:
# Each tree sees only a subset of features.

# Result:
# Trees become less correlated → better performance.


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

# METRIC EXPLANATION

# =========================================


# Accuracy → overall performance

# Precision → important when false positives matter
# Recall → important when missing positives is risky

# F1 Score → balance between precision and recall


---

# =========================================

# CONFUSION MATRIX

# =========================================

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# Helps analyze types of classification errors.
# Useful in imbalanced datasets.


---

# =========================================

# FEATURE IMPORTANCE

# =========================================

import pandas as pd

feature_importance = pd.Series(model.feature_importances_, index=feature_names)

feature_importance.sort_values(ascending=False).plot(kind='bar')
plt.title("Feature Importance")
plt.show()


# Shows which features contribute most to prediction.
# Helps in feature selection and interpretation.


---

# =========================================

# HYPERPARAMETER TUNING

# =========================================

param_grid = {
    'n_estimators': [50, 100, 200],      # number of trees
    'max_depth': [None, 5, 10],          # depth of each tree
    'min_samples_split': [2, 5],         # split condition
    'min_samples_leaf': [1, 2],          # leaf size
    'max_features': ['sqrt', 'log2']     # number of features per split
}

grid = GridSearchCV(
    RandomForestClassifier(),
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
# Number of trees in forest.
# More trees → better performance but slower.

# max_depth:
# Limits depth of trees.
# Prevents overfitting.

# min_samples_split:
# Minimum samples required to split node.

# min_samples_leaf:
# Minimum samples at leaf → smoother model.

# max_features:
# Number of features used per split.
# Controls randomness and diversity.


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

# Random Forest usually reduces overfitting compared to decision tree.
# But still possible if trees are too deep.


---

# =========================================

# CROSS VALIDATION

# =========================================

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print("CV Scores:", cv_scores)
print("Mean CV:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

# Low std → stable model
# High std → variance still exists


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

# Smaller gap than decision tree → better generalization


---

# =========================================

# NOISE

# =========================================

# Random Forest handles noise better than decision trees.
# Because averaging multiple trees reduces impact of noisy data.


---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

# Random Forest is an ensemble of decision trees.

# Key strength:
# Reduces overfitting and improves accuracy.

# Works well on most datasets → safe default model.

# Key tuning:
# n_estimators, max_depth, max_features

# Advantage:
# Handles noise and non-linearity well


---

