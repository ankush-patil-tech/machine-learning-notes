# =========================================

# LIGHTGBM (LIGHT GRADIENT BOOSTING)

# =========================================

# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve


# =========================================
# 1. LIGHTGBM MODEL
# =========================================
model = LGBMClassifier(n_estimators=100)
# LightGBM is a faster and more efficient version of Gradient Boosting.
# It uses a leaf-wise tree growth strategy instead of level-wise.

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Used for: large datasets, real-time prediction systems


---

# =========================================

# MODEL EXPLANATION

# =========================================

# Key Idea:
# Builds trees leaf-wise (grows deeper branches first)

# Difference from XGBoost:
# XGBoost → level-wise growth
# LightGBM → leaf-wise growth (faster and more accurate)

# Advantage:
# Faster training and better performance on large data


---

# =========================================

# KEY CONCEPTS

# =========================================

# Leaf-wise growth:
# Splits the leaf with maximum loss reduction

# Histogram-based learning:
# Converts continuous values into bins → faster computation

# Gradient-based sampling:
# Focuses on important data points


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
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [-1, 5, 10],
    'num_leaves': [31, 50],
    'subsample': [0.8, 1.0]
}

grid = GridSearchCV(
    LGBMClassifier(),
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

# num_leaves:
# Controls complexity of tree (more leaves → more complex)

# learning_rate:
# Controls contribution of each tree

# max_depth:
# Limits depth to prevent overfitting

# subsample:
# Uses fraction of data → improves generalization


---

# =========================================

# OVERFITTING / UNDERFITTING CHECK

# =========================================

y_train_pred = best_model.predict(X_train)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# LightGBM can overfit if num_leaves is too large.


---

# =========================================

# CROSS VALIDATION

# =========================================

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print("Mean CV:", cv_scores.mean())
print("Std Dev:", cv_scores.std())


---

# =========================================

# NOISE

# =========================================

# LightGBM is sensitive to noise due to aggressive leaf-wise growth.
# Proper regularization is required.


---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

# LightGBM is a fast and efficient boosting algorithm.

# Key strength:
# High speed and performance

# Key tuning:
# num_leaves, learning_rate

# Best for:
# Large datasets


---

