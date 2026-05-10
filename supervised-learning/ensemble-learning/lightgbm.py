

# =========================================

# 13. LIGHTGBM (LIGHT GRADIENT BOOSTING)

# =========================================

```python id="lgb001"
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
```

---

# =========================================

# MODEL EXPLANATION

# =========================================

```python id="lgb002"
# Key Idea:
# Builds trees leaf-wise (grows deeper branches first)

# Difference from XGBoost:
# XGBoost → level-wise growth
# LightGBM → leaf-wise growth (faster and more accurate)

# Advantage:
# Faster training and better performance on large data
```

---

# =========================================

# KEY CONCEPTS

# =========================================

```python id="lgb003"
# Leaf-wise growth:
# Splits the leaf with maximum loss reduction

# Histogram-based learning:
# Converts continuous values into bins → faster computation

# Gradient-based sampling:
# Focuses on important data points
```

---

# =========================================

# EVALUATION METRICS

# =========================================

```python id="lgb004"
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
```

---

# =========================================

# CONFUSION MATRIX

# =========================================

```python id="lgb005"
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
```

---

# =========================================

# HYPERPARAMETER TUNING

# =========================================

```python id="lgb006"
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
```

---

# =========================================

# HYPERPARAMETER EXPLANATION

# =========================================

```python id="lgb007"
# num_leaves:
# Controls complexity of tree (more leaves → more complex)

# learning_rate:
# Controls contribution of each tree

# max_depth:
# Limits depth to prevent overfitting

# subsample:
# Uses fraction of data → improves generalization
```

---

# =========================================

# OVERFITTING / UNDERFITTING CHECK

# =========================================

```python id="lgb008"
y_train_pred = best_model.predict(X_train)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# LightGBM can overfit if num_leaves is too large.
```

---

# =========================================

# CROSS VALIDATION

# =========================================

```python id="lgb009"
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print("Mean CV:", cv_scores.mean())
print("Std Dev:", cv_scores.std())
```

---

# =========================================

# NOISE

# =========================================

```python id="lgb010"
# LightGBM is sensitive to noise due to aggressive leaf-wise growth.
# Proper regularization is required.
```

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

```python id="lgb011"
# LightGBM is a fast and efficient boosting algorithm.

# Key strength:
# High speed and performance

# Key tuning:
# num_leaves, learning_rate

# Best for:
# Large datasets
```

---

# =========================================

# 14. CATBOOST

# =========================================

```python id="cat001"
# =========================================
# IMPORTS
# =========================================
from catboost import CatBoostClassifier


# =========================================
# 1. CATBOOST MODEL
# =========================================
model = CatBoostClassifier(verbose=0)
# CatBoost is designed to handle categorical features automatically.
# It reduces need for preprocessing like encoding.

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Used for: business datasets with many categorical variables
```

---

# =========================================

# MODEL EXPLANATION

# =========================================

```python id="cat002"
# Key Idea:
# Handles categorical data internally

# Uses ordered boosting:
# Prevents data leakage

# Advantage:
# No need for label encoding or one-hot encoding
```

---

# =========================================

# EVALUATION METRICS

# =========================================

```python id="cat003"
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

---

# =========================================

# HYPERPARAMETER TUNING

# =========================================

```python id="cat004"
param_grid = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [4, 6, 10]
}

grid = GridSearchCV(
    CatBoostClassifier(verbose=0),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Params:", grid.best_params_)
```

---

# =========================================

# HYPERPARAMETER EXPLANATION

# =========================================

```python id="cat005"
# iterations:
# Number of boosting steps

# learning_rate:
# Step size for learning

# depth:
# Tree depth → controls complexity
```

---

# =========================================

# NOISE

# =========================================

```python id="cat006"
# CatBoost handles noise better due to ordered boosting.
# It reduces overfitting compared to other boosting models.
```

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

```python id="cat007"
# CatBoost is best for categorical data.

# Key strength:
# Minimal preprocessing required

# Advantage:
# Handles categorical variables automatically

# Used in:
# Business and structured datasets
```

---


