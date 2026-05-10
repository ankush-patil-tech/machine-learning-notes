# =========================================

# GRADIENT BOOSTING (GBM)

# =========================================


# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve


# =========================================
# 1. GRADIENT BOOSTING MODEL
# =========================================
model = GradientBoostingClassifier(n_estimators=100)
# Gradient Boosting builds models sequentially.
# Each new model tries to correct errors made by previous models.

model.fit(X_train, y_train)
# Learns patterns step-by-step by focusing on mistakes.

y_pred = model.predict(X_test)
# Final prediction combines all weak learners.

# Used for: fraud detection, ranking systems, structured data problems


---

# =========================================

# MODEL EXPLANATION

# =========================================

# Key Idea:
# Instead of building independent trees (like Random Forest),
# models are built sequentially.

# Step-by-step:
# 1. First model makes predictions
# 2. Calculate errors (residuals)
# 3. Next model learns from errors
# 4. Repeat and combine all models

# Goal:
# Reduce error gradually using gradient descent


---

# =========================================

# KEY CONCEPTS

# =========================================

# Weak Learners:
# Small decision trees (usually shallow)

# Learning Rate:
# Controls how much each tree contributes

# Residuals:
# Errors made by previous model

# Boosting:
# Focus on correcting mistakes


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

005"
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


---

# =========================================

# HYPERPARAMETER TUNING

# =========================================

param_grid = {
    'n_estimators': [50, 100, 200],      # number of trees
    'learning_rate': [0.01, 0.1, 0.2],   # step size
    'max_depth': [3, 5],                 # depth of trees
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    GradientBoostingClassifier(),
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
# Number of trees → more trees improve learning but increase time

# learning_rate:
# Controls contribution of each tree
# Small → slow learning but better generalization
# Large → faster but risk overfitting

# max_depth:
# Controls complexity of trees

# min_samples_split:
# Prevents unnecessary splits


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

# Train high, Test low → Overfitting
# Both low → Underfitting


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

# Large gap → Overfitting


---

# =========================================

# NOISE

# =========================================

# Gradient Boosting is sensitive to noise.
# Because it tries to correct every error, including noise.

# Solution:
# Use small learning rate + regularization


---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

# Gradient Boosting builds models sequentially.

# Key strength:
# High accuracy

# Key tuning:
# learning_rate, n_estimators, max_depth

# Advantage:
# Learns complex patterns

# Limitation:
# Slow and sensitive to noise


---


 