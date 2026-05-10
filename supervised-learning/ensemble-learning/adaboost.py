# =========================================

# ADABOOST (ADAPTIVE BOOSTING)

# =========================================

# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve


# =========================================
# 1. ADABOOST MODEL
# =========================================
model = AdaBoostClassifier(n_estimators=50)
# AdaBoost is a boosting algorithm that focuses more on incorrectly predicted samples.
# Each new model gives more importance (weight) to mistakes.

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Used for: classification tasks where boosting simple models improves accuracy


---

# =========================================

# MODEL EXPLANATION

# =========================================

# Key Idea:
# Sequentially train models and adjust weights of data points

# Step-by-step:
# 1. Train first model
# 2. Increase weight of wrong predictions
# 3. Train next model focusing on mistakes
# 4. Combine all models

# Goal:
# Reduce overall classification error


---

# =========================================

# KEY CONCEPTS

# =========================================

# Weak Learners:
# Usually decision stumps (trees with depth=1)

# Weights:
# Misclassified points get higher importance

# Final Prediction:
# Weighted combination of all models


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
    'n_estimators': [50, 100, 200],     # number of weak learners
    'learning_rate': [0.01, 0.1, 1.0]   # controls contribution of each model
}

grid = GridSearchCV(
    AdaBoostClassifier(),
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
# Number of weak learners

# learning_rate:
# Controls how much each model contributes
# Lower → slower learning, better generalization
# Higher → faster but risk overfitting


---

# =========================================

# OVERFITTING / UNDERFITTING CHECK

# =========================================

y_train_pred = best_model.predict(X_train)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# AdaBoost can overfit if too many estimators are used


---

# =========================================

# CROSS VALIDATION

# =========================================

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print("Mean CV:", cv_scores.mean())
print("Std Dev:", cv_scores.std())


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

# AdaBoost is very sensitive to noise and outliers.
# Because it focuses more on misclassified points (including noisy ones).

# Noisy data can mislead the model.


---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

# AdaBoost is a boosting algorithm that focuses on mistakes.

# Key strength:
# Improves weak models

# Key tuning:
# n_estimators, learning_rate

# Limitation:
# Sensitive to noise


---
