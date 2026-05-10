# =========================================

# LOGISTIC REGRESSION

# =========================================

# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve


# =========================================
# 1. LOGISTIC REGRESSION MODEL
# =========================================
model = LogisticRegression()
# Logistic Regression is used for classification problems (not regression).
# It predicts probability of a class using a sigmoid function.

model.fit(X_train, y_train)
# The model learns weights that best separate classes.

y_pred = model.predict(X_test)
# Predicts class labels (0 or 1)

y_prob = model.predict_proba(X_test)[:, 1]
# Gives probability of class 1 (useful for threshold tuning)

# Used for: spam detection, disease prediction, fraud detection

---

# =========================================

# MODEL EXPLANATION

# =========================================

# Logistic Function (Sigmoid):
# p = 1 / (1 + e^(-z))

# What it does:
# Converts linear output into probability between 0 and 1.

# How it works:
# If probability > 0.5 → class 1
# If probability < 0.5 → class 0

# Example:
# Predict if email is spam (1) or not (0)

---

# =========================================

# TYPES OF LOGISTIC REGRESSION

# =========================================

# Binary Logistic Regression:
# Only 2 classes (0/1)

# Multinomial Logistic Regression:
# More than 2 classes

multi_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

multi_model.fit(X_train, y_train)


# Binary → spam vs not spam
# Multinomial → classify digits (0–9)


---

# =========================================

# EVALUATION METRICS (CLASSIFICATION)

# =========================================

accuracy = accuracy_score(y_test, y_pred)
# Accuracy = correct predictions / total predictions

precision = precision_score(y_test, y_pred)
# Precision = how many predicted positives are actually correct

recall = recall_score(y_test, y_pred)
# Recall = how many actual positives are correctly identified

f1 = f1_score(y_test, y_pred)
# F1 Score = balance between precision and recall

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

---

# =========================================

# EVALUATION EXPLANATION

# =========================================

# Accuracy → good when classes are balanced

# Precision → important when false positives are costly
# Example: spam filter (don't mark important mail as spam)

# Recall → important when false negatives are costly
# Example: disease detection (don't miss a sick patient)

# F1 Score → best when you need balance between precision & recall

---

# =========================================

# CONFUSION MATRIX

# =========================================

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Confusion Matrix:
# [[TN  FP]
#  [FN  TP]]

# TN → correct negatives
# FP → false positives
# FN → false negatives
# TP → correct positives


---

# =========================================

# VISUALIZATION (SIGMOID CURVE)

# =========================================

# Visualizing sigmoid function
x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.exp(-x))

plt.plot(x, y)
plt.title("Sigmoid Function")
plt.xlabel("Input")
plt.ylabel("Probability")
plt.show()


# Graph Explanation:
# Curve converts any value into probability (0 to 1)
# Middle point (0) → probability = 0.5

---

# =========================================

# HYPERPARAMETER TUNING

# =========================================

param_grid = {
    'C': [0.01, 0.1, 1, 10],        # inverse of regularization strength
    'penalty': ['l1', 'l2'],        # type of regularization
    'solver': ['liblinear']         # required for l1
}

grid = GridSearchCV(
    LogisticRegression(),
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

# C:
# Smaller value → stronger regularization → simpler model
# Larger value → weaker regularization → complex model

# penalty:
# l1 → feature selection (removes features)
# l2 → reduces feature weights

# solver:
# optimization algorithm used to train model

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
# Both high → Good model

---

# =========================================

# CROSS VALIDATION

# =========================================

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print("CV Scores:", cv_scores)
print("Mean CV:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

# High std → unstable model
# Close CV and test → good generalization

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

# Gap large → Overfitting
# Both low → Underfitting

---

# =========================================

# NOISE

# =========================================

# Noise refers to random or incorrect data points that do not follow pattern.
# Logistic Regression may struggle if data has too much noise or overlap.

# Example:
# Incorrect labels in training data

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

# Logistic Regression is used for classification problems.
# It predicts probability using sigmoid function.

# Key metrics:
# Accuracy, Precision, Recall, F1 Score

# Use regularization (C parameter) to prevent overfitting.

# Always check confusion matrix and cross-validation.

---
