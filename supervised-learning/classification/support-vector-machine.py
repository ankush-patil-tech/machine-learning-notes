
# =========================================

# 5. SUPPORT VECTOR MACHINE (SVM)

# =========================================

```python id="svm001"
# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve


# =========================================
# 1. SVM MODEL
# =========================================
model = SVC(kernel='linear', probability=True)
# SVM tries to find the best boundary (hyperplane) that separates classes.
# It focuses on maximizing the margin between classes.

model.fit(X_train, y_train)
# Learns the optimal boundary using training data.

y_pred = model.predict(X_test)
# Predicts class labels based on which side of boundary the point lies.

y_prob = model.predict_proba(X_test)[:, 1]
# Gives probability estimates (useful for threshold tuning)

# Used for: text classification, image recognition, bioinformatics
```

---

# =========================================

# MODEL EXPLANATION

# =========================================

```python id="svm002"
# Hyperplane:
# A line (2D) or plane (3D) that separates classes.

# Margin:
# Distance between hyperplane and nearest data points.

# Goal:
# Maximize margin → better generalization.

# Support Vectors:
# Data points closest to boundary.
# These points decide the position of hyperplane.

# Example:
# Classifying emails into spam/not spam
```

---

# =========================================

# TYPES OF KERNELS

# =========================================

```python id="svm003"
# Linear Kernel:
linear_model = SVC(kernel='linear')
# Used when data is linearly separable.

# Polynomial Kernel:
poly_model = SVC(kernel='poly', degree=3)
# Creates curved decision boundaries.

# RBF (Gaussian) Kernel:
rbf_model = SVC(kernel='rbf')
# Most commonly used kernel.
# Handles complex non-linear data.

# Sigmoid Kernel:
sigmoid_model = SVC(kernel='sigmoid')
```

```python id="svm004"
# Kernel trick:
# Transforms data into higher dimensions to make it separable.

# Choose:
# Linear → simple data
# RBF → complex data (default best choice)
```

---

# =========================================

# EVALUATION METRICS

# =========================================

```python id="svm005"
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

# METRIC EXPLANATION

# =========================================

```python id="svm006"
# Accuracy → overall correctness

# Precision → important when false positives matter
# Recall → important when false negatives matter

# F1 Score → balances both
```

---

# =========================================

# CONFUSION MATRIX

# =========================================

```python id="svm007"
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
```

---

# =========================================

# VISUALIZATION (DECISION BOUNDARY - 2D ONLY)

# =========================================

```python id="svm008"
# Only works for 2 features
if X.shape[1] == 2:
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)

    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, levels=[0])
    plt.title("Decision Boundary")
    plt.show()
```

```python id="svm009"
# Boundary separates classes.
# Support vectors lie near boundary.
# Wider margin → better model
```

---

# =========================================

# HYPERPARAMETER TUNING

# =========================================

```python id="svm010"
param_grid = {
    'C': [0.1, 1, 10],         # regularization parameter
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'] # for rbf kernel
}

grid = GridSearchCV(
    SVC(probability=True),
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

```python id="svm011"
# C:
# Controls trade-off between margin and classification error.
# Small C → wider margin → less overfitting
# Large C → narrow margin → may overfit

# gamma:
# Controls influence of single data point.
# High gamma → complex boundary → overfitting
# Low gamma → smooth boundary

# kernel:
# Determines type of decision boundary
```

---

# =========================================

# FINAL MODEL EVALUATION

# =========================================

```python id="svm012"
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Final Accuracy:", accuracy)
```

---

# =========================================

# OVERFITTING / UNDERFITTING CHECK

# =========================================

```python id="svm013"
y_train_pred = best_model.predict(X_train)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# Train high, Test low → Overfitting
# Both low → Underfitting
```

---

# =========================================

# CROSS VALIDATION

# =========================================

```python id="svm014"
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print("CV Scores:", cv_scores)
print("Mean CV:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

# High std → unstable model
```

---

# =========================================

# LEARNING CURVE

# =========================================

```python id="svm015"
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
```

---

# =========================================

# NOISE

# =========================================

```python id="svm016"
# SVM is sensitive to noise, especially when using hard margins.
# Outliers can affect boundary significantly.

# Using soft margin (C parameter) helps handle noise.
```

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

```python id="svm017"
# SVM finds optimal boundary with maximum margin.

# Works well for high-dimensional data.

# Key tuning:
# C, kernel, gamma

# Best for:
# text classification, image data

# Limitation:
# Slow on large datasets
```

---
