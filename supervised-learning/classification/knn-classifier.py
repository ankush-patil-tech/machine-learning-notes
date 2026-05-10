Perfect — now let’s cover a **simple but very important algorithm (often asked in interviews)** 🔥

---

# =========================================

# 6. K-NEAREST NEIGHBORS (KNN)

# =========================================

```python id="knn001"
# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler


# =========================================
# 1. FEATURE SCALING (VERY IMPORTANT FOR KNN)
# =========================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# KNN is distance-based, so all features must be on same scale.
# Without scaling, features with larger values dominate the distance.


# =========================================
# 2. KNN MODEL
# =========================================
model = KNeighborsClassifier(n_neighbors=5)
# KNN stores training data and makes predictions based on nearest neighbors.
# It is a lazy learning algorithm (no explicit training phase).

model.fit(X_train, y_train)
# Stores data (no real "learning" like other models).

y_pred = model.predict(X_test)
# Predicts class based on majority vote of nearest neighbors.

# Used for: recommendation systems, pattern recognition
```

---

# =========================================

# MODEL EXPLANATION

# =========================================

```python id="knn002"
# How it works:
# Step 1: Choose value of K (number of neighbors)
# Step 2: Calculate distance (usually Euclidean)
# Step 3: Find K nearest points
# Step 4: Take majority vote

# Example:
# If 3 out of 5 neighbors are class A → prediction = A
```

---

# =========================================

# DISTANCE METRICS

# =========================================

```python id="knn003"
# Euclidean Distance (default):
# Straight-line distance between points

# Manhattan Distance:
# Distance along grid (like city blocks)

model = KNeighborsClassifier(metric='manhattan')
```

```python id="knn004"
# Choice of distance affects model performance.
# Euclidean → most common
```

---

# =========================================

# EVALUATION METRICS

# =========================================

```python id="knn005"
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

```python id="knn006"
# Accuracy → overall correctness

# Precision → important when false positives matter
# Recall → important when false negatives matter

# F1 Score → balance between precision and recall
```

---

# =========================================

# CONFUSION MATRIX

# =========================================

```python id="knn007"
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
```

---

# =========================================

# VISUALIZATION (2D ONLY)

# =========================================

```python id="knn008"
if X.shape[1] == 2:
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.title("KNN Data Distribution")
    plt.show()
```

```python id="knn009"
# KNN does not create a clear boundary like SVM.
# It depends on local neighbors for prediction.
```

---

# =========================================

# HYPERPARAMETER TUNING

# =========================================

```python id="knn010"
param_grid = {
    'n_neighbors': [3, 5, 7, 9],         # number of neighbors
    'weights': ['uniform', 'distance'],  # voting method
    'metric': ['euclidean', 'manhattan'] # distance type
}

grid = GridSearchCV(
    KNeighborsClassifier(),
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

```python id="knn011"
# n_neighbors (K):
# Small K → sensitive to noise (overfitting)
# Large K → smoother decision (may underfit)

# weights:
# uniform → all neighbors equal
# distance → closer neighbors have more influence

# metric:
# defines how distance is calculated
```

---

# =========================================

# FINAL MODEL EVALUATION

# =========================================

```python id="knn012"
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Final Accuracy:", accuracy)
```

---

# =========================================

# OVERFITTING / UNDERFITTING CHECK

# =========================================

```python id="knn013"
y_train_pred = best_model.predict(X_train)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# Small K → Overfitting
# Large K → Underfitting
```

---

# =========================================

# CROSS VALIDATION

# =========================================

```python id="knn014"
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

```python id="knn015"
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

```python id="knn016"
# KNN is very sensitive to noise.
# Incorrect data points can affect nearest neighbors and predictions.

# Larger K can reduce impact of noise.
```

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

```python id="knn017"
# KNN is a simple, non-parametric algorithm.

# No training phase → stores data

# Key tuning:
# K (n_neighbors), distance metric

# Important:
# Always scale data

# Limitation:
# Slow for large datasets
```

---
