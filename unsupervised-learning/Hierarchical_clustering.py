You’ve already covered almost the **entire ML landscape 🔥**, but let’s add a few **final important algorithms/concepts** that are often asked and useful in real projects.

---

# =========================================

# 16. HIERARCHICAL CLUSTERING

# =========================================

```python id="hc001"
# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


# =========================================
# 1. HIERARCHICAL CLUSTERING MODEL
# =========================================
Z = linkage(X, method='ward')
# Linkage builds a hierarchy (tree) of clusters.
# 'ward' minimizes variance within clusters.

# Used for: grouping similar patterns, bioinformatics, document clustering
```

---

# =========================================

# MODEL EXPLANATION

# =========================================

```python id="hc002"
# Key Idea:
# Build clusters step-by-step in a tree structure

# Types:
# Agglomerative → bottom-up (most common)
# Divisive → top-down

# Process:
# Start with each point as its own cluster
# Merge closest clusters step-by-step
```

---

# =========================================

# DENDROGRAM (VISUALIZATION)

# =========================================

```python id="hc003"
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
```

```python id="hc004"
# Dendrogram shows how clusters merge.
# Cut the tree at a certain height to get clusters.
```

---

# =========================================

# EXTRACT CLUSTERS

# =========================================

```python id="hc005"
labels = fcluster(Z, t=3, criterion='maxclust')
# t=3 → number of clusters
```

---

# =========================================

# HYPERPARAMETERS

# =========================================

```python id="hc006"
# method:
# ward → minimizes variance (best choice)
# single → nearest distance
# complete → farthest distance
```

---

# =========================================

# NOISE

# =========================================

```python id="hc007"
# Sensitive to noise and outliers.
# Once clusters merge, cannot be undone.
```

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

```python id="hc008"
# Hierarchical clustering builds a tree of clusters.

# Advantage:
# No need to predefine number of clusters

# Limitation:
# Computationally expensive
```

---

# =========================================

# 17. STOCHASTIC GRADIENT DESCENT (SGD)

# =========================================

```python id="sgd001"
# =========================================
# IMPORTS
# =========================================
from sklearn.linear_model import SGDClassifier


# =========================================
# 1. SGD MODEL
# =========================================
model = SGDClassifier()
# SGD is an optimization algorithm used to train models efficiently.
# It updates model weights using small batches of data.

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Used for: large datasets, online learning
```

---

# =========================================

# MODEL EXPLANATION

# =========================================

```python id="sgd002"
# Key Idea:
# Instead of using all data at once,
# update model step-by-step using small samples

# Advantage:
# Faster and scalable for big data

# Works with:
# Linear models, logistic regression, SVM
```

---

# =========================================

# HYPERPARAMETERS

# =========================================

```python id="sgd003"
# learning_rate:
# Step size for updates

# max_iter:
# Number of passes over data

# loss:
# Type of model (hinge → SVM, log → logistic)
```

---

# =========================================

# NOISE

# =========================================

```python id="sgd004"
# SGD is noisy by nature (updates are random).
# But this helps escape local minima.
```

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

```python id="sgd005"
# SGD is used for fast optimization.

# Best for:
# Large-scale ML problems

# Advantage:
# Fast and scalable
```

---

# =========================================

# 18. POLYNOMIAL REGRESSION

# =========================================

```python id="poly001"
# =========================================
# IMPORTS
# =========================================
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# =========================================
# 1. POLYNOMIAL REGRESSION MODEL
# =========================================
poly = PolynomialFeatures(degree=2)
# Converts linear features into polynomial features.

X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

# Used for: non-linear relationships
```

---

# =========================================

# MODEL EXPLANATION

# =========================================

```python id="poly002"
# Key Idea:
# Fit a curve instead of a straight line

# Example:
# y = ax^2 + bx + c

# Captures non-linear patterns in data
```

---

# =========================================

# OVERFITTING

# =========================================

```python id="poly003"
# High degree → overfitting (complex curve)
# Low degree → underfitting (too simple)

# Choose optimal degree carefully
```

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

```python id="poly004"
# Polynomial Regression models curved relationships.

# Advantage:
# Captures non-linearity

# Limitation:
# Easily overfits
```

---
