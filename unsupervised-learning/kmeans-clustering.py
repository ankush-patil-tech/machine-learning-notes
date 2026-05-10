Perfect timing 🔥 — now we move to **UNSUPERVISED LEARNING (very important concept shift)**

---

# =========================================

# 8. K-MEANS CLUSTERING

# =========================================

```python id="km001"
# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV


# =========================================
# 1. K-MEANS MODEL
# =========================================
model = KMeans(n_clusters=3, random_state=42)
# K-Means groups data into K clusters based on similarity (distance).
# It is an unsupervised algorithm → no labels (y) required.

model.fit(X)
# Learns cluster centers (centroids) from data.

labels = model.labels_
# Assigns each data point to a cluster.

centroids = model.cluster_centers_
# Coordinates of cluster centers.

# Used for: customer segmentation, grouping users/products
```

---

# =========================================

# MODEL EXPLANATION

# =========================================

```python id="km002"
# How it works:
# Step 1: Choose number of clusters (K)
# Step 2: Randomly initialize centroids
# Step 3: Assign points to nearest centroid
# Step 4: Update centroid (mean of points)
# Step 5: Repeat until convergence

# Goal:
# Minimize distance within clusters (inertia)
```

---

# =========================================

# VISUALIZATION (2D ONLY)

# =========================================

```python id="km003"
if X.shape[1] == 2:
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')

    plt.title("K-Means Clustering")
    plt.show()
```

```python id="km004"
# Different colors → different clusters
# Red X → cluster centers (centroids)
```

---

# =========================================

# ELBOW METHOD (CHOOSING K)

# =========================================

```python id="km005"
inertia = []

K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
```

```python id="km006"
# Inertia = sum of squared distances within clusters

# Elbow point → optimal K
# After this point → diminishing improvement
```

---

# =========================================

# SILHOUETTE SCORE

# =========================================

```python id="km007"
score = silhouette_score(X, labels)
print("Silhouette Score:", score)
```

```python id="km008"
# Measures how well points fit in their cluster

# Range:
# +1 → well separated clusters
# 0 → overlapping clusters
# -1 → wrong clustering
```

---

# =========================================

# HYPERPARAMETER TUNING

# =========================================

```python id="km009"
# KMeans does not use GridSearchCV directly (unsupervised).
# We manually test different values of K.

best_score = -1
best_k = 2

for k in range(2, 10):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    if score > best_score:
        best_score = score
        best_k = k

print("Best K:", best_k)
```

---

# =========================================

# HYPERPARAMETER EXPLANATION

# =========================================

```python id="km010"
# n_clusters (K):
# Number of clusters to form

# init:
# Method to initialize centroids (k-means++ is default)

# max_iter:
# Maximum number of iterations

# random_state:
# Ensures reproducibility
```

---

# =========================================

# OVERFITTING / UNDERFITTING (CLUSTERING VIEW)

# =========================================

```python id="km011"
# Too small K → Underfitting (large clusters, poor separation)

# Too large K → Overfitting (too many small clusters)

# Balance needed using Elbow + Silhouette methods
```

---

# =========================================

# CROSS VALIDATION (LIMITATION)

# =========================================

```python id="km012"
# Traditional cross-validation is not used in clustering.
# Because there are no labels to compare predictions.

# Instead we use:
# - Inertia
# - Silhouette Score
```

---

# =========================================

# NOISE

# =========================================

```python id="km013"
# K-Means is sensitive to noise and outliers.
# Outliers can distort centroid positions.

# Solution:
# Remove outliers or use DBSCAN
```

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

```python id="km014"
# K-Means is an unsupervised clustering algorithm.

# Key idea:
# Group similar data points together

# Key tuning:
# Number of clusters (K)

# Evaluation:
# Elbow method, Silhouette score

# Limitation:
# Sensitive to noise and initial centroids
```

---
