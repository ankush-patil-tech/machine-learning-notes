Let’s go 🔥 — now we move to a **very powerful clustering algorithm that fixes K-Means problems**

---

# =========================================

# 9. DBSCAN (DENSITY-BASED CLUSTERING)

# =========================================

```python id="db001"
# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


# =========================================
# 1. DBSCAN MODEL
# =========================================
model = DBSCAN(eps=0.5, min_samples=5)
# DBSCAN groups points based on density (not distance like K-Means).
# It can automatically detect number of clusters and identify noise.

labels = model.fit_predict(X)
# Assigns cluster labels (-1 means noise/outlier).

# Used for: anomaly detection, spatial clustering, fraud detection
```

---

# =========================================

# MODEL EXPLANATION

# =========================================

```python id="db002"
# Key Idea:
# Points close together form clusters (dense regions)

# Types of points:
# Core Point → has many neighbors
# Border Point → near core point
# Noise Point → isolated point (label = -1)

# Goal:
# Find dense regions and separate noise
```

---

# =========================================

# VISUALIZATION (2D ONLY)

# =========================================

```python id="db003"
if X.shape[1] == 2:
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title("DBSCAN Clustering")
    plt.show()
```

```python id="db004"
# Different colors → clusters
# Label -1 → noise (outliers)
```

---

# =========================================

# EVALUATION (CLUSTERING)

# =========================================

```python id="db005"
# Remove noise points before evaluation
valid_points = labels != -1

if len(set(labels[valid_points])) > 1:
    score = silhouette_score(X[valid_points], labels[valid_points])
    print("Silhouette Score:", score)
```

---

```python id="db006"
# Higher silhouette score → better clustering
# DBSCAN works best when clusters are dense and well-separated
```

---

# =========================================

# HYPERPARAMETERS

# =========================================

```python id="db007"
# eps:
# Maximum distance between points to be considered neighbors

# min_samples:
# Minimum number of points required to form a cluster

model = DBSCAN(eps=0.3, min_samples=10)
```

---

# =========================================

# HYPERPARAMETER EXPLANATION

# =========================================

```python id="db008"
# eps:
# Small eps → many small clusters + more noise
# Large eps → fewer clusters, may merge clusters

# min_samples:
# Higher value → stricter clustering (fewer clusters)
# Lower value → more clusters
```

---

# =========================================

# PARAMETER TUNING (MANUAL)

# =========================================

```python id="db009"
best_score = -1
best_params = {}

for eps in [0.3, 0.5, 0.7]:
    for min_samples in [3, 5, 10]:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

        valid = labels != -1

        if len(set(labels[valid])) > 1:
            score = silhouette_score(X[valid], labels[valid])

            if score > best_score:
                best_score = score
                best_params = {"eps": eps, "min_samples": min_samples}

print("Best Params:", best_params)
```

---

# =========================================

# OVERFITTING / UNDERFITTING (CLUSTER VIEW)

# =========================================

```python id="db010"
# Too small eps → Underfitting (many points marked as noise)

# Too large eps → Overfitting (clusters merge incorrectly)

# Need balance for meaningful clusters
```

---

# =========================================

# CROSS VALIDATION (LIMITATION)

# =========================================

```python id="db011"
# DBSCAN does not use cross-validation.
# Because it is unsupervised (no labels available).

# Instead use:
# - Silhouette score
# - Visual inspection
```

---

# =========================================

# NOISE HANDLING

# =========================================

```python id="db012"
# DBSCAN is very good at handling noise.
# It automatically labels outliers as -1.

# Advantage over K-Means:
# Does not force every point into a cluster.
```

---

# =========================================

# K-MEANS vs DBSCAN (IMPORTANT)

# =========================================

```python id="db013"
# K-Means:
# - Needs number of clusters (K)
# - Sensitive to noise
# - Assumes spherical clusters

# DBSCAN:
# - No need to define clusters
# - Handles noise well
# - Works for irregular shapes
```

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

```python id="db014"
# DBSCAN is a density-based clustering algorithm.

# Key strength:
# Automatically detects clusters and noise

# Key tuning:
# eps, min_samples

# Advantage:
# Handles outliers and irregular cluster shapes

# Limitation:
# Sensitive to parameter selection
```

---

