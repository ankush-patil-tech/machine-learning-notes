Perfect 🔥 — now we move to a **VERY IMPORTANT concept used in almost every ML pipeline**

---

# =========================================

# 10. PCA (PRINCIPAL COMPONENT ANALYSIS)

# =========================================

```python id="pca001"
# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================================
# 1. FEATURE SCALING (IMPORTANT)
# =========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# PCA is sensitive to scale, so features must be standardized.
# Without scaling, features with larger values dominate components.


# =========================================
# 2. PCA MODEL
# =========================================
pca = PCA(n_components=2)
# PCA reduces number of features while preserving maximum information (variance).

X_reduced = pca.fit_transform(X_scaled)
# Transforms data into new feature space (principal components).

# Used for: dimensionality reduction, visualization, noise reduction
```

---

# =========================================

# MODEL EXPLANATION

# =========================================

```python id="pca002"
# Key Idea:
# Convert original features into new features (principal components)

# Principal Components:
# New axes that capture maximum variance in data

# PC1 → direction of maximum variance
# PC2 → second highest variance (orthogonal to PC1)

# Goal:
# Reduce dimensions while keeping most information
```

---

# =========================================

# EXPLAINED VARIANCE

# =========================================

```python id="pca003"
explained_variance = pca.explained_variance_ratio_

print("Explained Variance:", explained_variance)
print("Total Variance Retained:", sum(explained_variance))
```

```python id="pca004"
# Shows how much information each component retains.

# Example:
# [0.8, 0.15] → 95% information retained

# Higher total variance → better representation
```

---

# =========================================

# VISUALIZATION (2D)

# =========================================

```python id="pca005"
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title("PCA Reduced Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

```python id="pca006"
# Data is projected onto new axes (PC1, PC2)
# Helps visualize high-dimensional data in 2D
```

---

# =========================================

# CHOOSING NUMBER OF COMPONENTS

# =========================================

```python id="pca007"
pca = PCA()
pca.fit(X_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.title("Explained Variance Curve")
plt.show()
```

```python id="pca008"
# Choose number of components where curve flattens
# Usually retain 90%–95% variance
```

---

# =========================================

# HYPERPARAMETERS

# =========================================

```python id="pca009"
pca = PCA(
    n_components=2,     # number of components
    whiten=False        # decorrelate features
)
```

---

# =========================================

# HYPERPARAMETER EXPLANATION

# =========================================

```python id="pca010"
# n_components:
# Number of dimensions to keep

# whiten:
# Makes components uncorrelated with unit variance
# Useful for some ML models
```

---

# =========================================

# OVERFITTING / UNDERFITTING

# =========================================

```python id="pca011"
# Too few components → Underfitting (loss of information)

# Too many components → Overfitting (retain noise)

# Goal:
# Keep optimal number of components
```

---

# =========================================

# NOISE REDUCTION

# =========================================

```python id="pca012"
# PCA removes noise by keeping only important components.
# Low-variance components often represent noise.
```

---

# =========================================

# WHEN TO USE PCA

# =========================================

```python id="pca013"
# Use PCA when:
# - Too many features
# - Features are correlated
# - Want faster model training
# - Need visualization

# Example:
# Reducing 100 features → 10 features
```

---

# =========================================

# LIMITATIONS

# =========================================

```python id="pca014"
# PCA is linear → cannot capture complex relationships

# Loss of interpretability:
# New features (PCs) are not easily understandable

# Sensitive to scaling
```

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

```python id="pca015"
# PCA is used for dimensionality reduction.

# Key idea:
# Transform data into components with maximum variance

# Benefits:
# Reduces features, removes noise, improves performance

# Key tuning:
# n_components (variance threshold)

# Limitation:
# Hard to interpret components
```

---

