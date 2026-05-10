# =========================================

# HIERARCHICAL CLUSTERING

# =========================================

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


---

# =========================================

# MODEL EXPLANATION

# =========================================

# Key Idea:
# Build clusters step-by-step in a tree structure

# Types:
# Agglomerative → bottom-up (most common)
# Divisive → top-down

# Process:
# Start with each point as its own cluster
# Merge closest clusters step-by-step


---

# =========================================

# DENDROGRAM (VISUALIZATION)

# =========================================

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()


# Dendrogram shows how clusters merge.
# Cut the tree at a certain height to get clusters.


---

# =========================================

# EXTRACT CLUSTERS

# =========================================

labels = fcluster(Z, t=3, criterion='maxclust')
# t=3 → number of clusters


---

# =========================================

# HYPERPARAMETERS

# =========================================

# method:
# ward → minimizes variance (best choice)
# single → nearest distance
# complete → farthest distance


---

# =========================================

# NOISE

# =========================================

# Sensitive to noise and outliers.
# Once clusters merge, cannot be undone.


---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

# Hierarchical clustering builds a tree of clusters.

# Advantage:
# No need to predefine number of clusters

# Limitation:
# Computationally expensive


---
