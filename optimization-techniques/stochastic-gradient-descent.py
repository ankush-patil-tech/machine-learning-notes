
# =========================================

# STOCHASTIC GRADIENT DESCENT (SGD)

# =========================================

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


---

# =========================================

# MODEL EXPLANATION

# =========================================

# Key Idea:
# Instead of using all data at once,
# update model step-by-step using small samples

# Advantage:
# Faster and scalable for big data

# Works with:
# Linear models, logistic regression, SVM


---

# =========================================

# HYPERPARAMETERS

# =========================================

# learning_rate:
# Step size for updates

# max_iter:
# Number of passes over data

# loss:
# Type of model (hinge → SVM, log → logistic)


---

# =========================================

# NOISE

# =========================================

# SGD is noisy by nature (updates are random).
# But this helps escape local minima.


---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

# SGD is used for fast optimization.

# Best for:
# Large-scale ML problems

# Advantage:
# Fast and scalable


---