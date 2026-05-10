

# =========================================

# POLYNOMIAL REGRESSION

# =========================================

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


---

# =========================================

# MODEL EXPLANATION

# =========================================

# Key Idea:
# Fit a curve instead of a straight line

# Example:
# y = ax^2 + bx + c

# Captures non-linear patterns in data


---

# =========================================

# OVERFITTING

# =========================================

# High degree → overfitting (complex curve)
# Low degree → underfitting (too simple)

# Choose optimal degree carefully


---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

# Polynomial Regression models curved relationships.

# Advantage:
# Captures non-linearity

# Limitation:
# Easily overfits


---
