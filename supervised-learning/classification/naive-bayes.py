Love the momentum 🔥 — now we move to a **super fast + theory-heavy model (very common in interviews)**

---

# =========================================

# 7. NAIVE BAYES

# =========================================

```python id="nb001"
# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve


# =========================================
# 1. NAIVE BAYES MODEL
# =========================================
model = GaussianNB()
# Naive Bayes is a probabilistic algorithm based on Bayes Theorem.
# It assumes all features are independent (which is why it's called "naive").

model.fit(X_train, y_train)
# Learns probability distribution of features for each class.

y_pred = model.predict(X_test)
# Predicts class based on highest probability.

# Used for: spam detection, text classification, sentiment analysis
```

---

# =========================================

# MODEL EXPLANATION

# =========================================

```python id="nb002"
# Bayes Theorem:
# P(A|B) = (P(B|A) * P(A)) / P(B)

# What it means:
# Probability of class given data = likelihood * prior / evidence

# Naive assumption:
# Features are independent of each other

# Example:
# Email classification based on words (assumes each word is independent)
```

---

# =========================================

# TYPES OF NAIVE BAYES

# =========================================

```python id="nb003"
# Gaussian Naive Bayes:
gaussian = GaussianNB()
# Used when features are continuous (numerical)

# Multinomial Naive Bayes:
multinomial = MultinomialNB()
# Used for text data (word counts, frequencies)

multinomial.fit(X_train, y_train)
```

```python id="nb004"
# Gaussian → numeric data
# Multinomial → text data (NLP tasks)
```

---

# =========================================

# EVALUATION METRICS

# =========================================

```python id="nb005"
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

```python id="nb006"
# Accuracy → overall correctness

# Precision → important when false positives matter
# Recall → important when false negatives matter

# F1 Score → balance between precision and recall
```

---

# =========================================

# CONFUSION MATRIX

# =========================================

```python id="nb007"
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
```

---

# =========================================

# VISUALIZATION (PROBABILITY IDEA)

# =========================================

```python id="nb008"
# Naive Bayes does not have a simple geometric boundary like SVM.
# It works using probability distributions instead of distances.
```

---

# =========================================

# HYPERPARAMETER TUNING

# =========================================

```python id="nb009"
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7]
}
# var_smoothing helps handle numerical stability.

grid = GridSearchCV(
    GaussianNB(),
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

```python id="nb010"
# var_smoothing:
# Adds small value to variance to avoid division by zero.

# Helps stabilize probability calculations.
```

---

# =========================================

# FINAL MODEL EVALUATION

# =========================================

```python id="nb011"
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Final Accuracy:", accuracy)
```

---

# =========================================

# OVERFITTING / UNDERFITTING CHECK

# =========================================

```python id="nb012"
y_train_pred = best_model.predict(X_train)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# Usually Naive Bayes has low variance (less overfitting)
# But can underfit due to strong assumptions
```

---

# =========================================

# CROSS VALIDATION

# =========================================

```python id="nb013"
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print("CV Scores:", cv_scores)
print("Mean CV:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

# Low std → stable model
```

---

# =========================================

# LEARNING CURVE

# =========================================

```python id="nb014"
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

# Often shows small gap → low variance model
```

---

# =========================================

# NOISE

# =========================================

```python id="nb015"
# Naive Bayes handles noise moderately well.
# But independence assumption may fail in real-world data.

# Performance drops when features are highly correlated.
```

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

```python id="nb016"
# Naive Bayes is a fast probabilistic classifier.

# Key strength:
# Works well with high-dimensional data (like text)

# Key assumption:
# Features are independent

# Advantage:
# Very fast and simple

# Limitation:
# Can underfit due to strong assumptions
```

---
