# =========================================

# DECISION TREE

# =========================================


# =========================================
# IMPORTS
# =========================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve


# =========================================
# 1. DECISION TREE MODEL
# =========================================
model = DecisionTreeClassifier()
# Decision Tree splits data into branches based on feature conditions (if-else rules).
# It keeps splitting until it reaches pure nodes or stopping criteria.

model.fit(X_train, y_train)
# The model learns decision rules from training data.

y_pred = model.predict(X_test)
# Predicts class labels based on learned rules.

# Used for: credit approval, fraud detection, decision making systems
```

---

# =========================================

# MODEL EXPLANATION

# =========================================

```python id="dt002"
# How it works:
# The model splits data using questions like:
# "Is age > 30?" → left or right branch

# Goal:
# Create pure nodes (all data points in node belong to same class)

# Splitting Criteria:
# Gini Index → measures impurity (lower is better)
# Entropy → measures randomness

# Example:
# Loan approval → based on salary, age, credit score
```

---

# =========================================

# TYPES OF DECISION TREES

# =========================================

# Classification Tree:
# Used when target is categorical (yes/no)

# Regression Tree:
# Used when target is continuous (price, sales)

from sklearn.tree import DecisionTreeRegressor

reg_model = DecisionTreeRegressor()
reg_model.fit(X_train, y_train)
```

# Classification → spam detection
# Regression → house price prediction
```

---

# =========================================

# EVALUATION METRICS

# =========================================

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


# Accuracy → overall correctness of model

# Precision → how many predicted positives are correct
# Important when false positives are costly

# Recall → how many actual positives are detected
# Important when missing positive is risky

# F1 Score → balance between precision and recall
```

---

# =========================================

# CONFUSION MATRIX

# =========================================


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
```


# [[TN  FP]
#  [FN  TP]]

# Helps understand types of errors made by model
```

---

# =========================================

# VISUALIZATION (TREE STRUCTURE)

# =========================================

plt.figure(figsize=(12, 8))
plot_tree(model, filled=True)
plt.title("Decision Tree")
plt.show()
```

# Tree visualization shows:
# - Each split condition
# - Feature used for split
# - Class prediction at each node

# Deep tree → more complex model → risk of overfitting
```

---

# =========================================

# HYPERPARAMETER TUNING

# =========================================

param_grid = {
    'max_depth': [3, 5, 10, None],         # controls depth of tree
    'min_samples_split': [2, 5, 10],       # min samples required to split
    'min_samples_leaf': [1, 2, 4],         # min samples at leaf node
    'criterion': ['gini', 'entropy']       # measure of impurity
}

grid = GridSearchCV(
    DecisionTreeClassifier(),
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

# max_depth:
# Limits how deep the tree can grow.
# Smaller depth → simpler model → less overfitting.

# min_samples_split:
# Minimum samples required to split a node.
# Higher value → prevents unnecessary splits.

# min_samples_leaf:
# Minimum samples in a leaf node.
# Helps smooth the model and reduce noise.

# criterion:
# gini → faster, commonly used
# entropy → more informative but slower
```

---

# =========================================

# FINAL MODEL EVALUATION

# =========================================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Final Accuracy:", accuracy)
```

---

# =========================================

# OVERFITTING / UNDERFITTING CHECK

# =========================================

y_train_pred = best_model.predict(X_train)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# Train high, Test low → Overfitting (tree too deep)
# Both low → Underfitting (tree too shallow)
# Balanced → good model
```

---

# =========================================

# CROSS VALIDATION

# =========================================

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print("CV Scores:", cv_scores)
print("Mean CV:", cv_scores.mean())
print("Std Dev:", cv_scores.std())

# High std → unstable model (high variance)
```

---

# =========================================

# LEARNING CURVE

# =========================================

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
# Both low → Underfitting
```

---

# =========================================

# NOISE

# =========================================

# Decision Trees are very sensitive to noise.
# Even small changes in data can create very different trees.

# This makes them high variance models.
```

---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================


# Decision Tree is easy to understand and interpret.
# Works well on both classification and regression problems.

# Problem:
# High risk of overfitting (especially deep trees)

# Solution:
# Limit depth, use pruning, or use Random Forest

# Key tuning:
# max_depth, min_samples_split, criterion
```

---
