# =========================================

# CATBOOST 

# =========================================

# =========================================
# IMPORTS
# =========================================
from catboost import CatBoostClassifier


# =========================================
# 1. CATBOOST MODEL
# =========================================
model = CatBoostClassifier(verbose=0)
# CatBoost is designed to handle categorical features automatically.
# It reduces need for preprocessing like encoding.

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Used for: business datasets with many categorical variables


---

# =========================================

# MODEL EXPLANATION

# =========================================

# Key Idea:
# Handles categorical data internally

# Uses ordered boosting:
# Prevents data leakage

# Advantage:
# No need for label encoding or one-hot encoding


---

# =========================================

# EVALUATION METRICS

# =========================================

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


---

# =========================================

# HYPERPARAMETER TUNING

# =========================================

param_grid = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [4, 6, 10]
}

grid = GridSearchCV(
    CatBoostClassifier(verbose=0),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Params:", grid.best_params_)


---

# =========================================

# HYPERPARAMETER EXPLANATION

# =========================================

# iterations:
# Number of boosting steps

# learning_rate:
# Step size for learning

# depth:
# Tree depth → controls complexity


---

# =========================================

# NOISE

# =========================================

# CatBoost handles noise better due to ordered boosting.
# It reduces overfitting compared to other boosting models.


---

# =========================================

# FINAL INTERVIEW SUMMARY

# =========================================

# CatBoost is best for categorical data.

# Key strength:
# Minimal preprocessing required

# Advantage:
# Handles categorical variables automatically

# Used in:
# Business and structured datasets


---


