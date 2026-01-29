"""
CatBoost Model for Asset Allocation Performance Prediction
This script implements a CatBoost model to predict whether an asset allocation
will have positive or negative returns on the next trading day.

Usage:
    python src/models/catboost_model.py --imputation median
    python src/models/catboost_model.py --imputation mice
    python src/models/catboost_model.py --imputation kde
    python src/models/catboost_model.py --imputation no_imputation
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier as CatBoostBase
from datetime import datetime
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator
from src.preprocessing.argument_parser import parse_arguments
from src.preprocessing.data_imputation import impute_missing_values
from src.preprocessing.feature_engineering import create_features


# Create a wrapper to ensure CatBoost is compatible with sklearn's GridSearchCV
class CatBoostClassifier(CatBoostBase, BaseEstimator):
    pass


# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

print("=" * 70)
print("SECTION 1: Loading data and parsing arguments...")
print("=" * 70)

# Parse command-line arguments
args = parse_arguments()
IMPUTATION_STRATEGY = args.imputation

X_train = pd.read_csv("data/X_train.csv", index_col="ROW_ID")
X_test = pd.read_csv("data/X_test.csv", index_col="ROW_ID")
y_train = pd.read_csv("data/y_train.csv", index_col="ROW_ID")
sample_submission = pd.read_csv(
    "data/submissions/sample_submission.csv", index_col="ROW_ID"
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Target shape: {y_train.shape}")
print(f"Imputation strategy: {IMPUTATION_STRATEGY}")


# =============================================================================
# SECTION 2: FEATURE ENGINEERING
# =============================================================================

print("\n")
X_train, X_test, features = create_features(X_train, X_test)


# =============================================================================
# SECTION 3: DATA PREPROCESSING & MISSING VALUES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Handling missing values...")
print("=" * 70)

print(f"Using imputation strategy: {IMPUTATION_STRATEGY}")

# 3.1: Apply imputation using the selected strategy
# CatBoost is tree-based, so fill_with_zero=False (uses native NaN handling)
X_train, X_test, strategy_used = impute_missing_values(
    X_train, X_test, features, strategy=IMPUTATION_STRATEGY, fill_with_zero=False
)

# NOTE: CatBoost is tree-based and does not require feature scaling
print("Note: CatBoost is tree-based and does not require feature scaling")


# =============================================================================
# SECTION 4: MODEL TRAINING
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Hyperparameter Tuning with Cross-Validation...")
print("=" * 70)

# Convert to binary target (1 if positive return, 0 if negative)
y_train_binary = (y_train.values.ravel() > 0).astype(int)

# Phase 1: Tune iterations and depth
print("\nPhase 1: Tuning iterations and depth (5-fold CV)...")
param_grid_phase1 = {
    "iterations": [100, 200, 300],
    "depth": [4, 6, 8],
}

cat_base = CatBoostClassifier(
    learning_rate=0.1, random_seed=42, verbose=0, thread_count=-1
)

cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search_phase1 = GridSearchCV(
    cat_base,
    param_grid_phase1,
    cv=cv_splitter,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)

grid_search_phase1.fit(X_train[features], y_train_binary)

# Get best parameters from phase 1
best_depth = grid_search_phase1.best_params_["depth"]
best_iter = grid_search_phase1.best_params_["iterations"]
phase1_score = grid_search_phase1.best_score_

print(f"  ✓ Phase 1 complete - Best iterations: {best_iter}, Best depth: {best_depth}")
print(f"    CV Accuracy: {phase1_score:.4f}")

# Phase 2: Tune learning_rate and subsample
print("\nPhase 2: Tuning learning_rate and subsample (5-fold CV)...")
param_grid_phase2 = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
}

cat_base2 = CatBoostClassifier(
    iterations=best_iter,
    depth=best_depth,
    random_seed=42,
    verbose=0,
    thread_count=-1,
)

grid_search_phase2 = GridSearchCV(
    cat_base2,
    param_grid_phase2,
    cv=cv_splitter,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)

grid_search_phase2.fit(X_train[features], y_train_binary)

# Extract best parameters
CAT_ITERATIONS = grid_search_phase2.best_params_.get("iterations", best_iter)
CAT_DEPTH = grid_search_phase2.best_params_.get("depth", best_depth)
CAT_LEARNING_RATE = grid_search_phase2.best_params_["learning_rate"]
CAT_SUBSAMPLE = grid_search_phase2.best_params_["subsample"]
best_cv_score = grid_search_phase2.best_score_

print(f"  ✓ Phase 2 complete")
print(f"    Best learning_rate: {CAT_LEARNING_RATE}, Best subsample: {CAT_SUBSAMPLE}")
print(f"    CV Accuracy: {best_cv_score:.4f}")

# Train final model with best parameters
print(f"\n✓ Cross-validation complete! Training final model with best parameters...")
catboost_model = CatBoostClassifier(
    iterations=CAT_ITERATIONS,
    depth=CAT_DEPTH,
    learning_rate=CAT_LEARNING_RATE,
    subsample=CAT_SUBSAMPLE,
    bagging_temperature=1.0,
    random_seed=42,
    verbose=False,
    thread_count=-1,
)

catboost_model.fit(X_train[features], y_train_binary)

print(f"Model trained with best parameters")
print(f"  iterations: {CAT_ITERATIONS}, depth: {CAT_DEPTH}")
print(f"  learning_rate: {CAT_LEARNING_RATE}, subsample: {CAT_SUBSAMPLE}")


# =============================================================================
# SECTION 5: PREDICTION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Making predictions on test set...")
print("=" * 70)

# Predict probabilities
y_test_pred_proba = catboost_model.predict_proba(X_test[features])[:, 1]

# Convert to binary predictions (threshold = 0.5)
y_test_pred_binary = (y_test_pred_proba > 0.5).astype(int)

print(f"Predictions completed")
print(
    f"Positive predictions: {y_test_pred_binary.sum()} ({y_test_pred_binary.sum() / len(y_test_pred_binary) * 100:.2f}%)"
)
print(
    f"Negative predictions: {len(y_test_pred_binary) - y_test_pred_binary.sum()} ({(1 - y_test_pred_binary.mean()) * 100:.2f}%)"
)

# Calculate training accuracy for reference
y_train_pred_proba = catboost_model.predict_proba(X_train[features])[:, 1]
y_train_pred_binary = (y_train_pred_proba > 0.5).astype(int)
train_accuracy = np.mean(y_train_pred_binary == y_train_binary)

print(f"\nTraining accuracy: {train_accuracy * 100:.2f}%")


# =============================================================================
# SECTION 6: LOGGING RESULTS TO SUMMARY CSV
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Logging training statistics...")
print("=" * 70)

# Prepare results summary path
results_summary_path = "results/results_summary.csv"

# Create a dictionary with key model statistics
result_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "CatBoost",
    "num_features": len(features),
    "imputation_method": strategy_used,
    "scaling": "None (tree-based)",
    "train_accuracy": train_accuracy * 100,
    "test_positive_ratio": (y_test_pred_binary.sum() / len(y_test_pred_binary)) * 100,
}

# Check if results file exists
if os.path.exists(results_summary_path):
    results_df = pd.read_csv(results_summary_path)
    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
else:
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame([result_row])

results_df.to_csv(results_summary_path, index=False)
print(f"✓ Results logged to {results_summary_path}")
print(f"\nResults summary:")
print(results_df.iloc[-1])


# =============================================================================
# SECTION 7: SAVING PREDICTIONS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: Saving predictions...")
print("=" * 70)

# Create submission dataframe
submission = sample_submission.copy()
submission["TARGET"] = y_test_pred_binary

# Save predictions with parameters in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"data/submissions/preds_catboost_d{CAT_DEPTH}_lr{CAT_LEARNING_RATE}_it{CAT_ITERATIONS}_impute{IMPUTATION_STRATEGY}_{timestamp}.csv"

# Create submissions directory if it doesn't exist
os.makedirs("data/submissions", exist_ok=True)

submission.to_csv(output_filename)
print(f"✓ Predictions saved to {output_filename}")
print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:\n{submission.head()}")

print("\n" + "=" * 70)
print("CatBoost model training complete!")
print("=" * 70)
