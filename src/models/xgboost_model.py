"""
XGBoost Model for Asset Allocation Performance Prediction
This script implements an XGBoost model to predict whether an asset allocation
will have positive or negative returns on the next trading day.

Usage:
    python src/models/xgboost_model.py --imputation median
    python src/models/xgboost_model.py --imputation mice
    python src/models/xgboost_model.py --imputation kde
    python src/models/xgboost_model.py --imputation no_imputation
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.preprocessing.argument_parser import parse_arguments
from src.preprocessing.data_imputation import impute_missing_values
from src.preprocessing.feature_engineering import create_features


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
# XGBoost is tree-based, so fill_with_zero=False (uses native NaN handling)
X_train, X_test, strategy_used = impute_missing_values(
    X_train, X_test, features, strategy=IMPUTATION_STRATEGY, fill_with_zero=False
)

# NOTE: XGBoost does NOT require scaling (tree-based model)
print("Note: XGBoost is tree-based and does not require feature scaling")


# =============================================================================
# SECTION 4: MODEL TRAINING
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Hyperparameter Tuning with Cross-Validation...")
print("=" * 70)

# Convert to binary target (1 if positive return, 0 if negative)
y_train_binary = (y_train.values.ravel() > 0).astype(int)

# Phase 1: Tune n_estimators and max_depth
print("\nPhase 1: Tuning n_estimators and max_depth (5-fold CV)...")
param_grid_phase1 = {
    "n_estimators": [50, 100, 200],
    "max_depth": [4, 6, 8],
}

xgb_base = xgb.XGBClassifier(
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)

cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search_phase1 = GridSearchCV(
    xgb_base,
    param_grid_phase1,
    cv=cv_splitter,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)

grid_search_phase1.fit(X_train[features], y_train_binary)

# Get best parameters from phase 1
best_depth = grid_search_phase1.best_params_["max_depth"]
best_n_est = grid_search_phase1.best_params_["n_estimators"]
phase1_score = grid_search_phase1.best_score_

print(
    f"  ✓ Phase 1 complete - Best n_estimators: {best_n_est}, Best max_depth: {best_depth}"
)
print(f"    CV Accuracy: {phase1_score:.4f}")

# Phase 2: Tune learning_rate and subsample
print("\nPhase 2: Tuning learning_rate and subsample (5-fold CV)...")
param_grid_phase2 = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
}

xgb_base2 = xgb.XGBClassifier(
    n_estimators=best_n_est,
    max_depth=best_depth,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)

grid_search_phase2 = GridSearchCV(
    xgb_base2,
    param_grid_phase2,
    cv=cv_splitter,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)

grid_search_phase2.fit(X_train[features], y_train_binary)

# Extract best parameters
XGB_N_ESTIMATORS = grid_search_phase2.best_params_.get("n_estimators", best_n_est)
XGB_MAX_DEPTH = grid_search_phase2.best_params_.get("max_depth", best_depth)
XGB_LEARNING_RATE = grid_search_phase2.best_params_["learning_rate"]
XGB_SUBSAMPLE = grid_search_phase2.best_params_["subsample"]
best_cv_score = grid_search_phase2.best_score_

print(f"  ✓ Phase 2 complete")
print(f"    Best learning_rate: {XGB_LEARNING_RATE}, Best subsample: {XGB_SUBSAMPLE}")
print(f"    CV Accuracy: {best_cv_score:.4f}")

# Train final model with best parameters
print(f"\n✓ Cross-validation complete! Training final model with best parameters...")
xgb_model = xgb.XGBClassifier(
    n_estimators=XGB_N_ESTIMATORS,
    max_depth=XGB_MAX_DEPTH,
    learning_rate=XGB_LEARNING_RATE,
    subsample=XGB_SUBSAMPLE,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)

xgb_model.fit(X_train[features], y_train_binary)

print(f"Model trained with best parameters")
print(f"  n_estimators: {XGB_N_ESTIMATORS}, max_depth: {XGB_MAX_DEPTH}")
print(f"  learning_rate: {XGB_LEARNING_RATE}, subsample: {XGB_SUBSAMPLE}")


# =============================================================================
# SECTION 5: PREDICTION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Making predictions on test set...")
print("=" * 70)

# Predict probabilities
y_test_pred_proba = xgb_model.predict_proba(X_test[features])[:, 1]

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
y_train_pred_proba = xgb_model.predict_proba(X_train[features])[:, 1]
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
    "model": "XGBoost",
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

# Create submission dataframe with same format as Ridge
submission = pd.DataFrame({"ROW_ID": X_test.index, "target": y_test_pred_binary})
submission.set_index("ROW_ID", inplace=True)

# Save predictions with parameters in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"data/submissions/preds_xgboost_d{XGB_MAX_DEPTH}_lr{XGB_LEARNING_RATE}_ne{XGB_N_ESTIMATORS}_impute{IMPUTATION_STRATEGY}_{timestamp}.csv"

# Create submissions directory if it doesn't exist
os.makedirs("data/submissions", exist_ok=True)

submission.to_csv(output_filename)
print(f"✓ Predictions saved to {output_filename}")
print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:\n{submission.head()}")

print("\n" + "=" * 70)
print("XGBoost model training complete!")
print("=" * 70)
