"""
XGBoost Model for Asset Allocation - Direct Training (No GridSearch)
This script trains an XGBoost model with known parameters (no hyperparameter tuning).

Usage:
    python src/models/xgboost_model_direct.py --imputation median
    python src/models/xgboost_model_direct.py --imputation mice
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
from src.preprocessing.argument_parser import parse_arguments
from src.preprocessing.data_imputation import impute_missing_values
from src.preprocessing.feature_engineering import create_features


# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

print("=" * 70)
print("SECTION 1: Loading data...")
print("=" * 70)

# Parse command-line arguments
args = parse_arguments()
IMPUTATION_STRATEGY = args.imputation

X_train = pd.read_csv("data/X_train.csv", index_col="ROW_ID")
X_test = pd.read_csv("data/X_test.csv", index_col="ROW_ID")
y_train = pd.read_csv("data/y_train.csv", index_col="ROW_ID")

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Target shape: {y_train.shape}")


# =============================================================================
# SECTION 2: FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Feature engineering...")
print("=" * 70)

X_train, X_test, features = create_features(X_train, X_test)
print(f"Features created: {len(features)} features")


# =============================================================================
# SECTION 3: DATA PREPROCESSING
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Handling missing values...")
print("=" * 70)

if IMPUTATION_STRATEGY.lower() == "no_imputation":
    print(f"No imputation applied - XGBoost will handle NaN values natively")
    strategy_used = "no_imputation"
else:
    X_train, X_test, strategy_used = impute_missing_values(
        X_train, X_test, features, strategy=IMPUTATION_STRATEGY, fill_with_zero=False
    )
    print(f"Imputation strategy used: {strategy_used}")


# =============================================================================
# SECTION 4: TRAINING (DIRECT - NO GRIDSEARCH)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Training XGBoost with known parameters...")
print("=" * 70)

# Convert to binary target
y_train_binary = (y_train.values.ravel() > 0).astype(int)

# *** MODIFY THESE PARAMETERS WITH YOUR KNOWN VALUES ***
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 8
XGB_LEARNING_RATE = 0.2
XGB_SUBSAMPLE = 0.8
XGB_REG_ALPHA = 1.0  # L1 regularization
XGB_REG_LAMBDA = 0.5  # L2 regularization
XGB_COLSAMPLE_BYTREE = 0.8

print(f"Training parameters:")
print(f"  n_estimators: {XGB_N_ESTIMATORS}")
print(f"  max_depth: {XGB_MAX_DEPTH}")
print(f"  learning_rate: {XGB_LEARNING_RATE}")
print(f"  subsample: {XGB_SUBSAMPLE}")
print(f"  reg_alpha (L1): {XGB_REG_ALPHA}")
print(f"  reg_lambda (L2): {XGB_REG_LAMBDA}")
print(f"  colsample_bytree: {XGB_COLSAMPLE_BYTREE}")

# Create and train the model
xgb_model = xgb.XGBClassifier(
    n_estimators=XGB_N_ESTIMATORS,
    max_depth=XGB_MAX_DEPTH,
    learning_rate=XGB_LEARNING_RATE,
    subsample=XGB_SUBSAMPLE,
    reg_alpha=XGB_REG_ALPHA,
    reg_lambda=XGB_REG_LAMBDA,
    colsample_bytree=XGB_COLSAMPLE_BYTREE,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)

print(f"\nTraining model...")
xgb_model.fit(X_train[features], y_train_binary)
print(f"✓ Model trained successfully!")


# =============================================================================
# SECTION 5: PREDICTION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Making predictions...")
print("=" * 70)

# Predict on test set
y_test_pred_proba = xgb_model.predict_proba(X_test[features])[:, 1]
y_test_pred_binary = (y_test_pred_proba > 0.5).astype(int)

print(f"Predictions completed")
print(
    f"  Positive predictions: {y_test_pred_binary.sum()} ({y_test_pred_binary.sum() / len(y_test_pred_binary) * 100:.2f}%)"
)
print(
    f"  Negative predictions: {len(y_test_pred_binary) - y_test_pred_binary.sum()} ({(1 - y_test_pred_binary.mean()) * 100:.2f}%)"
)

# Training accuracy
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
    "model": "XGBoost (Direct)",
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

# Create submission
submission = pd.DataFrame({"ROW_ID": X_test.index, "target": y_test_pred_binary})
submission.set_index("ROW_ID", inplace=True)

# Save with parameters in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"data/submissions/preds_xgboost_direct_d{XGB_MAX_DEPTH}_lr{XGB_LEARNING_RATE}_ne{XGB_N_ESTIMATORS}_a{XGB_REG_ALPHA}_l{XGB_REG_LAMBDA}_impute{IMPUTATION_STRATEGY}_{timestamp}.csv"

os.makedirs("data/submissions", exist_ok=True)
submission.to_csv(output_filename)

print(f"✓ Predictions saved to {output_filename}")
print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:\n{submission.head()}")

print("\n" + "=" * 70)
print("XGBoost training complete!")
print("=" * 70)
