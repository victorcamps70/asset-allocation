"""
Ridge Regression Model for Asset Allocation Performance Prediction
This script implements a Ridge regression model to predict whether an asset allocation
will have positive or negative returns on the next trading day.

Usage:
  python src/models/ridge_regression.py                    # Uses default (median)
  python src/models/ridge_regression.py --imputation mice  # Uses MICE imputation
  python src/models/ridge_regression.py --imputation no_imputation  # Fills NaN with 0
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from datetime import datetime
import os
from src.preprocessing.data_imputation import impute_missing_values
from src.preprocessing.feature_engineering import create_features
from src.preprocessing.argument_parser import parse_arguments

# Parse command-line arguments
args = parse_arguments()
IMPUTATION_STRATEGY = (
    args.imputation
)  # =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

print("=" * 70)
print("SECTION 1: Loading data...")
print("=" * 70)

X_train = pd.read_csv("data/X_train.csv", index_col="ROW_ID")
X_test = pd.read_csv("data/X_test.csv", index_col="ROW_ID")
y_train = pd.read_csv("data/y_train.csv", index_col="ROW_ID")
sample_submission = pd.read_csv(
    "data/submissions/sample_submission.csv", index_col="ROW_ID"
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Target shape: {y_train.shape}")


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
# Ridge needs imputation, so fill_with_zero=True for no_imputation
X_train, X_test, strategy_used = impute_missing_values(
    X_train, X_test, features, strategy=IMPUTATION_STRATEGY, fill_with_zero=True
)

# 3.3: Normalize features (important for Ridge regression)
print("Scaling features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[features])
X_test_scaled = scaler.transform(X_test[features])

print(
    f"Features scaled. Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}"
)


# =============================================================================
# SECTION 4: MODEL TRAINING
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Hyperparameter Tuning with Cross-Validation...")
print("=" * 70)

# Convert to binary target
y_train_binary = (y_train.values.ravel() > 0).astype(int)

# Create a wrapper to convert Ridge predictions to binary for scoring
from sklearn.base import BaseEstimator, ClassifierMixin


class BinaryRidgeWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for Ridge to enable binary classification scoring"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.ridge = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.ridge.fit(X, y)
        return self

    def predict(self, X):
        # Ridge outputs continuous values, convert to binary
        predictions = self.ridge.predict(X)
        return (predictions > 0).astype(int)

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        if "alpha" in params:
            self.alpha = params["alpha"]
            self.ridge = Ridge(alpha=params["alpha"])
        return self


# Define parameter grid for Ridge
param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}

# Cross-validation with stratified folds
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearch to find best alpha
ridge_wrapper = BinaryRidgeWrapper()
grid_search = GridSearchCV(
    ridge_wrapper, param_grid, cv=cv_splitter, scoring="accuracy", n_jobs=-1, verbose=0
)

print("Running 5-fold cross-validation to find best alpha...")
grid_search.fit(X_train_scaled, y_train_binary)

# Get best parameters
RIDGE_ALPHA = grid_search.best_params_["alpha"]
best_cv_score = grid_search.best_score_

print(f"\n✓ Cross-validation complete!")
print(f"  Best alpha: {RIDGE_ALPHA}")
print(f"  Best CV Accuracy: {best_cv_score:.4f}")

# Train final model with best alpha (using raw Ridge for continuous predictions)
ridge_model = Ridge(alpha=RIDGE_ALPHA)
ridge_model.fit(X_train_scaled, y_train_binary)

print(f"Model trained with best parameters")
print(f"Ridge alpha parameter: {ridge_model.get_params()['alpha']}")


# =============================================================================
# SECTION 5: PREDICTION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Making predictions on test set...")
print("=" * 70)

# Predict on test set
y_pred = ridge_model.predict(X_test_scaled)

# Convert predictions to binary classification (positive/negative returns)
y_pred_binary = (y_pred > 0).astype(int)

print(f"Predictions shape: {y_pred.shape}")
print(
    f"Positive predictions: {y_pred_binary.sum()} ({y_pred_binary.sum() / len(y_pred_binary) * 100:.2f}%)"
)
print(
    f"Negative predictions: {(1 - y_pred_binary).sum()} ({(1 - y_pred_binary).sum() / len(y_pred_binary) * 100:.2f}%)"
)

# Calculate training set accuracy
y_train_pred_scaled = ridge_model.predict(X_train_scaled)
y_train_pred_binary = (y_train_pred_scaled > 0).astype(int)
train_accuracy = np.mean(
    y_train_pred_binary == (y_train.values.ravel() > 0).astype(int)
)

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
# NOTE: Keep this generic to support multiple model types
result_row = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "Ridge",
    "num_features": len(features),
    "imputation_method": strategy_used,
    "scaling": "StandardScaler",
    "train_accuracy": train_accuracy * 100,
    "test_positive_ratio": (y_pred_binary.sum() / len(y_pred_binary)) * 100,
}

# Convert to DataFrame
results_df = pd.DataFrame([result_row])

# Check if results summary file exists and append
if os.path.exists(results_summary_path):
    print(f"Appending to existing results summary: {results_summary_path}")
    existing_results = pd.read_csv(results_summary_path)
    results_df = pd.concat([existing_results, results_df], ignore_index=True)
else:
    print(f"Creating new results summary: {results_summary_path}")

# Save results summary
results_df.to_csv(results_summary_path, index=False)
print(f"Results summary saved successfully")
print(f"\nModel Summary:")
for key, value in result_row.items():
    print(f"  {key}: {value}")


# =============================================================================
# SECTION 7: SAVING PREDICTIONS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: Saving predictions...")
print("=" * 70)
# Create submission dataframe
submission = pd.DataFrame(
    y_pred_binary, index=sample_submission.index, columns=["target"]
)

# Save predictions with parameters in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"data/submissions/preds_ridge_alpha{RIDGE_ALPHA}_scaler_impute{IMPUTATION_STRATEGY}_{timestamp}.csv"

# Create submissions directory if it doesn't exist
os.makedirs("data/submissions", exist_ok=True)

submission.to_csv(output_path)

print(f"Predictions saved to: {output_path}")
print(f"Submission shape: {submission.shape}")
print("\n" + "=" * 70)
print("COMPLETED: Ridge Regression pipeline finished successfully!")
print("=" * 70)
