"""
Ridge Regression Model for Asset Allocation Performance Prediction
This script implements a Ridge regression model to predict whether an asset allocation
will have positive or negative returns on the next trading day.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os


# =============================================================================
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

print("\n" + "=" * 70)
print("SECTION 2: Creating new features...")
print("=" * 70)

# Define base feature groups
RET_features = [f"RET_{i}" for i in range(1, 21)]
SIGNED_VOLUME_features = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
TURNOVER_features = ["MEDIAN_DAILY_TURNOVER"]

# 2.1: Create average performance features on different time windows
print("Creating average performance features...")
for i in [3, 5, 10, 15, 20]:
    # Individual allocation average performance
    X_train[f"AVERAGE_PERF_{i}"] = X_train[RET_features[:i]].mean(1)
    X_test[f"AVERAGE_PERF_{i}"] = X_test[RET_features[:i]].mean(1)

    # Market-wide average performance on same day
    X_train[f"ALLOCATIONS_AVERAGE_PERF_{i}"] = X_train.groupby("TS")[
        f"AVERAGE_PERF_{i}"
    ].transform("mean")
    X_test[f"ALLOCATIONS_AVERAGE_PERF_{i}"] = X_test.groupby("TS")[
        f"AVERAGE_PERF_{i}"
    ].transform("mean")

# 2.2: Create volatility features
print("Creating volatility features...")
for i in [20]:
    # Individual allocation volatility
    X_train[f"STD_PERF_{i}"] = X_train[RET_features[:i]].std(1)
    X_test[f"STD_PERF_{i}"] = X_test[RET_features[:i]].std(1)

    # Market-wide average volatility on same day
    X_train[f"ALLOCATIONS_STD_PERF_{i}"] = X_train.groupby("TS")[
        f"STD_PERF_{i}"
    ].transform("mean")
    X_test[f"ALLOCATIONS_STD_PERF_{i}"] = X_test.groupby("TS")[
        f"STD_PERF_{i}"
    ].transform("mean")

# 2.3: Compile all features
features = (
    RET_features
    + SIGNED_VOLUME_features
    + TURNOVER_features
    + [f"AVERAGE_PERF_{i}" for i in [3, 5, 10, 15, 20]]
    + [f"ALLOCATIONS_AVERAGE_PERF_{i}" for i in [3, 5, 10, 15, 20]]
    + [f"STD_PERF_{i}" for i in [20]]
    + [f"ALLOCATIONS_STD_PERF_{i}" for i in [20]]
)

print(f"Total features created: {len(features)}")


# =============================================================================
# SECTION 3: DATA PREPROCESSING & MISSING VALUES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Handling missing values...")
print("=" * 70)

# 3.1: Identify missing values before imputation
missing_before = X_train[features].isnull().sum().sum()
print(f"Missing values before imputation: {missing_before}")

# 3.2: Impute missing values with column medians
print("Replacing NaN with column medians...")
for col in features:
    median_value = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)

missing_after = X_train[features].isnull().sum().sum()
print(f"Missing values after imputation: {missing_after}")

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
print("SECTION 4: Training Ridge Regression model...")
print("=" * 70)

# Initialize and train Ridge regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train.values.ravel())

print(f"Model trained successfully")
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
model_stats = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "Ridge",
    "num_features": len(features),
    "imputation": "median",
    "scaling": "StandardScaler",
    "train_accuracy": f"{train_accuracy * 100:.2f}%",
    "test_positive_ratio": f"{y_pred_binary.sum() / len(y_pred_binary) * 100:.2f}%",
}

# Convert to DataFrame
results_df = pd.DataFrame([model_stats])

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
for key, value in model_stats.items():
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

# Save predictions
output_path = "data/preds_ridge_median_imputation.csv"
submission.to_csv(output_path)

print(f"Predictions saved to: {output_path}")
print(f"Submission shape: {submission.shape}")
print("\n" + "=" * 70)
print("COMPLETED: Ridge Regression pipeline finished successfully!")
print("=" * 70)
