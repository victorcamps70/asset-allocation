"""
Feature Engineering Module for Asset Allocation Challenge
Provides a unified function to create all engineered features
used across all models (Ridge, XGBoost, CatBoost).
"""

import pandas as pd
import numpy as np


def create_features(X_train, X_test):
    """
    Create all engineered features for both training and test sets.

    Features created:
    - AVERAGE_PERF_* : Average performance over different time windows (3, 5, 10, 15, 20 days)
    - ALLOCATIONS_AVERAGE_PERF_* : Market-wide average performance per day
    - STD_PERF_* : Volatility over 20 days
    - ALLOCATIONS_STD_PERF_* : Market-wide volatility per day
    - Original features: RET_*, SIGNED_VOLUME_*, MEDIAN_DAILY_TURNOVER

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data
    X_test : pd.DataFrame
        Test data

    Returns
    -------
    X_train : pd.DataFrame
        Training data with engineered features
    X_test : pd.DataFrame
        Test data with engineered features
    features : list
        List of all feature names (base + engineered)
    """

    print("=" * 70)
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

    # 2.3: Create interactions with top 5 most important features
    print("Creating interaction features from top 5 most important features...")
    top_features = [
        "RET_1",
        "RET_4",
        "ALLOCATIONS_AVERAGE_PERF_3",
        "ALLOCATIONS_AVERAGE_PERF_15",
        "ALLOCATIONS_AVERAGE_PERF_5",
    ]

    # Products
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i + 1 :]:
            X_train[f"{feat1}_PROD_{feat2}"] = X_train[feat1] * X_train[feat2]
            X_test[f"{feat1}_PROD_{feat2}"] = X_test[feat1] * X_test[feat2]

    # Divisions (with safe handling of division by zero)
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i + 1 :]:
            X_train[f"{feat1}_DIV_{feat2}"] = np.where(
                X_train[feat2] != 0, X_train[feat1] / (X_train[feat2] + 1e-8), 0
            )
            X_test[f"{feat1}_DIV_{feat2}"] = np.where(
                X_test[feat2] != 0, X_test[feat1] / (X_test[feat2] + 1e-8), 0
            )

    # Subtractions
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i + 1 :]:
            X_train[f"{feat1}_MINUS_{feat2}"] = X_train[feat1] - X_train[feat2]
            X_test[f"{feat1}_MINUS_{feat2}"] = X_test[feat1] - X_test[feat2]

    # 2.4: Select only the most important features based on correlation analysis
    print("Selecting features based on target correlation analysis...\n")

    # Selected features (from feature selection analysis)
    selected_features = [
        "ALLOCATIONS_AVERAGE_PERF_10",
        "ALLOCATIONS_AVERAGE_PERF_3",
        "ALLOCATIONS_AVERAGE_PERF_3_MINUS_ALLOCATIONS_AVERAGE_PERF_5",
        "AVERAGE_PERF_10",
        "AVERAGE_PERF_3",
        "RET_1",
        "RET_1_PROD_ALLOCATIONS_AVERAGE_PERF_15",
        "RET_1_PROD_ALLOCATIONS_AVERAGE_PERF_3",
        "RET_4_MINUS_ALLOCATIONS_AVERAGE_PERF_3",
        "RET_4_PROD_ALLOCATIONS_AVERAGE_PERF_15",
        "RET_4_PROD_ALLOCATIONS_AVERAGE_PERF_3",
        "RET_4_PROD_ALLOCATIONS_AVERAGE_PERF_5",
        "RET_7",
        "RET_8",
        "RET_9",
        "SIGNED_VOLUME_1",  # One SIGNED_VOLUME feature
        "MEDIAN_DAILY_TURNOVER",  # TURNOVER feature
        "STD_PERF_20",  # STD_PERF feature
        "ALLOCATIONS_STD_PERF_20",  # ALLOCATIONS_STD_PERF feature
    ]

    features = selected_features

    print(f"Selected {len(features)} most important features\n")

    # 2.5: Clip features on first and last percentile to reduce outliers
    print("Clipping features on 1st and 99th percentiles...")
    for feat in features:
        if feat in X_train.columns:
            # Calculate percentiles from training data
            p1 = X_train[feat].quantile(0.01)
            p99 = X_train[feat].quantile(0.99)

            # Clip both train and test sets
            X_train[feat] = X_train[feat].clip(lower=p1, upper=p99)
            X_test[feat] = X_test[feat].clip(lower=p1, upper=p99)

    print(f"✓ All {len(features)} features clipped on 1st and 99th percentiles\n")
    print(f"Total features: {len(features)}\n")

    return X_train, X_test, features
