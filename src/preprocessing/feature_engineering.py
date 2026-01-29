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

    print(f"Total features created: {len(features)}\n")

    return X_train, X_test, features
