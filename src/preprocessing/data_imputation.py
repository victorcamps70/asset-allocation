"""
Data Imputation Module for Asset Allocation Challenge
Provides multiple strategies for handling missing values:
- no_imputation: Replace NaN with 0 (for Ridge) or leave as NaN (for tree models)
- median: Fill with column median
- kde: Fill with KDE-estimated mode (peak of distribution)
- mice: Multiple Imputation by Chained Equations

Optimization: Uses median for columns with <10% missing data (fast & sufficient)
and applies the selected strategy only to columns with >=10% missing data.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def impute_missing_values(
    X_train, X_test, features, strategy="median", fill_with_zero=False
):
    """
    Impute missing values in train and test sets using specified strategy.

    Optimization: Uses median for columns with <10% missing data and applies
    the selected strategy only to columns with >=10% missing data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data with missing values
    X_test : pd.DataFrame
        Test data with missing values
    features : list
        List of feature columns to impute
    strategy : str, default='median'
        Imputation strategy: 'no_imputation', 'median', 'kde', or 'mice'
    fill_with_zero : bool, default=False
        If True and strategy='no_imputation', fill NaN with 0 (for Ridge)
        If False and strategy='no_imputation', leave NaN (for tree models)

    Returns
    -------
    X_train_imputed : pd.DataFrame
        Training data with imputed values
    X_test_imputed : pd.DataFrame
        Test data with imputed values
    strategy : str
        The strategy used (for logging)
    """

    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()

    # Identify columns by missing data percentage
    missing_pct = (
        X_train_imputed[features].isnull().sum() / len(X_train_imputed)
    ) * 100
    cols_high_missing = missing_pct[missing_pct >= 10].index.tolist()  # >= 10%
    cols_low_missing = missing_pct[missing_pct < 10].index.tolist()  # < 10%

    print("=" * 70)
    print(f"IMPUTATION STRATEGY: {strategy.upper()}")
    if strategy == "no_imputation":
        fill_method = "zeros (for Ridge)" if fill_with_zero else "NaN (for tree models)"
        print(f"Missing values will be filled with: {fill_method}")
    print("=" * 70)
    print(f"Columns with <10% missing data (using median): {len(cols_low_missing)}")
    if strategy != "no_imputation":
        print(
            f"Columns with >=10% missing data (using {strategy}): {len(cols_high_missing)}"
        )
    print()

    missing_before = X_train_imputed[features].isnull().sum().sum()
    print(f"Missing values before imputation: {missing_before}")

    # Handle no_imputation strategy
    if strategy == "no_imputation":
        if fill_with_zero:
            print(f"→ Filling all NaN with ZERO (for Ridge regression)...")
            X_train_imputed[features] = X_train_imputed[features].fillna(0)
            X_test_imputed[features] = X_test_imputed[features].fillna(0)
        else:
            print(f"→ Leaving NaN as is (tree models handle NaN natively)...")

        missing_after = X_train_imputed[features].isnull().sum().sum()
        print(f"Missing values after imputation: {missing_after}")
        print(f"✓ Imputation complete with strategy: {strategy}\n")
        return X_train_imputed, X_test_imputed, strategy

    # STEP 1: Fill columns with <10% missing using median (fast)
    if cols_low_missing:
        print(
            f"\n→ Imputing {len(cols_low_missing)} columns with <10% missing using MEDIAN..."
        )
        for col in cols_low_missing:
            median_value = X_train_imputed[col].median()
            X_train_imputed[col] = X_train_imputed[col].fillna(median_value)
            X_test_imputed[col] = X_test_imputed[col].fillna(median_value)

    # STEP 2: Fill columns with >=10% missing using selected strategy
    if cols_high_missing:
        if strategy == "median":
            print(
                f"→ Imputing {len(cols_high_missing)} columns with >=10% missing using MEDIAN..."
            )
            for col in cols_high_missing:
                median_value = X_train_imputed[col].median()
                X_train_imputed[col] = X_train_imputed[col].fillna(median_value)
                X_test_imputed[col] = X_test_imputed[col].fillna(median_value)

        elif strategy == "kde":
            print(
                f"→ Imputing {len(cols_high_missing)} columns with >=10% missing using KDE MODE..."
            )
            for col in cols_high_missing:
                data_col = X_train_imputed[col].dropna()

                if len(data_col) > 0:
                    # Estimate mode using KDE (more efficient)
                    try:
                        kde = stats.gaussian_kde(data_col)
                        # Create a grid from min to max to find mode (more efficient than evaluate on all points)
                        x_min, x_max = data_col.min(), data_col.max()
                        x_range = np.linspace(
                            x_min, x_max, 200
                        )  # Sample 200 points instead of all
                        kde_values = kde(x_range)
                        mode_value = x_range[np.argmax(kde_values)]
                    except:
                        # Fallback to median if KDE fails
                        mode_value = data_col.median()

                    X_train_imputed[col] = X_train_imputed[col].fillna(mode_value)
                    X_test_imputed[col] = X_test_imputed[col].fillna(mode_value)

        elif strategy == "mice":
            print(
                f"→ Imputing {len(cols_high_missing)} columns with >=10% missing using MICE..."
            )
            # Use IterativeImputer (sklearn's implementation of MICE) only on high-missing columns
            imputer = IterativeImputer(max_iter=10, random_state=42, verbose=0)

            # Fit on train data and transform both train and test
            X_train_imputed[cols_high_missing] = imputer.fit_transform(
                X_train_imputed[cols_high_missing]
            )
            X_test_imputed[cols_high_missing] = imputer.transform(
                X_test_imputed[cols_high_missing]
            )

    missing_after = X_train_imputed[features].isnull().sum().sum()
    print(f"\nMissing values after imputation: {missing_after}")
    print(f"✓ Imputation complete with optimized strategy: {strategy}\n")

    return X_train_imputed, X_test_imputed, strategy
