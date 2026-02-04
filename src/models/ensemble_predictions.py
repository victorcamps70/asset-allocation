"""
Ensemble Predictions from Multiple Models
This script combines predictions from different models (XGBoost, CatBoost, Ridge, etc.)
using various ensembling strategies.

Usage:
    # Ensemble with 2 benchmarks + direct xgboost + direct catboost (no imputation)
    python src/models/ensemble_predictions.py --models "preds_ridge_bench,preds_lgbm_bench,preds_xgboost_direct*no_imputation*,preds_catboost_direct*no_imputation*" --strategy average

    # Weighted ensemble (give more weight to benchmarks)
    python src/models/ensemble_predictions.py --models "preds_ridge_bench,preds_lgbm_bench,preds_xgboost_direct*no_imputation*,preds_catboost_direct*no_imputation*" --strategy weighted --weights "0.3,0.3,0.2,0.2"

    # Voting with specific patterns
    python src/models/ensemble_predictions.py --models "preds_*_bench,preds_*_direct*no_imputation*" --strategy voting
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import os
from datetime import datetime
from glob import glob
import argparse


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Ensemble predictions from multiple models"
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of prediction files (or patterns) to use. "
        "Example: 'preds_ridge_bench,preds_lgbm_bench,preds_xgboost*no_imputation*,preds_catboost*no_imputation*'",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="average",
        choices=["average", "weighted", "voting"],
        help="Ensembling strategy: average, weighted, or voting",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="equal",
        help="Weights for weighted average (comma-separated, e.g., '0.5,0.3,0.2')",
    )
    return parser.parse_args()


def load_predictions(model_patterns, submissions_dir="data/submissions/"):
    """Load specific prediction files based on patterns

    Args:
        model_patterns: comma-separated string of file patterns (e.g., "preds_ridge_bench,preds_xgboost*")
        submissions_dir: directory containing prediction files

    Returns:
        Dictionary of model_name -> predictions array
    """
    predictions = {}

    # Parse patterns
    patterns = [p.strip() for p in model_patterns.split(",")]

    # Find matching files for each pattern
    all_matching_files = []
    for pattern in patterns:
        # Add .csv if not present
        if not pattern.endswith(".csv"):
            pattern = pattern + ".csv" if "*" not in pattern else pattern

        matching_files = glob(os.path.join(submissions_dir, pattern))
        if not matching_files:
            print(f"   ⚠️  No files matching pattern: {pattern}")
        else:
            all_matching_files.extend(matching_files)

    if not all_matching_files:
        print(f"❌ No prediction files found matching patterns: {model_patterns}")
        return None

    # Remove duplicates while preserving order
    all_matching_files = list(dict.fromkeys(all_matching_files))

    print(f"\n📂 Found {len(all_matching_files)} matching prediction file(s):")
    for csv_file in sorted(all_matching_files):
        try:
            df = pd.read_csv(csv_file, index_col="ROW_ID")
            model_name = Path(csv_file).stem
            predictions[model_name] = df["target"].values
            print(f"   ✓ {model_name}: {len(df)} predictions")
        except Exception as e:
            print(f"   ✗ {Path(csv_file).stem}: Error - {e}")

    return predictions if predictions else None


def ensemble_average(predictions_dict):
    """Simple averaging of all predictions"""
    print("\n" + "=" * 70)
    print("ENSEMBLE STRATEGY: AVERAGE")
    print("=" * 70)

    predictions_array = np.array(list(predictions_dict.values()))
    ensemble_pred = np.mean(predictions_array, axis=0)

    # Convert to binary (threshold = 0.5)
    ensemble_binary = (ensemble_pred > 0.5).astype(int)

    print(f"Average of {len(predictions_dict)} models")
    print(
        f"Ensemble predictions (probability): min={ensemble_pred.min():.4f}, max={ensemble_pred.max():.4f}, mean={ensemble_pred.mean():.4f}"
    )

    return ensemble_binary


def ensemble_weighted(predictions_dict, weights=None):
    """Weighted averaging of predictions"""
    print("\n" + "=" * 70)
    print("ENSEMBLE STRATEGY: WEIGHTED AVERAGE")
    print("=" * 70)

    n_models = len(predictions_dict)

    if weights is None:
        # Default: equal weights
        weights = np.ones(n_models) / n_models
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

    print(f"Weights: {', '.join([f'{w:.3f}' for w in weights])}")
    print(f"Models: {', '.join(list(predictions_dict.keys()))}")

    predictions_array = np.array(list(predictions_dict.values()))
    ensemble_pred = np.average(predictions_array, axis=0, weights=weights)

    # Convert to binary (threshold = 0.5)
    ensemble_binary = (ensemble_pred > 0.5).astype(int)

    print(
        f"Weighted ensemble predictions: min={ensemble_pred.min():.4f}, max={ensemble_pred.max():.4f}, mean={ensemble_pred.mean():.4f}"
    )

    return ensemble_binary


def ensemble_voting(predictions_dict):
    """Hard voting: use the majority vote"""
    print("\n" + "=" * 70)
    print("ENSEMBLE STRATEGY: VOTING")
    print("=" * 70)

    predictions_array = np.array(list(predictions_dict.values()))

    # Count votes for each sample
    ensemble_binary = np.apply_along_axis(
        lambda x: 1 if np.sum(x) > len(x) / 2 else 0, axis=0, arr=predictions_array
    )

    print(f"Hard voting with {len(predictions_dict)} models")
    print(f"Decision threshold: > {len(predictions_dict) / 2:.1f} votes")

    return ensemble_binary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("=" * 70)
print("ENSEMBLE PREDICTIONS")
print("=" * 70)

# Parse arguments
args = parse_arguments()

# Load predictions based on specified models
predictions_dict = load_predictions(args.models)
if predictions_dict is None or len(predictions_dict) < 2:
    print("❌ Need at least 2 models for ensembling!")
    sys.exit(1)

# Parse weights if provided
weights = None
if args.weights != "equal":
    try:
        weights = [float(w) for w in args.weights.split(",")]
        if len(weights) != len(predictions_dict):
            print(
                f"❌ Number of weights ({len(weights)}) must match number of models ({len(predictions_dict)})"
            )
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error parsing weights: {e}")
        sys.exit(1)

# Perform ensembling
ensemble_pred = None
if args.strategy == "average":
    ensemble_pred = ensemble_average(predictions_dict)
elif args.strategy == "weighted":
    ensemble_pred = ensemble_weighted(predictions_dict, weights)
elif args.strategy == "voting":
    ensemble_pred = ensemble_voting(predictions_dict)

if ensemble_pred is None:
    print(f"❌ Invalid strategy: {args.strategy}")
    sys.exit(1)

# Get sample submission format
sample_submission = pd.read_csv(
    "data/submissions/sample_submission.csv", index_col="ROW_ID"
)

# Create submission
submission = pd.DataFrame(
    ensemble_pred, index=sample_submission.index, columns=["target"]
)

# Save predictions
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"data/submissions/preds_ensemble_{args.strategy}_{len(predictions_dict)}models_{timestamp}.csv"

os.makedirs("data/submissions", exist_ok=True)
submission.to_csv(output_filename)

# Statistics
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\nEnsemble Statistics:")
print(
    f"  Positive predictions: {ensemble_pred.sum()} ({ensemble_pred.sum() / len(ensemble_pred) * 100:.2f}%)"
)
print(
    f"  Negative predictions: {len(ensemble_pred) - ensemble_pred.sum()} ({(1 - ensemble_pred.mean()) * 100:.2f}%)"
)

print(f"\n✓ Ensemble predictions saved to: {output_filename}")
print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:\n{submission.head()}")

print("\n" + "=" * 70)
print("Ensemble complete!")
print("=" * 70)
