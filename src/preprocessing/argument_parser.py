"""
Command-line argument parser for model training scripts.
Handles imputation method selection and other parameters.
"""

import argparse


def parse_arguments():
    """
    Parse command-line arguments for model training.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with imputation method and other parameters
    """

    parser = argparse.ArgumentParser(
        description="Train models for asset allocation prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/models/ridge_regression.py --imputation median
  python src/models/xgboost_model.py --imputation mice
  python src/models/catboost_model.py  # Uses default (median)
        """,
    )

    parser.add_argument(
        "--imputation",
        type=str,
        default="median",
        choices=["no_imputation", "median", "kde", "mice"],
        help="Imputation method for missing values (default: median)",
        metavar="METHOD",
    )

    args = parser.parse_args()

    return args
