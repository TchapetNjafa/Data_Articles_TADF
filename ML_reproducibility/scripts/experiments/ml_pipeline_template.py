#!/usr/bin/env python3
"""Template machine-learning pipeline for Article 3.

This script is a *template* for supervised regression on the tidy
feature tables produced by `data_processing/build_features.py`.

Current design:
- Input: CSV file (default: Article3_ML/Article3/data_processing/
  combined_features.csv).
- Target: user-specified numeric column (e.g. a future `k_RISC` or
  `DeltaE_ST` column).
- Features: all other numeric columns (except identifiers such as
  molecule / environment / transition / method).
- Model: RandomForestRegressor (default) or GaussianProcessRegressor
  from scikit-learn (if available).

The script can also be run in `--describe-only` mode, which inspects the
input table and prints candidate numeric columns without importing
scikit-learn or fitting a model.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]  # .../Article3_ML
ARTICLE_DIR = ROOT / "Article3"
DATA_PROC_DIR = ARTICLE_DIR / "data_processing"
DEFAULT_INPUT = DATA_PROC_DIR / "combined_features.csv"


ID_COLUMNS = {"molecule", "environment", "transition", "method"}


def load_table(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise SystemExit(f"Input file {path} is empty.")
    return rows, fieldnames


def _is_numeric_column(name: str, rows: Sequence[Dict[str, str]], max_samples: int = 50) -> bool:
    """Heuristically decide whether a column is numeric."""
    seen = 0
    for row in rows:
        if seen >= max_samples:
            break
        val = row.get(name)
        if val in (None, ""):
            continue
        seen += 1
        try:
            float(val)
        except (TypeError, ValueError):
            return False
    return seen > 0


def infer_numeric_columns(rows: Sequence[Dict[str, str]], fieldnames: Iterable[str]) -> List[str]:
    numeric: List[str] = []
    for name in fieldnames:
        if name in ID_COLUMNS:
            continue
        if _is_numeric_column(name, rows):
            numeric.append(name)
    return numeric


def build_X_y(
    rows: Sequence[Dict[str, str]],
    feature_cols: Sequence[str],
    target_col: str,
) -> Tuple[List[List[float]], List[float]]:
    X: List[List[float]] = []
    y: List[float] = []

    for r in rows:
        try:
            y_val = float(r[target_col])
        except (KeyError, TypeError, ValueError):
            continue

        feat_row: List[float] = []
        skip = False
        for c in feature_cols:
            try:
                feat_row.append(float(r[c]))
            except (KeyError, TypeError, ValueError):
                skip = True
                break
        if skip or not feat_row:
            continue

        X.append(feat_row)
        y.append(y_val)

    if not X:
        raise SystemExit(
            "No usable samples after building X and y. Check that the target "
            "and feature columns are numeric and sufficiently populated."
        )

    return X, y


def run_training(
    X: List[List[float]],
    y: List[float],
    test_size: float,
    random_state: int,
    model_type: str,
) -> None:
    """Train a regression model and report basic metrics.

    model_type:
        - "rf": RandomForestRegressor (default, robust baseline).
        - "gpr": GaussianProcessRegressor (aligned with the Article 3
          workflow for uncertainty-aware surrogates).
    """

    model_type = model_type.lower()
    if model_type not in {"rf", "gpr"}:
        raise SystemExit(f"Unsupported model_type: {model_type!r} (use 'rf' or 'gpr').")

    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        if model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
        else:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(
            "scikit-learn is required for training but is not available.\n"
            "Install it in your environment (e.g. `pip install scikit-learn`) "
            "or run this script with --describe-only to inspect the data.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        header = "# Random forest regression results"
    else:
        # Simple RBF + white-noise kernel; this is a template and can be
        # tuned further in concrete studies.
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        model = GaussianProcessRegressor(
            kernel=kernel,
            random_state=random_state,
            normalize_y=True,
        )
        model.fit(X_train, y_train)
        y_mean, y_std = model.predict(X_test, return_std=True)
        y_pred = y_mean
        header = "# Gaussian process regression results"

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(header)
    print(f"n_train = {len(X_train)}, n_test = {len(X_test)}")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R^2  = {r2:.4f}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Template ML pipeline for Article 3: train a regression model on "
            "a tidy feature table (e.g. combined_features.csv)."
        )
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to the input CSV file (default: combined_features.csv).",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help=(
            "Name of the numeric target column to predict (e.g. a future "
            "k_RISC column). If omitted together with --describe-only, the "
            "script only inspects the table."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for the test set (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for the train/test split and model (default: 0).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="rf",
        choices=["rf", "gpr"],
        help=(
            "Surrogate model type: 'rf' (RandomForest, default) or 'gpr' "
            "(GaussianProcess)."
        ),
    )
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help=(
            "Inspect the input table and print candidate numeric columns "
            "without training a model."
        ),
    )

    args = parser.parse_args(argv)
    input_path = Path(args.input_file)

    rows, fieldnames = load_table(input_path)
    numeric_cols = infer_numeric_columns(rows, fieldnames)

    print(f"[ml_pipeline_template] Loaded {len(rows)} rows from {input_path}.")
    print("[ml_pipeline_template] Numeric columns detected:")
    for name in numeric_cols:
        print(f"  - {name}")

    if args.describe_only or args.target_column is None:
        print(
            "[ml_pipeline_template] Describe-only mode or no target column "
            "specified; skipping model training."
        )
        return 0

    if args.target_column not in numeric_cols:
        raise SystemExit(
            f"Target column '{args.target_column}' is not among the detected "
            f"numeric columns: {numeric_cols}"
        )

    feature_cols = [c for c in numeric_cols if c != args.target_column]
    if not feature_cols:
        raise SystemExit(
            "No feature columns remain after excluding the target column. "
            "Add additional numeric descriptors or extend the data table."
        )

    print("[ml_pipeline_template] Using feature columns:")
    for name in feature_cols:
        print(f"  - {name}")

    X, y = build_X_y(rows, feature_cols, args.target_column)
    run_training(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        model_type=args.model_type,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

