#!/usr/bin/env python3
"""Active learning loop template for Article 3.

This script illustrates a simple pool-based active learning (AL) loop
built on top of the tidy feature tables produced by
`data_processing/build_features.py`.

Design assumptions
------------------
- Input: CSV file with numeric descriptors (e.g. overlaps and future
  CT/RespA/HTS descriptors).
- Target column: a numeric column that may be known only for a subset of
  rows (labeled set); missing/empty values indicate the unlabeled pool.
- Features: all other numeric columns except identifiers like
  molecule/environment/transition/method.

Two main modes
--------------
- `--describe-only` (no scikit-learn required):
  - Prints detected numeric columns.
  - If a target column is provided, reports the sizes of the labeled and
    unlabeled pools.
- AL suggestion mode (requires scikit-learn):
  - Fits either a RandomForestRegressor (baseline) or a
    GaussianProcessRegressor (GPR) on the labeled set.
  - Uses predictive uncertainty on the pool and combines it with a
    simple feature-space diversity score ("uncertainty × diversity"),
    following the ML workflow diagram.
  - Suggests indices / identifiers to query in batches.

The script does not modify any files; it only prints suggested query
points so that expensive calculations can be scheduled separately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

from ml_pipeline_template import ID_COLUMNS, infer_numeric_columns, load_table


ROOT = Path(__file__).resolve().parents[2]  # .../Article3_ML
ARTICLE_DIR = ROOT / "Article3"
DATA_PROC_DIR = ARTICLE_DIR / "data_processing"
DEFAULT_INPUT = DATA_PROC_DIR / "combined_features.csv"


def partition_labeled_pool(
    rows: Sequence[Dict[str, str]], target_col: str
) -> Tuple[List[int], List[int]]:
    labeled: List[int] = []
    pool: List[int] = []
    for i, r in enumerate(rows):
        val = r.get(target_col)
        if val is None or str(val).strip() == "":
            pool.append(i)
        else:
            labeled.append(i)
    return labeled, pool


def build_X_y_from_indices(
    rows: Sequence[Dict[str, str]],
    feature_cols: Sequence[str],
    target_col: str,
    indices: Sequence[int],
) -> Tuple[List[List[float]], List[float]]:
    X: List[List[float]] = []
    y: List[float] = []
    for idx in indices:
        r = rows[idx]
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
    return X, y


def build_X_from_indices(
    rows: Sequence[Dict[str, str]],
    feature_cols: Sequence[str],
    indices: Sequence[int],
) -> List[List[float]]:
    X: List[List[float]] = []
    for idx in indices:
        r = rows[idx]
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
    return X


def compute_uncertainty_from_rf(model, X_pool: List[List[float]]) -> List[float]:
    """Estimate uncertainty as the std. dev. of tree predictions for each point."""

    if not hasattr(model, "estimators_") or not model.estimators_:
        return [0.0 for _ in X_pool]

    all_preds: List[Sequence[float]] = [est.predict(X_pool) for est in model.estimators_]
    n_points = len(X_pool)
    uncertainties: List[float] = []
    for j in range(n_points):
        vals = [preds[j] for preds in all_preds]
        if len(vals) < 2:
            uncertainties.append(0.0)
        else:
            uncertainties.append(float(pstdev(vals)))
    return uncertainties


def compute_diversity_scores(
    X_labeled: List[List[float]], X_pool: List[List[float]]
) -> List[float]:
    """Compute a simple diversity score based on distance to labeled points.

    For each pool point we take the distance to its *closest* labeled point
    (in raw feature space) as a diversity proxy. Points far from the labeled
    set are considered more diverse.
    """

    if not X_labeled:
        # If nothing is labeled yet, treat all pool points as equally diverse.
        return [0.0 for _ in X_pool]

    scores: List[float] = []
    for x in X_pool:
        best_sq: float | None = None
        for y in X_labeled:
            # Euclidean distance squared.
            dist_sq = 0.0
            for a, b in zip(x, y):
                diff = a - b
                dist_sq += diff * diff
            if best_sq is None or dist_sq < best_sq:
                best_sq = dist_sq
        scores.append(best_sq if best_sq is not None else 0.0)
    return scores


def _normalize_scores(values: List[float]) -> List[float]:
    """Normalize a list of scores to [0, 1] for combination.

    If all values are identical and positive, they are mapped to 1.0 so that
    the other component (e.g. diversity) controls the ranking. If they are all
    zero, they stay zero.
    """

    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if v_max == v_min:
        if v_max <= 0.0:
            return [0.0 for _ in values]
        return [1.0 for _ in values]
    scale = v_max - v_min
    return [(v - v_min) / scale for v in values]


def combine_uncertainty_diversity(
    uncertainties: List[float], diversities: List[float]
) -> List[float]:
    """Combine uncertainty and diversity into a single acquisition score.

    Both components are first normalized to [0, 1]. The default combination is
    an elementwise product ("uncertainty × diversity"), with fallbacks when
    one component is entirely uninformative.
    """

    if len(uncertainties) != len(diversities):
        raise ValueError("uncertainties and diversities must have the same length")

    u_norm = _normalize_scores(uncertainties)
    d_norm = _normalize_scores(diversities)

    scores: List[float] = []
    for u, d in zip(u_norm, d_norm):
        if u == 0.0 and d == 0.0:
            scores.append(0.0)
        elif u == 0.0:
            scores.append(d)
        elif d == 0.0:
            scores.append(u)
        else:
            scores.append(u * d)
    return scores


def run_active_learning(
    rows: List[Dict[str, str]],
    numeric_cols: Sequence[str],
    target_col: str,
    batch_size: int,
    n_iterations: int,
    random_state: int,
    model_type: str,
) -> None:
    model_type = model_type.lower()
    if model_type not in {"rf", "gpr"}:
        raise SystemExit(f"Unsupported model_type: {model_type!r} (use 'rf' or 'gpr').")

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "scikit-learn is required for active learning suggestions but is "
            "not available in this environment."
        ) from exc

    feature_cols = [c for c in numeric_cols if c != target_col]
    if not feature_cols:
        raise SystemExit(
            "No feature columns remain after excluding the target column. "
            "Add additional numeric descriptors or extend the data table."
        )

    labeled, pool = partition_labeled_pool(rows, target_col)
    print(f"[active_learning] Initial labeled set size: {len(labeled)}")
    print(f"[active_learning] Initial unlabeled pool size: {len(pool)}")

    if len(labeled) < 5 or not pool:
        print(
            "[active_learning] Not enough labeled data or empty pool; "
            "nothing to suggest."
        )
        return

    for it in range(1, n_iterations + 1):
        print(f"[active_learning] Iteration {it}")

        X_labeled, y_labeled = build_X_y_from_indices(
            rows, feature_cols, target_col, labeled
        )
        if len(X_labeled) < 5:
            print("[active_learning] Too few usable labeled samples; stopping.")
            return

        if model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
            model = GaussianProcessRegressor(
                kernel=kernel,
                random_state=random_state,
                normalize_y=True,
            )
        model.fit(X_labeled, y_labeled)

        X_pool = build_X_from_indices(rows, feature_cols, pool)
        if not X_pool:
            print("[active_learning] No usable points left in pool; stopping.")
            return

        if model_type == "rf":
            uncertainties = compute_uncertainty_from_rf(model, X_pool)
        else:
            _, std = model.predict(X_pool, return_std=True)
            uncertainties = [float(s) for s in std]

        diversities = compute_diversity_scores(X_labeled, X_pool)
        scores = combine_uncertainty_diversity(uncertainties, diversities)

        order = sorted(range(len(pool)), key=lambda j: -scores[j])
        batch_indices = order[: min(batch_size, len(order))]

        print(
            "rank,global_index,molecule,environment,transition,method,"
            "uncertainty,diversity,score"
        )
        for rank, j in enumerate(batch_indices, start=1):
            row_idx = pool[j]
            r = rows[row_idx]
            print(
                f"{rank},{row_idx},{r.get('Molecule','')},{r.get('Environment','')},"
                f"{r.get('Transition','')},{r.get('Method','')},"
                f"{uncertainties[j]:.6f},{diversities[j]:.6f},{scores[j]:.6f}"
            )

        # Simulate moving the selected points from pool to labeled.
        new_labeled = [pool[j] for j in batch_indices]
        labeled.extend(new_labeled)
        for j in sorted(batch_indices, reverse=True):
            del pool[j]

        if not pool:
            print("[active_learning] Pool exhausted; stopping.")
            return


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Active learning loop template for Article 3: suggest new points "
            "to label based on a tidy feature table."
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
            "Name of the numeric target column to drive active learning (e.g. "
            "a future k_RISC column)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of points to suggest per AL iteration (default: 5).",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=3,
        help="Number of AL iterations to simulate (default: 3).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for the underlying model (default: 0).",
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
            "Inspect the table and report numeric columns and pool sizes "
            "without fitting a model."
        ),
    )

    args = parser.parse_args(argv)
    input_path = Path(args.input_file)

    rows, fieldnames = load_table(input_path)
    numeric_cols = infer_numeric_columns(rows, fieldnames)

    print(f"[active_learning] Loaded {len(rows)} rows from {input_path}.")
    print("[active_learning] Numeric columns detected:")
    for name in numeric_cols:
        print(f"  - {name}")

    if args.describe_only or args.target_column is None:
        if args.target_column and args.target_column in numeric_cols:
            labeled, pool = partition_labeled_pool(rows, args.target_column)
            print(
                f"[active_learning] Labeled = {len(labeled)}, "
                f"unlabeled pool = {len(pool)}"
            )
        else:
            print(
                "[active_learning] Describe-only mode or no valid target "
                "column specified; skipping AL suggestions."
            )
        return 0

    if args.target_column not in numeric_cols:
        raise SystemExit(
            f"Target column '{args.target_column}' is not among the detected "
            f"numeric columns: {numeric_cols}"
        )

    run_active_learning(
        rows,
        numeric_cols,
        args.target_column,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        random_state=args.random_state,
        model_type=args.model_type,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

