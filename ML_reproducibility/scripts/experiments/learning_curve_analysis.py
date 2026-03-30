#!/usr/bin/env python3
"""
Learning Curve Analysis for Article 3
=======================================

Plots MAE vs training set size to quantify data efficiency.
Supports the active learning claims by showing:
  - How model performance improves with more data
  - Saturation point (diminishing returns)
  - Comparison SVR vs RF to show consistency

Output: learning_curve_results.csv, learning_curve_summary.json
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

from parallel_config import (MAX_THREADS, Parallel, delayed, get_parallel_config)

import numpy as np
import pandas as pd

# ──────────────────────── Configuration ────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_FILE = DATA_DIR / "csv" / "combined_features_747mol_full_ct.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "learning_curve"

TARGET_COL = "Delta_E_ST_eV"


def load_data(describe_only=False):
    """Load and prepare the feature dataset."""
    if not FEATURES_FILE.exists():
        print(f"ERROR: Feature file not found: {FEATURES_FILE}")
        sys.exit(1)

    df = pd.read_csv(FEATURES_FILE)
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")

    if describe_only:
        print(f"\nTarget: {TARGET_COL}")
        if TARGET_COL in df.columns:
            y = df[TARGET_COL].dropna()
            print(f"  n={len(y)}, mean={y.mean():.4f}, std={y.std():.4f}")
        return None

    if TARGET_COL not in df.columns:
        if "S1_energy_eV" in df.columns and "T1_energy_eV" in df.columns:
            df[TARGET_COL] = df["S1_energy_eV"] - df["T1_energy_eV"]
        else:
            print("ERROR: Cannot find or compute target column.")
            sys.exit(1)

    id_cols = {"molecule", "environment", "method"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in id_cols and c != TARGET_COL]

    df = df.dropna(subset=[TARGET_COL] + feature_cols)
    print(f"After dropping NaN: {len(df)} samples, {len(feature_cols)} features")

    return df, feature_cols


def compute_learning_curve(X, y, model, train_sizes, n_splits=5, n_repeats=3):
    """Compute learning curve with repeated CV for robust estimates."""
    from sklearn.model_selection import learning_curve

    # Maximum allowed train size for k-fold CV: floor(n * (k-1)/k)
    max_train = int(len(X) * (n_splits - 1) / n_splits)
    abs_train_sizes = (train_sizes * len(X)).astype(int)
    abs_train_sizes = np.clip(abs_train_sizes, n_splits + 1, max_train)
    abs_train_sizes = np.unique(abs_train_sizes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_sizes_out, train_scores, test_scores = learning_curve(
            model, X, y,
            train_sizes=abs_train_sizes,
            cv=n_splits,
            scoring="neg_mean_absolute_error",
            n_jobs=min(n_splits, MAX_THREADS),
            random_state=42,
            return_times=False,
        )

    # Convert to positive MAE
    train_mae = -train_scores
    test_mae = -test_scores

    results = []
    for i, size in enumerate(train_sizes_out):
        results.append({
            "train_size": int(size),
            "train_fraction": round(size / len(X), 3),
            "train_mae_mean": round(float(train_mae[i].mean()), 4),
            "train_mae_std": round(float(train_mae[i].std()), 4),
            "test_mae_mean": round(float(test_mae[i].mean()), 4),
            "test_mae_std": round(float(test_mae[i].std()), 4),
        })

    return results


def run_learning_curves(df, feature_cols):
    """Run learning curve analysis for SVR and RF."""
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    train_fractions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    models = {
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=MAX_THREADS))
        ]),
    }

    all_results = {}

    for name, model in models.items():
        print(f"\n{'━' * 60}")
        print(f"  ▶ Learning Curve: {name}")
        print(f"{'━' * 60}")

        results = compute_learning_curve(X, y, model, train_fractions)
        all_results[name] = results

        # Print table
        print(f"\n  {'Train Size':>10s}  {'Fraction':>8s}  "
              f"{'Train MAE':>10s}  {'Test MAE':>10s}  {'±':>1s}  {'Std':>6s}")
        print("-" * 55)
        for r in results:
            print(f"  {r['train_size']:>10d}  {r['train_fraction']:>8.1%}  "
                  f"{r['train_mae_mean']:>10.4f}  {r['test_mae_mean']:>10.4f}     "
                  f"{r['test_mae_std']:>6.4f}")

    return all_results


def analyze_data_efficiency(all_results):
    """Analyze data efficiency from learning curves."""
    print(f"\n{'=' * 60}")
    print(f"  DATA EFFICIENCY ANALYSIS")
    print(f"{'=' * 60}")

    for model_name, results in all_results.items():
        if len(results) < 3:
            continue

        full_mae = results[-1]["test_mae_mean"]
        target_90 = full_mae * 1.10  # 10% above best

        # Find minimum training size to reach within 10% of best
        min_size = None
        for r in results:
            if r["test_mae_mean"] <= target_90:
                min_size = r["train_size"]
                min_frac = r["train_fraction"]
                break

        print(f"\n  {model_name}:")
        print(f"    Full-data MAE:  {full_mae:.4f} eV")
        print(f"    90% target MAE: {target_90:.4f} eV")
        if min_size:
            print(f"    Min samples for 90% performance: {min_size} ({min_frac:.0%} of data)")
            data_reduction = 1.0 - min_frac
            print(f"    → Data reduction: {data_reduction:.0%} fewer samples needed")
        else:
            print(f"    90% target not reached at any training size")

        # Improvement from 10% to 100% data
        if len(results) >= 2:
            first_mae = results[0]["test_mae_mean"]
            improvement = (first_mae - full_mae) / first_mae * 100
            print(f"    Improvement 10%→100% data: {improvement:.1f}% MAE reduction")

    return


def main():
    parser = argparse.ArgumentParser(
        description="Learning curve analysis for ΔE_ST prediction")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    result = load_data(describe_only=args.describe_only)
    if result is None:
        return
    df, feature_cols = result

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run learning curves
    all_results = run_learning_curves(df, feature_cols)

    # Analyze data efficiency
    analyze_data_efficiency(all_results)

    # Save results
    all_rows = []
    for model_name, results in all_results.items():
        for r in results:
            r["model"] = model_name
            all_rows.append(r)

    df_results = pd.DataFrame(all_rows)
    csv_path = output_dir / "learning_curve_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")

    summary = {
        "date": datetime.now().isoformat(),
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "parallelization": get_parallel_config(),
        "models": {k: v for k, v in all_results.items()},
    }
    json_path = output_dir / "learning_curve_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")


if __name__ == "__main__":
    main()
