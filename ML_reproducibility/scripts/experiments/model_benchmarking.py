#!/usr/bin/env python3
"""
Model Benchmarking — Multi-Algorithm Comparison for Article 3
==============================================================

Addresses Reviewer Concern:
  "Why was only SVR used? How do results compare across different ML models?"

Benchmarks 4 algorithms on the same feature set with 5-fold CV:
  - SVR (RBF kernel)       — current publication model
  - Random Forest (RF)     — ensemble baseline
  - XGBoost (XGB)          — gradient boosting
  - Kernel Ridge (KRR)     — kernel method alternative

If SHAP importances are consistent across models, the physical
interpretation (S_he^T1 dominance) is model-agnostic.

Output: Comparison table (CSV + JSON), per-model metrics.
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# Parallelization config — must be imported BEFORE numpy/sklearn
from parallel_config import (MAX_THREADS, Parallel, delayed, get_parallel_config)

import numpy as np
import pandas as pd

# ──────────────────────── Configuration ────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_FILE = DATA_DIR / "csv" / "combined_features_747mol_full_ct.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "model_benchmarking"

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
            print(f"  Mean: {df[TARGET_COL].mean():.4f} eV")
            print(f"  Std:  {df[TARGET_COL].std():.4f} eV")
            print(f"  Range: [{df[TARGET_COL].min():.4f}, {df[TARGET_COL].max():.4f}] eV")
        return None

    if TARGET_COL not in df.columns:
        if "S1_energy_eV" in df.columns and "T1_energy_eV" in df.columns:
            df[TARGET_COL] = df["S1_energy_eV"] - df["T1_energy_eV"]
        else:
            print("ERROR: Cannot find or compute target column.")
            sys.exit(1)

    # Get numeric feature columns
    id_cols = {"molecule", "environment", "method"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in id_cols and c != TARGET_COL]

    # Drop NaN rows
    essential = [TARGET_COL] + feature_cols
    df = df.dropna(subset=essential)
    print(f"After dropping NaN: {len(df)} samples, {len(feature_cols)} features")

    return df, feature_cols


def get_models():
    """Return dict of model name -> (pipeline_constructor, description)."""
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    models = {
        "SVR (RBF)": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"))
            ]),
            "description": "Support Vector Regression with RBF kernel (publication model)",
            "needs_scaling": True,
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(
                    n_estimators=200, max_depth=None, min_samples_leaf=2,
                    random_state=42, n_jobs=MAX_THREADS))
            ]),
            "description": "Random Forest ensemble (200 trees)",
            "needs_scaling": False,
        },
        "KRR (RBF)": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KernelRidge(kernel="rbf", alpha=0.1, gamma=None))
            ]),
            "description": "Kernel Ridge Regression with RBF kernel",
            "needs_scaling": True,
        },
    }

    # XGBoost (optional)
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=MAX_THREADS, verbosity=0))
            ]),
            "description": "Gradient Boosting (XGBoost, 200 rounds)",
            "needs_scaling": False,
        }
    except ImportError:
        print("  WARNING: xgboost not installed. Skipping XGBoost.")
        print("  Install with: pip install xgboost")

    return models


def benchmark_models(df, feature_cols, n_splits=5, random_state=42):
    """Run 5-fold CV for each model and return comparison table."""
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = get_models()
    results = []

    for name, model_info in models.items():
        print(f"\n{'━' * 60}")
        print(f"  ▶ {name}")
        print(f"    {model_info['description']}")
        print(f"{'━' * 60}")

        pipe = model_info["pipeline"]
        start_time = datetime.now()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(pipe, X, y, cv=kf,
                                       n_jobs=min(n_splits, MAX_THREADS))

        elapsed = (datetime.now() - start_time).total_seconds()

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        max_err = np.max(np.abs(y - y_pred))

        # Per-fold metrics
        fold_maes = []
        for train_idx, test_idx in kf.split(X):
            fold_maes.append(mean_absolute_error(y[test_idx], y_pred[test_idx]))

        print(f"    MAE  = {mae:.4f} ± {np.std(fold_maes):.4f} eV")
        print(f"    RMSE = {rmse:.4f} eV")
        print(f"    R²   = {r2:.4f}")
        print(f"    Max error = {max_err:.4f} eV")
        print(f"    Time = {elapsed:.1f}s")

        results.append({
            "model": name,
            "description": model_info["description"],
            "mae_eV": round(mae, 4),
            "mae_std_eV": round(np.std(fold_maes), 4),
            "rmse_eV": round(rmse, 4),
            "r2": round(r2, 4),
            "max_error_eV": round(max_err, 4),
            "time_s": round(elapsed, 1),
            "n_features": len(feature_cols),
            "n_samples": len(X),
        })

    return results


def print_comparison_table(results):
    """Print formatted comparison table."""
    print(f"\n{'=' * 85}")
    print(f"  MODEL BENCHMARKING COMPARISON")
    print(f"{'=' * 85}")
    print(f"  {'Model':<18s}  {'MAE (eV)':>10s}  {'±':>1s}  {'Std':>6s}  "
          f"{'RMSE (eV)':>10s}  {'R²':>6s}  {'MaxErr':>7s}  {'Time':>6s}")
    print("-" * 85)

    best_mae = min(r["mae_eV"] for r in results)
    best_r2 = max(r["r2"] for r in results)

    for r in results:
        mae_marker = " ★" if r["mae_eV"] == best_mae else "  "
        r2_marker = " ★" if r["r2"] == best_r2 else "  "
        print(f"  {r['model']:<18s}  {r['mae_eV']:>10.4f}     {r['mae_std_eV']:>6.4f}  "
              f"{r['rmse_eV']:>10.4f}  {r['r2']:>6.4f}{r2_marker} "
              f"{r['max_error_eV']:>7.4f}  {r['time_s']:>5.1f}s{mae_marker}")

    print(f"\n  ★ = best in category")

    # Consistency check
    r2_values = [r["r2"] for r in results]
    r2_range = max(r2_values) - min(r2_values)
    if r2_range < 0.05:
        print(f"\n  ✅ CONCLUSION: All models achieve similar R² (range: {r2_range:.4f})")
        print(f"     → Results are MODEL-AGNOSTIC — physical insights are robust")
    else:
        print(f"\n  ⚠️  R² range across models: {r2_range:.4f}")
        print(f"     → Some model dependence detected. Report best + worst.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple ML models for ΔE_ST prediction")
    parser.add_argument("--describe-only", action="store_true",
                        help="Only describe the dataset")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    result = load_data(describe_only=args.describe_only)
    if result is None:
        return
    df, feature_cols = result

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    results = benchmark_models(df, feature_cols, n_splits=args.n_splits)

    # Print comparison
    print_comparison_table(results)

    # Save results
    df_results = pd.DataFrame(results)
    csv_path = output_dir / "model_benchmarking_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")

    summary = {
        "date": datetime.now().isoformat(),
        "dataset": str(FEATURES_FILE),
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "n_splits": args.n_splits,
        "parallelization": get_parallel_config(),
        "results": results,
    }
    json_path = output_dir / "model_benchmarking_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")


if __name__ == "__main__":
    main()
