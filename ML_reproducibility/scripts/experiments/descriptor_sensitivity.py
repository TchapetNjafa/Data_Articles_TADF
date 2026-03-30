#!/usr/bin/env python3
"""
Descriptor Sensitivity / Noise Robustness Analysis for Article 3
=================================================================

Tests model robustness by adding Gaussian noise to features and
measuring prediction degradation:
  - Overall noise levels: 1%, 5%, 10% of feature σ
  - Per-feature sensitivity: which features are most noise-critical
  - Validates that xTB descriptor noise (~5-10%) doesn't ruin predictions

Output: sensitivity_results.csv, sensitivity_summary.json
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
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "descriptor_sensitivity"

TARGET_COL = "Delta_E_ST_eV"

NOISE_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]  # fraction of feature σ


def load_data(describe_only=False):
    """Load and prepare the feature dataset."""
    if not FEATURES_FILE.exists():
        print(f"ERROR: Feature file not found: {FEATURES_FILE}")
        sys.exit(1)

    df = pd.read_csv(FEATURES_FILE)
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")

    if describe_only:
        print(f"\nTarget: {TARGET_COL}")
        return None

    if TARGET_COL not in df.columns:
        if "S1_energy_eV" in df.columns and "T1_energy_eV" in df.columns:
            df[TARGET_COL] = df["S1_energy_eV"] - df["T1_energy_eV"]

    id_cols = {"molecule", "environment", "method"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in id_cols and c != TARGET_COL]

    df = df.dropna(subset=[TARGET_COL] + feature_cols)
    print(f"After dropping NaN: {len(df)} samples, {len(feature_cols)} features")

    return df, feature_cols


def evaluate_with_noise(X, y, noise_fraction, feature_stds, n_repeats=5):
    """Train SVR on clean data, predict on noisy data. Average over repeats."""
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import mean_absolute_error, r2_score

    maes = []
    r2s = []

    for seed in range(n_repeats):
        rng = np.random.RandomState(seed)
        noise = rng.randn(*X.shape) * feature_stds * noise_fraction
        X_noisy = X + noise

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"))
        ])

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(pipe, X_noisy, y, cv=kf,
                                       n_jobs=min(5, MAX_THREADS))

        maes.append(mean_absolute_error(y, y_pred))
        r2s.append(r2_score(y, y_pred))

    return np.mean(maes), np.std(maes), np.mean(r2s), np.std(r2s)


def global_noise_analysis(df, feature_cols, n_repeats=5):
    """Test model robustness at multiple global noise levels."""
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    feature_stds = X.std(axis=0)

    print(f"\n{'=' * 70}")
    print(f"  GLOBAL NOISE ROBUSTNESS ANALYSIS")
    print(f"{'=' * 70}")

    # Baseline (no noise)
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import mean_absolute_error, r2_score

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"))
    ])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(pipe, X, y, cv=kf,
                                   n_jobs=min(5, MAX_THREADS))
    baseline_mae = mean_absolute_error(y, y_pred)
    baseline_r2 = r2_score(y, y_pred)

    print(f"\n  Baseline (no noise): MAE = {baseline_mae:.4f} eV, R² = {baseline_r2:.4f}")
    print(f"\n  {'Noise Level':>12s}  {'MAE (eV)':>10s}  {'±':>1s}  {'Std':>6s}  "
          f"{'R²':>6s}  {'MAE Δ%':>8s}  {'R² Δ':>8s}")
    print("-" * 60)

    results = [{
        "noise_fraction": 0.0,
        "mae_mean": round(baseline_mae, 4),
        "mae_std": 0.0,
        "r2_mean": round(baseline_r2, 4),
        "r2_std": 0.0,
        "mae_pct_change": 0.0,
        "r2_change": 0.0,
    }]

    for noise_frac in NOISE_LEVELS:
        mae_mean, mae_std, r2_mean, r2_std = evaluate_with_noise(
            X, y, noise_frac, feature_stds, n_repeats=n_repeats
        )

        mae_pct = (mae_mean - baseline_mae) / baseline_mae * 100
        r2_change = r2_mean - baseline_r2

        print(f"  {noise_frac:>11.0%}  {mae_mean:>10.4f}     {mae_std:>6.4f}  "
              f"{r2_mean:>6.4f}  {mae_pct:>+7.1f}%  {r2_change:>+8.4f}")

        results.append({
            "noise_fraction": noise_frac,
            "mae_mean": round(mae_mean, 4),
            "mae_std": round(mae_std, 4),
            "r2_mean": round(r2_mean, 4),
            "r2_std": round(r2_std, 4),
            "mae_pct_change": round(mae_pct, 1),
            "r2_change": round(r2_change, 4),
        })

    return results, baseline_mae, baseline_r2


def per_feature_sensitivity(df, feature_cols, noise_fraction=0.10, n_repeats=3):
    """Test sensitivity by adding noise to one feature at a time."""
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import mean_absolute_error

    X = df[feature_cols].values
    y = df[TARGET_COL].values
    feature_stds = X.std(axis=0)

    # Baseline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"))
    ])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_base = cross_val_predict(pipe, X, y, cv=kf,
                                         n_jobs=min(5, MAX_THREADS))
    baseline_mae = mean_absolute_error(y, y_pred_base)

    print(f"\n{'=' * 70}")
    print(f"  PER-FEATURE SENSITIVITY (noise = {noise_fraction:.0%} of σ)")
    print(f"{'=' * 70}")

    results = []

    def _eval_feature(feat_idx):
        """Evaluate with noise on single feature."""
        maes = []
        for seed in range(n_repeats):
            rng = np.random.RandomState(seed)
            X_noisy = X.copy()
            X_noisy[:, feat_idx] += rng.randn(len(X)) * feature_stds[feat_idx] * noise_fraction

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = cross_val_predict(
                    Pipeline([
                        ("scaler", StandardScaler()),
                        ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"))
                    ]),
                    X_noisy, y, cv=kf, n_jobs=1
                )
            maes.append(mean_absolute_error(y, y_pred))
        return np.mean(maes)

    # Parallel evaluation
    print(f"  Evaluating {len(feature_cols)} features...")
    noisy_maes = Parallel(n_jobs=min(MAX_THREADS, len(feature_cols)))(
        delayed(_eval_feature)(i) for i in range(len(feature_cols))
    )

    for i, (col, noisy_mae) in enumerate(zip(feature_cols, noisy_maes)):
        pct_change = (noisy_mae - baseline_mae) / baseline_mae * 100
        results.append({
            "feature": col,
            "noisy_mae_eV": round(noisy_mae, 4),
            "mae_pct_change": round(pct_change, 2),
        })

    # Sort by sensitivity
    results.sort(key=lambda x: -x["mae_pct_change"])

    print(f"\n  TOP 15 MOST SENSITIVE FEATURES:")
    print(f"  {'Feature':<30s}  {'Noisy MAE':>10s}  {'Δ MAE%':>8s}")
    print("-" * 55)
    for r in results[:15]:
        print(f"  {r['feature']:<30s}  {r['noisy_mae_eV']:>10.4f}  {r['mae_pct_change']:>+7.2f}%")

    print(f"\n  LEAST SENSITIVE FEATURES:")
    for r in results[-5:]:
        print(f"  {r['feature']:<30s}  {r['noisy_mae_eV']:>10.4f}  {r['mae_pct_change']:>+7.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Descriptor sensitivity / noise robustness analysis")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--n-repeats", type=int, default=5,
                        help="Number of noise repeats (default: 5)")
    parser.add_argument("--skip-per-feature", action="store_true",
                        help="Skip per-feature analysis (much faster)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    result = load_data(describe_only=args.describe_only)
    if result is None:
        return
    df, feature_cols = result

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Global noise analysis
    global_results, baseline_mae, baseline_r2 = global_noise_analysis(
        df, feature_cols, n_repeats=args.n_repeats
    )

    # Interpretation
    r5 = next((r for r in global_results if r["noise_fraction"] == 0.05), None)
    r10 = next((r for r in global_results if r["noise_fraction"] == 0.10), None)
    print(f"\n  INTERPRETATION:")
    if r5 and abs(r5["mae_pct_change"]) < 10:
        print(f"    ✅ Model is robust to 5% noise (MAE change: {r5['mae_pct_change']:+.1f}%)")
    if r10 and abs(r10["mae_pct_change"]) < 20:
        print(f"    ✅ Model is robust to 10% noise (MAE change: {r10['mae_pct_change']:+.1f}%)")
    elif r10:
        print(f"    ⚠️  10% noise causes {r10['mae_pct_change']:+.1f}% MAE degradation")

    # 2. Per-feature sensitivity
    per_feature_results = []
    if not args.skip_per_feature:
        per_feature_results = per_feature_sensitivity(
            df, feature_cols, n_repeats=min(args.n_repeats, 3)
        )

    # Save
    df_global = pd.DataFrame(global_results)
    df_global.to_csv(output_dir / "noise_robustness_global.csv", index=False)

    if per_feature_results:
        df_per_feat = pd.DataFrame(per_feature_results)
        df_per_feat.to_csv(output_dir / "noise_sensitivity_per_feature.csv", index=False)

    summary = {
        "date": datetime.now().isoformat(),
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "n_repeats": args.n_repeats,
        "baseline_mae": baseline_mae,
        "baseline_r2": baseline_r2,
        "noise_levels": NOISE_LEVELS,
        "parallelization": get_parallel_config(),
        "global_results": global_results,
        "per_feature_top10": per_feature_results[:10] if per_feature_results else [],
    }
    json_path = output_dir / "sensitivity_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
