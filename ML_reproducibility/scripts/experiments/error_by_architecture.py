#!/usr/bin/env python3
"""
Error Analysis by Molecular Architecture for Article 3
========================================================

Stratifies model prediction errors by architecture class (D-A, D-A-D, MR, etc.)
to identify:
  - Which architectures are well-predicted vs. problematic
  - Systematic biases per architecture
  - Outlier rates per class
  - Whether MR-TADF (different physics) is systematically worse

Output: error_by_architecture_results.csv, architecture_error_summary.json
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
ARCH_FILE = DATA_DIR / "csv" / "tadf_architecture_analysis.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "error_by_architecture"

TARGET_COL = "Delta_E_ST_eV"


def load_data(describe_only=False):
    """Load features and architecture classification."""
    if not FEATURES_FILE.exists():
        print(f"ERROR: Feature file not found: {FEATURES_FILE}")
        sys.exit(1)

    df = pd.read_csv(FEATURES_FILE)
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")

    # Load architecture classification
    arch_col = None
    if ARCH_FILE.exists():
        arch_df = pd.read_csv(ARCH_FILE)
        print(f"Loaded architecture data: {len(arch_df)} entries")
        if "Molecule_id" in arch_df.columns:
            arch_df = arch_df.rename(columns={"Molecule_id": "molecule"})
        for c in ["Architecture", "architecture", "Category", "category"]:
            if c in arch_df.columns:
                arch_col = c
                break
        if arch_col and "molecule" in arch_df.columns and "molecule" in df.columns:
            df = df.merge(arch_df[["molecule", arch_col]].drop_duplicates(),
                          on="molecule", how="left")
            print(f"  Merged architecture info ({arch_col}): "
                  f"{df[arch_col].notna().sum()} matches")
    else:
        print(f"WARNING: Architecture file not found at {ARCH_FILE}")

    if describe_only:
        if arch_col:
            print(f"\nArchitecture distribution:")
            print(df[arch_col].value_counts().to_string())
        return None

    if TARGET_COL not in df.columns:
        if "S1_energy_eV" in df.columns and "T1_energy_eV" in df.columns:
            df[TARGET_COL] = df["S1_energy_eV"] - df["T1_energy_eV"]

    id_cols = {"molecule", "environment", "method", arch_col} if arch_col else {"molecule", "environment", "method"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in id_cols and c != TARGET_COL]

    df = df.dropna(subset=[TARGET_COL] + feature_cols)
    print(f"After dropping NaN: {len(df)} samples")

    return df, feature_cols, arch_col


def analyze_errors_by_architecture(df, feature_cols, arch_col):
    """Run CV and stratify errors by architecture."""
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import mean_absolute_error, r2_score

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"))
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(pipe, X, y, cv=kf,
                                   n_jobs=min(5, MAX_THREADS))

    df = df.copy()
    df["y_pred"] = y_pred
    df["residual"] = y_pred - y
    df["abs_error"] = np.abs(y_pred - y)

    # Overall metrics
    overall_mae = mean_absolute_error(y, y_pred)
    overall_r2 = r2_score(y, y_pred)
    print(f"\n  Overall: MAE = {overall_mae:.4f} eV, R² = {overall_r2:.4f}")

    # Outlier threshold (2× MAE)
    outlier_threshold = 2 * overall_mae

    results = []

    if arch_col and arch_col in df.columns:
        print(f"\n{'=' * 80}")
        print(f"  ERROR ANALYSIS BY ARCHITECTURE ({arch_col})")
        print(f"{'=' * 80}")
        print(f"  {'Architecture':<20s}  {'n':>5s}  {'MAE':>8s}  {'RMSE':>8s}  "
              f"{'R²':>6s}  {'Bias':>8s}  {'Outliers':>8s}")
        print("-" * 80)

        for arch in sorted(df[arch_col].dropna().unique()):
            mask = df[arch_col] == arch
            subset = df[mask]
            if len(subset) < 5:
                continue

            y_true_s = subset[TARGET_COL].values
            y_pred_s = subset["y_pred"].values

            mae = mean_absolute_error(y_true_s, y_pred_s)
            rmse = np.sqrt(np.mean((y_true_s - y_pred_s)**2))
            r2 = r2_score(y_true_s, y_pred_s) if len(y_true_s) > 2 else float("nan")
            bias = np.mean(y_pred_s - y_true_s)
            n_outliers = (subset["abs_error"] > outlier_threshold).sum()
            pct_outliers = 100 * n_outliers / len(subset)

            print(f"  {str(arch):<20s}  {len(subset):>5d}  {mae:>8.4f}  {rmse:>8.4f}  "
                  f"{r2:>6.4f}  {bias:>+8.4f}  {n_outliers:>3d} ({pct_outliers:>4.1f}%)")

            results.append({
                "architecture": str(arch),
                "n_samples": len(subset),
                "mae_eV": round(mae, 4),
                "rmse_eV": round(rmse, 4),
                "r2": round(r2, 4) if not np.isnan(r2) else None,
                "bias_eV": round(bias, 4),
                "n_outliers": int(n_outliers),
                "pct_outliers": round(pct_outliers, 1),
            })

        # Worst architectures
        if results:
            worst = max(results, key=lambda x: x["mae_eV"])
            best = min(results, key=lambda x: x["mae_eV"])
            print(f"\n  Best predicted:  {best['architecture']} (MAE = {best['mae_eV']:.4f} eV)")
            print(f"  Worst predicted: {worst['architecture']} (MAE = {worst['mae_eV']:.4f} eV)")
            print(f"  Error ratio (worst/best): {worst['mae_eV']/max(best['mae_eV'], 1e-10):.2f}x")
    else:
        print("\n  No architecture classification available.")
        print("  Running analysis by ΔE_ST range instead.")

        # Fallback: analyze by ΔE_ST range
        bins = [(-np.inf, 0), (0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, np.inf)]
        bin_labels = ["<0 (inv.)", "0-0.1", "0.1-0.3", "0.3-0.5", ">0.5"]

        print(f"\n  {'ΔE_ST Range':<15s}  {'n':>5s}  {'MAE':>8s}  {'Bias':>8s}")
        print("-" * 40)

        for (lo, hi), label in zip(bins, bin_labels):
            mask = (y >= lo) & (y < hi)
            if mask.sum() < 5:
                continue
            mae = np.mean(np.abs(y_pred[mask] - y[mask]))
            bias = np.mean(y_pred[mask] - y[mask])
            print(f"  {label:<15s}  {mask.sum():>5d}  {mae:>8.4f}  {bias:>+8.4f}")

            results.append({
                "architecture": f"ΔE_ST_{label}",
                "n_samples": int(mask.sum()),
                "mae_eV": round(mae, 4),
                "bias_eV": round(bias, 4),
            })

    return results, df


def main():
    parser = argparse.ArgumentParser(
        description="Error analysis stratified by molecular architecture")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    result = load_data(describe_only=args.describe_only)
    if result is None:
        return
    df, feature_cols, arch_col = result

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, df_with_errors = analyze_errors_by_architecture(df, feature_cols, arch_col)

    # Save
    df_results = pd.DataFrame(results)
    csv_path = output_dir / "error_by_architecture_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")

    # Save per-sample errors
    error_cols = ["molecule", "abs_error", "residual", "y_pred", TARGET_COL]
    if arch_col and arch_col in df_with_errors.columns:
        error_cols.append(arch_col)
    per_sample = df_with_errors[[c for c in error_cols if c in df_with_errors.columns]]
    per_sample.to_csv(output_dir / "per_sample_errors.csv", index=False)

    summary = {
        "date": datetime.now().isoformat(),
        "n_samples": len(df),
        "architecture_column": arch_col,
        "parallelization": get_parallel_config(),
        "results": results,
    }
    json_path = output_dir / "architecture_error_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")


if __name__ == "__main__":
    main()
