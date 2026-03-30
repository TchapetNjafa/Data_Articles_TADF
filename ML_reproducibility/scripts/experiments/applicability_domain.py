#!/usr/bin/env python3
"""
Applicability Domain Analysis for Article 3
=============================================

Defines the reliable prediction region using:
  1. k-NN distance to training set (distance-based AD)
  2. Leverage approach (hat matrix, Williams plot)
  3. Correlation between distance and prediction error

Identifies out-of-domain molecules where predictions are unreliable.

Output: applicability_domain_results.csv, ad_summary.json
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

from parallel_config import (MAX_THREADS, NUMBA_AVAILABLE, jit, prange,
                              Parallel, delayed, get_parallel_config)

import numpy as np
import pandas as pd

# ──────────────────────── Configuration ────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_FILE = DATA_DIR / "csv" / "combined_features_747mol_full_ct.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "applicability_domain"

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


# Numba-accelerated k-NN distance computation
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _knn_distances(X, k=5):
        """Compute average distance to k nearest neighbors for each sample."""
        n = X.shape[0]
        avg_distances = np.empty(n)
        for i in prange(n):
            # Compute all distances from point i
            dists = np.empty(n - 1)
            idx = 0
            for j in range(n):
                if j == i:
                    continue
                d = 0.0
                for f in range(X.shape[1]):
                    diff = X[i, f] - X[j, f]
                    d += diff * diff
                dists[idx] = np.sqrt(d)
                idx += 1
            # Sort and take k nearest
            dists.sort()
            avg_distances[i] = np.mean(dists[:k])
        return avg_distances
else:
    def _knn_distances(X, k=5):
        """Compute average distance to k nearest neighbors (numpy fallback)."""
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=MAX_THREADS)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        return distances[:, 1:].mean(axis=1)  # exclude self


def compute_leverage(X):
    """Compute leverage (hat matrix diagonal) for each sample."""
    # H = X (X^T X)^{-1} X^T
    # h_ii = diagonal of H
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        H = X @ XtX_inv @ X.T
        return np.diag(H)
    except np.linalg.LinAlgError:
        print("  WARNING: Leverage computation failed (singular matrix)")
        return np.full(X.shape[0], np.nan)


def ad_analysis(df, feature_cols, k=5, n_splits=5):
    """Run full applicability domain analysis."""
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict, KFold
    from scipy import stats

    X = df[feature_cols].values
    y = df[TARGET_COL].values
    mol_ids = df["molecule"].values if "molecule" in df.columns else np.arange(len(y))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. k-NN Distance
    print(f"\n  Computing k-NN distances (k={k})...")
    knn_dist = _knn_distances(np.ascontiguousarray(X_scaled), k=k)

    # 2. Leverage
    print(f"  Computing leverage (hat matrix)...")
    leverage = compute_leverage(X_scaled)

    # 3. Cross-validation predictions for error analysis
    print(f"  Running {n_splits}-fold CV for error estimation...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"))
    ])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(pipe, X, y, cv=kf,
                                   n_jobs=min(n_splits, MAX_THREADS))

    abs_errors = np.abs(y - y_pred)

    # 4. Correlation: distance vs error
    r_dist, p_dist = stats.pearsonr(knn_dist, abs_errors)
    r_lev, p_lev = stats.pearsonr(leverage[~np.isnan(leverage)],
                                   abs_errors[~np.isnan(leverage)])

    print(f"\n  CORRELATION: distance vs |error|")
    print(f"    k-NN distance:  r = {r_dist:.4f}  (p = {p_dist:.2e})")
    print(f"    Leverage:       r = {r_lev:.4f}  (p = {p_lev:.2e})")

    # 5. Define AD threshold (mean + 3*std of k-NN distance)
    dist_threshold = knn_dist.mean() + 3 * knn_dist.std()
    leverage_threshold = 3 * X_scaled.shape[1] / X_scaled.shape[0]  # 3p/n

    in_domain = knn_dist <= dist_threshold
    out_domain = ~in_domain

    print(f"\n  APPLICABILITY DOMAIN (distance threshold = {dist_threshold:.4f})")
    print(f"    In-domain:  {in_domain.sum()} samples ({100*in_domain.mean():.1f}%)")
    print(f"    Out-domain: {out_domain.sum()} samples ({100*out_domain.mean():.1f}%)")

    if in_domain.sum() > 0:
        print(f"    In-domain MAE:  {abs_errors[in_domain].mean():.4f} eV")
    if out_domain.sum() > 0:
        print(f"    Out-domain MAE: {abs_errors[out_domain].mean():.4f} eV")
        print(f"    Error ratio (out/in): "
              f"{abs_errors[out_domain].mean() / max(abs_errors[in_domain].mean(), 1e-10):.2f}x")

    # 6. Identify worst out-of-domain molecules
    if out_domain.sum() > 0:
        ood_idx = np.where(out_domain)[0]
        ood_sorted = ood_idx[np.argsort(-abs_errors[ood_idx])]
        print(f"\n  TOP OUT-OF-DOMAIN MOLECULES (highest error):")
        for idx in ood_sorted[:10]:
            print(f"    {mol_ids[idx]:20s}  dist={knn_dist[idx]:.4f}  "
                  f"|error|={abs_errors[idx]:.4f} eV")

    # 7. Bin analysis (error vs distance bins)
    print(f"\n  ERROR BY DISTANCE BIN:")
    n_bins = 5
    dist_bins = np.percentile(knn_dist, np.linspace(0, 100, n_bins + 1))
    print(f"  {'Bin':>20s}  {'n':>5s}  {'MAE':>8s}  {'Max Err':>8s}")
    print("-" * 50)
    for i in range(n_bins):
        mask = (knn_dist >= dist_bins[i]) & (knn_dist < dist_bins[i + 1])
        if i == n_bins - 1:
            mask = (knn_dist >= dist_bins[i]) & (knn_dist <= dist_bins[i + 1])
        if mask.sum() > 0:
            label = f"[{dist_bins[i]:.2f}, {dist_bins[i+1]:.2f})"
            print(f"  {label:>20s}  {mask.sum():>5d}  "
                  f"{abs_errors[mask].mean():>8.4f}  {abs_errors[mask].max():>8.4f}")

    # Build per-sample results
    sample_results = pd.DataFrame({
        "molecule": mol_ids,
        "knn_distance": np.round(knn_dist, 4),
        "leverage": np.round(leverage, 6),
        "abs_error_eV": np.round(abs_errors, 4),
        "y_true": np.round(y, 4),
        "y_pred": np.round(y_pred, 4),
        "in_domain": in_domain,
    })

    summary = {
        "k": k,
        "n_samples": len(X),
        "n_features": len(feature_cols),
        "dist_threshold": round(float(dist_threshold), 4),
        "leverage_threshold": round(float(leverage_threshold), 6),
        "n_in_domain": int(in_domain.sum()),
        "n_out_domain": int(out_domain.sum()),
        "pct_in_domain": round(float(100 * in_domain.mean()), 1),
        "in_domain_mae": round(float(abs_errors[in_domain].mean()), 4) if in_domain.sum() > 0 else None,
        "out_domain_mae": round(float(abs_errors[out_domain].mean()), 4) if out_domain.sum() > 0 else None,
        "correlation_dist_error_r": round(float(r_dist), 4),
        "correlation_dist_error_p": float(p_dist),
        "correlation_lev_error_r": round(float(r_lev), 4),
        "correlation_lev_error_p": float(p_lev),
    }

    return sample_results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Applicability domain analysis for ΔE_ST predictions")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of nearest neighbors (default: 5)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    result = load_data(describe_only=args.describe_only)
    if result is None:
        return
    df, feature_cols = result

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  APPLICABILITY DOMAIN ANALYSIS")
    print("=" * 60)

    sample_results, summary = ad_analysis(df, feature_cols, k=args.k)

    # Save
    csv_path = output_dir / "applicability_domain_results.csv"
    sample_results.to_csv(csv_path, index=False)
    print(f"\n  Per-sample results saved to {csv_path}")

    summary["date"] = datetime.now().isoformat()
    summary["parallelization"] = get_parallel_config()
    json_path = output_dir / "ad_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {json_path}")


if __name__ == "__main__":
    main()
