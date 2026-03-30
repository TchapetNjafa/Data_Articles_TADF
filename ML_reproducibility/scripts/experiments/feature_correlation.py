#!/usr/bin/env python3
"""
Feature Correlation Analysis for Article 3
============================================

Computes Pearson and Spearman correlation matrices for all 42 features.
Identifies:
  - Highly correlated pairs (|r| > 0.9) — redundant features
  - Energy–CT independence — supports ablation study
  - Feature clusters for dimensionality understanding

Output: correlation_matrix.csv, highly_correlated_pairs.csv, correlation_summary.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from parallel_config import (MAX_THREADS, get_parallel_config)

import numpy as np
import pandas as pd

# ──────────────────────── Configuration ────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_FILE = DATA_DIR / "csv" / "combined_features_747mol_full_ct.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "feature_correlation"

TARGET_COL = "Delta_E_ST_eV"

ENERGY_FEATURES = [
    "S1_energy_eV", "T1_energy_eV", "HOMO_LUMO_gap_eV", "S1_osc_strength"
]

CT_FEATURES = [
    "S1_CT_number", "S1_Lambda_D", "S1_Lambda_A",
    "S1_hole_on_A", "S1_particle_on_D", "S1_Delta_r", "S1_S_he",
    "T1_CT_number", "T1_Lambda_D", "T1_Lambda_A",
    "T1_hole_on_A", "T1_particle_on_D", "T1_Delta_r", "T1_S_he",
    "Delta_CT_number", "Abs_Delta_CT_number",
    "Delta_Lambda_D", "Abs_Delta_Lambda_D",
    "Delta_Lambda_A", "Abs_Delta_Lambda_A",
    "Delta_Delta_r", "Abs_Delta_Delta_r",
    "Delta_S_he", "Abs_Delta_S_he"
]

OVERLAP_FEATURES = [
    "S1_overlap", "T1_overlap", "Delta_S_NTO", "Abs_Delta_S_NTO",
    "Char_diff_squared", "S_NTO_sum", "S_NTO_product",
    "Log_Abs_S1", "Log_Abs_T1", "S_NTO_ratio"
]


def load_data(describe_only=False):
    """Load and prepare the feature dataset."""
    if not FEATURES_FILE.exists():
        print(f"ERROR: Feature file not found: {FEATURES_FILE}")
        sys.exit(1)

    df = pd.read_csv(FEATURES_FILE)
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")

    if describe_only:
        numeric = df.select_dtypes(include=[np.number]).columns
        print(f"\nNumeric features: {len(numeric)}")
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


def classify_feature(name):
    """Classify a feature into its group."""
    if name in ENERGY_FEATURES:
        return "Energy"
    elif name in CT_FEATURES:
        return "CT"
    elif name in OVERLAP_FEATURES:
        return "Overlap"
    else:
        return "Other"


def compute_correlations(df, feature_cols, output_dir):
    """Compute Pearson and Spearman correlation matrices."""
    X = df[feature_cols]

    # Pearson
    print("\n  Computing Pearson correlations...")
    pearson_corr = X.corr(method="pearson")

    # Spearman
    print("  Computing Spearman correlations...")
    spearman_corr = X.corr(method="spearman")

    # Save full matrices
    pearson_corr.to_csv(output_dir / "pearson_correlation_matrix.csv")
    spearman_corr.to_csv(output_dir / "spearman_correlation_matrix.csv")

    return pearson_corr, spearman_corr


def find_highly_correlated(corr_matrix, threshold=0.9):
    """Find pairs with |r| > threshold."""
    pairs = []
    n = corr_matrix.shape[0]
    cols = corr_matrix.columns

    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix.iloc[i, j]
            if abs(r) > threshold:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "group_1": classify_feature(cols[i]),
                    "group_2": classify_feature(cols[j]),
                    "pearson_r": round(r, 4),
                    "cross_group": classify_feature(cols[i]) != classify_feature(cols[j]),
                })

    return sorted(pairs, key=lambda x: -abs(x["pearson_r"]))


def analyze_group_correlations(corr_matrix, feature_cols):
    """Analyze correlations between feature groups (Energy vs CT vs Overlap)."""
    groups = {}
    for col in feature_cols:
        g = classify_feature(col)
        if g not in groups:
            groups[g] = []
        groups[g].append(col)

    print(f"\n{'=' * 70}")
    print(f"  INTER-GROUP CORRELATION ANALYSIS")
    print(f"{'=' * 70}")

    group_names = sorted(groups.keys())
    results = {}

    print(f"\n  {'Group 1':<12s}  {'Group 2':<12s}  {'Mean |r|':>10s}  {'Max |r|':>10s}  {'Pairs':>6s}")
    print("-" * 55)

    for i, g1 in enumerate(group_names):
        for g2 in group_names[i:]:
            cols1 = groups[g1]
            cols2 = groups[g2]

            r_values = []
            for c1 in cols1:
                for c2 in cols2:
                    if c1 != c2:
                        r_values.append(abs(corr_matrix.loc[c1, c2]))

            if r_values:
                mean_r = np.mean(r_values)
                max_r = np.max(r_values)
                key = f"{g1}_vs_{g2}"
                results[key] = {
                    "mean_abs_r": round(mean_r, 4),
                    "max_abs_r": round(max_r, 4),
                    "n_pairs": len(r_values),
                }
                print(f"  {g1:<12s}  {g2:<12s}  {mean_r:>10.4f}  {max_r:>10.4f}  {len(r_values):>6d}")

    return results


def correlations_with_target(df, feature_cols):
    """Compute feature-target correlations."""
    from scipy import stats

    print(f"\n{'=' * 70}")
    print(f"  CORRELATIONS WITH ΔE_ST")
    print(f"{'=' * 70}")
    print(f"  {'Feature':<30s}  {'Group':<8s}  {'Pearson r':>10s}  {'p-value':>10s}  {'|r|':>6s}")
    print("-" * 70)

    results = []
    for col in feature_cols:
        valid = df[[col, TARGET_COL]].dropna()
        if len(valid) > 10:
            r, p = stats.pearsonr(valid[col], valid[TARGET_COL])
            group = classify_feature(col)
            results.append({
                "feature": col,
                "group": group,
                "pearson_r": round(r, 4),
                "p_value": float(p),
                "abs_r": round(abs(r), 4),
            })

    results.sort(key=lambda x: -x["abs_r"])

    for r in results[:15]:  # Top 15
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"  {r['feature']:<30s}  {r['group']:<8s}  {r['pearson_r']:>10.4f}  "
              f"{r['p_value']:>10.2e}  {r['abs_r']:>6.4f}  {sig}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Feature correlation analysis")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Threshold for 'highly correlated' pairs (default: 0.9)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    result = load_data(describe_only=args.describe_only)
    if result is None:
        return
    df, feature_cols = result

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Correlation matrices
    pearson, spearman = compute_correlations(df, feature_cols, output_dir)

    # 2. Highly correlated pairs
    print(f"\n{'=' * 70}")
    print(f"  HIGHLY CORRELATED PAIRS (|r| > {args.threshold})")
    print(f"{'=' * 70}")

    pairs = find_highly_correlated(pearson, threshold=args.threshold)
    if pairs:
        print(f"  Found {len(pairs)} highly correlated pairs:")
        for p in pairs[:20]:
            cross = " [CROSS-GROUP]" if p["cross_group"] else ""
            print(f"    {p['feature_1']:25s} ↔ {p['feature_2']:25s}  "
                  f"r = {p['pearson_r']:+.4f}  ({p['group_1']}÷{p['group_2']}){cross}")

        pd.DataFrame(pairs).to_csv(
            output_dir / "highly_correlated_pairs.csv", index=False)
    else:
        print(f"  No pairs found with |r| > {args.threshold}")

    # 3. Inter-group correlations
    group_results = analyze_group_correlations(pearson, feature_cols)

    # 4. Target correlations
    target_results = correlations_with_target(df, feature_cols)

    # Key finding: Energy-CT independence
    energy_ct_pairs = [p for p in pairs if p["cross_group"]
                       and "Energy" in (p["group_1"], p["group_2"])
                       and "CT" in (p["group_1"], p["group_2"])]

    print(f"\n{'=' * 70}")
    print(f"  KEY FINDING: Energy–CT Feature Independence")
    print(f"{'=' * 70}")
    if not energy_ct_pairs:
        print(f"  ✅ No Energy-CT pairs with |r| > {args.threshold}")
        print(f"     → CT descriptors are INDEPENDENT from energy features")
        print(f"     → Supports ablation study: CT descriptors add non-redundant info")
    else:
        print(f"  ⚠️  {len(energy_ct_pairs)} Energy-CT pairs with |r| > {args.threshold}")

    # Save summary
    summary = {
        "date": datetime.now().isoformat(),
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "correlation_threshold": args.threshold,
        "n_highly_correlated_pairs": len(pairs),
        "n_cross_group_pairs": sum(1 for p in pairs if p["cross_group"]),
        "group_correlations": group_results,
        "top_target_correlations": target_results[:10],
        "parallelization": get_parallel_config(),
    }
    json_path = output_dir / "correlation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {json_path}")


if __name__ == "__main__":
    main()
