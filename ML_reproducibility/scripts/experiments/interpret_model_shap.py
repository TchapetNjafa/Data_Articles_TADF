#!/usr/bin/env python3
"""
SHAP-based model interpretability for Article 3 ML pipeline.

This script trains a model on the combined_features.csv dataset and uses
SHAP (SHapley Additive exPlanations) to explain feature importance.

Features are grouped into categories for aggregate importance analysis:
- NTO features: S1_overlap, T1_overlap
- Energy features: S1_energy_eV, T1_energy_eV, Delta_E_ST_eV, HOMO_LUMO_gap_eV
- Oscillator: S1_osc_strength

Usage:
    python interpret_model_shap.py --target-column Delta_E_ST_eV --model-type rf

Dependencies: scikit-learn, shap, matplotlib
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# Feature group definitions for aggregate importance
FEATURE_GROUPS = {
    'NTO': ['S1_overlap', 'T1_overlap'],
    'Energy': ['S1_energy_eV', 'T1_energy_eV', 'HOMO_LUMO_gap_eV'],
    'Oscillator': ['S1_osc_strength'],
}


def load_data(input_file: Path, target_column: str) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Load data and return X, y arrays and feature names."""
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"No data rows in {input_file}")

    # Identify numeric columns
    numeric_cols = []
    sample_row = rows[0]
    for col in sample_row:
        if col in ['molecule', 'environment', 'method']:
            continue
        try:
            float(sample_row[col])
            numeric_cols.append(col)
        except (ValueError, TypeError):
            pass

    if target_column not in numeric_cols:
        raise SystemExit(f"Target column '{target_column}' not found in numeric columns: {numeric_cols}")

    # Feature columns = numeric columns minus target
    feature_cols = [c for c in numeric_cols if c != target_column]

    # Build arrays
    X = []
    y = []
    for row in rows:
        try:
            target_val = float(row[target_column])
            features = [float(row[c]) for c in feature_cols]
            X.append(features)
            y.append(target_val)
        except (ValueError, TypeError):
            continue

    return np.array(X), np.array(y), feature_cols


def compute_shap_values(model, X: np.ndarray, model_type: str) -> np.ndarray:
    """Compute SHAP values for the model predictions."""
    import shap

    if model_type == 'rf':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:  # GPR or other
        explainer = shap.KernelExplainer(model.predict, X)
        shap_values = explainer.shap_values(X)

    return shap_values


def compute_feature_importance(shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Compute mean absolute SHAP values for each feature."""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance = {name: float(val) for name, val in zip(feature_names, mean_abs_shap)}
    return importance


def compute_group_importance(feature_importance: Dict[str, float]) -> Dict[str, float]:
    """Aggregate feature importance by group."""
    group_importance = {}
    for group_name, features in FEATURE_GROUPS.items():
        total = sum(feature_importance.get(f, 0.0) for f in features)
        group_importance[group_name] = total

    # Add any ungrouped features
    all_grouped = set(f for features in FEATURE_GROUPS.values() for f in features)
    ungrouped = {f: v for f, v in feature_importance.items() if f not in all_grouped}
    if ungrouped:
        group_importance['Other'] = sum(ungrouped.values())

    return group_importance


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray, feature_names: List[str],
                      output_path: Optional[Path] = None):
    """Create SHAP summary plot."""
    import shap
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[shap] Saved summary plot to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_group_importance(group_importance: Dict[str, float], output_path: Optional[Path] = None):
    """Create bar plot of grouped feature importance."""
    import matplotlib.pyplot as plt

    groups = list(group_importance.keys())
    values = list(group_importance.values())
    total = sum(values)
    percentages = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(groups, percentages, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'][:len(groups)])

    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=10)

    ax.set_xlabel('Feature Group Importance (%)')
    ax.set_title('SHAP Feature Group Importance')
    ax.set_xlim(0, max(percentages) * 1.2)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[shap] Saved group importance plot to {output_path}")
    else:
        plt.show()
    plt.close()


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="SHAP-based model interpretability for Article 3 ML pipeline."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(Path(__file__).parent.parent / "data_processing" / "combined_features.csv"),
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        required=True,
        help="Target column name for regression.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["rf", "gpr"],
        default="rf",
        help="Model type: rf (Random Forest) or gpr (Gaussian Process).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots. If not specified, displays plots interactively.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (just print importance values).",
    )

    args = parser.parse_args(argv)

    input_file = Path(args.input_file)
    if not input_file.is_file():
        raise SystemExit(f"Input file not found: {input_file}")

    print(f"[shap] Loading data from {input_file}")
    X, y, feature_names = load_data(input_file, args.target_column)
    print(f"[shap] Loaded {len(y)} samples with {len(feature_names)} features")
    print(f"[shap] Features: {feature_names}")
    print(f"[shap] Target: {args.target_column}")

    # Train model on full data for SHAP analysis
    print(f"[shap] Training {args.model_type.upper()} model...")

    if args.model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
        model = GaussianProcessRegressor(kernel=kernel, random_state=42, normalize_y=True)

    model.fit(X, y)
    print("[shap] Model trained successfully")

    # Compute SHAP values
    print("[shap] Computing SHAP values...")
    shap_values = compute_shap_values(model, X, args.model_type)

    # Compute importance
    feature_importance = compute_feature_importance(shap_values, feature_names)
    group_importance = compute_group_importance(feature_importance)

    # Print results
    print("\n" + "=" * 50)
    print("SHAP Feature Importance (mean |SHAP|)")
    print("=" * 50)
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for name, val in sorted_features:
        print(f"  {name:20s}: {val:.4f}")

    print("\n" + "=" * 50)
    print("SHAP Group Importance")
    print("=" * 50)
    total = sum(group_importance.values())
    sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
    for name, val in sorted_groups:
        pct = val / total * 100
        print(f"  {name:15s}: {val:.4f} ({pct:.1f}%)")

    # Generate plots
    if not args.no_plots:
        output_dir = Path(args.output_dir) if args.output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / f"shap_summary_{args.target_column}.png" if output_dir else None
        group_path = output_dir / f"shap_groups_{args.target_column}.png" if output_dir else None

        try:
            plot_shap_summary(shap_values, X, feature_names, summary_path)
            plot_group_importance(group_importance, group_path)
        except Exception as e:
            print(f"[shap] Warning: Could not generate plots: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
