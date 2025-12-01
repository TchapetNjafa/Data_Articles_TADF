#!/usr/bin/env python3
"""
Generate publication figures for advanced ML and AL experiments.
Outputs in PNG, PDF, and PGF formats.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results_747mol"
FIGURES_DIR = SCRIPT_DIR.parent / "figures_747mol"

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

COLORS = {
    'random': '#e74c3c',
    'uncertainty': '#3498db',
    'expected_improvement': '#2ecc71',
    'ucb': '#9b59b6',
    'qbc': '#f39c12',
    'diversity': '#1abc9c',
    'hybrid': '#34495e',
}

MODEL_COLORS = {
    'Random Forest': '#3498db',
    'Gradient Boosting': '#2ecc71',
    'SVR (RBF)': '#9b59b6',
    'Neural Network': '#e74c3c',
    'GPR (Matern)': '#f39c12',
}


def save_figure(fig, name):
    """Save figure in PNG, PDF, and PGF formats."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches='tight')
    try:
        fig.savefig(FIGURES_DIR / f"{name}.pgf", backend='pgf')
    except Exception as e:
        print(f"    PGF warning: {e}")
    print(f"  Saved: {name}")


def plot_model_comparison(results):
    """Plot cross-validation comparison of all ML models."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    cv = results['cross_validation']
    models = list(cv.keys())

    # Filter models with mae_mean
    models = [m for m in models if 'mae_mean' in cv[m] or 'mae_test' in cv[m]]

    mae_vals = []
    mae_stds = []
    r2_vals = []
    r2_stds = []
    colors = []

    for model in models:
        if 'mae_mean' in cv[model]:
            mae_vals.append(cv[model]['mae_mean'])
            mae_stds.append(cv[model].get('mae_std', 0))
            r2_vals.append(cv[model]['r2_mean'])
            r2_stds.append(cv[model].get('r2_std', 0))
        else:
            mae_vals.append(cv[model].get('mae_test', 0))
            mae_stds.append(0)
            r2_vals.append(cv[model].get('r2_test', 0))
            r2_stds.append(0)
        colors.append(MODEL_COLORS.get(model, '#7f8c8d'))

    x = np.arange(len(models))

    # MAE plot
    ax = axes[0]
    bars = ax.bar(x, mae_vals, yerr=mae_stds, color=colors, alpha=0.8, capsize=3)
    ax.set_ylabel('MAE (eV)')
    ax.set_title('Cross-Validation MAE')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=8)
    ax.axhline(0.05, color='gray', linestyle='--', alpha=0.5, label='Target')

    # R² plot
    ax = axes[1]
    bars = ax.bar(x, r2_vals, yerr=r2_stds, color=colors, alpha=0.8, capsize=3)
    ax.set_ylabel(r'R$^2$ Score')
    ax.set_title(r'Cross-Validation R$^2$')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=8)
    ax.set_ylim(0.7, 1.0)

    plt.tight_layout()
    save_figure(fig, 'model_comparison_advanced')
    plt.close(fig)


def plot_al_acquisition_comparison(results):
    """Plot comparison of all AL acquisition functions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    al_results = results['results']

    # Learning curves
    ax = axes[0]
    for acq_fn, data in al_results.items():
        n_samples = data['n_samples']
        mae_mean = data['mae_mean']
        mae_std = data['mae_std']
        color = COLORS.get(acq_fn, '#7f8c8d')
        linestyle = '--' if acq_fn == 'random' else '-'
        ax.plot(n_samples, mae_mean, linestyle, color=color, label=acq_fn.replace('_', ' ').title(), linewidth=2)
        ax.fill_between(n_samples,
                       np.array(mae_mean) - np.array(mae_std),
                       np.array(mae_mean) + np.array(mae_std),
                       alpha=0.15, color=color)

    ax.set_xlabel('Training Samples')
    ax.set_ylabel('MAE (eV)')
    ax.set_title('Active Learning: Acquisition Function Comparison')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Final MAE comparison
    ax = axes[1]
    acq_fns = list(al_results.keys())
    final_maes = [al_results[a]['final_mae'] for a in acq_fns]
    final_stds = [al_results[a]['final_mae_std'] for a in acq_fns]
    colors = [COLORS.get(a, '#7f8c8d') for a in acq_fns]

    x = np.arange(len(acq_fns))
    bars = ax.bar(x, final_maes, yerr=final_stds, color=colors, alpha=0.8, capsize=3)

    ax.set_ylabel('Final MAE (eV)')
    ax.set_title('Final MAE by Acquisition Function')
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('_', '\n') for a in acq_fns], fontsize=8)

    # Highlight best
    best_idx = np.argmin(final_maes)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    save_figure(fig, 'al_acquisition_comparison')
    plt.close(fig)


def plot_al_improvement_heatmap(results):
    """Plot improvement over random as heatmap."""
    fig, ax = plt.subplots(figsize=(8, 5))

    al_results = results['results']
    improvements = results.get('improvements_over_random', {})

    acq_fns = [a for a in al_results.keys() if a != 'random']
    if not improvements:
        random_mae = al_results['random']['final_mae']
        improvements = {a: (random_mae - al_results[a]['final_mae']) / random_mae * 100 for a in acq_fns}

    # Bar plot
    x = np.arange(len(acq_fns))
    colors = [COLORS.get(a, '#7f8c8d') for a in acq_fns]
    imps = [improvements[a] for a in acq_fns]

    bars = ax.bar(x, imps, color=colors, alpha=0.8)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Improvement over Random (%)')
    ax.set_title('Active Learning: Improvement over Random Sampling')
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('_', ' ').title() for a in acq_fns], fontsize=9)

    # Add value labels
    for bar, imp in zip(bars, imps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                f'{imp:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'al_improvement_comparison')
    plt.close(fig)


def plot_comprehensive_summary(ml_results, al_results):
    """Create comprehensive summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. ML Model comparison (MAE)
    ax = axes[0, 0]
    cv = ml_results['cross_validation']
    models = [m for m in cv.keys() if 'mae_mean' in cv[m]]
    mae_vals = [cv[m]['mae_mean'] for m in models]
    colors = [MODEL_COLORS.get(m, '#7f8c8d') for m in models]
    bars = ax.barh(range(len(models)), mae_vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('MAE (eV)')
    ax.set_title('ML Models: Cross-Validation MAE')
    ax.axvline(0.05, color='red', linestyle='--', alpha=0.5, label='Target')
    ax.invert_yaxis()

    # 2. Feature importance
    ax = axes[0, 1]
    if 'feature_importance' in ml_results:
        importance = ml_results['feature_importance']
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features = [x[0] for x in sorted_items]
        values = [x[1] for x in sorted_items]
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))[::-1]
        ax.barh(range(len(features)), values, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('SHAP Importance')
        ax.set_title('Feature Importance (RF)')
        ax.invert_yaxis()

    # 3. AL Learning curves
    ax = axes[1, 0]
    al_data = al_results['results']
    for acq_fn in ['random', 'uncertainty', 'hybrid']:
        if acq_fn in al_data:
            data = al_data[acq_fn]
            color = COLORS.get(acq_fn, '#7f8c8d')
            linestyle = '--' if acq_fn == 'random' else '-'
            ax.plot(data['n_samples'], data['mae_mean'], linestyle,
                   color=color, label=acq_fn.title(), linewidth=2)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('MAE (eV)')
    ax.set_title('Active Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. AL Improvement
    ax = axes[1, 1]
    improvements = al_results.get('improvements_over_random', {})
    if improvements:
        sorted_imps = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
        acq_fns = [x[0] for x in sorted_imps]
        imps = [x[1] for x in sorted_imps]
        colors = [COLORS.get(a, '#7f8c8d') for a in acq_fns]
        bars = ax.barh(range(len(acq_fns)), imps, color=colors, alpha=0.8)
        ax.set_yticks(range(len(acq_fns)))
        ax.set_yticklabels([a.replace('_', ' ').title() for a in acq_fns])
        ax.set_xlabel('Improvement over Random (%)')
        ax.set_title('AL Acquisition Function Comparison')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.invert_yaxis()

    plt.tight_layout()
    save_figure(fig, 'comprehensive_summary')
    plt.close(fig)


def main():
    print("=" * 60)
    print("GENERATING ADVANCED FIGURES")
    print("=" * 60)

    # Load results
    ml_file = RESULTS_DIR / "advanced_ml_results.json"
    al_file = RESULTS_DIR / "advanced_al_results.json"

    print("\nLoading results...")

    ml_results = None
    al_results = None

    if ml_file.exists():
        with open(ml_file, 'r') as f:
            ml_results = json.load(f)
        print(f"  Loaded ML results")
    else:
        print(f"  ML results not found: {ml_file}")

    if al_file.exists():
        with open(al_file, 'r') as f:
            al_results = json.load(f)
        print(f"  Loaded AL results")
    else:
        print(f"  AL results not found: {al_file}")

    print("\nGenerating figures...")

    if ml_results:
        plot_model_comparison(ml_results)

    if al_results:
        plot_al_acquisition_comparison(al_results)
        plot_al_improvement_heatmap(al_results)

    if ml_results and al_results:
        plot_comprehensive_summary(ml_results, al_results)

    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
