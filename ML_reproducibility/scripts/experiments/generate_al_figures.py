#!/usr/bin/env python3
"""
Generate Active Learning figures for 747 TADF molecules study.
Creates publication-quality figures in PNG, PDF, and PGF formats.
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

# Style settings
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


def load_al_results():
    with open(RESULTS_DIR / "al_results_747mol.json", 'r') as f:
        return json.load(f)


def save_figure(fig, name):
    """Save figure in PNG, PDF, and PGF formats."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches='tight')

    try:
        import matplotlib.backends.backend_pgf
        fig.savefig(FIGURES_DIR / f"{name}.pgf", backend='pgf')
    except Exception as e:
        print(f"    PGF warning: {e}")

    print(f"  Saved: {name}")


def plot_learning_curves(results):
    """Plot AL vs Random learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n_samples = np.array(results['n_samples'])
    al_mae = np.array(results['active_learning']['mae_mean'])
    al_mae_std = np.array(results['active_learning']['mae_std'])
    random_mae = np.array(results['random_sampling']['mae_mean'])
    random_mae_std = np.array(results['random_sampling']['mae_std'])

    al_r2 = np.array(results['active_learning']['r2_mean'])
    random_r2 = np.array(results['random_sampling']['r2_mean'])

    # MAE plot
    ax = axes[0]
    ax.plot(n_samples, al_mae, 'b-', linewidth=2, label='Active Learning')
    ax.fill_between(n_samples, al_mae - al_mae_std, al_mae + al_mae_std, alpha=0.2, color='blue')
    ax.plot(n_samples, random_mae, 'r--', linewidth=2, label='Random Sampling')
    ax.fill_between(n_samples, random_mae - random_mae_std, random_mae + random_mae_std, alpha=0.2, color='red')

    ax.set_xlabel('Training Samples')
    ax.set_ylabel('MAE (eV)')
    ax.set_title(r'Learning Curves: MAE for $\Delta E_{ST}$ Prediction')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # R2 plot
    ax = axes[1]
    ax.plot(n_samples, al_r2, 'b-', linewidth=2, label='Active Learning')
    ax.plot(n_samples, random_r2, 'r--', linewidth=2, label='Random Sampling')

    ax.set_xlabel('Training Samples')
    ax.set_ylabel(r'R$^2$ Score')
    ax.set_title(r'Learning Curves: R$^2$ for $\Delta E_{ST}$ Prediction')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 0.95)

    plt.tight_layout()
    save_figure(fig, 'al_learning_curves_747mol')
    plt.close(fig)


def plot_al_improvement(results):
    """Plot improvement of AL over random sampling."""
    fig, ax = plt.subplots(figsize=(6, 4))

    n_samples = np.array(results['n_samples'])
    al_mae = np.array(results['active_learning']['mae_mean'])
    random_mae = np.array(results['random_sampling']['mae_mean'])

    # Calculate improvement percentage
    improvement = (random_mae - al_mae) / random_mae * 100

    ax.plot(n_samples, improvement, 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.fill_between(n_samples, 0, improvement, where=improvement > 0, alpha=0.3, color='green')
    ax.fill_between(n_samples, 0, improvement, where=improvement < 0, alpha=0.3, color='red')

    ax.set_xlabel('Training Samples')
    ax.set_ylabel('MAE Improvement (%)')
    ax.set_title('Active Learning Improvement Over Random Sampling')
    ax.grid(True, alpha=0.3)

    # Add final improvement annotation
    final_imp = results['comparison']['final_improvement_percent']
    ax.text(0.95, 0.95, f'Final: {final_imp:.1f}%', transform=ax.transAxes,
            fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_figure(fig, 'al_improvement_747mol')
    plt.close(fig)


def plot_al_summary(results):
    """Plot summary bar chart comparing final metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    methods = ['Active\nLearning', 'Random\nSampling']
    colors = ['#3498db', '#e74c3c']

    # MAE comparison
    ax = axes[0]
    mae_vals = [results['active_learning']['final_mae'],
                results['random_sampling']['final_mae']]
    bars = ax.bar(methods, mae_vals, color=colors)
    ax.set_ylabel('MAE (eV)')
    ax.set_title(r'Final MAE at 880 Samples')

    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # R2 comparison
    ax = axes[1]
    r2_vals = [results['active_learning']['final_r2'],
               results['random_sampling']['final_r2']]
    bars = ax.bar(methods, r2_vals, color=colors)
    ax.set_ylabel(r'R$^2$ Score')
    ax.set_title(r'Final R$^2$ at 880 Samples')
    ax.set_ylim(0.7, 0.95)

    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_figure(fig, 'al_summary_747mol')
    plt.close(fig)


def main():
    print("=" * 60)
    print("GENERATING ACTIVE LEARNING FIGURES")
    print("=" * 60)

    print("\nLoading AL results...")
    results = load_al_results()

    print(f"\nAL Summary:")
    print(f"  Final AL MAE:     {results['active_learning']['final_mae']:.4f} eV")
    print(f"  Final Random MAE: {results['random_sampling']['final_mae']:.4f} eV")
    print(f"  Improvement:      {results['comparison']['final_improvement_percent']:.1f}%")

    print("\nGenerating figures (PNG, PDF, PGF)...")
    plot_learning_curves(results)
    plot_al_improvement(results)
    plot_al_summary(results)

    print(f"\nAll AL figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
