#!/usr/bin/env python3
"""
Generate publication-quality figures for 747 TADF molecules ML study.
Outputs figures in PNG, PDF, and PGF (TikZ) formats.
"""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results_747mol"
FIGURES_DIR = SCRIPT_DIR.parent / "figures_747mol"
DATA_FILE = SCRIPT_DIR.parent / "data_processing" / "combined_features_747mol.csv"

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


def load_results():
    with open(RESULTS_DIR / "ml_results_747mol.json", 'r') as f:
        return json.load(f)


def load_predictions():
    preds = {'y_true': [], 'rf_pred': [], 'gpr_pred': []}
    with open(RESULTS_DIR / "predictions_747mol.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds['y_true'].append(float(row['y_true']))
            preds['rf_pred'].append(float(row['rf_pred']))
            preds['gpr_pred'].append(float(row['gpr_pred']))
    return preds


def load_data():
    data = []
    with open(DATA_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def save_figure(fig, name):
    """Save figure in PNG, PDF, and PGF formats."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Save PNG and PDF with Agg backend
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches='tight')

    # Save PGF with pgf backend
    try:
        import matplotlib.backends.backend_pgf
        fig.savefig(FIGURES_DIR / f"{name}.pgf", backend='pgf')
    except Exception as e:
        print(f"    PGF warning: {e}")

    print(f"  Saved: {name}")


def plot_parity(preds, results):
    """Plot predicted vs actual Delta_E_ST."""
    fig, ax = plt.subplots(figsize=(5, 5))

    y_true = np.array(preds['y_true'])
    y_pred = np.array(preds['rf_pred'])

    ax.scatter(y_true, y_pred, alpha=0.5, s=20, c='steelblue', edgecolors='none')

    lims = [min(y_true.min(), y_pred.min()) - 0.05,
            max(y_true.max(), y_pred.max()) + 0.05]
    ax.plot(lims, lims, 'k--', lw=1.5, label='Perfect prediction')

    rf_metrics = results['random_forest']['metrics']
    ax.text(0.05, 0.95, f"R$^2$ = {rf_metrics['r2']:.3f}\nMAE = {rf_metrics['mae']:.3f} eV",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(r'Actual $\Delta E_{ST}$ (eV)')
    ax.set_ylabel(r'Predicted $\Delta E_{ST}$ (eV)')
    ax.set_title('Random Forest: Predicted vs Actual')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    plt.tight_layout()
    save_figure(fig, 'parity_plot_747mol')
    plt.close(fig)


def plot_feature_importance(results):
    """Plot SHAP feature importance."""
    fig, ax = plt.subplots(figsize=(7, 4))

    importance = results['random_forest']['feature_importance']
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    display_names = {
        'S1_energy_eV': r'$S_1$ Energy',
        'T1_energy_eV': r'$T_1$ Energy',
        'S1_overlap': r'$S_1$ NTO Overlap',
        'T1_overlap': r'$T_1$ NTO Overlap',
        'S1_osc_strength': 'Oscillator Strength',
        'HOMO_LUMO_gap_eV': 'HOMO-LUMO Gap',
        'environment': 'Environment',
        'method': 'Method',
    }
    features_display = [display_names.get(f, f) for f in features]

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))[::-1]

    ax.barh(range(len(features)), values, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features_display)
    ax.set_xlabel('SHAP Feature Importance')
    ax.set_title(r'Feature Importance for $\Delta E_{ST}$ Prediction')
    ax.invert_yaxis()

    plt.tight_layout()
    save_figure(fig, 'feature_importance_747mol')
    plt.close(fig)


def plot_feature_group_importance(results):
    """Plot grouped feature importance as pie chart."""
    fig, ax = plt.subplots(figsize=(6, 6))

    importance = results['random_forest']['feature_importance']

    groups = {
        'Energy\nFeatures': importance['S1_energy_eV'] + importance['T1_energy_eV'] + importance['HOMO_LUMO_gap_eV'],
        'Oscillator\nStrength': importance['S1_osc_strength'],
        'NTO\nOverlaps': importance['S1_overlap'] + importance['T1_overlap'],
        'Environment\n& Method': importance['environment'] + importance['method'],
    }

    labels = list(groups.keys())
    sizes = list(groups.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.75)
    ax.set_title(r'Feature Group Importance for $\Delta E_{ST}$ Prediction')

    plt.tight_layout()
    save_figure(fig, 'feature_group_importance_747mol')
    plt.close(fig)


def plot_delta_est_distribution(data):
    """Plot distribution of Delta_E_ST values."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    delta_est = [float(row['Delta_E_ST_eV']) for row in data if row['Delta_E_ST_eV']]
    gas_delta = [float(row['Delta_E_ST_eV']) for row in data
                 if row['Delta_E_ST_eV'] and row['environment'] == 'gas']
    toluene_delta = [float(row['Delta_E_ST_eV']) for row in data
                     if row['Delta_E_ST_eV'] and row['environment'] == 'toluene']

    ax = axes[0]
    ax.hist(delta_est, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(delta_est), color='red', linestyle='--', label=f'Mean: {np.mean(delta_est):.3f} eV')
    ax.set_xlabel(r'$\Delta E_{ST}$ (eV)')
    ax.set_ylabel('Count')
    ax.set_title(r'Distribution of $\Delta E_{ST}$ (747 Molecules)')
    ax.legend()

    ax = axes[1]
    ax.hist(gas_delta, bins=40, alpha=0.7, label='Gas', color='#3498db')
    ax.hist(toluene_delta, bins=40, alpha=0.7, label='Toluene', color='#e74c3c')
    ax.set_xlabel(r'$\Delta E_{ST}$ (eV)')
    ax.set_ylabel('Count')
    ax.set_title(r'$\Delta E_{ST}$ Distribution by Environment')
    ax.legend()

    plt.tight_layout()
    save_figure(fig, 'delta_est_distribution_747mol')
    plt.close(fig)


def plot_residuals(preds):
    """Plot residual analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    y_true = np.array(preds['y_true'])
    y_pred = np.array(preds['rf_pred'])
    residuals = y_true - y_pred

    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    ax.axhline(0, color='k', linestyle='--', lw=1)
    ax.set_xlabel(r'Predicted $\Delta E_{ST}$ (eV)')
    ax.set_ylabel('Residual (eV)')
    ax.set_title('Residuals vs Predicted')

    ax = axes[1]
    ax.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='k', linestyle='--', lw=1)
    ax.set_xlabel('Residual (eV)')
    ax.set_ylabel('Count')
    ax.set_title(f'Residual Distribution (std = {np.std(residuals):.3f} eV)')

    plt.tight_layout()
    save_figure(fig, 'residual_analysis_747mol')
    plt.close(fig)


def plot_model_comparison(results):
    """Plot model comparison bar chart."""
    fig, ax = plt.subplots(figsize=(6, 4))

    models = ['Random Forest', 'Gaussian Process*']
    rf = results['random_forest']['metrics']
    gpr = results['gaussian_process']['metrics']

    x = np.arange(len(models))
    width = 0.35

    r2_vals = [rf['r2'], min(gpr['r2'], 1.0)]
    mae_vals = [rf['mae'] * 10, gpr['mae'] * 10]

    bars1 = ax.bar(x - width/2, r2_vals, width, label=r'R$^2$', color='#3498db')
    bars2 = ax.bar(x + width/2, mae_vals, width, label='MAE x 10 (eV)', color='#e74c3c')

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)

    for bar, val in zip(bars1, [rf['r2'], gpr['r2']]):
        ax.text(bar.get_x() + bar.get_width()/2, min(bar.get_height(), 1.05) + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, [rf['mae'], gpr['mae']]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.text(0.5, -0.15, '*GPR results may show overfitting due to subsampling',
            transform=ax.transAxes, fontsize=8, ha='center', style='italic')

    plt.tight_layout()
    save_figure(fig, 'model_comparison_747mol')
    plt.close(fig)


def main():
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES FOR 747 MOLECULES")
    print("=" * 60)

    print("\nLoading data...")
    results = load_results()
    preds = load_predictions()
    data = load_data()

    print("\nGenerating figures (PNG, PDF, PGF)...")
    plot_parity(preds, results)
    plot_feature_importance(results)
    plot_feature_group_importance(results)
    plot_delta_est_distribution(data)
    plot_residuals(preds)
    plot_model_comparison(results)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
