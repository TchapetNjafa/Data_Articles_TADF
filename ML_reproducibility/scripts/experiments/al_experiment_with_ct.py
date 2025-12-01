#!/usr/bin/env python3
"""
Active Learning Experiment with CT Descriptors for 747 TADF Molecules.

Compares active learning (uncertainty sampling) vs random sampling
using the enhanced CT descriptor feature set.
"""

import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR.parent / "data_processing" / "combined_features_747mol_full_ct.csv"
RESULTS_DIR = SCRIPT_DIR.parent / "figures_747mol"

# Feature categories
FEATURE_COLS = [
    # Energy features
    'S1_energy_eV', 'T1_energy_eV', 'HOMO_LUMO_gap_eV',
    # Oscillator
    'S1_osc_strength',
    # NTO overlap
    'S1_overlap', 'T1_overlap',
    # CT descriptors
    'S1_CT_number', 'T1_CT_number',
    'S1_Lambda_D', 'T1_Lambda_D',
    'S1_Lambda_A', 'T1_Lambda_A',
    'S1_Delta_r', 'T1_Delta_r',
    'S1_S_he', 'T1_S_he',
    'Delta_CT_number', 'Delta_Lambda_D', 'Delta_Lambda_A',
    'Delta_Delta_r', 'Delta_S_he',
]


def save_figure(fig, output_dir, basename):
    """Save figure in PDF, PNG, and PGF formats."""
    for ext in ['pdf', 'png', 'pgf']:
        filepath = output_dir / f"{basename}.{ext}"
        dpi = 300 if ext == 'png' else None
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        except Exception as e:
            print(f"  Warning: Could not save {ext}: {e}")


def load_data():
    """Load and prepare data."""
    data = []
    with open(DATA_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def prepare_features(data):
    """Prepare feature matrix X and target vector y."""
    X, y = [], []
    for row in data:
        try:
            features = []
            for col in FEATURE_COLS:
                val = row.get(col, '')
                if val == '' or val == 'nan':
                    features.append(np.nan)
                else:
                    features.append(float(val))

            target = float(row['Delta_E_ST_eV'])

            # Skip if any feature is NaN
            if any(np.isnan(f) for f in features):
                continue

            X.append(features)
            y.append(target)
        except (ValueError, KeyError):
            continue

    return np.array(X), np.array(y)


def train_rf(X_train, y_train):
    """Train Random Forest model."""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, max_depth=12,
                                  min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def uncertainty_sampling(model, X_pool, n_samples=1):
    """Select samples with highest prediction uncertainty."""
    predictions = np.array([tree.predict(X_pool) for tree in model.estimators_])
    uncertainties = np.std(predictions, axis=0)
    indices = np.argsort(uncertainties)[-n_samples:]
    return indices


def run_al_experiment(X, y, n_init=50, n_iterations=40, batch_size=10, seed=42):
    """Run single active learning experiment."""
    from sklearn.metrics import mean_absolute_error, r2_score

    np.random.seed(seed)

    # Split into pool and test
    n_total = len(X)
    indices = np.random.permutation(n_total)
    test_size = int(0.2 * n_total)
    test_indices = indices[:test_size]
    pool_indices = indices[test_size:]

    X_test, y_test = X[test_indices], y[test_indices]

    # Initialize with random samples
    init_indices = np.random.choice(len(pool_indices), n_init, replace=False)
    labeled_mask = np.zeros(len(pool_indices), dtype=bool)
    labeled_mask[init_indices] = True

    al_maes, al_r2s = [], []
    random_maes, random_r2s = [], []

    # Active learning loop
    for iteration in range(n_iterations):
        labeled_idx = pool_indices[labeled_mask]
        X_train, y_train = X[labeled_idx], y[labeled_idx]

        model = train_rf(X_train, y_train)

        y_pred = model.predict(X_test)
        al_maes.append(mean_absolute_error(y_test, y_pred))
        al_r2s.append(r2_score(y_test, y_pred))

        unlabeled_mask = ~labeled_mask
        if np.sum(unlabeled_mask) < batch_size:
            break

        unlabeled_pool_idx = np.where(unlabeled_mask)[0]
        X_unlabeled = X[pool_indices[unlabeled_pool_idx]]

        selected = uncertainty_sampling(model, X_unlabeled, batch_size)
        actual_indices = unlabeled_pool_idx[selected]
        labeled_mask[actual_indices] = True

    # Random sampling comparison
    np.random.seed(seed)
    labeled_mask_random = np.zeros(len(pool_indices), dtype=bool)
    labeled_mask_random[init_indices] = True

    for iteration in range(n_iterations):
        labeled_idx = pool_indices[labeled_mask_random]
        X_train, y_train = X[labeled_idx], y[labeled_idx]

        model = train_rf(X_train, y_train)
        y_pred = model.predict(X_test)
        random_maes.append(mean_absolute_error(y_test, y_pred))
        random_r2s.append(r2_score(y_test, y_pred))

        unlabeled_mask = ~labeled_mask_random
        if np.sum(unlabeled_mask) < batch_size:
            break

        unlabeled_pool_idx = np.where(unlabeled_mask)[0]
        selected = np.random.choice(len(unlabeled_pool_idx), batch_size, replace=False)
        actual_indices = unlabeled_pool_idx[selected]
        labeled_mask_random[actual_indices] = True

    return {
        'al_maes': al_maes,
        'al_r2s': al_r2s,
        'random_maes': random_maes,
        'random_r2s': random_r2s,
        'n_samples': [n_init + i * batch_size for i in range(len(al_maes))],
    }


def generate_figures(results, output_dir):
    """Generate AL comparison figures in PDF, PNG, and PGF formats."""
    n_samples = results['n_samples']
    al_maes_mean = np.array(results['active_learning']['mae_mean'])
    al_maes_std = np.array(results['active_learning']['mae_std'])
    random_maes_mean = np.array(results['random_sampling']['mae_mean'])
    random_maes_std = np.array(results['random_sampling']['mae_std'])
    al_r2s_mean = np.array(results['active_learning']['r2_mean'])
    random_r2s_mean = np.array(results['random_sampling']['r2_mean'])

    # Figure 1: Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MAE plot
    ax = axes[0]
    ax.plot(n_samples, al_maes_mean, 'b-', label='Active Learning', linewidth=2)
    ax.fill_between(n_samples, al_maes_mean - al_maes_std, al_maes_mean + al_maes_std,
                    alpha=0.2, color='blue')
    ax.plot(n_samples, random_maes_mean, 'r--', label='Random Sampling', linewidth=2)
    ax.fill_between(n_samples, random_maes_mean - random_maes_std,
                    random_maes_mean + random_maes_std, alpha=0.2, color='red')
    ax.set_xlabel('Training Samples', fontsize=12)
    ax.set_ylabel('MAE (eV)', fontsize=12)
    ax.set_title('Learning Curves - MAE (with CT Descriptors)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R² plot
    ax = axes[1]
    ax.plot(n_samples, al_r2s_mean, 'b-', label='Active Learning', linewidth=2)
    ax.plot(n_samples, random_r2s_mean, 'r--', label='Random Sampling', linewidth=2)
    ax.set_xlabel('Training Samples', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('Learning Curves - R² (with CT Descriptors)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, 'al_learning_curves_ct')
    plt.close()

    # Figure 2: Summary comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Final MAE comparison
    ax = axes[0]
    methods = ['Active\nLearning', 'Random\nSampling']
    final_maes = [results['active_learning']['final_mae'],
                  results['random_sampling']['final_mae']]
    colors = ['#1f77b4', '#d62728']
    bars = ax.bar(methods, final_maes, color=colors, width=0.6)
    ax.set_ylabel('Final MAE (eV)', fontsize=12)
    ax.set_title(f'Final MAE at {n_samples[-1]} Samples')
    for bar, mae in zip(bars, final_maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=11)

    # Final R² comparison
    ax = axes[1]
    final_r2s = [results['active_learning']['final_r2'],
                 results['random_sampling']['final_r2']]
    bars = ax.bar(methods, final_r2s, color=colors, width=0.6)
    ax.set_ylabel('Final R²', fontsize=12)
    ax.set_title(f'Final R² at {n_samples[-1]} Samples')
    for bar, r2 in zip(bars, final_r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    save_figure(fig, output_dir, 'al_summary_ct')
    plt.close()

    print(f"Figures saved to {output_dir}")


def main():
    print("=" * 70)
    print("ACTIVE LEARNING WITH CT DESCRIPTORS - 747 TADF MOLECULES")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")

    # Load data
    print("\nLoading data...")
    data = load_data()
    X, y = prepare_features(data)
    print(f"  Dataset: {len(X)} samples with {X.shape[1]} CT features")

    # Run experiments with multiple seeds
    n_seeds = 10
    n_init = 100
    n_iterations = 40
    batch_size = 20

    print(f"\nRunning {n_seeds} experiments...")
    print(f"  Initial samples: {n_init}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Batch size: {batch_size}")

    all_results = []
    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}...")
        result = run_al_experiment(X, y, n_init=n_init, n_iterations=n_iterations,
                                   batch_size=batch_size, seed=seed)
        all_results.append(result)

    # Aggregate results
    n_samples = all_results[0]['n_samples']
    al_maes_mean = np.mean([r['al_maes'] for r in all_results], axis=0)
    al_maes_std = np.std([r['al_maes'] for r in all_results], axis=0)
    random_maes_mean = np.mean([r['random_maes'] for r in all_results], axis=0)
    random_maes_std = np.std([r['random_maes'] for r in all_results], axis=0)

    al_r2s_mean = np.mean([r['al_r2s'] for r in all_results], axis=0)
    al_r2s_std = np.std([r['al_r2s'] for r in all_results], axis=0)
    random_r2s_mean = np.mean([r['random_r2s'] for r in all_results], axis=0)
    random_r2s_std = np.std([r['random_r2s'] for r in all_results], axis=0)

    # Calculate improvement
    final_improvement = (random_maes_mean[-1] - al_maes_mean[-1]) / random_maes_mean[-1] * 100

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_seeds': n_seeds,
            'n_init': n_init,
            'n_iterations': n_iterations,
            'batch_size': batch_size,
            'n_features': X.shape[1],
            'n_samples': len(X),
        },
        'n_samples': n_samples,
        'active_learning': {
            'mae_mean': al_maes_mean.tolist(),
            'mae_std': al_maes_std.tolist(),
            'r2_mean': al_r2s_mean.tolist(),
            'r2_std': al_r2s_std.tolist(),
            'final_mae': float(al_maes_mean[-1]),
            'final_r2': float(al_r2s_mean[-1]),
        },
        'random_sampling': {
            'mae_mean': random_maes_mean.tolist(),
            'mae_std': random_maes_std.tolist(),
            'r2_mean': random_r2s_mean.tolist(),
            'r2_std': random_r2s_std.tolist(),
            'final_mae': float(random_maes_mean[-1]),
            'final_r2': float(random_r2s_mean[-1]),
        },
        'comparison': {
            'final_improvement_percent': float(final_improvement),
        }
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "al_results_ct.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(results, RESULTS_DIR)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nFinal MAE (at {n_samples[-1]} samples):")
    print(f"  Active Learning: {al_maes_mean[-1]:.4f} ± {al_maes_std[-1]:.4f} eV")
    print(f"  Random Sampling: {random_maes_mean[-1]:.4f} ± {random_maes_std[-1]:.4f} eV")
    print(f"  Improvement: {final_improvement:.1f}%")

    print(f"\nFinal R²:")
    print(f"  Active Learning: {al_r2s_mean[-1]:.3f} ± {al_r2s_std[-1]:.3f}")
    print(f"  Random Sampling: {random_r2s_mean[-1]:.3f} ± {random_r2s_std[-1]:.3f}")

    print(f"\nResults saved to: {results_file}")
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
