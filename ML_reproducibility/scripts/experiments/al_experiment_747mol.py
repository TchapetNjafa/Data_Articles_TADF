#!/usr/bin/env python3
"""
Active Learning Experiment for 747 TADF Molecules.

Compares active learning (uncertainty sampling) vs random sampling
for efficient model training with limited labeled data.
"""

import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR.parent / "data_processing" / "combined_features_747mol.csv"
RESULTS_DIR = SCRIPT_DIR / "results_747mol"


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
    feature_cols = ['S1_overlap', 'T1_overlap', 'S1_energy_eV', 'T1_energy_eV',
                    'S1_osc_strength', 'HOMO_LUMO_gap_eV']

    X, y = [], []
    for row in data:
        try:
            features = [float(row[col]) for col in feature_cols]
            target = float(row['Delta_E_ST_eV'])
            env_encoded = 1 if row['environment'] == 'toluene' else 0
            method_encoded = 1 if row['method'] == 'stddft' else 0
            features.extend([env_encoded, method_encoded])
            X.append(features)
            y.append(target)
        except (ValueError, KeyError):
            continue

    return np.array(X), np.array(y)


def train_rf(X_train, y_train):
    """Train Random Forest model."""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def uncertainty_sampling(model, X_pool, n_samples=1):
    """Select samples with highest prediction uncertainty."""
    # Use variance of tree predictions as uncertainty
    predictions = np.array([tree.predict(X_pool) for tree in model.estimators_])
    uncertainties = np.std(predictions, axis=0)
    indices = np.argsort(uncertainties)[-n_samples:]
    return indices


def run_al_experiment(X, y, n_init=50, n_iterations=50, batch_size=10, seed=42):
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
        # Get current labeled data
        labeled_idx = pool_indices[labeled_mask]
        X_train, y_train = X[labeled_idx], y[labeled_idx]

        # Train model
        model = train_rf(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        al_maes.append(mean_absolute_error(y_test, y_pred))
        al_r2s.append(r2_score(y_test, y_pred))

        # Select next batch using uncertainty sampling
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

        # Random selection
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


def main():
    print("=" * 70)
    print("ACTIVE LEARNING EXPERIMENT FOR 747 TADF MOLECULES")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")

    # Load data
    print("\nLoading data...")
    data = load_data()
    X, y = prepare_features(data)
    print(f"  Dataset: {len(X)} samples")

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
    random_r2s_mean = np.mean([r['random_r2s'] for r in all_results], axis=0)

    # Calculate improvement
    final_improvement = (random_maes_mean[-1] - al_maes_mean[-1]) / random_maes_mean[-1] * 100

    # Find samples needed for target MAE
    target_mae = 0.04  # eV
    al_samples_to_target = next((n for n, mae in zip(n_samples, al_maes_mean) if mae <= target_mae), None)
    random_samples_to_target = next((n for n, mae in zip(n_samples, random_maes_mean) if mae <= target_mae), None)

    if al_samples_to_target and random_samples_to_target:
        data_efficiency = (random_samples_to_target - al_samples_to_target) / random_samples_to_target * 100
    else:
        data_efficiency = None

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_seeds': n_seeds,
            'n_init': n_init,
            'n_iterations': n_iterations,
            'batch_size': batch_size,
        },
        'n_samples': n_samples,
        'active_learning': {
            'mae_mean': al_maes_mean.tolist(),
            'mae_std': al_maes_std.tolist(),
            'r2_mean': al_r2s_mean.tolist(),
            'final_mae': float(al_maes_mean[-1]),
            'final_r2': float(al_r2s_mean[-1]),
        },
        'random_sampling': {
            'mae_mean': random_maes_mean.tolist(),
            'mae_std': random_maes_std.tolist(),
            'r2_mean': random_r2s_mean.tolist(),
            'final_mae': float(random_maes_mean[-1]),
            'final_r2': float(random_r2s_mean[-1]),
        },
        'comparison': {
            'final_improvement_percent': float(final_improvement),
            'target_mae': target_mae,
            'al_samples_to_target': al_samples_to_target,
            'random_samples_to_target': random_samples_to_target,
            'data_efficiency_percent': float(data_efficiency) if data_efficiency else None,
        }
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "al_results_747mol.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nFinal MAE (at {n_samples[-1]} samples):")
    print(f"  Active Learning: {al_maes_mean[-1]:.4f} ± {al_maes_std[-1]:.4f} eV")
    print(f"  Random Sampling: {random_maes_mean[-1]:.4f} ± {random_maes_std[-1]:.4f} eV")
    print(f"  Improvement: {final_improvement:.1f}%")

    print(f"\nFinal R²:")
    print(f"  Active Learning: {al_r2s_mean[-1]:.3f}")
    print(f"  Random Sampling: {random_r2s_mean[-1]:.3f}")

    if al_samples_to_target and random_samples_to_target:
        print(f"\nSamples to reach MAE ≤ {target_mae} eV:")
        print(f"  Active Learning: {al_samples_to_target}")
        print(f"  Random Sampling: {random_samples_to_target}")
        print(f"  Data Efficiency: {data_efficiency:.1f}%")

    print(f"\nResults saved to: {results_file}")
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
