#!/usr/bin/env python3
"""
Advanced Active Learning Experiment for 747 TADF Molecules.

Implements multiple acquisition functions:
1. Uncertainty Sampling (US) - variance of predictions
2. Expected Improvement (EI) - balance exploration/exploitation
3. Upper Confidence Bound (UCB) - optimistic acquisition
4. Query by Committee (QBC) - disagreement among ensemble
5. Diversity Sampling - maximize feature space coverage
6. Hybrid (Uncertainty × Diversity)

Compares against random sampling baseline.
"""

import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

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


class ActiveLearner:
    """Active learning with multiple acquisition functions."""

    def __init__(self, model_type='rf', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def _create_model(self):
        if self.model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=50, max_depth=10,
                random_state=self.random_state, n_jobs=-1
            )
        elif self.model_type == 'gpr':
            kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(0.1)
            return GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=3,
                random_state=self.random_state, normalize_y=True
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X, y):
        """Fit the model."""
        self.model = self._create_model()
        if self.model_type == 'gpr':
            X_scaled = self.scaler.fit_transform(X)
            # Subsample for GPR
            if len(X_scaled) > 500:
                idx = np.random.choice(len(X_scaled), 500, replace=False)
                self.model.fit(X_scaled[idx], y[idx])
            else:
                self.model.fit(X_scaled, y)
        else:
            self.model.fit(X, y)

    def predict(self, X, return_std=False):
        """Predict with optional uncertainty."""
        if self.model_type == 'gpr':
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled, return_std=return_std)
        elif self.model_type == 'rf':
            predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            mean_pred = predictions.mean(axis=0)
            if return_std:
                std_pred = predictions.std(axis=0)
                return mean_pred, std_pred
            return mean_pred

    def uncertainty_sampling(self, X_pool, n_samples=1):
        """Select samples with highest prediction uncertainty."""
        _, uncertainties = self.predict(X_pool, return_std=True)
        indices = np.argsort(uncertainties)[-n_samples:]
        return indices

    def expected_improvement(self, X_pool, y_best, n_samples=1, xi=0.01):
        """Expected Improvement acquisition function."""
        mu, sigma = self.predict(X_pool, return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        # For minimization (lower ΔE_ST is better for TADF)
        imp = y_best - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        indices = np.argsort(ei)[-n_samples:]
        return indices

    def upper_confidence_bound(self, X_pool, n_samples=1, beta=2.0):
        """Upper Confidence Bound (for exploration)."""
        mu, sigma = self.predict(X_pool, return_std=True)
        # UCB for minimization: select lowest (mu - beta*sigma)
        ucb = mu - beta * sigma
        indices = np.argsort(ucb)[:n_samples]
        return indices

    def query_by_committee(self, X_pool, n_samples=1, n_committee=5):
        """Query by Committee - disagreement among ensemble."""
        if self.model_type != 'rf':
            return self.uncertainty_sampling(X_pool, n_samples)

        # Use subset of trees as committee
        n_trees = len(self.model.estimators_)
        committee_idx = np.random.choice(n_trees, min(n_committee, n_trees), replace=False)

        predictions = np.array([
            self.model.estimators_[i].predict(X_pool) for i in committee_idx
        ])
        disagreement = predictions.std(axis=0)
        indices = np.argsort(disagreement)[-n_samples:]
        return indices

    def diversity_sampling(self, X_pool, X_labeled, n_samples=1):
        """Select samples maximizing diversity (distance from labeled)."""
        if len(X_labeled) == 0:
            return np.random.choice(len(X_pool), n_samples, replace=False)

        # Compute minimum distance to any labeled point
        distances = cdist(X_pool, X_labeled, metric='euclidean')
        min_distances = distances.min(axis=1)

        indices = np.argsort(min_distances)[-n_samples:]
        return indices

    def hybrid_acquisition(self, X_pool, X_labeled, n_samples=1, alpha=0.5):
        """Hybrid: uncertainty × diversity."""
        _, uncertainties = self.predict(X_pool, return_std=True)

        if len(X_labeled) == 0:
            diversity_scores = np.ones(len(X_pool))
        else:
            distances = cdist(X_pool, X_labeled, metric='euclidean')
            diversity_scores = distances.min(axis=1)

        # Normalize
        uncertainties = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-9)
        diversity_scores = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min() + 1e-9)

        combined = alpha * uncertainties + (1 - alpha) * diversity_scores
        indices = np.argsort(combined)[-n_samples:]
        return indices


def run_al_experiment(X, y, acquisition_fn, n_init=50, n_iterations=40, batch_size=10, seed=42):
    """Run single AL experiment with specified acquisition function."""
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

    maes, r2s = [], []
    learner = ActiveLearner(model_type='rf', random_state=seed)

    for iteration in range(n_iterations):
        # Get current labeled data
        labeled_idx = pool_indices[labeled_mask]
        X_train, y_train = X[labeled_idx], y[labeled_idx]

        # Train model
        learner.fit(X_train, y_train)

        # Evaluate
        y_pred = learner.predict(X_test)
        maes.append(mean_absolute_error(y_test, y_pred))
        r2s.append(r2_score(y_test, y_pred))

        # Select next batch
        unlabeled_mask = ~labeled_mask
        if np.sum(unlabeled_mask) < batch_size:
            break

        unlabeled_pool_idx = np.where(unlabeled_mask)[0]
        X_unlabeled = X[pool_indices[unlabeled_pool_idx]]
        X_labeled = X[pool_indices[labeled_mask]]

        # Apply acquisition function
        if acquisition_fn == 'uncertainty':
            selected = learner.uncertainty_sampling(X_unlabeled, batch_size)
        elif acquisition_fn == 'expected_improvement':
            y_best = y_train.min()
            selected = learner.expected_improvement(X_unlabeled, y_best, batch_size)
        elif acquisition_fn == 'ucb':
            selected = learner.upper_confidence_bound(X_unlabeled, batch_size)
        elif acquisition_fn == 'qbc':
            selected = learner.query_by_committee(X_unlabeled, batch_size)
        elif acquisition_fn == 'diversity':
            selected = learner.diversity_sampling(X_unlabeled, X_labeled, batch_size)
        elif acquisition_fn == 'hybrid':
            selected = learner.hybrid_acquisition(X_unlabeled, X_labeled, batch_size)
        elif acquisition_fn == 'random':
            selected = np.random.choice(len(unlabeled_pool_idx), batch_size, replace=False)
        else:
            raise ValueError(f"Unknown acquisition: {acquisition_fn}")

        actual_indices = unlabeled_pool_idx[selected]
        labeled_mask[actual_indices] = True

    return {
        'maes': maes,
        'r2s': r2s,
        'n_samples': [n_init + i * batch_size for i in range(len(maes))],
    }


def main():
    print("=" * 70)
    print("ADVANCED ACTIVE LEARNING EXPERIMENT")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")

    # Load data
    print("\nLoading data...")
    data = load_data()
    X, y = prepare_features(data)
    print(f"  Dataset: {len(X)} samples")

    # Experiment parameters
    n_seeds = 10
    n_init = 50
    n_iterations = 40
    batch_size = 10

    acquisition_functions = [
        'random',
        'uncertainty',
        'expected_improvement',
        'ucb',
        'qbc',
        'diversity',
        'hybrid',
    ]

    print(f"\nRunning experiments...")
    print(f"  Seeds: {n_seeds}")
    print(f"  Initial samples: {n_init}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Batch size: {batch_size}")
    print(f"  Acquisition functions: {len(acquisition_functions)}")

    all_results = {}

    for acq_fn in acquisition_functions:
        print(f"\n  {acq_fn.upper()}:")
        results_list = []
        for seed in range(n_seeds):
            result = run_al_experiment(
                X, y, acq_fn, n_init=n_init,
                n_iterations=n_iterations, batch_size=batch_size, seed=seed
            )
            results_list.append(result)
            print(f"    Seed {seed + 1}/{n_seeds} done")

        # Aggregate
        n_samples = results_list[0]['n_samples']
        maes_mean = np.mean([r['maes'] for r in results_list], axis=0)
        maes_std = np.std([r['maes'] for r in results_list], axis=0)
        r2s_mean = np.mean([r['r2s'] for r in results_list], axis=0)
        r2s_std = np.std([r['r2s'] for r in results_list], axis=0)

        all_results[acq_fn] = {
            'n_samples': n_samples,
            'mae_mean': maes_mean.tolist(),
            'mae_std': maes_std.tolist(),
            'r2_mean': r2s_mean.tolist(),
            'r2_std': r2s_std.tolist(),
            'final_mae': float(maes_mean[-1]),
            'final_mae_std': float(maes_std[-1]),
            'final_r2': float(r2s_mean[-1]),
            'final_r2_std': float(r2s_std[-1]),
        }

        print(f"    Final MAE: {maes_mean[-1]:.4f} ± {maes_std[-1]:.4f} eV")
        print(f"    Final R²:  {r2s_mean[-1]:.4f} ± {r2s_std[-1]:.4f}")

    # Compute improvements over random
    random_mae = all_results['random']['final_mae']
    improvements = {}
    for acq_fn in acquisition_functions:
        if acq_fn != 'random':
            acq_mae = all_results[acq_fn]['final_mae']
            improvement = (random_mae - acq_mae) / random_mae * 100
            improvements[acq_fn] = improvement

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_seeds': n_seeds,
            'n_init': n_init,
            'n_iterations': n_iterations,
            'batch_size': batch_size,
        },
        'results': all_results,
        'improvements_over_random': improvements,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "advanced_al_results.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - IMPROVEMENT OVER RANDOM SAMPLING")
    print("=" * 70)
    print(f"\nRandom baseline MAE: {random_mae:.4f} eV")
    print("\nImprovement by acquisition function:")
    for acq_fn, imp in sorted(improvements.items(), key=lambda x: -x[1]):
        print(f"  {acq_fn:25s}: {imp:+.1f}%")

    print(f"\nBest acquisition: {max(improvements, key=improvements.get)}")
    print(f"\nResults saved to: {results_file}")
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
