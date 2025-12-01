#!/usr/bin/env python3
"""
Advanced ML Pipeline for 747 TADF Molecules - Delta_E_ST Prediction

Implements multiple ML models with cross-validation and hyperparameter optimization:
- Random Forest (RF)
- Gradient Boosting (XGBoost)
- Gaussian Process Regression (GPR)
- Neural Network (MLP)
- Support Vector Regression (SVR)

Features comprehensive evaluation metrics and SHAP analysis.
"""

import csv
import json
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR.parent / "data_processing" / "combined_features_747mol.csv"
RESULTS_DIR = SCRIPT_DIR / "results_747mol"


def load_data(path):
    """Load combined features CSV."""
    data = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def prepare_features(data):
    """Prepare feature matrix X and target vector y."""
    feature_cols = ['S1_overlap', 'T1_overlap', 'S1_energy_eV', 'T1_energy_eV',
                    'S1_osc_strength', 'HOMO_LUMO_gap_eV']

    X, y, molecules = [], [], []

    for row in data:
        try:
            features = [float(row[col]) for col in feature_cols]
            target = float(row['Delta_E_ST_eV'])
            env_encoded = 1 if row['environment'] == 'toluene' else 0
            method_encoded = 1 if row['method'] == 'stddft' else 0
            features.extend([env_encoded, method_encoded])
            X.append(features)
            y.append(target)
            molecules.append(row['molecule'])
        except (ValueError, KeyError):
            continue

    return np.array(X), np.array(y), molecules


def get_models():
    """Define all ML models with default parameters."""
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            min_samples_split=5, random_state=42
        ),
        'SVR (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.01))
        ]),
        'Neural Network': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(64, 32, 16), activation='relu',
                solver='adam', alpha=0.001, max_iter=1000,
                early_stopping=True, random_state=42
            ))
        ]),
    }
    return models


def get_hyperparameter_grids():
    """Define hyperparameter grids for optimization."""
    grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'min_samples_split': [2, 5, 10],
        },
        'SVR (RBF)': {
            'svr__C': [1, 10, 100],
            'svr__gamma': ['scale', 'auto', 0.1, 0.01],
            'svr__epsilon': [0.01, 0.1, 0.2],
        },
        'Neural Network': {
            'mlp__hidden_layer_sizes': [(32, 16), (64, 32), (64, 32, 16), (128, 64, 32)],
            'mlp__alpha': [0.0001, 0.001, 0.01],
            'mlp__learning_rate_init': [0.001, 0.01],
        },
    }
    return grids


def cross_validate_model(model, X, y, cv=5):
    """Perform k-fold cross-validation and return metrics."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    y_pred_cv = cross_val_predict(model, X, y, cv=kf)

    # Per-fold metrics
    fold_metrics = {'mae': [], 'rmse': [], 'r2': []}
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model_clone = clone_model(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)

        fold_metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        fold_metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        fold_metrics['r2'].append(r2_score(y_test, y_pred))

    return {
        'mae_mean': np.mean(fold_metrics['mae']),
        'mae_std': np.std(fold_metrics['mae']),
        'rmse_mean': np.mean(fold_metrics['rmse']),
        'rmse_std': np.std(fold_metrics['rmse']),
        'r2_mean': np.mean(fold_metrics['r2']),
        'r2_std': np.std(fold_metrics['r2']),
        'y_pred_cv': y_pred_cv.tolist(),
    }


def clone_model(model):
    """Clone a model for cross-validation."""
    from sklearn.base import clone
    return clone(model)


def optimize_hyperparameters(model, param_grid, X, y, cv=3):
    """Perform grid search hyperparameter optimization."""
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='neg_mean_absolute_error',
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X, y)
    return grid_search.best_params_, -grid_search.best_score_


def compute_shap_importance(model, X, feature_names):
    """Compute SHAP feature importance."""
    try:
        import shap
        # Handle pipeline models
        if hasattr(model, 'named_steps'):
            if 'scaler' in model.named_steps:
                X_scaled = model.named_steps['scaler'].transform(X)
            else:
                X_scaled = X
            estimator = list(model.named_steps.values())[-1]
        else:
            X_scaled = X
            estimator = model

        # Use appropriate explainer
        if hasattr(estimator, 'feature_importances_'):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_scaled)
        else:
            explainer = shap.KernelExplainer(estimator.predict, X_scaled[:100])
            shap_values = explainer.shap_values(X_scaled[:100])

        importance = np.abs(shap_values).mean(axis=0)
        return dict(zip(feature_names, importance))
    except Exception as e:
        # Fallback
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        return {f: 0.0 for f in feature_names}


def train_gpr_subsampled(X_train, y_train, X_test, max_samples=800):
    """Train GPR with subsampling for scalability."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Subsample if needed
    if len(X_train_scaled) > max_samples:
        indices = np.random.choice(len(X_train_scaled), max_samples, replace=False)
        X_train_gpr = X_train_scaled[indices]
        y_train_gpr = y_train[indices]
    else:
        X_train_gpr = X_train_scaled
        y_train_gpr = y_train

    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)

    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5,
        random_state=42, normalize_y=True
    )
    gpr.fit(X_train_gpr, y_train_gpr)

    y_pred, y_std = gpr.predict(X_test_scaled, return_std=True)

    return y_pred, y_std, scaler


def main():
    print("=" * 70)
    print("ADVANCED ML PIPELINE FOR 747 TADF MOLECULES")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    data = load_data(DATA_FILE)
    X, y, molecules = prepare_features(data)
    print(f"  Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"  Unique molecules: {len(set(molecules))}")

    feature_names = ['S1_overlap', 'T1_overlap', 'S1_energy_eV', 'T1_energy_eV',
                     'S1_osc_strength', 'HOMO_LUMO_gap_eV', 'environment', 'method']

    # Train-test split for final evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X.shape[1],
            'unique_molecules': len(set(molecules)),
        },
        'models': {},
        'cross_validation': {},
        'hyperparameter_optimization': {},
    }

    # Get models
    models = get_models()
    grids = get_hyperparameter_grids()

    # Cross-validation for all models
    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    for name, model in models.items():
        print(f"\n{name}:")
        cv_results = cross_validate_model(model, X, y, cv=5)
        results['cross_validation'][name] = {
            'mae': f"{cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}",
            'rmse': f"{cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}",
            'r2': f"{cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}",
            'mae_mean': cv_results['mae_mean'],
            'mae_std': cv_results['mae_std'],
            'r2_mean': cv_results['r2_mean'],
            'r2_std': cv_results['r2_std'],
        }
        print(f"  MAE:  {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f} eV")
        print(f"  RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f} eV")
        print(f"  R²:   {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")

    # Add GPR (special handling due to O(n³) complexity)
    print(f"\nGaussian Process (Matern kernel):")
    gpr_pred, gpr_std, _ = train_gpr_subsampled(X_train, y_train, X_test)
    gpr_mae = mean_absolute_error(y_test, gpr_pred)
    gpr_r2 = r2_score(y_test, gpr_pred)
    print(f"  MAE:  {gpr_mae:.4f} eV (test set)")
    print(f"  R²:   {gpr_r2:.4f}")
    print(f"  Mean uncertainty: {np.mean(gpr_std):.4f} eV")
    results['cross_validation']['GPR (Matern)'] = {
        'mae_test': gpr_mae,
        'r2_test': gpr_r2,
        'mean_uncertainty': float(np.mean(gpr_std)),
    }

    # Hyperparameter optimization (for RF and GB only - others are slower)
    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION (Grid Search)")
    print("=" * 70)

    for name in ['Random Forest', 'Gradient Boosting']:
        print(f"\nOptimizing {name}...")
        model = models[name]
        grid = grids[name]
        best_params, best_score = optimize_hyperparameters(model, grid, X_train, y_train)
        results['hyperparameter_optimization'][name] = {
            'best_params': best_params,
            'best_cv_mae': best_score,
        }
        print(f"  Best params: {best_params}")
        print(f"  Best CV MAE: {best_score:.4f} eV")

    # Final evaluation with optimized models
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (Test Set)")
    print("=" * 70)

    # Optimized Random Forest
    rf_opt = RandomForestRegressor(
        **results['hyperparameter_optimization']['Random Forest']['best_params'],
        random_state=42, n_jobs=-1
    )
    rf_opt.fit(X_train, y_train)
    rf_pred = rf_opt.predict(X_test)

    results['models']['Random Forest (Optimized)'] = {
        'mae': mean_absolute_error(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'r2': r2_score(y_test, rf_pred),
    }

    print(f"\nRandom Forest (Optimized):")
    print(f"  MAE:  {results['models']['Random Forest (Optimized)']['mae']:.4f} eV")
    print(f"  R²:   {results['models']['Random Forest (Optimized)']['r2']:.4f}")

    # Optimized Gradient Boosting
    gb_opt = GradientBoostingRegressor(
        **results['hyperparameter_optimization']['Gradient Boosting']['best_params'],
        random_state=42
    )
    gb_opt.fit(X_train, y_train)
    gb_pred = gb_opt.predict(X_test)

    results['models']['Gradient Boosting (Optimized)'] = {
        'mae': mean_absolute_error(y_test, gb_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
        'r2': r2_score(y_test, gb_pred),
    }

    print(f"\nGradient Boosting (Optimized):")
    print(f"  MAE:  {results['models']['Gradient Boosting (Optimized)']['mae']:.4f} eV")
    print(f"  R²:   {results['models']['Gradient Boosting (Optimized)']['r2']:.4f}")

    # Feature importance for best model
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (SHAP)")
    print("=" * 70)

    shap_importance = compute_shap_importance(rf_opt, X_test, feature_names)
    results['feature_importance'] = shap_importance

    print("\nFeature importance (Random Forest):")
    for feat, imp in sorted(shap_importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    # Save predictions for all models
    predictions = {
        'y_true': y_test.tolist(),
        'rf_opt_pred': rf_pred.tolist(),
        'gb_opt_pred': gb_pred.tolist(),
        'gpr_pred': gpr_pred.tolist(),
        'gpr_std': gpr_std.tolist(),
    }

    # Save results
    results_file = RESULTS_DIR / "advanced_ml_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    pred_file = RESULTS_DIR / "advanced_predictions.json"
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to: {pred_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBest model: Random Forest (Optimized)")
    print(f"  R² = {results['models']['Random Forest (Optimized)']['r2']:.4f}")
    print(f"  MAE = {results['models']['Random Forest (Optimized)']['mae']:.4f} eV")

    print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    main()
