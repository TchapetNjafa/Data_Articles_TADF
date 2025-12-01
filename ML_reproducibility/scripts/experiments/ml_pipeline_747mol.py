#!/usr/bin/env python3
"""
ML Pipeline for 747 TADF Molecules - Delta_E_ST Prediction

This script trains and evaluates Random Forest and Gaussian Process models
for predicting singlet-triplet energy gaps (Delta_E_ST) from NTO overlaps
and sTDA features.

Features:
- S1_overlap, T1_overlap: NTO hole-electron overlaps
- S1_energy_eV, T1_energy_eV: Excitation energies
- S1_osc_strength: Oscillator strength
- HOMO_LUMO_gap_eV: HOMO-LUMO gap
- environment: gas vs toluene (encoded)
- method: stda vs stddft (encoded)

Target: Delta_E_ST_eV
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

    X = []
    y = []
    molecules = []
    environments = []
    methods = []

    for row in data:
        try:
            features = [float(row[col]) for col in feature_cols]
            target = float(row['Delta_E_ST_eV'])

            # Encode environment: gas=0, toluene=1
            env_encoded = 1 if row['environment'] == 'toluene' else 0
            features.append(env_encoded)

            # Encode method: stda=0, stddft=1
            method_encoded = 1 if row['method'] == 'stddft' else 0
            features.append(method_encoded)

            X.append(features)
            y.append(target)
            molecules.append(row['molecule'])
            environments.append(row['environment'])
            methods.append(row['method'])

        except (ValueError, KeyError) as e:
            continue

    return np.array(X), np.array(y), molecules, environments, methods


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model and return metrics."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
    }

    return model, y_pred, metrics


def train_gpr(X_train, y_train, X_test, y_test):
    """Train Gaussian Process Regressor and return metrics."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    # Scale features for GPR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Subsample for GPR if too large (GPR is O(n^3))
    max_samples = 1000
    if len(X_train_scaled) > max_samples:
        indices = np.random.choice(len(X_train_scaled), max_samples, replace=False)
        X_train_gpr = X_train_scaled[indices]
        y_train_gpr = y_train[indices]
    else:
        X_train_gpr = X_train_scaled
        y_train_gpr = y_train

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        random_state=42,
        normalize_y=True
    )

    model.fit(X_train_gpr, y_train_gpr)
    y_pred, y_std = model.predict(X_test_scaled, return_std=True)

    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'mean_uncertainty': float(np.mean(y_std)),
    }

    return model, y_pred, metrics, scaler


def compute_shap_importance(model, X, feature_names):
    """Compute SHAP feature importance."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        importance = np.abs(shap_values).mean(axis=0)
        return dict(zip(feature_names, importance))
    except ImportError:
        # Fallback to built-in feature importance
        return dict(zip(feature_names, model.feature_importances_))


def main():
    print("=" * 70)
    print("ML PIPELINE FOR 747 TADF MOLECULES - Delta_E_ST Prediction")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    data = load_data(DATA_FILE)
    print(f"  Loaded {len(data)} rows")

    # Prepare features
    X, y, molecules, environments, methods = prepare_features(data)
    print(f"  Prepared {len(X)} samples with {X.shape[1]} features")

    feature_names = ['S1_overlap', 'T1_overlap', 'S1_energy_eV', 'T1_energy_eV',
                     'S1_osc_strength', 'HOMO_LUMO_gap_eV', 'environment', 'method']

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Train Random Forest
    print("\n" + "-" * 50)
    print("Training Random Forest...")
    rf_model, rf_pred, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    print(f"  MAE:  {rf_metrics['mae']:.4f} eV")
    print(f"  RMSE: {rf_metrics['rmse']:.4f} eV")
    print(f"  R²:   {rf_metrics['r2']:.4f}")

    # SHAP importance for RF
    print("\nComputing SHAP feature importance...")
    rf_importance = compute_shap_importance(rf_model, X_test, feature_names)
    print("  Feature importance (SHAP):")
    for feat, imp in sorted(rf_importance.items(), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.4f}")

    # Train GPR
    print("\n" + "-" * 50)
    print("Training Gaussian Process Regressor...")
    gpr_model, gpr_pred, gpr_metrics, scaler = train_gpr(X_train, y_train, X_test, y_test)
    print(f"  MAE:  {gpr_metrics['mae']:.4f} eV")
    print(f"  RMSE: {gpr_metrics['rmse']:.4f} eV")
    print(f"  R²:   {gpr_metrics['r2']:.4f}")
    print(f"  Mean uncertainty: {gpr_metrics['mean_uncertainty']:.4f} eV")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X.shape[1],
            'unique_molecules': len(set(molecules)),
        },
        'random_forest': {
            'metrics': rf_metrics,
            'feature_importance': rf_importance,
        },
        'gaussian_process': {
            'metrics': gpr_metrics,
        },
        'target_statistics': {
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'min': float(np.min(y)),
            'max': float(np.max(y)),
        }
    }

    results_file = RESULTS_DIR / "ml_results_747mol.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save predictions
    pred_file = RESULTS_DIR / "predictions_747mol.csv"
    with open(pred_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['y_true', 'rf_pred', 'gpr_pred'])
        for yt, rfp, gprp in zip(y_test, rf_pred, gpr_pred):
            writer.writerow([yt, rfp, gprp])
    print(f"Predictions saved to: {pred_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Dataset: {len(set(molecules))} molecules, {len(X)} samples")
    print(f"\nRandom Forest:  R² = {rf_metrics['r2']:.3f}, MAE = {rf_metrics['mae']:.4f} eV")
    print(f"Gaussian Process: R² = {gpr_metrics['r2']:.3f}, MAE = {gpr_metrics['mae']:.4f} eV")

    # Group feature importance
    nto_imp = rf_importance['S1_overlap'] + rf_importance['T1_overlap']
    energy_imp = rf_importance['S1_energy_eV'] + rf_importance['T1_energy_eV'] + rf_importance['HOMO_LUMO_gap_eV']
    osc_imp = rf_importance['S1_osc_strength']
    env_imp = rf_importance['environment'] + rf_importance['method']

    total_imp = nto_imp + energy_imp + osc_imp + env_imp
    print(f"\nFeature Group Importance:")
    print(f"  Energy features: {100*energy_imp/total_imp:.1f}%")
    print(f"  NTO overlaps:    {100*nto_imp/total_imp:.1f}%")
    print(f"  Oscillator:      {100*osc_imp/total_imp:.1f}%")
    print(f"  Environment:     {100*env_imp/total_imp:.1f}%")

    print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    main()
