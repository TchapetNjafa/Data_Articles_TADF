#!/usr/bin/env python3
"""
ML Pipeline with CT (Charge-Transfer) Descriptors for Delta_E_ST Prediction.
Full 747mol dataset version with 2943 samples.

This script trains ML models using the enhanced feature set that includes
NTO-derived CT descriptors and performs SHAP analysis.

Feature Categories:
- Energy: S1_energy_eV, T1_energy_eV, HOMO_LUMO_gap_eV
- Oscillator: S1_osc_strength
- NTO_overlap: S1_overlap, T1_overlap
- CT_descriptors: CT_number, Lambda_D, Lambda_A, Delta_r, S_he (for S1 and T1)

Author: Generated for Article 3 ML pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import shap
import warnings
import json
warnings.filterwarnings('ignore')

# Define feature categories for SHAP grouping
FEATURE_CATEGORIES = {
    'Energy': ['S1_energy_eV', 'T1_energy_eV', 'HOMO_LUMO_gap_eV'],
    'Oscillator': ['S1_osc_strength'],
    'NTO_overlap': ['S1_overlap', 'T1_overlap'],
    'CT_descriptors': [
        'S1_CT_number', 'T1_CT_number',
        'S1_Lambda_D', 'T1_Lambda_D',
        'S1_Lambda_A', 'T1_Lambda_A',
        'S1_Delta_r', 'T1_Delta_r',
        'S1_S_he', 'T1_S_he',
        'Delta_CT_number', 'Delta_Lambda_D', 'Delta_Lambda_A',
        'Delta_Delta_r', 'Delta_S_he'
    ],
    'Proxy_RespA': [
        'Delta_S_NTO', 'Abs_Delta_S_NTO', 'Char_diff_squared',
        'S_NTO_sum', 'S_NTO_product', 'Log_Abs_S1', 'Log_Abs_T1', 'S_NTO_ratio'
    ]
}

def load_data(data_path):
    """Load enhanced feature data."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    return df

def prepare_features(df, target='Delta_E_ST_eV', use_ct=True):
    """Prepare feature matrix and target vector."""

    # Get all available features from categories
    all_category_features = []
    for cat_name, cat_features in FEATURE_CATEGORIES.items():
        if not use_ct and cat_name == 'CT_descriptors':
            continue
        all_category_features.extend(cat_features)

    # Filter to features that exist in dataframe
    feature_cols = [c for c in all_category_features if c in df.columns]

    # Check for non-NaN samples
    df_clean = df.dropna(subset=feature_cols + [target])
    print(f"Samples with complete data: {len(df_clean)} (dropped {len(df) - len(df_clean)})")

    print(f"Using {len(feature_cols)} features: {feature_cols}")

    X = df_clean[feature_cols].values
    y = df_clean[target].values
    feature_names = feature_cols

    # Handle any remaining NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    return X, y, feature_names, df_clean

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """Train multiple models and evaluate performance."""
    results = {}

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15,
                                               min_samples_split=5, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                        learning_rate=0.1, random_state=42),
        'SVR': SVR(C=10, gamma='scale', epsilon=0.01),
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Use scaled data for SVR
        if name in ['SVR']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Cross-validation
        if name in ['SVR']:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5,
                                        scoring='neg_mean_absolute_error')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                        scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()

        results[name] = {
            'model': model,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_MAE': cv_mae,
            'y_pred': y_pred
        }

        print(f"  MAE: {mae:.4f} eV, RMSE: {rmse:.4f} eV, R²: {r2:.4f}")
        print(f"  CV MAE: {cv_mae:.4f} eV (5-fold)")

    return results, scaler

def compute_shap_importance(model, X, feature_names, model_name='Random Forest'):
    """Compute SHAP values and group by category."""
    print(f"\nComputing SHAP values for {model_name}...")

    if model_name in ['Random Forest', 'Gradient Boosting']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        # Sample for kernel explainer (faster)
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X[:200])  # Limit for speed

    # Compute mean absolute SHAP values per feature
    mean_shap = np.abs(shap_values).mean(axis=0)

    # Create feature importance dict
    feature_importance = dict(zip(feature_names, mean_shap))

    # Group by category
    category_importance = {}
    total_importance = sum(mean_shap)

    for category, cat_features in FEATURE_CATEGORIES.items():
        cat_importance = sum(feature_importance.get(f, 0) for f in cat_features)
        category_importance[category] = {
            'absolute': cat_importance,
            'percentage': 100 * cat_importance / total_importance if total_importance > 0 else 0
        }

    return shap_values, feature_importance, category_importance

def save_figure(fig, output_dir, basename):
    """Save figure in PDF, PNG, and PGF formats."""
    for ext in ['pdf', 'png', 'pgf']:
        filepath = output_dir / f"{basename}.{ext}"
        dpi = 300 if ext == 'png' else None
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        except Exception as e:
            print(f"  Warning: Could not save {ext}: {e}")

def plot_results(results, y_test, feature_importance, category_importance, output_dir):
    """Generate and save plots in PDF, PNG, and PGF formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parity plot for best model
    best_model = max(results.items(), key=lambda x: x[1]['R2'])
    model_name, model_results = best_model

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, model_results['y_pred'], alpha=0.5, s=20)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual $\\Delta E_{ST}$ (eV)', fontsize=12)
    ax.set_ylabel('Predicted $\\Delta E_{ST}$ (eV)', fontsize=12)
    ax.set_title(f'{model_name}: R² = {model_results["R2"]:.3f}, MAE = {model_results["MAE"]:.3f} eV')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    save_figure(fig, output_dir, 'parity_plot_747mol')
    plt.close()

    # 2. Feature importance bar plot (top 20)
    fig, ax = plt.subplots(figsize=(12, 10))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    features, importances = zip(*sorted_features)

    # Color by category
    category_colors = {
        'Energy': '#1f77b4',
        'Oscillator': '#ff7f0e',
        'NTO_overlap': '#2ca02c',
        'CT_descriptors': '#d62728',
        'Proxy_RespA': '#9467bd'
    }

    colors = []
    for f in features:
        for cat, cat_features in FEATURE_CATEGORIES.items():
            if f in cat_features:
                colors.append(category_colors.get(cat, '#7f7f7f'))
                break
        else:
            colors.append('#7f7f7f')

    bars = ax.barh(range(len(features)), importances, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title('Top 20 Feature Importance (SHAP) - 747mol Dataset')
    ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat)
                      for cat, color in category_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    save_figure(fig, output_dir, 'feature_importance_747mol')
    plt.close()

    # 3. Category importance pie chart
    fig, ax = plt.subplots(figsize=(10, 10))
    categories = [c for c in category_importance.keys() if category_importance[c]['percentage'] > 0.1]
    percentages = [category_importance[c]['percentage'] for c in categories]
    colors = [category_colors.get(c, '#7f7f7f') for c in categories]

    wedges, texts, autotexts = ax.pie(percentages, labels=categories, autopct='%.1f%%',
                                       colors=colors, startangle=90)
    ax.set_title('Feature Category Importance (SHAP) - 747mol Dataset')
    plt.tight_layout()
    save_figure(fig, output_dir, 'category_importance_747mol')
    plt.close()

    print(f"\nPlots saved to {output_dir}")

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data_processing' / 'combined_features_747mol_full_ct.csv'
    output_dir = base_dir / 'figures_747mol'

    # Load data
    df = load_data(data_path)

    # Prepare features (with CT descriptors)
    X, y, feature_names, df_clean = prepare_features(df, use_ct=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Train models
    results, scaler = train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names)

    # SHAP analysis on best tree model
    best_tree_model = 'Gradient Boosting' if results['Gradient Boosting']['R2'] > results['Random Forest']['R2'] else 'Random Forest'
    model = results[best_tree_model]['model']
    shap_values, feature_importance, category_importance = compute_shap_importance(
        model, X_test, feature_names, best_tree_model
    )

    # Print category importance
    print("\n" + "="*60)
    print("FEATURE CATEGORY IMPORTANCE (SHAP)")
    print("="*60)
    for category, imp in sorted(category_importance.items(), key=lambda x: x[1]['percentage'], reverse=True):
        if imp['percentage'] > 0.1:
            print(f"  {category}: {imp['percentage']:.1f}%")

    # Print top features
    print("\nTop 15 Individual Features:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, imp) in enumerate(sorted_features[:15], 1):
        pct = 100 * imp / sum(feature_importance.values())
        print(f"  {i:2d}. {feature}: {pct:.1f}%")

    # Generate plots
    plot_results(results, y_test, feature_importance, category_importance, output_dir)

    # Save results to JSON
    results_summary = {
        'models': {name: {'MAE': r['MAE'], 'RMSE': r['RMSE'], 'R2': r['R2'], 'CV_MAE': r['CV_MAE']}
                   for name, r in results.items()},
        'category_importance': {k: v['percentage'] for k, v in category_importance.items()},
        'feature_importance': {k: float(v) for k, v in feature_importance.items()},
        'n_samples': len(df_clean),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': len(feature_names),
        'feature_names': feature_names
    }

    with open(output_dir / 'ml_results_747mol.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to {output_dir / 'ml_results_747mol.json'}")

if __name__ == '__main__':
    main()
