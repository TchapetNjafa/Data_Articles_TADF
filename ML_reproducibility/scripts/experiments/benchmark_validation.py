#!/usr/bin/env python3
"""
Benchmark validation: Compare ML predictions with HLT reference values.

Uses STGABS27 benchmark data for validation:
- 4CzIPN: ROKS/PCM experimental value = 0.16 eV (from Kunze et al. 2021)
- DMAC-TRZ: SCS-CC2/PCM = 0.05 eV (from STGABS27)

Reference: Kunze et al., J. Phys. Chem. Lett. 2021, 12, 8470-8480
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# HLT reference values (from STGABS27 benchmark)
HLT_REFERENCES = {
    '4CzIPN': {'method': 'ROKS/PCM', 'value': 0.16, 'solvent': 'toluene', 'source': 'Kunze2021'},
    'DMAC-TRZ': {'method': 'SCS-CC2/PCM', 'value': 0.05, 'solvent': 'toluene', 'source': 'Kunze2021'},
}

# Experimental references (where available)
EXP_REFERENCES = {
    '4CzIPN': {'value': 0.09, 'solvent': 'toluene', 'source': 'Uoyama2012'},  # ~0.09 eV in toluene
    'DMAC-TRZ': {'value': 0.05, 'solvent': 'toluene', 'source': 'Lin2017'},
}


def load_features(data_path):
    """Load combined features with CT descriptors."""
    df = pd.read_csv(data_path)
    return df


def train_ml_model(df, target='Delta_E_ST_eV'):
    """Train SVR model on full dataset and return predictions."""
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Feature columns (numeric only, excluding identifiers and target)
    exclude_cols = ['molecule', 'environment', 'method', target]
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]

    X = df[feature_cols].values
    y = df[target].values

    # Train SVR model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=10, epsilon=0.01, gamma='scale'))
    ])
    model.fit(X, y)

    # Get predictions
    df['ML_prediction'] = model.predict(X)

    return df, model, feature_cols


def generate_validation_table(df, output_dir):
    """Generate benchmark validation table in LaTeX format."""
    # Filter for reference molecules in toluene
    ref_molecules = list(HLT_REFERENCES.keys())

    validation_data = []
    for mol in ref_molecules:
        mol_df = df[(df['molecule'] == mol) & (df['environment'] == 'toluene')]
        if len(mol_df) == 0:
            continue

        for _, row in mol_df.iterrows():
            hlt_info = HLT_REFERENCES.get(mol, {})
            exp_info = EXP_REFERENCES.get(mol, {})

            validation_data.append({
                'Molecule': mol,
                'Method': row['method'],
                'Computed_DEST': row['Delta_E_ST_eV'],
                'ML_Pred': row['ML_prediction'],
                'HLT_Ref': hlt_info.get('value', np.nan),
                'HLT_Method': hlt_info.get('method', '-'),
                'Exp_Ref': exp_info.get('value', np.nan),
            })

    val_df = pd.DataFrame(validation_data)

    # Calculate errors
    val_df['Error_vs_HLT'] = val_df['Computed_DEST'] - val_df['HLT_Ref']
    val_df['ML_Error_vs_HLT'] = val_df['ML_Pred'] - val_df['HLT_Ref']

    # Generate LaTeX table
    latex_table = generate_latex_table(val_df)

    # Save table
    table_path = Path(output_dir) / 'tables' / 'table_hlt_validation.tex'
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, 'w') as f:
        f.write(latex_table)

    print(f"Validation table saved to: {table_path}")

    return val_df


def generate_latex_table(val_df):
    """Generate LaTeX table for HLT validation."""
    latex = r"""% Benchmark validation table: ML predictions vs HLT references
% Reference: Kunze et al., J. Phys. Chem. Lett. 2021, 12, 8470-8480

\begin{table}[ht]
\small
\caption{Benchmark validation of computed $\Delta E_{\mathrm{ST}}$ values against high-level theory (HLT) references from the STGABS27 benchmark set. All values in eV.}
\label{tab:hlt_validation}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}} l l c c c c}
\toprule
Molecule & Method & Computed & ML Pred. & HLT Ref. & $|\Delta|$ \\
\midrule
"""

    for _, row in val_df.iterrows():
        mol = row['Molecule']
        method = 'sTDA' if row['Method'] == 'stda' else 'sTD-DFT'
        computed = f"{row['Computed_DEST']:.3f}"
        ml_pred = f"{row['ML_Pred']:.3f}"
        hlt = f"{row['HLT_Ref']:.2f}" if not np.isnan(row['HLT_Ref']) else '-'
        error = f"{abs(row['Error_vs_HLT']):.3f}" if not np.isnan(row['Error_vs_HLT']) else '-'

        latex += f"{mol} & {method} & {computed} & {ml_pred} & {hlt} & {error} \\\\\n"

    latex += r"""\bottomrule
\end{tabular*}
\begin{tablenotes}
\small
\item HLT methods: ROKS/PCM (4CzIPN), SCS-CC2/PCM (DMAC-TRZ) from Kunze et al. 2021.
\end{tablenotes}
\end{table}
"""
    return latex


def generate_validation_figure(val_df, output_dir):
    """Generate parity plot comparing computed vs HLT values."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Filter valid HLT data
    valid_df = val_df.dropna(subset=['HLT_Ref'])

    # Plot 1: Computed vs HLT
    ax1 = axes[0]
    colors = {'stda': '#2E86AB', 'stddft': '#A23B72'}
    markers = {'stda': 'o', 'stddft': 's'}

    for method in valid_df['Method'].unique():
        subset = valid_df[valid_df['Method'] == method]
        label = 'sTDA-xTB' if method == 'stda' else 'sTD-DFT-xTB'
        ax1.scatter(subset['HLT_Ref'], subset['Computed_DEST'],
                   c=colors[method], marker=markers[method], s=100,
                   label=label, edgecolors='black', linewidth=0.5)

    # Perfect agreement line
    lims = [0, 0.3]
    ax1.plot(lims, lims, 'k--', alpha=0.5, label='Perfect agreement')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel(r'HLT Reference $\Delta E_{\mathrm{ST}}$ (eV)', fontsize=11)
    ax1.set_ylabel(r'Computed $\Delta E_{\mathrm{ST}}$ (eV)', fontsize=11)
    ax1.legend(loc='upper left', frameon=True)
    ax1.set_title('(a) xTB Methods vs HLT', fontsize=12)

    # Add molecule labels
    for _, row in valid_df.iterrows():
        offset = (0.01, 0.01) if row['Method'] == 'stda' else (0.01, -0.02)
        ax1.annotate(row['Molecule'], (row['HLT_Ref'], row['Computed_DEST']),
                    xytext=offset, textcoords='offset points', fontsize=8)

    # Plot 2: ML Predictions vs HLT
    ax2 = axes[1]
    for method in valid_df['Method'].unique():
        subset = valid_df[valid_df['Method'] == method]
        label = 'sTDA' if method == 'stda' else 'sTD-DFT'
        ax2.scatter(subset['HLT_Ref'], subset['ML_Pred'],
                   c=colors[method], marker=markers[method], s=100,
                   label=label, edgecolors='black', linewidth=0.5)

    ax2.plot(lims, lims, 'k--', alpha=0.5)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel(r'HLT Reference $\Delta E_{\mathrm{ST}}$ (eV)', fontsize=11)
    ax2.set_ylabel(r'ML Predicted $\Delta E_{\mathrm{ST}}$ (eV)', fontsize=11)
    ax2.legend(loc='upper left', frameon=True)
    ax2.set_title('(b) ML Predictions vs HLT', fontsize=12)

    plt.tight_layout()

    # Save in multiple formats
    output_path = Path(output_dir) / 'figures_747mol'
    output_path.mkdir(parents=True, exist_ok=True)

    for fmt in ['pdf', 'png', 'pgf']:
        fig_path = output_path / f'hlt_validation.{fmt}'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_path}")

    plt.close()


def compute_statistics(val_df):
    """Compute validation statistics."""
    valid_df = val_df.dropna(subset=['HLT_Ref'])

    print("\n" + "="*60)
    print("BENCHMARK VALIDATION STATISTICS")
    print("="*60)

    for method in valid_df['Method'].unique():
        subset = valid_df[valid_df['Method'] == method]
        method_name = 'sTDA-xTB' if method == 'stda' else 'sTD-DFT-xTB'

        mae_computed = np.mean(np.abs(subset['Error_vs_HLT']))
        mae_ml = np.mean(np.abs(subset['ML_Error_vs_HLT']))

        print(f"\n{method_name}:")
        print(f"  Computed MAE vs HLT: {mae_computed:.4f} eV")
        print(f"  ML Pred MAE vs HLT:  {mae_ml:.4f} eV")

    # Overall statistics
    print("\n" + "-"*60)
    overall_mae = np.mean(np.abs(valid_df['Error_vs_HLT']))
    overall_ml_mae = np.mean(np.abs(valid_df['ML_Error_vs_HLT']))
    print(f"Overall Computed MAE vs HLT: {overall_mae:.4f} eV")
    print(f"Overall ML Pred MAE vs HLT:  {overall_ml_mae:.4f} eV")
    print("="*60)

    return {
        'computed_mae': overall_mae,
        'ml_mae': overall_ml_mae
    }


def main():
    # Paths
    script_dir = Path(__file__).parent
    article_dir = script_dir.parent
    data_path = article_dir / 'data_processing' / 'combined_features_747mol_full_ct.csv'
    output_dir = article_dir

    print("Loading data...")
    df = load_features(data_path)
    print(f"Loaded {len(df)} samples")

    print("\nTraining ML model...")
    df, model, feature_cols = train_ml_model(df)

    print("\nGenerating validation table...")
    val_df = generate_validation_table(df, output_dir)

    print("\nGenerating validation figure...")
    generate_validation_figure(val_df, output_dir)

    print("\nComputing statistics...")
    stats = compute_statistics(val_df)

    print("\n" + "="*60)
    print("BENCHMARK VALIDATION COMPLETE")
    print("="*60)

    return stats


if __name__ == '__main__':
    main()
