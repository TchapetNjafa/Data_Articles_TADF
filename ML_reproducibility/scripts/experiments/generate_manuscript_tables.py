#!/usr/bin/env python3
"""
Generate LaTeX Tables for Manuscript
"""

import pandas as pd
import json
from pathlib import Path

# Output directory
TABLES_DIR = Path("tables_manuscript")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def save_latex_table(latex_str, filename, caption, label):
    """Save LaTeX table with caption and label"""
    
    full_table = f"""% Table: {caption}
% Generated automatically - do not edit manually

\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex_str}
\\end{{table}}
"""
    
    output_file = TABLES_DIR / f"{filename}.tex"
    with open(output_file, 'w') as f:
        f.write(full_table)
    
    print(f"  ✓ Saved: {filename}.tex")

# ============================================================================
# TABLE 1: Model Benchmarking Comparison
# ============================================================================
def create_model_comparison_table():
    """Table comparing 4 ML algorithms"""
    
    print("\n=== Creating Table 1: Model Benchmarking ===")
    
    # Load data
    with open("results/model_benchmarking/model_benchmarking_summary.json") as f:
        data = json.load(f)
    
    # Extract results
    models_data = []
    for model_name, metrics in data['model_performance'].items():
        models_data.append({
            'Model': model_name,
            'MAE (eV)': f"{metrics['mae_eV']:.4f}",
            'RMSE (eV)': f"{metrics['rmse_eV']:.4f}",
            'R²': f"{metrics['r2']:.4f}",
            'Max Error (eV)': f"{metrics['max_error_eV']:.3f}",
            'Training Time (s)': f"{metrics.get('training_time_s', 0):.1f}"
        })
    
    df = pd.DataFrame(models_data)
    
    # Sort by MAE
    df = df.sort_values('MAE (eV)')
    
    # Generate LaTeX
    latex = df.to_latex(index=False, 
                       column_format='lccccc',
                       escape=False,
                       caption='Model Benchmarking Results',
                       label='tab:model_comparison')
    
    # Clean up LaTeX
    latex = latex.replace('\\toprule', '\\hline\\hline')
    latex = latex.replace('\\midrule', '\\hline')
    latex = latex.replace('\\bottomrule', '\\hline\\hline')
    
    save_latex_table(latex, 
                    'table1_model_comparison',
                    'Comparison of machine learning algorithms for $\\Delta E_\\mathrm{ST}$ prediction using 5-fold cross-validation on 2943 samples.',
                    'tab:model_comparison')

# ============================================================================
# TABLE 2: Ablation Study Results
# ============================================================================
def create_ablation_table():
    """Table showing ablation study results"""
    
    print("\n=== Creating Table 2: Ablation Study ===")
    
    # Load data
    with open("results/ablation_study/ablation_summary.json") as f:
        data = json.load(f)
    
    # Extract results
    ablation_data = []
    for item in data['results']:
        ablation_data.append({
            'Model': item['model_name'].replace('_', ' '),
            'N Features': item['n_features'],
            'MAE (eV)': f"{item['mae_eV']:.4f}",
            'RMSE (eV)': f"{item['rmse_eV']:.4f}",
            'R²': f"{item['r2']:.4f}"
        })
    
    df = pd.DataFrame(ablation_data)
    
    # Generate LaTeX
    latex = df.to_latex(index=False,
                       column_format='lccccc',
                       escape=False)
    
    # Clean up
    latex = latex.replace('\\toprule', '\\hline\\hline')
    latex = latex.replace('\\midrule', '\\hline')
    latex = latex.replace('\\bottomrule', '\\hline\\hline')
    
    save_latex_table(latex,
                    'table2_ablation_study',
                    'Ablation study comparing feature sets: Model A (energy features only), Model C (full features), Model D (CT descriptors only, no energy features).',
                    'tab:ablation')

# ============================================================================
# TABLE 3: Application Candidates
# ============================================================================
def create_candidates_table():
    """Table summarizing application-specific candidates"""
    
    print("\n=== Creating Table 3: Application Candidates ===")
    
    # Load data
    with open("results/candidate_filtering/candidate_summary.json") as f:
        data = json.load(f)
    
    # Create summary table
    candidates_data = []
    for app_name, app_data in data['candidates'].items():
        candidates_data.append({
            'Application': app_name.replace('_', ' ').title(),
            'Criteria': app_data['criteria'],
            'N Candidates': app_data['count'],
            'Top Molecule': app_data['top_10'][0] if app_data['top_10'] else 'None'
        })
    
    df = pd.DataFrame(candidates_data)
    
    # Generate LaTeX
    latex = df.to_latex(index=False,
                       column_format='lp{5cm}cc',
                       escape=False)
    
    # Clean up
    latex = latex.replace('\\toprule', '\\hline\\hline')
    latex = latex.replace('\\midrule', '\\hline')
    latex = latex.replace('\\bottomrule', '\\hline\\hline')
    latex = latex.replace('_', '\\_')  # Escape underscores in molecule names
    
    save_latex_table(latex,
                    'table3_candidates',
                    'Application-specific TADF candidates identified from 747-molecule database through multi-objective filtering.',
                    'tab:candidates')

# ============================================================================
# TABLE 4: Experimental Validation
# ============================================================================
def create_experimental_validation_table():
    """Table comparing computed vs experimental ΔE_ST"""
    
    print("\n=== Creating Table 4: Experimental Validation ===")
    
    # Load data
    validation_file = Path("results/experimental_validation/validation_summary.json")
    if not validation_file.exists():
        print("  ⚠ Experimental validation file not found - skipping")
        return
    
    with open(validation_file) as f:
        data = json.load(f)
    
    # Create table
    exp_data = []
    for mol_name, mol_data in data['molecules'].items():
        exp_data.append({
            'Molecule': mol_name,
            'Phase': mol_data['phase'],
            'Exp. $\\Delta E_\\mathrm{ST}$ (eV)': f"{mol_data['experimental_eV']:.3f}",
            'Computed (eV)': f"{mol_data['computed_vertical_eV']:.3f}",
            'Error (eV)': f"{mol_data['error_eV']:+.3f}",
            'Reference': mol_data.get('reference', 'N/A')
        })
    
    df = pd.DataFrame(exp_data)
    
    # Generate LaTeX
    latex = df.to_latex(index=False,
                       column_format='llcccp{3cm}',
                       escape=False)
    
    # Clean up
    latex = latex.replace('\\toprule', '\\hline\\hline')
    latex = latex.replace('\\midrule', '\\hline')
    latex = latex.replace('\\bottomrule', '\\hline\\hline')
    latex = latex.replace('_', '\\_')
    
    save_latex_table(latex,
                    'table4_experimental_validation',
                    'Comparison of vertical CAM-B3LYP/def2-TZVP $\\Delta E_\\mathrm{ST}$ predictions with experimental emission energies.',
                    'tab:experimental')

# ============================================================================
# Main Execution
# ============================================================================
def main():
    print("="*70)
    print("GENERATING MANUSCRIPT TABLES")
    print("="*70)
    
    create_model_comparison_table()
    create_ablation_table()
    create_candidates_table()
    create_experimental_validation_table()
    
    print("\n" + "="*70)
    print("ALL TABLES CREATED SUCCESSFULLY")
    print(f"Output directory: {TABLES_DIR}")
    print("="*70)
    print("\nTables generated:")
    for table_file in sorted(TABLES_DIR.glob("*.tex")):
        print(f"  - {table_file.name}")

if __name__ == "__main__":
    main()
