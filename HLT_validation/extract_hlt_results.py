#!/usr/bin/env python3
"""
Extract HLT results and prepare for Article3 integration.

This script:
1. Parses ORCA TD-DFT output files
2. Extracts S1 and T1 energies
3. Calculates ΔE_ST
4. Compares with sTDA/sTD-DFT-xTB values
5. Generates summary tables and figures

Usage:
    python extract_hlt_results.py

Output:
    - hlt_results_summary.csv
    - hlt_validation_table.tex
    - hlt_parity_plot.pdf
"""

import os
import re
import pandas as pd
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
ARTICLE_DIR = os.path.dirname(SCRIPT_DIR)

# Reference values from sTDA/sTD-DFT-xTB (from our dataset)
XTB_REFERENCE = {
    "DMAC-DPS": {
        "stda": {"S1_eV": 3.73, "T1_eV": 3.29, "Delta_EST": 0.44},
        "stddft": {"S1_eV": 3.60, "T1_eV": 3.17, "Delta_EST": 0.43},
    },
    "DMAC-TRZ": {
        "stda": {"S1_eV": 3.05, "T1_eV": 2.96, "Delta_EST": 0.085},
        "stddft": {"S1_eV": 2.93, "T1_eV": 2.85, "Delta_EST": 0.077},
    },
    "4CzIPN": {
        "stda": {"S1_eV": 2.98, "T1_eV": 2.77, "Delta_EST": 0.21},
        "stddft": {"S1_eV": 2.85, "T1_eV": 2.63, "Delta_EST": 0.22},
    },
}

# Experimental reference values
EXPERIMENTAL = {
    "DMAC-DPS": {"Delta_EST": 0.08, "source": "Adachi 2014"},
    "DMAC-TRZ": {"Delta_EST": 0.05, "source": "STGABS27"},
    "4CzIPN": {"Delta_EST": 0.08, "source": "STGABS27"},
}


def parse_orca_tddft(output_file):
    """Parse ORCA TD-DFT output file and extract excited state energies."""

    if not os.path.exists(output_file):
        print(f"Warning: Output file not found: {output_file}")
        return None

    with open(output_file, 'r') as f:
        content = f.read()

    results = {
        "S1_eV": None,
        "T1_eV": None,
        "S1_osc": None,
    }

    # Look for singlet excited states
    # Pattern: STATE  1:  E=   0.123456 au    3.3579 eV
    singlet_pattern = r"SINGLET.*?STATE\s+1:.*?(\d+\.\d+)\s*eV"
    singlet_match = re.search(singlet_pattern, content, re.DOTALL | re.IGNORECASE)

    if singlet_match:
        results["S1_eV"] = float(singlet_match.group(1))

    # Alternative pattern for singlet
    if results["S1_eV"] is None:
        alt_pattern = r"STATE\s+1:.*?E=.*?(\d+\.\d+)\s*eV"
        match = re.search(alt_pattern, content)
        if match:
            results["S1_eV"] = float(match.group(1))

    # Look for triplet excited states
    triplet_pattern = r"TRIPLET.*?STATE\s+1:.*?(\d+\.\d+)\s*eV"
    triplet_match = re.search(triplet_pattern, content, re.DOTALL | re.IGNORECASE)

    if triplet_match:
        results["T1_eV"] = float(triplet_match.group(1))

    # Extract oscillator strength
    osc_pattern = r"STATE\s+1:.*?f=\s*(\d+\.\d+)"
    osc_match = re.search(osc_pattern, content)
    if osc_match:
        results["S1_osc"] = float(osc_match.group(1))

    return results


def parse_optimal_omega(mol_dir):
    """Parse optimal omega value from tuning results."""
    omega_file = os.path.join(mol_dir, "optimal_omega.txt")

    if not os.path.exists(omega_file):
        return None

    with open(omega_file, 'r') as f:
        content = f.read()

    match = re.search(r"Optimal omega:\s*(\d+\.\d+)", content)
    if match:
        return float(match.group(1))

    return None


def extract_all_results():
    """Extract results for all molecules."""

    molecules = ["DMAC-DPS", "DMAC-TRZ", "4CzIPN"]
    results = []

    for mol in molecules:
        mol_dir = os.path.join(RESULTS_DIR, mol)

        if not os.path.exists(mol_dir):
            print(f"Warning: No results directory for {mol}")
            continue

        # Parse omega
        omega = parse_optimal_omega(mol_dir)

        # Parse TD-DFT results
        tddft_file = os.path.join(mol_dir, f"{mol}_TDDFT.out")
        tddft_results = parse_orca_tddft(tddft_file)

        if tddft_results is None:
            print(f"Warning: Could not parse TD-DFT results for {mol}")
            continue

        # Calculate Delta_EST
        delta_est = None
        if tddft_results["S1_eV"] and tddft_results["T1_eV"]:
            delta_est = tddft_results["S1_eV"] - tddft_results["T1_eV"]

        # Add reference values
        xtb_stda = XTB_REFERENCE.get(mol, {}).get("stda", {})
        xtb_stddft = XTB_REFERENCE.get(mol, {}).get("stddft", {})
        exp = EXPERIMENTAL.get(mol, {})

        results.append({
            "Molecule": mol,
            "Optimal_omega": omega,
            "HLT_S1_eV": tddft_results["S1_eV"],
            "HLT_T1_eV": tddft_results["T1_eV"],
            "HLT_Delta_EST": delta_est,
            "HLT_S1_osc": tddft_results["S1_osc"],
            "sTDA_Delta_EST": xtb_stda.get("Delta_EST"),
            "sTDDFT_Delta_EST": xtb_stddft.get("Delta_EST"),
            "Exp_Delta_EST": exp.get("Delta_EST"),
            "Exp_source": exp.get("source"),
        })

    return pd.DataFrame(results)


def calculate_mae(df):
    """Calculate MAE for different methods vs experiment."""

    mae_results = {}

    # Filter rows with experimental data
    df_exp = df.dropna(subset=["Exp_Delta_EST"])

    if len(df_exp) > 0:
        if df_exp["HLT_Delta_EST"].notna().any():
            mae_results["HLT_vs_Exp"] = np.mean(np.abs(
                df_exp["HLT_Delta_EST"] - df_exp["Exp_Delta_EST"]
            ))

        if df_exp["sTDA_Delta_EST"].notna().any():
            mae_results["sTDA_vs_Exp"] = np.mean(np.abs(
                df_exp["sTDA_Delta_EST"] - df_exp["Exp_Delta_EST"]
            ))

        if df_exp["sTDDFT_Delta_EST"].notna().any():
            mae_results["sTDDFT_vs_Exp"] = np.mean(np.abs(
                df_exp["sTDDFT_Delta_EST"] - df_exp["Exp_Delta_EST"]
            ))

    return mae_results


def generate_latex_table(df, output_file):
    """Generate LaTeX table for Article3."""

    latex = r"""\begin{table}[h]
\centering
\caption{Comparison of $\Delta E_{\text{ST}}$ values (eV) from different computational methods and experiment.}
\label{tab:hlt_validation}
\begin{tabular}{lcccccc}
\hline
Molecule & $\omega_{\text{opt}}$ & OT-LC-$\omega$PBE & sTDA-xTB & sTD-DFT-xTB & Exp. \\
\hline
"""

    for _, row in df.iterrows():
        omega = f"{row['Optimal_omega']:.3f}" if pd.notna(row['Optimal_omega']) else "-"
        hlt = f"{row['HLT_Delta_EST']:.3f}" if pd.notna(row['HLT_Delta_EST']) else "-"
        stda = f"{row['sTDA_Delta_EST']:.3f}" if pd.notna(row['sTDA_Delta_EST']) else "-"
        stddft = f"{row['sTDDFT_Delta_EST']:.3f}" if pd.notna(row['sTDDFT_Delta_EST']) else "-"
        exp = f"{row['Exp_Delta_EST']:.2f}" if pd.notna(row['Exp_Delta_EST']) else "-"

        latex += f"{row['Molecule']} & {omega} & {hlt} & {stda} & {stddft} & {exp} \\\\\n"

    latex += r"""\hline
\end{tabular}
\end{table}
"""

    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"LaTeX table saved to: {output_file}")


def main():
    print("="*60)
    print("Extracting HLT Results for Article3")
    print("="*60)

    # Extract results
    df = extract_all_results()

    if len(df) == 0:
        print("\nNo results found. Run HLT calculations first:")
        print("  ./run_hlt_calculations.sh")
        return

    # Save CSV
    csv_file = os.path.join(RESULTS_DIR, "hlt_results_summary.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to: {csv_file}")

    # Print summary
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(df.to_string(index=False))

    # Calculate MAE
    mae = calculate_mae(df)
    print("\n" + "="*60)
    print("MAE vs Experiment")
    print("="*60)
    for method, value in mae.items():
        print(f"  {method}: {value:.3f} eV")

    # Generate LaTeX table
    latex_file = os.path.join(ARTICLE_DIR, "tables", "table_hlt_validation_new.tex")
    os.makedirs(os.path.dirname(latex_file), exist_ok=True)
    generate_latex_table(df, latex_file)

    print("\n" + "="*60)
    print("Integration Instructions")
    print("="*60)
    print("""
1. Update Article3_draft1.tex:
   - Replace/update benchmark validation section
   - Include new MAE values

2. Update Article3_SI.tex:
   - Add OT-LC-ωPBE methodology details
   - Include optimal ω values

3. Regenerate figures if needed:
   - Update hlt_validation.pdf with new data

4. Recompile manuscripts:
   cd Article3 && pdflatex Article3_draft1.tex && pdflatex Article3_SI.tex
""")


if __name__ == "__main__":
    main()
