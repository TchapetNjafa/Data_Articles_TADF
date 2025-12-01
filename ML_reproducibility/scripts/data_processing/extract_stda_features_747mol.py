#!/usr/bin/env python3
"""
Extract excitation features from sTDA/sTD-DFT log files for 747 TADF molecules.

Parses .log files from the Data_calculation_747Mol folder to extract:
- S1 and T1 excitation energies (eV)
- Delta_E_ST = E(S1) - E(T1)
- Oscillator strengths
- HOMO-LUMO gap

Output: stda_features_747mol.csv with one row per molecule/environment/method combination.

Usage:
    python extract_stda_features_747mol.py
"""

import os
import re
import csv
from pathlib import Path


# Path to the 747 molecules data
DATA_747_DIR = Path("/home/tchapet/Post-Doc/ARTICLES DOSSIERS DES ARTICLES EN REDACTION/NOUVELS AXES DE RECHERCHE A REGARDER URGEMMENT/ARTICLES EN REDACTIONS/ARTICLE1/redaction/Article3_ML/Result_article1_TADF_xTB/Data_calculation_747Mol")


def parse_log_file(log_path):
    """
    Parse an sTDA/sTD-DFT log file to extract excitation data.
    """
    result = {
        'energy_ev': None,
        'wavelength_nm': None,
        'osc_strength': None,
        'is_triplet': False,
        'homo_energy': None,
        'lumo_energy': None,
        'homo_lumo_gap': None,
    }

    if not os.path.exists(log_path):
        return result

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return result

    # Check if triplet calculation
    if re.search(r'triplet\s*:\s*T', content):
        result['is_triplet'] = True

    # Extract frontier orbital energies
    frontier_match = re.search(
        r'ordered frontier orbitals\s*\n\s*eV\s*#\s*centers\s*\n(.*?)\n\s*\n(.*?)\n\s*\n',
        content, re.DOTALL
    )
    if frontier_match:
        occ_lines = frontier_match.group(1).strip().split('\n')
        virt_lines = frontier_match.group(2).strip().split('\n')

        occ_energies = []
        virt_energies = []

        for line in occ_lines:
            match = re.match(r'\s*(\d+)\s+(-?\d+\.\d+)\s+', line)
            if match:
                orb_idx = int(match.group(1))
                orb_energy = float(match.group(2))
                occ_energies.append((orb_idx, orb_energy))

        for line in virt_lines:
            match = re.match(r'\s*(\d+)\s+(-?\d+\.\d+)\s+', line)
            if match:
                orb_idx = int(match.group(1))
                orb_energy = float(match.group(2))
                virt_energies.append((orb_idx, orb_energy))

        if occ_energies:
            occ_energies.sort(key=lambda x: x[0])
            result['homo_energy'] = occ_energies[-1][1]

        if virt_energies:
            virt_energies.sort(key=lambda x: x[0])
            result['lumo_energy'] = virt_energies[0][1]

        if result['homo_energy'] is not None and result['lumo_energy'] is not None:
            result['homo_lumo_gap'] = result['lumo_energy'] - result['homo_energy']

    # Extract first excitation from the final results section
    exc_pattern = r'^\s*1\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+[-\d.]+\s+(.*?)$'
    exc_match = re.search(exc_pattern, content, re.MULTILINE)

    if exc_match:
        result['energy_ev'] = float(exc_match.group(1))
        result['wavelength_nm'] = float(exc_match.group(2))
        result['osc_strength'] = float(exc_match.group(3))

    return result


def discover_molecules(data_dir):
    """Discover all molecule folders in the gas directory."""
    gas_dir = data_dir / "gas"
    if not gas_dir.exists():
        return []

    molecules = []
    for item in sorted(gas_dir.iterdir()):
        if item.is_dir():
            molecules.append(item.name)
    return molecules


def extract_all_features(data_dir):
    """
    Extract features from all 747 molecules.
    """
    molecules = discover_molecules(data_dir)
    environments = ["gas", "toluene"]
    methods = ["stda", "stddft"]

    print(f"Found {len(molecules)} molecules")

    features = []
    processed = 0

    for mol in molecules:
        processed += 1
        if processed % 100 == 0:
            print(f"Processing {processed}/{len(molecules)}: {mol}")

        for env in environments:
            for method in methods:
                mol_dir = data_dir / env / mol

                # Get S1 data (S0S1 transition)
                s1_log = mol_dir / f"{mol}_{env}_S0S1_{method}.log"
                s1_data = parse_log_file(str(s1_log))

                # Get T1 data (S0T1 transition)
                t1_log = mol_dir / f"{mol}_{env}_S0T1_{method}.log"
                t1_data = parse_log_file(str(t1_log))

                # Calculate Delta_E_ST
                delta_est = None
                if s1_data['energy_ev'] is not None and t1_data['energy_ev'] is not None:
                    delta_est = s1_data['energy_ev'] - t1_data['energy_ev']

                feature_row = {
                    'molecule': mol,
                    'environment': env,
                    'method': method,
                    'S1_energy_eV': s1_data['energy_ev'],
                    'S1_wavelength_nm': s1_data['wavelength_nm'],
                    'S1_osc_strength': s1_data['osc_strength'],
                    'T1_energy_eV': t1_data['energy_ev'],
                    'T1_wavelength_nm': t1_data['wavelength_nm'],
                    'Delta_E_ST_eV': delta_est,
                    'HOMO_eV': s1_data['homo_energy'],
                    'LUMO_eV': s1_data['lumo_energy'],
                    'HOMO_LUMO_gap_eV': s1_data['homo_lumo_gap'],
                }

                features.append(feature_row)

    return features


def main():
    script_dir = Path(__file__).parent
    output_file = script_dir / "stda_features_747mol.csv"

    print(f"Extracting features from: {DATA_747_DIR}")

    features = extract_all_features(DATA_747_DIR)

    # Count successful extractions
    n_complete = sum(1 for f in features if f['Delta_E_ST_eV'] is not None)
    print(f"\nExtracted {len(features)} rows, {n_complete} complete with Delta_E_ST")

    # Write to CSV
    if features:
        fieldnames = list(features[0].keys())
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(features)
        print(f"Output saved to: {output_file}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for key in ['S1_energy_eV', 'T1_energy_eV', 'Delta_E_ST_eV', 'S1_osc_strength', 'HOMO_LUMO_gap_eV']:
        values = [f[key] for f in features if f[key] is not None]
        if values:
            print(f"{key}: min={min(values):.3f}, max={max(values):.3f}, mean={sum(values)/len(values):.3f}, count={len(values)}")


if __name__ == "__main__":
    main()
