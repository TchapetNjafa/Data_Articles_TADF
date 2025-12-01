#!/usr/bin/env python3
"""
Build combined feature table for 747 TADF molecules.

Combines:
- NTO orbital overlaps from nto_orbital_overlap_747mol.csv
- sTDA excitation features from stda_features_747mol.csv

Output: combined_features_747mol.csv with columns:
- molecule, environment, method
- S1_overlap, T1_overlap (NTO hole-electron overlaps)
- S1_energy_eV, T1_energy_eV, Delta_E_ST_eV
- S1_osc_strength, HOMO_LUMO_gap_eV
"""

import csv
from pathlib import Path


# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_747_DIR = Path("/home/tchapet/Post-Doc/ARTICLES DOSSIERS DES ARTICLES EN REDACTION/NOUVELS AXES DE RECHERCHE A REGARDER URGEMMENT/ARTICLES EN REDACTIONS/ARTICLE1/redaction/Article3_ML/Result_article1_TADF_xTB/Data_calculation_747Mol")

NTO_OVERLAP_FILE = DATA_747_DIR / "nto_orbital_overlap_747mol.csv"
STDA_FEATURES_FILE = SCRIPT_DIR / "stda_features_747mol.csv"
OUTPUT_FILE = SCRIPT_DIR / "combined_features_747mol.csv"


def load_nto_overlaps(path):
    """Load NTO overlaps and aggregate by (molecule, environment, method)."""
    overlap_map = {}

    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get('Status', '').strip()
            if 'success' not in status.lower():
                continue

            molecule = row.get('Molecule', '').strip()
            environment = row.get('Environment', '').strip()
            transition = row.get('Transition', '').strip()
            method = row.get('Method', '').strip()

            try:
                overlap = float(row.get('Overlap_1_2', ''))
            except (ValueError, TypeError):
                continue

            key = (molecule, environment, method)
            if key not in overlap_map:
                overlap_map[key] = {}

            if transition == 'S0S1':
                overlap_map[key]['S1_overlap'] = overlap
            elif transition == 'S0T1':
                overlap_map[key]['T1_overlap'] = overlap

    return overlap_map


def load_stda_features(path):
    """Load sTDA features keyed by (molecule, environment, method)."""
    stda_map = {}

    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row.get('molecule', '').strip(),
                row.get('environment', '').strip(),
                row.get('method', '').strip(),
            )

            stda_map[key] = {
                'S1_energy_eV': row.get('S1_energy_eV'),
                'T1_energy_eV': row.get('T1_energy_eV'),
                'Delta_E_ST_eV': row.get('Delta_E_ST_eV'),
                'S1_osc_strength': row.get('S1_osc_strength'),
                'HOMO_LUMO_gap_eV': row.get('HOMO_LUMO_gap_eV'),
            }

    return stda_map


def build_combined_features(overlap_map, stda_map):
    """Combine NTO overlaps and sTDA features into a single table."""
    # Get all unique keys
    all_keys = set(overlap_map.keys()) | set(stda_map.keys())

    rows = []
    for key in sorted(all_keys):
        molecule, environment, method = key

        overlaps = overlap_map.get(key, {})
        stda = stda_map.get(key, {})

        # Skip if no overlap data
        if 'S1_overlap' not in overlaps or 'T1_overlap' not in overlaps:
            continue

        # Skip if no sTDA data
        if not stda.get('Delta_E_ST_eV'):
            continue

        row = {
            'molecule': molecule,
            'environment': environment,
            'method': method,
            'S1_overlap': overlaps.get('S1_overlap', ''),
            'T1_overlap': overlaps.get('T1_overlap', ''),
            'S1_energy_eV': stda.get('S1_energy_eV', ''),
            'T1_energy_eV': stda.get('T1_energy_eV', ''),
            'Delta_E_ST_eV': stda.get('Delta_E_ST_eV', ''),
            'S1_osc_strength': stda.get('S1_osc_strength', ''),
            'HOMO_LUMO_gap_eV': stda.get('HOMO_LUMO_gap_eV', ''),
        }

        rows.append(row)

    return rows


def main():
    print(f"Loading NTO overlaps from: {NTO_OVERLAP_FILE}")
    overlap_map = load_nto_overlaps(NTO_OVERLAP_FILE)
    print(f"  Loaded overlaps for {len(overlap_map)} (mol, env, method) combinations")

    print(f"\nLoading sTDA features from: {STDA_FEATURES_FILE}")
    stda_map = load_stda_features(STDA_FEATURES_FILE)
    print(f"  Loaded sTDA features for {len(stda_map)} entries")

    print("\nBuilding combined feature table...")
    rows = build_combined_features(overlap_map, stda_map)
    print(f"  Built {len(rows)} complete feature rows")

    # Write output
    fieldnames = [
        'molecule', 'environment', 'method',
        'S1_overlap', 'T1_overlap',
        'S1_energy_eV', 'T1_energy_eV', 'Delta_E_ST_eV',
        'S1_osc_strength', 'HOMO_LUMO_gap_eV'
    ]

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOutput saved to: {OUTPUT_FILE}")

    # Summary statistics
    print("\n=== Dataset Summary ===")
    molecules = set(r['molecule'] for r in rows)
    print(f"Unique molecules: {len(molecules)}")
    print(f"Total rows: {len(rows)}")
    print(f"  - Gas: {sum(1 for r in rows if r['environment'] == 'gas')}")
    print(f"  - Toluene: {sum(1 for r in rows if r['environment'] == 'toluene')}")
    print(f"  - sTDA: {sum(1 for r in rows if r['method'] == 'stda')}")
    print(f"  - sTD-DFT: {sum(1 for r in rows if r['method'] == 'stddft')}")


if __name__ == "__main__":
    main()
