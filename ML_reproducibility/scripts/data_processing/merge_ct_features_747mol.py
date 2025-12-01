#!/usr/bin/env python3
"""
Merge CT descriptors with existing combined features for 747mol dataset.

This script:
1. Pivots CT descriptors from transition (S0S1/S0T1) format to S1/T1 columns
2. Merges with existing combined_features_747mol_with_respa.csv
3. Creates enhanced feature set with CT descriptors

Author: Generated for Article 3 ML pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

def pivot_ct_descriptors(ct_df):
    """Pivot CT descriptors from row-per-transition to columns."""

    # Separate S1 and T1 data
    s1_df = ct_df[ct_df['transition'] == 'S0S1'].copy()
    t1_df = ct_df[ct_df['transition'] == 'S0T1'].copy()

    # Rename columns for S1
    s1_cols = {col: f'S1_{col}' for col in ['CT_number', 'Lambda_D', 'Lambda_A',
                                             'hole_on_A', 'particle_on_D', 'Delta_r', 'S_he']}
    s1_df = s1_df.rename(columns=s1_cols)
    s1_df = s1_df.drop(columns=['transition'])

    # Rename columns for T1
    t1_cols = {col: f'T1_{col}' for col in ['CT_number', 'Lambda_D', 'Lambda_A',
                                             'hole_on_A', 'particle_on_D', 'Delta_r', 'S_he']}
    t1_df = t1_df.rename(columns=t1_cols)
    t1_df = t1_df.drop(columns=['transition'])

    # Merge S1 and T1 data
    merged = pd.merge(s1_df, t1_df, on=['molecule', 'environment', 'method'], how='outer')

    return merged

def main():
    script_dir = Path(__file__).parent

    # Load CT descriptors
    ct_file = script_dir / 'ct_descriptors_747mol.csv'
    print(f"Loading CT descriptors from {ct_file}")
    ct_df = pd.read_csv(ct_file)
    print(f"  Loaded {len(ct_df)} rows")

    # Pivot CT descriptors
    print("Pivoting CT descriptors...")
    ct_pivoted = pivot_ct_descriptors(ct_df)
    print(f"  Pivoted to {len(ct_pivoted)} rows")

    # Load combined features (747 molecules version with proxy RespA)
    combined_file = script_dir / 'combined_features_747mol_with_respa.csv'
    print(f"Loading combined features from {combined_file}")
    combined_df = pd.read_csv(combined_file)
    print(f"  Loaded {len(combined_df)} rows")

    # Merge CT descriptors with combined features
    print("Merging datasets...")
    merged_df = pd.merge(combined_df, ct_pivoted,
                         on=['molecule', 'environment', 'method'],
                         how='left')

    # Count how many rows got CT data
    ct_cols = [c for c in merged_df.columns if c.startswith('S1_CT') or c.startswith('T1_CT')]
    if ct_cols:
        n_with_ct = merged_df[ct_cols[0]].notna().sum()
        print(f"  {n_with_ct} rows have CT descriptors")

    # Compute derived CT features
    print("Computing derived CT features...")

    # Delta features (difference between T1 and S1)
    for base_col in ['CT_number', 'Lambda_D', 'Lambda_A', 'Delta_r', 'S_he']:
        s1_col = f'S1_{base_col}'
        t1_col = f'T1_{base_col}'
        if s1_col in merged_df.columns and t1_col in merged_df.columns:
            merged_df[f'Delta_{base_col}'] = merged_df[t1_col] - merged_df[s1_col]
            merged_df[f'Abs_Delta_{base_col}'] = np.abs(merged_df[f'Delta_{base_col}'])

    # Save enhanced features
    output_file = script_dir / 'combined_features_747mol_full_ct.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nSaved enhanced features to {output_file}")
    print(f"  Total columns: {len(merged_df.columns)}")
    print(f"  Total rows: {len(merged_df)}")

    # Print new CT columns statistics
    new_cols = [c for c in merged_df.columns if 'CT' in c or 'Lambda' in c or
                c.startswith('S1_Delta_r') or c.startswith('T1_Delta_r') or
                c.startswith('S1_S_he') or c.startswith('T1_S_he') or
                (c.startswith('Delta_') and 'S_NTO' not in c)]
    print(f"\nNew CT-related columns ({len(new_cols)}):")
    for col in sorted(new_cols):
        if col in merged_df.columns:
            valid = merged_df[col].notna()
            if valid.any():
                print(f"  {col}: mean={merged_df.loc[valid, col].mean():.4f}, "
                      f"std={merged_df.loc[valid, col].std():.4f}, n={valid.sum()}")

if __name__ == '__main__':
    main()
