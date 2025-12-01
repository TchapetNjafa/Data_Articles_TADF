#!/usr/bin/env python3
"""
Identify top TADF candidates for different applications from our 747-molecule dataset.

Applications:
1. Biomedical imaging: NIR emission (low S1 energy, high PLQY proxy)
2. Photocatalysis: Long-lived triplets (low Delta_E_ST, moderate S_he)
3. Photodetection: Fast k_RISC (low Delta_E_ST, high T1_S_he)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Load combined features dataset."""
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / 'data_processing' / 'combined_features_747mol_full_ct.csv'
    return pd.read_csv(data_path)


def identify_nir_candidates(df, top_n=5):
    """
    NIR emission candidates: Low S1 energy (<2.5 eV, λ > 500 nm)
    Additional criteria: High oscillator strength (good emission)
    """
    print("\n" + "="*60)
    print("BIOMEDICAL IMAGING (NIR) CANDIDATES")
    print("="*60)
    print("Criteria: Low S1_energy_eV (<2.5 eV), High S1_osc_strength")

    # Filter for toluene (biologically relevant solvent mimic)
    nir_df = df[df['environment'] == 'toluene'].copy()

    # Low S1 energy and good oscillator strength
    nir_df = nir_df[nir_df['S1_energy_eV'] < 2.5]
    nir_df = nir_df[nir_df['S1_osc_strength'] > 0.05]

    # Sort by S1 energy (lower = redder emission)
    nir_df = nir_df.sort_values('S1_energy_eV').head(top_n)

    print(f"\nTop {top_n} candidates:")
    for i, (_, row) in enumerate(nir_df.iterrows(), 1):
        lambda_nm = 1239.8 / row['S1_energy_eV']  # Energy to wavelength
        print(f"{i}. {row['molecule']}")
        print(f"   S1 = {row['S1_energy_eV']:.3f} eV (λ ≈ {lambda_nm:.0f} nm)")
        print(f"   f = {row['S1_osc_strength']:.4f}")
        print(f"   ΔE_ST = {row['Delta_E_ST_eV']:.3f} eV")

    return nir_df


def identify_photocatalysis_candidates(df, top_n=5):
    """
    Photocatalysis candidates: Long-lived triplets for bimolecular reactions
    Criteria: Low Delta_E_ST, moderate energy for visible absorption
    """
    print("\n" + "="*60)
    print("PHOTOCATALYSIS CANDIDATES")
    print("="*60)
    print("Criteria: Low Delta_E_ST (<0.15 eV), Visible absorption (2.0-3.5 eV)")

    cat_df = df[df['environment'] == 'toluene'].copy()

    # Low singlet-triplet gap
    cat_df = cat_df[cat_df['Delta_E_ST_eV'] < 0.15]
    # Visible absorption range
    cat_df = cat_df[(cat_df['S1_energy_eV'] > 2.0) & (cat_df['S1_energy_eV'] < 3.5)]

    # Sort by Delta_E_ST
    cat_df = cat_df.sort_values('Delta_E_ST_eV').head(top_n)

    print(f"\nTop {top_n} candidates:")
    for i, (_, row) in enumerate(cat_df.iterrows(), 1):
        print(f"{i}. {row['molecule']}")
        print(f"   ΔE_ST = {row['Delta_E_ST_eV']:.3f} eV")
        print(f"   S1 = {row['S1_energy_eV']:.3f} eV, T1 = {row['T1_energy_eV']:.3f} eV")
        print(f"   T1_S_he = {row['T1_S_he']:.3f}")

    return cat_df


def identify_photodetection_candidates(df, top_n=5):
    """
    Photodetection candidates: Fast k_RISC
    Criteria: Low Delta_E_ST AND high T1_S_he (strong exchange -> high SOC)
    k_RISC ∝ |<S1|H_SOC|T1>|² × exp(-ΔE_ST/kT)
    """
    print("\n" + "="*60)
    print("PHOTODETECTION CANDIDATES")
    print("="*60)
    print("Criteria: Low Delta_E_ST AND high T1_S_he (fast k_RISC)")

    det_df = df[df['environment'] == 'toluene'].copy()

    # Low Delta_E_ST and high T1_S_he
    det_df = det_df[det_df['Delta_E_ST_eV'] < 0.20]
    det_df = det_df[det_df['T1_S_he'] > 0.4]

    # Create k_RISC proxy score (higher is better)
    # Score = T1_S_he / Delta_E_ST (normalized)
    det_df['k_RISC_proxy'] = det_df['T1_S_he'] / (det_df['Delta_E_ST_eV'] + 0.01)

    # Sort by k_RISC proxy
    det_df = det_df.sort_values('k_RISC_proxy', ascending=False).head(top_n)

    print(f"\nTop {top_n} candidates:")
    for i, (_, row) in enumerate(det_df.iterrows(), 1):
        print(f"{i}. {row['molecule']}")
        print(f"   ΔE_ST = {row['Delta_E_ST_eV']:.3f} eV")
        print(f"   T1_S_he = {row['T1_S_he']:.3f}")
        print(f"   k_RISC proxy score = {row['k_RISC_proxy']:.2f}")

    return det_df


def main():
    print("Loading dataset...")
    df = load_data()
    print(f"Loaded {len(df)} samples from {df['molecule'].nunique()} molecules")

    # Identify candidates
    nir_candidates = identify_nir_candidates(df)
    cat_candidates = identify_photocatalysis_candidates(df)
    det_candidates = identify_photodetection_candidates(df)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: TOP CANDIDATES FOR EACH APPLICATION")
    print("="*60)

    if len(nir_candidates) > 0:
        top_nir = nir_candidates.iloc[0]
        print(f"\nBiomedical imaging: {top_nir['molecule']}")
        print(f"  λ_em ≈ {1239.8/top_nir['S1_energy_eV']:.0f} nm")

    if len(cat_candidates) > 0:
        top_cat = cat_candidates.iloc[0]
        print(f"\nPhotocatalysis: {top_cat['molecule']}")
        print(f"  ΔE_ST = {top_cat['Delta_E_ST_eV']:.3f} eV")

    if len(det_candidates) > 0:
        top_det = det_candidates.iloc[0]
        print(f"\nPhotodetection: {top_det['molecule']}")
        print(f"  ΔE_ST = {top_det['Delta_E_ST_eV']:.3f} eV, T1_S_he = {top_det['T1_S_he']:.3f}")

    return nir_candidates, cat_candidates, det_candidates


if __name__ == '__main__':
    main()
