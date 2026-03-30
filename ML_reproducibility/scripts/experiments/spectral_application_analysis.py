#!/usr/bin/env python3
"""
Spectral Application Analysis for TADF Emitters

Calculates application-specific metrics:
1. Photosynthetic Photon Efficacy (PPE) for horticulture
2. Melanopic Efficacy Ratio (MER) for human-centric lighting

Uses existing wavelength data from combined_features_747mol_full_ct.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import simpson
from scipy.stats import norm

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "csv"
RESULTS_DIR = BASE_DIR / "results" / "spectral_applications"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
WAVELENGTHS = np.arange(350, 800, 1)  # nm, 1 nm resolution
FWHM_DEFAULT = 60  # nm, typical for TADF emitters


def gaussian_spectrum(peak_nm, fwhm=FWHM_DEFAULT, wavelengths=WAVELENGTHS):
    """Generate Gaussian emission spectrum"""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return norm.pdf(wavelengths, loc=peak_nm, scale=sigma)


def mccree_action_spectrum(wavelengths):
    """
    McCree (1972) photosynthetic action spectrum
    Approximation for 400-700 nm range
    """
    action = np.zeros_like(wavelengths, dtype=float)
    
    # Blue region (400-500 nm): high efficiency
    blue_mask = (wavelengths >= 400) & (wavelengths < 500)
    action[blue_mask] = 0.9 - 0.4 * (wavelengths[blue_mask] - 400) / 100
    
    # Green region (500-600 nm): low efficiency
    green_mask = (wavelengths >= 500) & (wavelengths < 600)
    action[green_mask] = 0.5 - 0.2 * np.sin(np.pi * (wavelengths[green_mask] - 500) / 100)
    
    # Red region (600-700 nm): high efficiency
    red_mask = (wavelengths >= 600) & (wavelengths <= 700)
    action[red_mask] = 0.6 + 0.3 * (wavelengths[red_mask] - 600) / 100
    
    return action


def melanopic_sensitivity(wavelengths):
    """
    CIE S026:2018 melanopic sensitivity
    Peak at 490 nm
    """
    # Simplified Gaussian approximation
    return norm.pdf(wavelengths, loc=490, scale=50)


def photopic_sensitivity(wavelengths):
    """
    CIE photopic luminosity function V(λ)
    Peak at 555 nm
    """
    return norm.pdf(wavelengths, loc=555, scale=80)


def calculate_ppe(peak_nm, fwhm=FWHM_DEFAULT):
    """
    Calculate Photosynthetic Photon Efficacy (PPE)
    
    PPE = ∫ spectrum(λ) × action(λ) dλ / ∫ spectrum(λ) dλ
    
    Returns: PPE (relative units, 0-1 scale)
    """
    spectrum = gaussian_spectrum(peak_nm, fwhm)
    action = mccree_action_spectrum(WAVELENGTHS)
    
    # Normalize
    spectrum_norm = spectrum / simpson(spectrum, WAVELENGTHS)
    
    # Weighted integral
    ppe = simpson(spectrum_norm * action, WAVELENGTHS)
    
    return ppe


def calculate_mer(peak_nm, fwhm=FWHM_DEFAULT):
    """
    Calculate Melanopic Efficacy Ratio (MER)
    
    MER = melanopic_lux / photopic_lux
    
    Returns: MER (ratio)
    """
    spectrum = gaussian_spectrum(peak_nm, fwhm)
    melanopic = melanopic_sensitivity(WAVELENGTHS)
    photopic = photopic_sensitivity(WAVELENGTHS)
    
    melanopic_lux = simpson(spectrum * melanopic, WAVELENGTHS)
    photopic_lux = simpson(spectrum * photopic, WAVELENGTHS)
    
    return melanopic_lux / photopic_lux if photopic_lux > 0 else 0


def estimate_cri(peak_nm):
    """
    Rough CRI estimate based on wavelength
    Single emitters have low CRI, but useful for ranking
    """
    if 450 <= peak_nm <= 480:  # Blue
        return 40
    elif 480 <= peak_nm <= 520:  # Cyan
        return 50
    elif 520 <= peak_nm <= 570:  # Green-yellow
        return 60
    elif 570 <= peak_nm <= 620:  # Yellow-orange
        return 70
    elif 620 <= peak_nm <= 700:  # Red
        return 50
    else:
        return 30


def main():
    print("=" * 80)
    print("SPECTRAL APPLICATION ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    
    # Load gas phase results (has wavelength data)
    gas_file = DATA_DIR / "Data_AllGas_results.csv"
    df = pd.read_csv(gas_file)
    
    print(f"   Loaded {len(df)} molecules")
    
    # Rename columns for consistency
    df = df.rename(columns={
        'Molecule_id': 'Molecule',
        '$\\underset{sTDA}{\\lambda_{PL}}$': 'lambda_S1_nm',
        '$\\underset{sTDA}{f_{12}}(S_0\\to S_1)$': 'f_S1',
        '$\\underset{sTDA}{\\Delta E_{ST}}$': 'Delta_E_ST_eV'
    })
    
    # Check required columns
    required = ['Molecule', 'lambda_S1_nm', 'f_S1', 'Delta_E_ST_eV']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"   ERROR: Missing columns: {missing}")
        print(f"   Available columns: {list(df.columns)[:10]}")
        return
    
    # Remove invalid wavelengths
    df = df[df['lambda_S1_nm'].notna()].copy()
    df = df[(df['lambda_S1_nm'] >= 350) & (df['lambda_S1_nm'] <= 800)].copy()
    print(f"   Valid wavelengths: {len(df)} molecules")
    
    # Calculate metrics
    print("\n2. Calculating spectral metrics...")
    df['PPE'] = df['lambda_S1_nm'].apply(calculate_ppe)
    df['MER'] = df['lambda_S1_nm'].apply(calculate_mer)
    df['CRI_estimate'] = df['lambda_S1_nm'].apply(estimate_cri)
    
    # Application scores
    print("\n3. Computing application scores...")
    
    # Horticulture score: PPE × f_S1 × efficiency_factor
    df['efficiency_factor'] = np.exp(-df['Delta_E_ST_eV'].abs() / 0.1)  # Favor small ΔE_ST
    df['horticulture_score'] = df['PPE'] * df['f_S1'] * df['efficiency_factor']
    
    # Human-centric score: MER × f_S1 × efficiency_factor × CRI_factor
    df['CRI_factor'] = df['CRI_estimate'] / 100
    df['human_centric_score'] = df['MER'] * df['f_S1'] * df['efficiency_factor'] * df['CRI_factor']
    
    # Identify candidates
    print("\n4. Identifying top candidates...")
    
    # Horticulture: Blue (430-450 nm) or Red (630-680 nm)
    hort_blue = df[(df['lambda_S1_nm'] >= 430) & (df['lambda_S1_nm'] <= 450)].copy()
    hort_red = df[(df['lambda_S1_nm'] >= 630) & (df['lambda_S1_nm'] <= 680)].copy()
    hort_candidates = pd.concat([hort_blue, hort_red])
    hort_candidates = hort_candidates.nlargest(20, 'horticulture_score')
    
    print(f"   Horticulture candidates: {len(hort_candidates)}")
    print(f"     Blue (430-450 nm): {len(hort_blue)}")
    print(f"     Red (630-680 nm): {len(hort_red)}")
    
    # Human-centric: Blue-cyan (460-490 nm) with small ΔE_ST
    human_candidates = df[
        (df['lambda_S1_nm'] >= 460) & 
        (df['lambda_S1_nm'] <= 490) & 
        (df['Delta_E_ST_eV'].abs() < 0.20)
    ].copy()
    human_candidates = human_candidates.nlargest(20, 'human_centric_score')
    
    print(f"   Human-centric candidates: {len(human_candidates)}")
    
    # Summary statistics
    print("\n5. Summary statistics...")
    print(f"\n   PPE range: {df['PPE'].min():.3f} - {df['PPE'].max():.3f}")
    print(f"   PPE mean: {df['PPE'].mean():.3f} ± {df['PPE'].std():.3f}")
    
    print(f"\n   MER range: {df['MER'].min():.3f} - {df['MER'].max():.3f}")
    print(f"   MER mean: {df['MER'].mean():.3f} ± {df['MER'].std():.3f}")
    
    # Save results
    print("\n6. Saving results...")
    
    # Full dataset with metrics
    output_file = RESULTS_DIR / "spectral_metrics_all.csv"
    df.to_csv(output_file, index=False)
    print(f"   Saved: {output_file}")
    
    # Horticulture candidates
    hort_file = RESULTS_DIR / "horticulture_candidates.csv"
    hort_candidates.to_csv(hort_file, index=False)
    print(f"   Saved: {hort_file}")
    
    # Human-centric candidates
    human_file = RESULTS_DIR / "human_centric_candidates.csv"
    human_candidates.to_csv(human_file, index=False)
    print(f"   Saved: {human_file}")
    
    # Summary JSON
    summary = {
        'total_molecules': len(df),
        'horticulture_candidates': len(hort_candidates),
        'horticulture_blue': len(hort_blue),
        'horticulture_red': len(hort_red),
        'human_centric_candidates': len(human_candidates),
        'ppe_mean': float(df['PPE'].mean()),
        'ppe_std': float(df['PPE'].std()),
        'ppe_min': float(df['PPE'].min()),
        'ppe_max': float(df['PPE'].max()),
        'mer_mean': float(df['MER'].mean()),
        'mer_std': float(df['MER'].std()),
        'mer_min': float(df['MER'].min()),
        'mer_max': float(df['MER'].max()),
        'top_horticulture': hort_candidates.head(10)[['Molecule', 'lambda_S1_nm', 'PPE', 'horticulture_score']].to_dict('records'),
        'top_human_centric': human_candidates.head(10)[['Molecule', 'lambda_S1_nm', 'MER', 'human_centric_score']].to_dict('records')
    }
    
    import json
    summary_file = RESULTS_DIR / "spectral_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved: {summary_file}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
