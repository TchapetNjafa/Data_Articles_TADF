#!/usr/bin/env python3
"""
Extract ORCA CAM-B3LYP results and prepare validation table
Ready for when adiabatic calculations complete
"""

import pandas as pd
import json
from pathlib import Path

# Paths
ORCA_RESULTS = Path("/home/tchapet/Tmp_orca_results")
OUTPUT_DIR = Path("/home/tchapet/Post-Doc/ARTICLES DOSSIERS DES ARTICLES EN REDACTION/NOUVELS AXES DE RECHERCHE A REGARDER URGEMMENT/ARTICLES EN REDACTIONS/ARTICLE1/redaction/Article3_ML/ML-IMPROVEMENT/orca_package/results")

def extract_vertical_results():
    """Extract vertical excitation energies from existing ORCA outputs"""
    
    molecules = ["4CzIPN", "DMAC-TRZ", "DMAC-DPS", "PXZ-NAI", "TPA-APy", 
                 "BACN", "BMZ-TZ", "2CzPN", "APPT-PXZ", "2PXZP", 
                 "ACRSA", "ACRFLCN", "2CzTPE", "BBPA"]
    environments = ["gas", "toluene"]
    
    data = []
    
    for mol in molecules:
        for env in environments:
            summary_file = ORCA_RESULTS / mol / env / "summary.json"
            
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    result = json.load(f)
                
                data.append({
                    'Molecule': mol,
                    'Environment': env,
                    'S1_vertical_eV': result.get('S1_energy_eV'),
                    'T1_vertical_eV': result.get('T1_energy_eV'),
                    'Delta_EST_vertical_eV': result.get('Delta_EST_eV'),
                    'Method': 'CAM-B3LYP/def2-TZVP',
                    'Type': 'Vertical'
                })
    
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_DIR / "orca_vertical_results.csv", index=False)
    print(f"✓ Extracted {len(df)} vertical results")
    return df

def prepare_adiabatic_template():
    """Create template for adiabatic results (to be filled when calculations complete)"""
    
    molecules = ["4CzIPN", "DMAC-TRZ", "DMAC-DPS", "PXZ-NAI", "TPA-APy", 
                 "BACN", "BMZ-TZ", "2CzPN", "APPT-PXZ", "2PXZP", 
                 "ACRSA", "ACRFLCN", "2CzTPE", "BBPA"]
    environments = ["gas", "toluene"]
    
    data = []
    
    for mol in molecules:
        for env in environments:
            data.append({
                'Molecule': mol,
                'Environment': env,
                'S1_adiabatic_eV': None,
                'T1_adiabatic_eV': None,
                'Delta_EST_adiabatic_eV': None,
                'S1_relaxation_eV': None,  # S1_vertical - S1_adiabatic
                'T1_relaxation_eV': None,  # T1_vertical - T1_adiabatic
                'Method': 'CAM-B3LYP/def2-TZVP',
                'Type': 'Adiabatic',
                'Status': 'PENDING'
            })
    
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_DIR / "orca_adiabatic_template.csv", index=False)
    print(f"✓ Created template for {len(df)} adiabatic calculations")
    return df

def create_validation_table_template():
    """Create validation table comparing xTB, CAM-B3LYP, and experimental"""
    
    # Molecules with experimental data
    exp_data = {
        '4CzIPN': {'Delta_EST_exp': 0.08, 'ref': 'Uoyama et al., Nature 2012'},
        'DMAC-TRZ': {'Delta_EST_exp': 0.05, 'ref': 'Noda et al., Sci. Adv. 2018'}
    }
    
    data = []
    
    for mol, exp in exp_data.items():
        data.append({
            'Molecule': mol,
            'Delta_EST_xTB_eV': None,  # To be filled from xTB dataset
            'Delta_EST_CAM-B3LYP_vertical_eV': None,  # From vertical results
            'Delta_EST_CAM-B3LYP_adiabatic_eV': None,  # From adiabatic results
            'Delta_EST_exp_eV': exp['Delta_EST_exp'],
            'Reference': exp['ref'],
            'Error_xTB_eV': None,
            'Error_vertical_eV': None,
            'Error_adiabatic_eV': None
        })
    
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_DIR / "validation_table_template.csv", index=False)
    print(f"✓ Created validation table template for {len(df)} molecules")
    return df

def main():
    print("=" * 60)
    print("ORCA Results Extraction & Template Preparation")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract existing vertical results
    print("\n1. Extracting vertical results...")
    vertical_df = extract_vertical_results()
    
    # Create adiabatic template
    print("\n2. Creating adiabatic template...")
    adiabatic_df = prepare_adiabatic_template()
    
    # Create validation table template
    print("\n3. Creating validation table template...")
    validation_df = create_validation_table_template()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Vertical results: {len(vertical_df)} configurations")
    print(f"Adiabatic template: {len(adiabatic_df)} configurations (PENDING)")
    print(f"Validation template: {len(validation_df)} molecules")
    print(f"\nFiles created in: {OUTPUT_DIR}")
    print("  - orca_vertical_results.csv")
    print("  - orca_adiabatic_template.csv")
    print("  - validation_table_template.csv")
    print("\nReady for adiabatic calculations to complete!")

if __name__ == "__main__":
    main()
