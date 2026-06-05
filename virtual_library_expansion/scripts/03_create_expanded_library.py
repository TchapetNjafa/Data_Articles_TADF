#!/usr/bin/env python3
"""
Step 3: Create final expanded library

This script combines all sources and creates the final expanded library:
1. Original 747 SMILES
2. PubChem filtered TADF compounds
3. Literature curated compounds (if available)

Adds metadata and prepares for Phase 2 (cluster computation).

Expected output: Final library with 2,000-20,000 molecules
Estimated time: 10 minutes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Set
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen
    RDKIT_AVAILABLE = True
except ImportError:
    print("❌ RDKit not installed. Install with: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False
    exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

EXISTING_SMILES = Path("../../SMILES_molecules.csv")
PUBCHEM_FILTERED = Path("../data/pubchem_tadf_filtered.csv")
LITERATURE_CURATED = Path("../data/literature_validated_smiles.csv")
OUTPUT_FILE = Path("../data/expanded_library_final.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def calculate_full_properties(smiles: str) -> dict:
    """Calculate comprehensive molecular properties."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {}
        
        return {
            'MW': round(Descriptors.MolWt(mol), 2),
            'LogP': round(Crippen.MolLogP(mol), 2),
            'n_aromatic_rings': Lipinski.NumAromaticRings(mol),
            'n_heteroatoms': sum(1 for atom in mol.GetAtoms() 
                                if atom.GetSymbol() in ['N', 'O', 'S']),
            'n_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            'n_hbd': Lipinski.NumHDonors(mol),
            'n_hba': Lipinski.NumHAcceptors(mol),
            'tpsa': round(Descriptors.TPSA(mol), 2),
            'n_rings': Lipinski.RingCount(mol),
            'n_atoms': mol.GetNumAtoms()
        }
    except:
        return {}


def load_and_process_source(filepath: Path, source_name: str) -> pd.DataFrame:
    """Load a source file and standardize columns."""
    if not filepath.exists():
        print(f"   ⚠️  {source_name} not found: {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    print(f"   ✅ {source_name}: {len(df):,} molecules")
    
    # Standardize column names
    if 'SMILES' not in df.columns and 'smiles' in df.columns:
        df.rename(columns={'smiles': 'SMILES'}, inplace=True)
    
    # Add source column
    df['source'] = source_name
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main Workflow
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("STEP 3: CREATE FINAL EXPANDED LIBRARY")
    print("=" * 80)
    
    if not RDKIT_AVAILABLE:
        print("❌ RDKit is required. Exiting.")
        return
    
    # Step 1: Load all sources
    print("\n📂 Loading all sources...")
    
    # Original 747
    df_original = load_and_process_source(EXISTING_SMILES, "Original_747")
    
    # PubChem filtered
    df_pubchem = load_and_process_source(PUBCHEM_FILTERED, "PubChem_filtered")
    
    # Literature curated (optional)
    df_literature = load_and_process_source(LITERATURE_CURATED, "Literature_curated")
    
    # Step 2: Combine all sources
    print("\n🔗 Combining all sources...")
    
    all_dfs = [df for df in [df_original, df_pubchem, df_literature] if len(df) > 0]
    
    if not all_dfs:
        print("❌ No data sources found! Exiting.")
        return
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"   Combined: {len(df_combined):,} molecules")
    
    # Step 3: Deduplicate by canonical SMILES
    print("\n🔍 Deduplicating by canonical SMILES...")
    
    canonical_smiles = []
    for smi in df_combined['SMILES']:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            else:
                canonical_smiles.append(None)
        except:
            canonical_smiles.append(None)
    
    df_combined['SMILES_canonical'] = canonical_smiles
    
    # Remove invalid SMILES
    df_combined = df_combined[df_combined['SMILES_canonical'].notna()].copy()
    print(f"   After removing invalid: {len(df_combined):,}")
    
    # Remove duplicates
    df_combined.drop_duplicates(subset=['SMILES_canonical'], keep='first', inplace=True)
    print(f"   After deduplication: {len(df_combined):,}")
    
    # Step 4: Calculate full properties for all molecules
    print("\n📊 Calculating molecular properties...")
    print("   This may take a few minutes...")
    
    properties_list = []
    for i, smi in enumerate(df_combined['SMILES_canonical'], 1):
        if i % 500 == 0:
            print(f"   Progress: {i}/{len(df_combined)} ({i/len(df_combined)*100:.1f}%)")
        
        props = calculate_full_properties(smi)
        properties_list.append(props)
    
    # Add properties to dataframe
    df_props = pd.DataFrame(properties_list)
    df_final = pd.concat([df_combined.reset_index(drop=True), df_props], axis=1)
    
    # Step 5: Add molecule ID
    df_final.insert(0, 'molecule_id', [f'MOL_{i:06d}' for i in range(1, len(df_final) + 1)])
    
    # Step 6: Reorder columns
    column_order = [
        'molecule_id', 'SMILES_canonical', 'source',
        'MW', 'LogP', 'n_aromatic_rings', 'n_heteroatoms',
        'n_rotatable_bonds', 'n_hbd', 'n_hba', 'tpsa',
        'n_rings', 'n_atoms'
    ]
    
    # Keep only columns that exist
    column_order = [col for col in column_order if col in df_final.columns]
    df_final = df_final[column_order]
    
    # Step 7: Save final library
    print(f"\n💾 Saving final library...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    # Step 8: Generate statistics
    print("\n" + "=" * 80)
    print("FINAL LIBRARY STATISTICS")
    print("=" * 80)
    print()
    print(f"Total molecules: {len(df_final):,}")
    print()
    print("Source breakdown:")
    source_counts = df_final['source'].value_counts()
    for source, count in source_counts.items():
        print(f"   {source}: {count:,} ({count/len(df_final)*100:.1f}%)")
    print()
    print("Molecular properties:")
    
    # Check which columns exist and have data - using more robust approach
    try:
        if 'MW' in df_final.columns:
            mw_series = df_final['MW']
            if not mw_series.empty and mw_series.notna().any():
                mw_min = float(mw_series.min())
                mw_max = float(mw_series.max())
                mw_mean = float(mw_series.mean())
                print(f"   MW: {mw_min:.1f} - {mw_max:.1f} Da (mean: {mw_mean:.1f})")
    except Exception as e:
        print(f"   MW: Error calculating properties: {e}")
    
    try:
        if 'LogP' in df_final.columns:
            logp_series = df_final['LogP']
            if not logp_series.empty and logp_series.notna().any():
                logp_min = float(logp_series.min())
                logp_max = float(logp_series.max())
                logp_mean = float(logp_series.mean())
                print(f"   LogP: {logp_min:.1f} - {logp_max:.1f} (mean: {logp_mean:.1f})")
    except Exception as e:
        print(f"   LogP: Error calculating properties: {e}")
    
    try:
        if 'n_aromatic_rings' in df_final.columns:
            ar_series = df_final['n_aromatic_rings']
            if not ar_series.empty and ar_series.notna().any():
                ar_min = int(ar_series.min())
                ar_max = int(ar_series.max())
                ar_mean = float(ar_series.mean())
                print(f"   Aromatic rings: {ar_min} - {ar_max} (mean: {ar_mean:.1f})")
    except Exception as e:
        print(f"   Aromatic rings: Error calculating properties: {e}")
    
    try:
        if 'n_heteroatoms' in df_final.columns:
            het_series = df_final['n_heteroatoms']
            if not het_series.empty and het_series.notna().any():
                het_min = int(het_series.min())
                het_max = int(het_series.max())
                het_mean = float(het_series.mean())
                print(f"   Heteroatoms: {het_min} - {het_max} (mean: {het_mean:.1f})")
    except Exception as e:
        print(f"   Heteroatoms: Error calculating properties: {e}")
    
    print()
    print(f"📁 Output: {OUTPUT_FILE}")
    print()
    
    # Step 9: Create summary report
    summary_file = OUTPUT_FILE.parent / "library_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPANDED LIBRARY SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total molecules: {len(df_final):,}\n\n")
        f.write("Source breakdown:\n")
        for source, count in source_counts.items():
            f.write(f"   {source}: {count:,} ({count/len(df_final)*100:.1f}%)\n")
        f.write("\n")
        f.write(df_final.describe().to_string())
    
    print(f"📄 Summary report: {summary_file}")
    print()
    
    # Step 10: Check if target met
    target_new = 2000
    new_molecules = len(df_final) - len(df_original)
    
    print("=" * 80)
    print("TARGET ASSESSMENT")
    print("=" * 80)
    print(f"Original library: {len(df_original):,}")
    print(f"New molecules: {new_molecules:,}")
    print(f"Total library: {len(df_final):,}")
    print()
    
    if new_molecules >= target_new:
        print(f"✅ TARGET MET: {new_molecules:,} new molecules (target: {target_new:,})")
        print("   Ready for Phase 2 (cluster computation)")
    else:
        print(f"⚠️  TARGET NOT MET: {new_molecules:,} new molecules (target: {target_new:,})")
        print(f"   Shortfall: {target_new - new_molecules:,} molecules")
        print("   Options:")
        print("   1. Proceed with current library (still demonstrates scalability)")
        print("   2. Lower MAX_OUTPUT limit in filter script and rerun")
        print("   3. Add manual curation from literature")
    
    print()
    print("=" * 80)
    print("✅ STEP 3 COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review library_summary.txt")
    print("2. Proceed to Phase 2: Cluster computation (GFN2-xTB + sTDA)")


if __name__ == "__main__":
    main()
