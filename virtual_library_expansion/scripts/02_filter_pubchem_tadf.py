#!/usr/bin/env python3
"""
Step 2: Filter PubChem for TADF-relevant compounds

This script processes the PubChem CID-SMILES file (~110 million compounds)
and extracts only TADF-relevant molecules based on structural features.

TADF-relevant criteria:
1. Contains donor fragments (carbazole, phenoxazine, phenothiazine, acridine, etc.)
2. Contains acceptor fragments (triazine, benzonitrile, pyridine, quinoxaline, etc.)
3. Molecular weight < 900 Da
4. ≥2 aromatic rings
5. ≥1 N/O/S heteroatom

Expected output: 5,000-20,000 TADF-relevant compounds
Estimated time: 1-2 hours (processing 110M compounds)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Set
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    RDKIT_AVAILABLE = True
except ImportError:
    print("❌ RDKit not installed. Install with: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False
    exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PUBCHEM_FILE = Path("../data/pubchem/CID-SMILES")
EXISTING_SMILES = Path("../../SMILES_molecules.csv")
OUTPUT_FILE = Path("../data/pubchem_tadf_filtered.csv")

# Molecular filters
MAX_MW = 900
MAX_ROTATABLE_BONDS = 25
MIN_AROMATIC_RINGS = 2
MIN_HETEROATOMS = 1

# Processing batch size (to avoid memory issues)
BATCH_SIZE = 100000  # Process 100k compounds at a time
MAX_OUTPUT = 20000   # Stop after finding 20k TADF compounds

# ─────────────────────────────────────────────────────────────────────────────
# SMARTS Patterns for TADF Fragments
# ─────────────────────────────────────────────────────────────────────────────

# Donor fragments (electron-rich aromatic systems)
DONOR_SMARTS = [
    'c1ccc2c(c1)[nH]c3ccccc23',           # Carbazole
    'c1ccc2c(c1)Nc3ccccc3S2',             # Phenothiazine
    'c1ccc2c(c1)Nc3ccccc3O2',             # Phenoxazine
    'c1ccc2c(c1)nc3ccccc3c2',             # Acridine
    'c1ccc(cc1)Nc2ccccc2',                # Diphenylamine
    'c1ccc(cc1)N(c2ccccc2)c3ccccc3',      # Triphenylamine
    'CN(C)c1ccccc1',                      # Dimethylaniline
]

# Acceptor fragments (electron-deficient aromatic systems)
ACCEPTOR_SMARTS = [
    'c1nc(nc(n1)*)*',                     # Triazine
    'c1ccc(cc1)C#N',                      # Benzonitrile
    'c1ccncc1',                           # Pyridine
    'c1cncnc1',                           # Pyrimidine
    'c1ccc2c(c1)nccn2',                   # Quinoxaline
    'c1ccc2c(c1)nsn2',                    # Benzothiadiazole
    'C#N',                                # Cyano group
    'c1ccc2c(c1)C(=O)c3ccccc3C2=O',      # Naphthoquinone
]

# Compile SMARTS patterns
DONOR_PATTERNS = [Chem.MolFromSmarts(s) for s in DONOR_SMARTS if Chem.MolFromSmarts(s)]
ACCEPTOR_PATTERNS = [Chem.MolFromSmarts(s) for s in ACCEPTOR_SMARTS if Chem.MolFromSmarts(s)]

print(f"✅ Loaded {len(DONOR_PATTERNS)} donor patterns")
print(f"✅ Loaded {len(ACCEPTOR_PATTERNS)} acceptor patterns")

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_smiles() -> Set[str]:
    """Load existing 747 SMILES for deduplication."""
    print(f"\n📂 Loading existing 747 SMILES for deduplication...")
    
    df = pd.read_csv(EXISTING_SMILES)
    canonical_smiles = set()
    
    for smi in df['compound.SMILES'].dropna():
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical_smiles.add(Chem.MolToSmiles(mol, canonical=True))
        except:
            continue
    
    print(f"   Existing molecules: {len(canonical_smiles):,}")
    return canonical_smiles


def has_donor_fragment(mol) -> bool:
    """Check if molecule contains at least one donor fragment."""
    return any(mol.HasSubstructMatch(pattern) for pattern in DONOR_PATTERNS)


def has_acceptor_fragment(mol) -> bool:
    """Check if molecule contains at least one acceptor fragment."""
    return any(mol.HasSubstructMatch(pattern) for pattern in ACCEPTOR_PATTERNS)


def is_tadf_relevant(smiles: str, existing_smiles: Set[str]) -> tuple:
    """
    Check if SMILES represents a TADF-relevant molecule.
    
    Returns: (is_relevant, canonical_smiles, properties_dict)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False, None, {}
        
        # Get canonical SMILES
        canonical = Chem.MolToSmiles(mol, canonical=True)
        
        # Skip if already in our 747
        if canonical in existing_smiles:
            return False, None, {}
        
        # Calculate properties
        mw = Descriptors.MolWt(mol)
        n_aromatic_rings = Lipinski.NumAromaticRings(mol)
        n_rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        n_heteroatoms = sum(1 for atom in mol.GetAtoms() 
                           if atom.GetSymbol() in ['N', 'O', 'S'])
        
        # Apply basic filters first (fast)
        if (mw > MAX_MW or 
            n_aromatic_rings < MIN_AROMATIC_RINGS or
            n_heteroatoms < MIN_HETEROATOMS or
            n_rotatable_bonds > MAX_ROTATABLE_BONDS):
            return False, None, {}
        
        # Check for D-A structure (slower, but only on filtered molecules)
        has_donor = has_donor_fragment(mol)
        has_acceptor = has_acceptor_fragment(mol)
        
        if not (has_donor and has_acceptor):
            return False, None, {}
        
        # Passed all filters!
        properties = {
            'MW': round(mw, 2),
            'n_aromatic_rings': n_aromatic_rings,
            'n_heteroatoms': n_heteroatoms,
            'n_rotatable_bonds': n_rotatable_bonds,
            'has_donor': has_donor,
            'has_acceptor': has_acceptor
        }
        
        return True, canonical, properties
        
    except Exception as e:
        return False, None, {}


def process_batch(lines: list, existing_smiles: Set[str]) -> list:
    """Process a batch of CID-SMILES lines."""
    results = []
    
    for line in lines:
        try:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            
            cid, smiles = parts
            
            is_relevant, canonical, properties = is_tadf_relevant(smiles, existing_smiles)
            
            if is_relevant:
                results.append({
                    'CID': cid,
                    'SMILES': canonical,
                    **properties
                })
                existing_smiles.add(canonical)  # Add to set to avoid duplicates
                
        except Exception as e:
            continue
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main Workflow
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("STEP 2: FILTER PUBCHEM FOR TADF-RELEVANT COMPOUNDS")
    print("=" * 80)
    
    if not RDKIT_AVAILABLE:
        print("❌ RDKit is required. Exiting.")
        return
    
    if not PUBCHEM_FILE.exists():
        print(f"❌ PubChem file not found: {PUBCHEM_FILE}")
        print("   Run download_pubchem.sh first!")
        return
    
    print(f"\n📂 PubChem file: {PUBCHEM_FILE}")
    print(f"   Size: {PUBCHEM_FILE.stat().st_size / 1e9:.2f} GB")
    
    # Load existing SMILES
    existing_smiles = load_existing_smiles()
    
    # Process PubChem file in batches
    print(f"\n🔍 Filtering PubChem compounds...")
    print(f"   Batch size: {BATCH_SIZE:,} compounds")
    print(f"   Target: {MAX_OUTPUT:,} TADF compounds")
    print(f"   This will take 1-2 hours...")
    print()
    
    all_results = []
    total_processed = 0
    batch_lines = []
    
    with open(PUBCHEM_FILE, 'r') as f:
        for i, line in enumerate(f, 1):
            batch_lines.append(line)
            
            # Process batch
            if len(batch_lines) >= BATCH_SIZE:
                results = process_batch(batch_lines, existing_smiles)
                all_results.extend(results)
                total_processed += len(batch_lines)
                batch_lines = []
                
                # Progress update
                print(f"   Processed: {total_processed:,} | Found: {len(all_results):,} TADF compounds")
                
                # Stop if we have enough
                if len(all_results) >= MAX_OUTPUT:
                    print(f"\n✅ Reached target of {MAX_OUTPUT:,} compounds. Stopping.")
                    break
        
        # Process remaining lines
        if batch_lines and len(all_results) < MAX_OUTPUT:
            results = process_batch(batch_lines, existing_smiles)
            all_results.extend(results)
            total_processed += len(batch_lines)
    
    # Save results
    print(f"\n💾 Saving results...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total PubChem compounds processed: {total_processed:,}")
    print(f"TADF-relevant compounds found:     {len(all_results):,}")
    print(f"Hit rate:                          {len(all_results)/total_processed*100:.4f}%")
    print()
    
    if len(all_results) > 0:
        print("Molecular weight distribution:")
        print(f"   Min: {df['MW'].min():.1f} Da")
        print(f"   Max: {df['MW'].max():.1f} Da")
        print(f"   Mean: {df['MW'].mean():.1f} Da")
        print()
        print("Aromatic rings distribution:")
        print(f"   Min: {df['n_aromatic_rings'].min()}")
        print(f"   Max: {df['n_aromatic_rings'].max()}")
        print(f"   Mean: {df['n_aromatic_rings'].mean():.1f}")
    
    print()
    print(f"📁 Output: {OUTPUT_FILE}")
    print()
    print("=" * 80)
    print("✅ STEP 2 COMPLETE")
    print("=" * 80)
    print()
    print(f"Total library size: {len(existing_smiles) + len(all_results):,} molecules")
    print(f"   Original 747: {len(existing_smiles):,}")
    print(f"   New from PubChem: {len(all_results):,}")
    print()
    print("Next: Run 03_create_expanded_library.py to finalize")


if __name__ == "__main__":
    main()
