#!/usr/bin/env python3
"""
Compute charge-transfer descriptors from NTO molden files for 747mol dataset.

This script processes the larger 747mol dataset with parallel processing.
Computes RespA-like CT descriptors from NTO hole/particle orbitals.

Descriptors computed:
- CT_number: Charge transfer number (fraction of electron transferred from D to A)
- Lambda_D: Hole localization on donor fragment
- Lambda_A: Particle localization on acceptor fragment
- Delta_r: Charge transfer distance (hole-particle centroid distance)
- S_he: Hole-electron overlap integral

Author: Generated for Article 3 ML pipeline
"""

import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import sys

def parse_molden_atoms(filepath: str) -> List[Tuple[str, np.ndarray]]:
    """Parse atom coordinates from molden file."""
    atoms = []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception:
        return atoms

    # Find [Atoms] section
    atoms_match = re.search(r'\[Atoms\]\s*(Angs|AU)\s*\n(.*?)(?=\[|\Z)', content, re.DOTALL | re.IGNORECASE)
    if not atoms_match:
        return atoms

    unit = atoms_match.group(1).lower()
    atoms_section = atoms_match.group(2)

    # Parse each atom line: Element Index AtomicNum X Y Z
    for line in atoms_section.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 6:
            element = parts[0]
            try:
                coords = np.array([float(parts[3]), float(parts[4]), float(parts[5])])
                if unit == 'au':
                    coords *= 0.529177  # Convert to Angstrom
                atoms.append((element, coords))
            except ValueError:
                continue

    return atoms

def parse_molden_orbitals(filepath: str) -> List[Dict]:
    """Parse orbital information from molden file."""
    orbitals = []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception:
        return orbitals

    # Find [MO] section
    mo_match = re.search(r'\[MO\]\s*\n(.*?)(?=\[|\Z)', content, re.DOTALL | re.IGNORECASE)
    if not mo_match:
        return orbitals

    mo_section = mo_match.group(1)

    # Split by orbital headers (Sym=, Ene=, etc.)
    orbital_blocks = re.split(r'(?=Sym=)', mo_section)

    for block in orbital_blocks:
        if not block.strip():
            continue

        orbital = {}

        # Parse orbital properties
        sym_match = re.search(r'Sym=\s*(\S+)', block)
        ene_match = re.search(r'Ene=\s*([\d.E+-]+)', block)
        spin_match = re.search(r'Spin=\s*(\S+)', block)
        occup_match = re.search(r'Occup=\s*([\d.E+-]+)', block)

        if sym_match:
            orbital['sym'] = sym_match.group(1)
        if ene_match:
            try:
                orbital['energy'] = float(ene_match.group(1))
            except ValueError:
                pass
        if spin_match:
            orbital['spin'] = spin_match.group(1)
        if occup_match:
            try:
                orbital['occup'] = float(occup_match.group(1))
            except ValueError:
                pass

        # Parse coefficients
        coeffs = []
        for line in block.split('\n'):
            coeff_match = re.match(r'\s*(\d+)\s+([\d.E+-]+)', line)
            if coeff_match:
                try:
                    idx = int(coeff_match.group(1))
                    coeff = float(coeff_match.group(2))
                    coeffs.append((idx, coeff))
                except ValueError:
                    continue

        if coeffs:
            orbital['coefficients'] = coeffs
            orbitals.append(orbital)

    return orbitals

def compute_atomic_populations(orbital: Dict, n_atoms: int, basis_per_atom: int = 7) -> np.ndarray:
    """
    Compute approximate atomic populations from orbital coefficients.
    Simplified Mulliken-like analysis assuming equal basis functions per atom.
    """
    coeffs = orbital.get('coefficients', [])
    if not coeffs:
        return np.zeros(n_atoms)

    # Create coefficient array
    max_idx = max(idx for idx, _ in coeffs)
    coeff_array = np.zeros(max_idx + 1)
    for idx, coeff in coeffs:
        coeff_array[idx] = coeff

    # Sum squared coefficients per atom (simplified Mulliken)
    populations = np.zeros(n_atoms)
    for atom_idx in range(n_atoms):
        start_basis = atom_idx * basis_per_atom + 1
        end_basis = min(start_basis + basis_per_atom, len(coeff_array))
        if start_basis < len(coeff_array):
            populations[atom_idx] = np.sum(coeff_array[start_basis:end_basis]**2)

    # Normalize
    total = np.sum(populations)
    if total > 0:
        populations /= total

    return populations

def compute_ct_descriptors_from_nto(nto_file: str, atoms: List[Tuple[str, np.ndarray]],
                                    fragment_D: List[int], fragment_A: List[int]) -> Dict:
    """
    Compute CT descriptors from NTO molden file.
    """
    orbitals = parse_molden_orbitals(nto_file)
    n_atoms = len(atoms)

    # Find hole and particle orbitals
    hole_orb = None
    particle_orb = None

    for orb in orbitals:
        sym = orb.get('sym', '').lower()
        if 'hole' in sym:
            hole_orb = orb
        elif 'particle' in sym or 'electron' in sym:
            particle_orb = orb

    if hole_orb is None or particle_orb is None:
        # Fall back to first two orbitals
        if len(orbitals) >= 2:
            hole_orb = orbitals[0]
            particle_orb = orbitals[1]
        else:
            return {'error': 'Could not identify hole/particle orbitals'}

    # Compute atomic populations
    hole_pop = compute_atomic_populations(hole_orb, n_atoms)
    particle_pop = compute_atomic_populations(particle_orb, n_atoms)

    # Convert fragment indices to 0-based
    D_idx = [i - 1 for i in fragment_D if i - 1 < n_atoms]
    A_idx = [i - 1 for i in fragment_A if i - 1 < n_atoms]

    # Compute fragment populations
    Lambda_D = np.sum(hole_pop[D_idx]) if D_idx else 0.0  # Hole on donor
    Lambda_A = np.sum(particle_pop[A_idx]) if A_idx else 0.0  # Particle on acceptor

    hole_on_A = np.sum(hole_pop[A_idx]) if A_idx else 0.0
    particle_on_D = np.sum(particle_pop[D_idx]) if D_idx else 0.0

    # CT number: electron transfer from D to A
    CT_number = Lambda_D * Lambda_A

    # Compute centroids
    coords = np.array([atom[1] for atom in atoms])

    hole_centroid = np.average(coords, weights=hole_pop + 1e-10, axis=0)
    particle_centroid = np.average(coords, weights=particle_pop + 1e-10, axis=0)

    # Delta_r: hole-particle distance
    Delta_r = np.linalg.norm(particle_centroid - hole_centroid)

    # S_he: hole-electron overlap (simplified - product of populations)
    S_he = np.sum(np.sqrt(hole_pop * particle_pop))

    return {
        'CT_number': CT_number,
        'Lambda_D': Lambda_D,
        'Lambda_A': Lambda_A,
        'hole_on_A': hole_on_A,
        'particle_on_D': particle_on_D,
        'Delta_r': Delta_r,
        'S_he': S_he,
    }

def process_single_nto(args: Tuple[Path, str, str, str, str]) -> Optional[Dict]:
    """Process a single NTO file (for parallel execution)."""
    nto_dir, molecule, environment, method, transition = args

    nto_file = nto_dir / "nto001.molden"
    if not nto_file.exists():
        # Try to find any nto file
        nto_files = list(nto_dir.glob("nto*.molden"))
        if nto_files:
            nto_file = nto_files[0]
        else:
            return None

    # Parse atoms
    atoms = parse_molden_atoms(str(nto_file))
    if not atoms:
        return None

    # Default fragment definition: first half is D, second half is A
    n_atoms = len(atoms)
    mid = n_atoms // 2
    fragment_D = list(range(1, mid + 1))
    fragment_A = list(range(mid + 1, n_atoms + 1))

    # Compute descriptors
    try:
        descriptors = compute_ct_descriptors_from_nto(str(nto_file), atoms, fragment_D, fragment_A)
        if 'error' in descriptors:
            return None
        descriptors['molecule'] = molecule
        descriptors['environment'] = environment
        descriptors['method'] = method
        descriptors['transition'] = transition
        return descriptors
    except Exception:
        return None

def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description='Compute CT descriptors from NTO files (747mol dataset)')
    parser.add_argument('--data-dir', type=str,
                        default='../../Result_article1_TADF_xTB/Data_calculation_747Mol',
                        help='Path to Data_calculation_747Mol directory')
    parser.add_argument('--output', '-o', type=str,
                        default='ct_descriptors_747mol.csv',
                        help='Output CSV file')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='Number of parallel workers')
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_path = script_dir / args.output

    print(f"Data directory: {data_dir}")

    # Collect all NTO directories
    nto_tasks = []
    for env_dir in ['gas', 'toluene']:
        env_path = data_dir / env_dir
        if not env_path.exists():
            continue

        for mol_dir in env_path.iterdir():
            if not mol_dir.is_dir():
                continue
            molecule = mol_dir.name

            for method in ['stda', 'stddft']:
                for transition in ['S0S1', 'S0T1']:
                    nto_dir = mol_dir / f"{molecule}_{env_dir}_{transition}_NTO_{method}"
                    if nto_dir.exists():
                        nto_tasks.append((nto_dir, molecule, env_dir, method, transition))

    print(f"Found {len(nto_tasks)} NTO directories to process")

    # Process in parallel
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_nto, task): task for task in nto_tasks}

        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                results.append(result)

            if completed % 500 == 0:
                print(f"  Processed {completed}/{len(nto_tasks)} ({len(results)} successful)")

    print(f"\nCompleted: {len(results)} successful out of {len(nto_tasks)}")

    if results:
        # Create DataFrame
        df = pd.DataFrame(results)

        # Select columns for output
        numeric_cols = ['CT_number', 'Lambda_D', 'Lambda_A', 'hole_on_A',
                       'particle_on_D', 'Delta_r', 'S_he']
        id_cols = ['molecule', 'environment', 'method', 'transition']

        df_out = df[id_cols + [c for c in numeric_cols if c in df.columns]]

        # Save
        df_out.to_csv(output_path, index=False)
        print(f"\nSaved {len(df_out)} rows to {output_path}")

        # Print summary
        print("\nCT Descriptor Statistics:")
        for col in numeric_cols:
            if col in df_out.columns:
                print(f"  {col}: mean={df_out[col].mean():.4f}, std={df_out[col].std():.4f}")
    else:
        print("No results to save")

if __name__ == '__main__':
    main()
