#!/usr/bin/env python3
"""
Utility functions for PySCF-based HLT validation calculations.
Includes checkpoint management, HDF5 storage, and helper functions.

Author: Generated for Article3 HLT validation
Date: 2025
"""

import os
import h5py
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage checkpoints and restart capability for calculations."""
    
    def __init__(self, checkpoint_dir: str = "./HLT_calculations/pyscf_temp"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, molecule_name: str, calc_type: str, data: Dict[str, Any]):
        """Save calculation checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{molecule_name}_{calc_type}_checkpoint.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        
    def load_checkpoint(self, molecule_name: str, calc_type: str) -> Optional[Dict[str, Any]]:
        """Load calculation checkpoint if exists."""
        checkpoint_file = self.checkpoint_dir / f"{molecule_name}_{calc_type}_checkpoint.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint_file}")
            return data
        return None
    
    def checkpoint_exists(self, molecule_name: str, calc_type: str) -> bool:
        """Check if checkpoint exists."""
        checkpoint_file = self.checkpoint_dir / f"{molecule_name}_{calc_type}_checkpoint.pkl"
        return checkpoint_file.exists()
    
    def clear_checkpoint(self, molecule_name: str, calc_type: str):
        """Remove checkpoint file."""
        checkpoint_file = self.checkpoint_dir / f"{molecule_name}_{calc_type}_checkpoint.pkl"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info(f"Checkpoint cleared: {checkpoint_file}")


class HDF5ResultsManager:
    """Manage HDF5 database for storing calculation results."""
    
    def __init__(self, hdf5_file: str = "./HLT_calculations/hlt_results.h5"):
        self.hdf5_file = Path(hdf5_file)
        self.hdf5_file.parent.mkdir(parents=True, exist_ok=True)
        
    def save_result(self, molecule_name: str, calc_type: str, results: Dict[str, Any]):
        """Save calculation results to HDF5."""
        with h5py.File(self.hdf5_file, 'a') as f:
            # Create group for molecule if doesn't exist
            if molecule_name not in f:
                mol_group = f.create_group(molecule_name)
            else:
                mol_group = f[molecule_name]
            
            # Create/update dataset for calculation type
            if calc_type in mol_group:
                del mol_group[calc_type]
            
            calc_group = mol_group.create_group(calc_type)
            
            # Store all results
            for key, value in results.items():
                if isinstance(value, (int, float, np.ndarray)):
                    calc_group.create_dataset(key, data=value)
                elif isinstance(value, str):
                    calc_group.attrs[key] = value
                elif isinstance(value, dict):
                    # Store nested dict as attributes
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            calc_group.attrs[f"{key}_{subkey}"] = subvalue
                        elif isinstance(subvalue, str):
                            calc_group.attrs[f"{key}_{subkey}"] = subvalue
            
            # Add timestamp
            calc_group.attrs['timestamp'] = datetime.now().isoformat()
            
        logger.info(f"Results saved to HDF5: {molecule_name}/{calc_type}")
    
    def load_result(self, molecule_name: str, calc_type: str) -> Optional[Dict[str, Any]]:
        """Load calculation results from HDF5."""
        if not self.hdf5_file.exists():
            return None
            
        with h5py.File(self.hdf5_file, 'r') as f:
            if molecule_name not in f or calc_type not in f[molecule_name]:
                return None
            
            calc_group = f[molecule_name][calc_type]
            results = {}
            
            # Load datasets
            for key in calc_group.keys():
                results[key] = calc_group[key][()]
            
            # Load attributes
            for key, value in calc_group.attrs.items():
                results[key] = value
            
        logger.info(f"Results loaded from HDF5: {molecule_name}/{calc_type}")
        return results
    
    def result_exists(self, molecule_name: str, calc_type: str) -> bool:
        """Check if result exists in HDF5."""
        if not self.hdf5_file.exists():
            return False
            
        with h5py.File(self.hdf5_file, 'r') as f:
            return molecule_name in f and calc_type in f[molecule_name]
    
    def list_molecules(self) -> list:
        """List all molecules in database."""
        if not self.hdf5_file.exists():
            return []
            
        with h5py.File(self.hdf5_file, 'r') as f:
            return list(f.keys())
    
    def list_calculations(self, molecule_name: str) -> list:
        """List all calculations for a molecule."""
        if not self.hdf5_file.exists():
            return []
            
        with h5py.File(self.hdf5_file, 'r') as f:
            if molecule_name not in f:
                return []
            return list(f[molecule_name].keys())


def read_xyz_file(xyz_file: str) -> Tuple[list, np.ndarray]:
    """
    Read XYZ file and return atom symbols and coordinates.
    
    Args:
        xyz_file: Path to XYZ file
        
    Returns:
        Tuple of (atom_symbols, coordinates_in_angstrom)
    """
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    # Skip first two lines (number of atoms and comment)
    n_atoms = int(lines[0].strip())
    atom_lines = lines[2:2+n_atoms]
    
    atoms = []
    coords = []
    
    for line in atom_lines:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return atoms, np.array(coords)


def xyz_to_pyscf_format(atoms: list, coords: np.ndarray) -> str:
    """
    Convert atom list and coordinates to PySCF format string.
    
    Args:
        atoms: List of atom symbols
        coords: Coordinates in Angstrom (N x 3 array)
        
    Returns:
        PySCF format string
    """
    pyscf_str = ""
    for atom, coord in zip(atoms, coords):
        pyscf_str += f"{atom} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}; "
    return pyscf_str.rstrip("; ")


def hartree_to_ev(energy_hartree: float) -> float:
    """Convert energy from Hartree to eV."""
    return energy_hartree * 27.211386245988  # CODATA 2018


def ev_to_hartree(energy_ev: float) -> float:
    """Convert energy from eV to Hartree."""
    return energy_ev / 27.211386245988


def bohr_to_angstrom(distance_bohr: float) -> float:
    """Convert distance from Bohr to Angstrom."""
    return distance_bohr * 0.529177210903


def angstrom_to_bohr(distance_angstrom: float) -> float:
    """Convert distance from Angstrom to Bohr."""
    return distance_angstrom / 0.529177210903

def setup_pyscf_molecule(atoms: list, coords: np.ndarray, charge: int = 0, 
                         spin: int = 0, basis: str = 'def2-svp'):
    """
    Setup PySCF molecule object.
    
    Args:
        atoms: List of atom symbols
        coords: Coordinates in Angstrom
        charge: Molecular charge
        spin: Spin multiplicity - 1 (0 for singlet, 1 for doublet, 2 for triplet)
        basis: Basis set name
        
    Returns:
        PySCF Mole object
    """
    from pyscf import gto
    from pathlib import Path
    import os
    
    mol = gto.Mole()
    mol.atom = xyz_to_pyscf_format(atoms, coords)
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.lindep = 1e-7  # ← ADD THIS LINE! It will remove functions with eigenvalues below this threshold (The leading minor of order 65 of B is not positive definite)
    # Default is 1e-14, too strict for difficult cases
    
    # Set temporary directory to persist through sleep/wake cycles
    # Check if PYSCF_TMPDIR is set in environment, otherwise use default
    if 'PYSCF_TMPDIR' in os.environ:
        mol.tmpdir = os.environ['PYSCF_TMPDIR']
    else:
        pyscf_tmpdir = Path.cwd() / "HLT_calculations" / "pyscf_temp"
        pyscf_tmpdir.mkdir(parents=True, exist_ok=True)
        mol.tmpdir = str(pyscf_tmpdir.absolute())
    
    # For very difficult cases, can be even more aggressive
    # mol.lindep = 1e-6
    mol.build()
    
    return mol


def get_molecule_info(molecule_name: str, solvent: str = None, data_root: str = None) -> Dict[str, str]:
    """
    Get file paths for a molecule.
    
    Args:
        molecule_name: Name of molecule (e.g., 'DMAC-DPS', 'DMAC-TRZ', '4CzIPN')
        solvent: Solvent name ('toluene' or None for gas phase)
        data_root: Root directory for data (default: auto-detect)
        
    Returns:
        Dictionary with file paths
    """
    if data_root is None:
        data_root = "/home/tchapet/Documents/GitHub/TADF/smiEmpirical-TADF/Public_Results/Result_article1_TADF_xTB/Data_calculation_747Mol"
    
    data_root = Path(data_root)
    
    # Determine solvent folder
    if solvent is None or solvent.lower() == 'gas':
        solvent_folder = "gas"
        solvent_label = "gas"
    else:
        solvent_folder = solvent.lower()
        solvent_label = solvent.lower()
    
    # Find S0 optimized geometry
    s0_xyz = data_root / solvent_folder / molecule_name / f"{molecule_name}_{solvent_label}_S0_finalOpt.xtbopt.xyz"
    
    if not s0_xyz.exists():
        # Try alternative naming
        s0_xyz = data_root / solvent_folder / molecule_name / f"{molecule_name}.xyz"
    
    # Find T1 xTB geometry (starting point for optimization)
    t1_xyz = data_root / solvent_folder / molecule_name / f"{molecule_name}_{solvent_label}_T1_finalOpt.xtbopt.xyz"
    
    info = {
        'name': molecule_name,
        'solvent': solvent_label,
        's0_geometry': str(s0_xyz) if s0_xyz.exists() else None,
        't1_geometry': str(t1_xyz) if t1_xyz.exists() else None,
        'data_dir': str(data_root / solvent_folder / molecule_name)
    }
    
    return info


def create_progress_log(log_file: str = "./HLT_calculations/pyscf_results/progress.log"):
    """Create or append to progress log."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Progress Log Entry: {datetime.now().isoformat()}\n")
        f.write(f"{'='*80}\n")
    
    return str(log_path)


def log_progress(message: str, log_file: str = "./HLT_calculations/pyscf_results/progress.log"):
    """Append message to progress log."""
    with open(log_file, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    logger.info(message)


if __name__ == "__main__":
    # Test utilities
    print("Testing PySCF utilities...")
    
    # Test checkpoint manager
    cm = CheckpointManager()
    test_data = {'energy': -100.5, 'converged': True}
    cm.save_checkpoint('test_mol', 'omega_tuning', test_data)
    loaded = cm.load_checkpoint('test_mol', 'omega_tuning')
    print(f"Checkpoint test: {loaded}")
    cm.clear_checkpoint('test_mol', 'omega_tuning')
    
    # Test HDF5 manager
    hm = HDF5ResultsManager()
    test_results = {
        'omega': 0.15,
        'energy_s0': -100.5,
        'energy_s1': -100.3,
        'delta_est': 0.2
    }
    hm.save_result('test_mol', 'omega_tuning', test_results)
    loaded_results = hm.load_result('test_mol', 'omega_tuning')
    print(f"HDF5 test: {loaded_results}")
    
    # Test molecule info
    for mol in ['DMAC-DPS', 'DMAC-TRZ', '4CzIPN']:
        print(f"\n{mol}:")
        for solvent in [None, 'toluene']:
            solvent_label = solvent if solvent else 'gas'
            info = get_molecule_info(mol, solvent=solvent)
            print(f"  {solvent_label}:")
            print(f"    S0 geometry: {info['s0_geometry']}")
            print(f"    Exists: {Path(info['s0_geometry']).exists() if info['s0_geometry'] else False}")
    
    print("\nUtilities test complete!")
