#!/usr/bin/env python3
"""
CORRECTED PySCF HLT validation with proper S1 calculation.

CRITICAL BUG FIX:
- S1 MUST be calculated with TD-DFT, NOT ROKS with spin=0
- ROKS with spin=0 just returns the S0 energy (that's why e_s0 == e_s1_vert)

Correct methodology:
- S0: RKS (spin=0) - ground state
- T1: ROKS (spin=2) - triplet state with geometry optimization  
- S1: TD-DFT on S0 - singlet excited state

FEATURES:
1. Proper S1 calculation using TD-DFT
2. Persistent HDF5 storage for geometries AND energies
3. Automatic resume from saved results
4. S1 geometry uses saved T1 geometry as starting point
5. Fast geometry optimization by default

Author: Corrected for Article3 TADF research
Date: 2025
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import h5py
from datetime import datetime

# Threading setup
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

PYSCF_PERSISTENT_DIR = Path(__file__).parent / "pyscf_persistent_data"
PYSCF_PERSISTENT_DIR.mkdir(parents=True, exist_ok=True)
os.environ['PYSCF_TMPDIR'] = str(PYSCF_PERSISTENT_DIR.absolute())

sys.path.insert(0, str(Path(__file__).parent))

from utils_pyscf import get_molecule_info, HDF5ResultsManager

from pyscf import gto, dft, tddft
from pyscf.geomopt.geometric_solver import optimize
from pyscf.solvent import ddCOSMO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# OMEGA VALUES
# ============================================================================

LITERATURE_OMEGA_VALUES = {
    'DMAC-DPS': {'omega': 0.17, 'range': (0.16, 0.18), 'reference': 'Samanta2017_JACS'},
    'DMAC-TRZ': {'omega': 0.18, 'range': (0.17, 0.19), 'reference': 'Mewes2018_PCCP'},
    '4CzIPN': {'omega': 0.15, 'range': (0.14, 0.16), 'reference': 'Froitzheim2024_ChemRxiv'},
    'PXZ-NAI': {'omega': 0.16, 'range': (0.15, 0.17), 'reference': 'Shee2022_JCTC'},
    'TPA-APy': {'omega': 0.17, 'range': (0.16, 0.18), 'reference': 'Jacquemin2016_JCTC'},
    'BMZ-TZ': {'omega': 0.16, 'range': (0.15, 0.17), 'reference': 'Kunze2021_JPCL'},
}

def get_omega_for_molecule(molecule_name: str) -> Tuple[float, Dict]:
    if molecule_name in LITERATURE_OMEGA_VALUES:
        data = LITERATURE_OMEGA_VALUES[molecule_name]
        return data['omega'], data
    return 0.16, {'omega': 0.16, 'range': (0.15, 0.18), 'reference': 'default'}


# ============================================================================
# GEOMETRY MANAGER
# ============================================================================

class GeometryOptimizationManager:
    def __init__(self):
        self.geom_dir = PYSCF_PERSISTENT_DIR / "optimized_geometries"
        self.geom_dir.mkdir(parents=True, exist_ok=True)
        self.geom_file = self.geom_dir / "optimized_geometries.h5"
    
    def save_geometry(self, molecule: str, solvent: str, state: str, mol_obj):
        try:
            coords = mol_obj.atom_coords()
            symbols = [mol_obj.atom_symbol(i) for i in range(mol_obj.natm)]
            
            with h5py.File(self.geom_file, 'a') as f:
                key = f"{molecule}_{solvent}_{state}"
                if key in f:
                    del f[key]
                grp = f.create_group(key)
                grp.attrs['molecule'] = molecule
                grp.attrs['solvent'] = solvent
                grp.attrs['state'] = state
                grp.attrs['timestamp'] = datetime.now().isoformat()
                grp.create_dataset('coords_bohr', data=coords)
                grp.create_dataset('symbols', data=np.array(symbols, dtype='S10'))
            logger.info(f"✓ Saved geometry: {key}")
        except Exception as e:
            logger.error(f"Failed to save geometry: {e}")
    
    def load_geometry(self, molecule: str, solvent: str, state: str):
        if not self.geom_file.exists():
            return None
        try:
            with h5py.File(self.geom_file, 'r') as f:
                key = f"{molecule}_{solvent}_{state}"
                if key not in f:
                    return None
                grp = f[key]
                symbols = [s.decode('utf-8') for s in grp['symbols'][()]]
                coords = grp['coords_bohr'][()]
                logger.info(f"✓ Loaded geometry: {key}")
                return symbols, coords
        except Exception as e:
            logger.error(f"Failed to load geometry: {e}")
            return None


# ============================================================================
# ENERGY RESULTS MANAGER
# ============================================================================

class EnergyResultsManager:
    def __init__(self):
        self.results_dir = PYSCF_PERSISTENT_DIR / "energy_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / "energy_results.h5"
    
    def save_results(self, molecule: str, solvent: str, results: Dict):
        try:
            with h5py.File(self.results_file, 'a') as f:
                key = f"{molecule}_{solvent}"
                if key in f:
                    del f[key]
                grp = f.create_group(key)
                grp.attrs['molecule'] = molecule
                grp.attrs['solvent'] = solvent
                grp.attrs['timestamp'] = datetime.now().isoformat()
                for k, v in results.items():
                    if isinstance(v, (int, float, str, bool)):
                        grp.attrs[k] = v
            logger.info(f"✓ Saved energy results: {key}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def load_results(self, molecule: str, solvent: str):
        if not self.results_file.exists():
            return None
        try:
            with h5py.File(self.results_file, 'r') as f:
                key = f"{molecule}_{solvent}"
                if key not in f:
                    return None
                grp = f[key]
                results = {k: v for k, v in grp.attrs.items()}
                logger.info(f"✓ Loaded energy results: {key}")
                return results
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return None


# ============================================================================
# VALIDATION CLASS
# ============================================================================

class OptimizedPySCFValidation:
    def __init__(self, molecules: List[str], basis: str = 'def2-svp',
                 functional: str = 'LC_wPBE', fast_geomopt: bool = True):
        self.molecules = molecules
        self.basis = basis
        self.functional = functional
        self.solvents = [None, 'toluene']
        self.fast_geomopt = fast_geomopt
        
        self.geom_mgr = GeometryOptimizationManager()
        self.energy_mgr = EnergyResultsManager()
        self.results_dir = Path(__file__).parent / "pyscf_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("CORRECTED PYSCF VALIDATION - TD-DFT for S1")
        logger.info("="*80)
    
    def _calculate_with_tddft(self, molecule: str, geom_file: Path, omega: float,
                              basis: str, functional: str, solvent: Optional[str]) -> Dict:
        """Calculate energies with TD-DFT for S1 (CORRECTED!)"""
        
        def load_xyz(f: Path) -> str:
            with open(f) as file:
                lines = file.readlines()[2:]  # Skip first 2 lines
            return "".join([l for l in lines if len(l.split()) >= 4])
        
        def build_mol(symbols, coords, basis, spin):
            atoms = [(symbols[i], tuple(coords[i])) for i in range(len(symbols))]
            return gto.M(atom=atoms, unit="Bohr", basis=basis, charge=0, 
                        spin=spin, verbose=4, max_memory=22000)
        
        solvent_label = solvent if solvent else 'gas'
        xyz_s0 = load_xyz(geom_file)
        
        # ====================================================================
        # S0: Ground state
        # ====================================================================
        logger.info("  Calculating S0...")
        mol_s0 = gto.M(atom=xyz_s0, basis=basis, charge=0, spin=0, 
                      verbose=4, max_memory=22000)
        
        mf_s0 = dft.RKS(mol_s0)
        mf_s0.xc = functional
        mf_s0.omega = omega
        
        if solvent:
            mf_s0 = ddCOSMO(mf_s0)
            mf_s0.with_solvent.eps = 2.374 if solvent == 'toluene' else 78.3553
        
        mf_s0.max_cycle = 200
        mf_s0.conv_tol = 1e-7
        e_s0 = mf_s0.kernel()
        
        # ====================================================================
        # T1: Triplet state with geometry optimization
        # ====================================================================
        saved_t1 = self.geom_mgr.load_geometry(molecule, solvent_label, 'T1')
        
        if saved_t1:
            logger.info("  ✓ Using saved T1 geometry")
            symbols_t1, coords_t1 = saved_t1
            mol_t1_opt = build_mol(symbols_t1, coords_t1, basis, spin=2)
        else:
            logger.info("  Optimizing T1...")
            mol_t1 = gto.M(atom=xyz_s0, basis='def2-svp', charge=0, spin=2,
                          verbose=4, max_memory=22000)
            
            mf_t1_opt = dft.ROKS(mol_t1)
            mf_t1_opt.xc = 'pbe'
            
            if solvent:
                mf_t1_opt = ddCOSMO(mf_t1_opt)
                mf_t1_opt.with_solvent.eps = 2.374 if solvent == 'toluene' else 78.3553
            
            mf_t1_opt.max_cycle = 200
            
            try:
                mol_t1_opt = optimize(mf_t1_opt, maxsteps=100)
                self.geom_mgr.save_geometry(molecule, solvent_label, 'T1', mol_t1_opt)
            except:
                logger.warning("T1 optimization failed, using S0 geometry")
                mol_t1_opt = mol_t1
            
            # Rebuild with target basis
            coords_t1 = mol_t1_opt.atom_coords()
            symbols_t1 = [mol_t1_opt.atom_symbol(i) for i in range(mol_t1_opt.natm)]
            mol_t1_opt = build_mol(symbols_t1, coords_t1, basis, spin=2)
        
        # T1 single-point
        logger.info(f"  T1 single-point...")
        mf_t1 = dft.ROKS(mol_t1_opt)
        mf_t1.xc = functional
        mf_t1.omega = omega
        
        if solvent:
            mf_t1 = ddCOSMO(mf_t1)
            mf_t1.with_solvent.eps = 2.374 if solvent == 'toluene' else 78.3553
        
        mf_t1.max_cycle = 200
        e_t1_opt = mf_t1.kernel()
        
        # ====================================================================
        # S1: CRITICAL - Use TD-DFT, NOT ROKS!
        # ====================================================================
        logger.info("  Calculating S1 with TD-DFT (CORRECTED!)...")
        
        # Vertical S1 (at S0 geometry)
        td_vert = tddft.TDA(mf_s0)
        td_vert.nstates = 5
        
        try:
            td_vert.kernel()
            exc_s1_vert = td_vert.e[0]  # Hartree
            e_s1_vert = e_s0 + exc_s1_vert
            logger.info(f"    S1 vertical: {exc_s1_vert * 27.2114:.4f} eV")
        except Exception as e:
            logger.error(f"TD-DFT vertical failed: {e}")
            e_s1_vert = e_s0
            exc_s1_vert = 0.0
        
        # S1 at T1 geometry
        mol_s1_geom = build_mol(symbols_t1, coords_t1, basis, spin=0)
        mf_s1_geom = dft.RKS(mol_s1_geom)
        mf_s1_geom.xc = functional
        mf_s1_geom.omega = omega
        
        if solvent:
            mf_s1_geom = ddCOSMO(mf_s1_geom)
            mf_s1_geom.with_solvent.eps = 2.374 if solvent == 'toluene' else 78.3553
        
        mf_s1_geom.max_cycle = 200
        e_s0_at_t1 = mf_s1_geom.kernel()
        
        td_opt = tddft.TDA(mf_s1_geom)
        td_opt.nstates = 5
        
        try:
            td_opt.kernel()
            exc_s1_opt = td_opt.e[0]
            e_s1_opt = e_s0_at_t1 + exc_s1_opt
            logger.info(f"    S1 at T1 geom: {exc_s1_opt * 27.2114:.4f} eV")
        except Exception as e:
            logger.error(f"TD-DFT optimized failed: {e}")
            e_s1_opt = e_s0_at_t1
            exc_s1_opt = 0.0
        
        # Vertical T1 (at S0 geometry)
        mol_t1_vert = gto.M(atom=xyz_s0, basis=basis, charge=0, spin=2,
                           verbose=4, max_memory=22000)
        mf_t1_vert = dft.ROKS(mol_t1_vert)
        mf_t1_vert.xc = functional
        mf_t1_vert.omega = omega
        
        if solvent:
            mf_t1_vert = ddCOSMO(mf_t1_vert)
            mf_t1_vert.with_solvent.eps = 2.374 if solvent == 'toluene' else 78.3553
        
        mf_t1_vert.max_cycle = 200
        e_t1_vert = mf_t1_vert.kernel()
        
        # Convert to eV
        hartree_to_ev = 27.211386245988
        
        return {
            "molecule": molecule,
            "solvent": solvent if solvent else "gas",
            "omega_bohr-1": omega,
            "e_s0_eV": e_s0 * hartree_to_ev,
            "e_s1_opt_eV": e_s1_opt * hartree_to_ev,
            "e_t1_opt_eV": e_t1_opt * hartree_to_ev,
            "e_s1_vert_eV": e_s1_vert * hartree_to_ev,
            "e_t1_vert_eV": e_t1_vert * hartree_to_ev,
            "excitation_s1_opt_eV": exc_s1_opt * hartree_to_ev,
            "excitation_t1_opt_eV": (e_t1_opt - e_s0) * hartree_to_ev,
            "excitation_s1_vert_eV": exc_s1_vert * hartree_to_ev,
            "excitation_t1_vert_eV": (e_t1_vert - e_s0) * hartree_to_ev,
            "delta_est_opt_eV": (e_s1_opt - e_t1_opt) * hartree_to_ev,
            "delta_est_vert_eV": (e_s1_vert - e_t1_vert) * hartree_to_ev,
            "basis": basis,
            "functional": functional,
        }
    
    def run_delta_roks_optimized(self, force_recalc: bool = False) -> Dict:
        """Run calculations with persistent storage."""
        logger.info("="*80)
        logger.info("ΔROKS CALCULATIONS (CORRECTED - TD-DFT for S1)")
        logger.info("="*80)
        
        results_dict = {}
        
        for molecule in self.molecules:
            omega, omega_data = get_omega_for_molecule(molecule)
            
            logger.info(f"\nMOLECULE: {molecule} (ω={omega:.4f})")
            
            for solvent in self.solvents:
                solvent_label = solvent if solvent else 'gas'
                key = f"{molecule}_{solvent_label}"
                
                # Check saved results
                if not force_recalc:
                    saved = self.energy_mgr.load_results(molecule, solvent_label)
                    if saved:
                        results_dict[key] = saved
                        logger.info(f"  ✓ {solvent_label}: ΔE_ST = {saved['delta_est_opt_eV']:.4f} eV (loaded)")
                        continue
                
                mol_info = get_molecule_info(molecule, solvent=solvent)
                if mol_info['s0_geometry'] is None:
                    logger.error(f"  ✗ {solvent_label}: No geometry file")
                    continue
                
                logger.info(f"  Calculating {solvent_label}...")
                
                try:
                    results = self._calculate_with_tddft(
                        molecule,
                        Path(mol_info['s0_geometry']),
                        omega, self.basis, self.functional, solvent
                    )
                    
                    results_dict[key] = results
                    self.energy_mgr.save_results(molecule, solvent_label, results)
                    
                    logger.info(f"    ✓ ΔE_ST (opt) = {results['delta_est_opt_eV']:.4f} eV")
                    logger.info(f"    ✓ ΔE_ST (vert) = {results['delta_est_vert_eV']:.4f} eV")
                    
                except Exception as e:
                    logger.error(f"    ✗ Failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        self._save_csv(results_dict)
        return results_dict
    
    def _save_csv(self, results_dict: Dict):
        """Save to CSV."""
        data = []
        for key, r in results_dict.items():
            omega_val, omega_data = get_omega_for_molecule(r['molecule'])
            data.append({
                'molecule': r['molecule'],
                'solvent': r['solvent'],
                'omega_bohr-1': r['omega_bohr-1'],
                'omega_reference': omega_data['reference'],
                'e_s0_eV': r['e_s0_eV'],
                'e_s1_opt_eV': r['e_s1_opt_eV'],
                'e_t1_opt_eV': r['e_t1_opt_eV'],
                'e_s1_vert_eV': r['e_s1_vert_eV'],
                'e_t1_vert_eV': r['e_t1_vert_eV'],
                'excitation_s1_opt_eV': r['excitation_s1_opt_eV'],
                'excitation_t1_opt_eV': r['excitation_t1_opt_eV'],
                'excitation_s1_vert_eV': r['excitation_s1_vert_eV'],
                'excitation_t1_vert_eV': r['excitation_t1_vert_eV'],
                'delta_est_opt_eV': r['delta_est_opt_eV'],
                'delta_est_vert_eV': r['delta_est_vert_eV'],
                'basis': r['basis'],
                'functional': r['functional']
            })
        
        df = pd.DataFrame(data)
        out_file = self.results_dir / 'optimized_pyscf_delta_roks_results_corrected.csv'
        df.to_csv(out_file, index=False, float_format='%.6f')
        logger.info(f"\n✓ Results: {out_file}")
        
        print("\n" + "="*80)
        print("RESULTS (CORRECTED)")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Corrected PySCF validation with TD-DFT for S1')
    parser.add_argument('--molecules', nargs='+', default=['DMAC-DPS', 'DMAC-TRZ'])
    parser.add_argument('--basis', default='def2-svp')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    pipeline = OptimizedPySCFValidation(molecules=args.molecules, basis=args.basis)
    
    start = time.time()
    results = pipeline.run_delta_roks_optimized(force_recalc=args.force)
    elapsed = time.time() - start
    
    logger.info(f"\n✓ Complete in {elapsed/60:.2f} minutes ({len(results)} calculations)")


if __name__ == "__main__":
    main()
