#!/usr/bin/env python3
"""
ORCA-based TADF Calculations - CAM-B3LYP Protocol
==================================================

Replaces PySCF with ORCA for faster, more accurate calculations.
Uses CAM-B3LYP/def2-TZVP as recommended by reviewers (Hal et al. JPCA 2023).

Protocol:
  1. Ground state optimization (CAM-B3LYP/def2-TZVP with D3BJ)
  2. TD-DFT (10 singlets + 10 triplets)
  3. SOC calculation ⟨S₁|Ĥ_SOC|T₁⟩
  4. k_RISC estimation (Marcus formula, λ=0.1 eV)

Server ORCA path: /home/penavora/orca_6_1_0/orca
"""

import os
import sys
import json
import re
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ──────────────────────── Configuration ────────────────────────
ORCA_PATH = "/home/penavora/orca_6_1_0/orca"
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
INPUTS_DIR = SCRIPT_DIR / "inputs"
GEOM_DIR = SCRIPT_DIR / "xyz_files"

RESULTS_DIR.mkdir(exist_ok=True)
INPUTS_DIR.mkdir(exist_ok=True)

# Physical constants
AU_TO_EV = 27.2114
CM1_TO_EV = 1.23984e-4
HBAR_EVS = 6.582119e-16
KB_EV = 8.617333e-5

# Molecules to process
ALL_MOLECULES = [
    # New validation (9 molecules)
    "DMAC-DPS", "APPT-PXZ", "2CzPN", "2PXZP", "ACRSA", "ACRFLCN", 
    "2CzTPE", "BMZ-TZ", "BBPA",
    # Re-run existing with CAM-B3LYP for consistency (6 molecules)
    "4CzIPN", "BACN", "DMAC-TRZ", "PXZ-NAI", "TPA-APy"
]

# ──────────────────────── Input Generation ────────────────────────
def generate_tddft_input(mol_name, phase, xyz_file, nprocs=8):
    """Generate ORCA input for TD-DFT + SOC calculation
    
    Uses CAM-B3LYP/def2-TZVP with:
    - RIJCOSX approximation (fast)
    - D3BJ dispersion
    - CPCM for toluene
    - 10 singlets + 10 triplets
    - SOC calculation
    """
    
    # Solvent block
    if phase == "toluene":
        solvent_block = """
%cpcm
  epsilon 2.374
  refrac 1.497
end
"""
    else:
        solvent_block = ""
    
    # Main input
    inp = f"""# TADF calculation: {mol_name} ({phase})
# CAM-B3LYP/def2-TZVP with RIJCOSX, D3BJ, TD-DFT, SOC

%pal nprocs {nprocs} end

%maxcore 4000

! CAM-B3LYP def2-TZVP def2/J RIJCOSX D3BJ
! TightSCF SlowConv
{solvent_block}

%tddft
  nroots 10      # 10 singlets
  triplets true
  maxdim 5
  dosoc true     # Enable SOC calculation
end

* xyzfile 0 1 {xyz_file.name}
"""
    
    return inp

def generate_opt_input(mol_name, phase, xyz_file, state="S0", nprocs=8):
    """Generate ORCA input for geometry optimization
    
    For S0: singlet ground state (not used, we use xTB geometries)
    For S1: singlet excited state (root 1)
    For T1: triplet state (multiplicity=3)
    """
    
    if state == "S1":
        # Excited state optimization for S1
        mult = 1
        opt_block = """
%geom
  MaxIter 250
  TolE 5e-6
  TolRMSG 1e-4
  TolMaxG 3e-4
end

%tddft
  nroots 1
  iroot 1      # Optimize S1
  maxdim 5
end
"""
        task = "! Opt TightSCF"
    elif state == "T1":
        # Triplet ground state optimization
        mult = 3
        opt_block = """
%geom
  MaxIter 250
  TolE 5e-6
  TolRMSG 1e-4
  TolMaxG 3e-4
end
"""
        task = "! Opt TightSCF"
    else:
        # S0 ground state (not used in this workflow)
        mult = 1
        opt_block = """
%geom
  MaxIter 250
  TolE 5e-6
  TolRMSG 1e-4
  TolMaxG 3e-4
end
"""
        task = "! Opt TightSCF"
    
    if phase == "toluene":
        solvent_block = """
%cpcm
  epsilon 2.374
  refrac 1.497
end
"""
    else:
        solvent_block = ""
    
    inp = f"""# Geometry optimization: {mol_name} {state} ({phase})
# CAM-B3LYP/def2-TZVP with D3BJ

{opt_block}

%pal nprocs {nprocs} end

%maxcore 4000

! CAM-B3LYP def2-TZVP def2/J RIJCOSX D3BJ
{task}
{solvent_block}

* xyzfile 0 {mult} {xyz_file.name}
"""
    
    return inp

# ──────────────────────── Output Parsing ────────────────────────
def parse_orca_energy(out_file):
    """Extract final SCF energy from ORCA output"""
    with open(out_file) as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                return float(line.split()[-1])
    return None

def parse_optimized_geometry(out_file):
    """Extract optimized geometry from ORCA output
    
    Returns path to optimized xyz file if found, None otherwise
    """
    out_path = Path(out_file)
    # ORCA creates .xyz file with optimized geometry
    opt_xyz = out_path.parent / (out_path.stem + ".xyz")
    
    if opt_xyz.exists():
        return opt_xyz
    
    # Fallback: extract from output file
    return None

def parse_tddft_states(out_file):
    """Extract TD-DFT excitation energies and SOC matrix"""
    singlets = []
    triplets = []
    soc_matrix = None
    
    with open(out_file) as f:
        content = f.read()
    
    # Parse singlet states
    import re
    s_matches = re.findall(
        r"STATE\s+(\d+):\s+E=\s+[\d.]+\s+au\s+([\d.]+)\s+eV",
        content
    )
    if s_matches:
        for state_num, energy_ev in s_matches[:10]:
            singlets.append(float(energy_ev))
    
    # Parse triplet states
    t_section = re.search(
        r"TD-DFT/TDA EXCITED STATES \(TRIPLETS\)(.*?)(?:TD-DFT EXCITED STATES \(SINGLETS\)|ABSORPTION SPECTRUM|$)",
        content, re.DOTALL
    )
    if t_section:
        t_matches = re.findall(
            r"STATE\s+(\d+):\s+E=\s+[\d.]+\s+au\s+([\d.]+)\s+eV",
            t_section.group(1)
        )
        for state_num, energy_ev in t_matches[:10]:
            triplets.append(float(energy_ev))
    
    # Parse SOC matrix
    soc_matrix = parse_soc_matrix(content)
    
    return singlets, triplets, soc_matrix

def parse_soc_matrix(content):
    """Extract SOC matrix from ORCA output
    
    Returns dict with:
      - S1_T1_ms0, S1_T1_msm1, S1_T1_msp1 (in cm^-1)
      - total SOC
    """
    
    # Find SOC matrix section
    real_match = re.search(
        r"The full SOC matrix:.*?Real part:\s*\n(.*?)(?:Image? part:|$)",
        content, re.DOTALL
    )
    imag_match = re.search(
        r"Image? part:\s*\n(.*?)(?:Diagonalizing|$)",
        content, re.DOTALL
    )
    
    if not real_match or not imag_match:
        return None
    
    # Parse matrices (simplified - assumes 5 singlets + 15 triplet sublevels)
    # Index: S0=0, S1=1, T1_ms0=6, T1_msm1=11, T1_msp1=16
    real_matrix = _parse_matrix_block(real_match.group(1))
    imag_matrix = _parse_matrix_block(imag_match.group(1))
    
    if real_matrix is None or imag_matrix is None:
        return None
    
    # Extract S1-T1 elements
    s1_idx = 1
    t1_ms0 = 6
    t1_msm1 = 11
    t1_msp1 = 16
    
    AU_TO_CM1 = 219474.63
    
    def get_soc(i, j):
        re_val = real_matrix[i, j] if i < len(real_matrix) and j < len(real_matrix[0]) else 0
        im_val = imag_matrix[i, j] if i < len(imag_matrix) and j < len(imag_matrix[0]) else 0
        return np.sqrt(re_val**2 + im_val**2) * AU_TO_CM1
    
    soc_ms0 = get_soc(s1_idx, t1_ms0)
    soc_msm1 = get_soc(s1_idx, t1_msm1)
    soc_msp1 = get_soc(s1_idx, t1_msp1)
    
    soc_total = np.sqrt(soc_ms0**2 + soc_msm1**2 + soc_msp1**2)
    
    return {
        "SOC_ms0_cm1": soc_ms0,
        "SOC_msm1_cm1": soc_msm1,
        "SOC_msp1_cm1": soc_msp1,
        "SOC_total_cm1": soc_total
    }

def _parse_matrix_block(text):
    """Parse ORCA matrix block (6 columns per block)"""
    lines = text.strip().split('\n')
    matrix = []
    current_cols = []
    
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        
        # Column headers
        if all(p.isdigit() for p in parts):
            current_cols = [int(p) for p in parts]
            continue
        
        # Data rows
        if parts[0].isdigit():
            row_idx = int(parts[0])
            while len(matrix) <= row_idx:
                matrix.append([0.0] * 50)  # Assume max 50x50
            
            for i, val in enumerate(parts[1:]):
                if i < len(current_cols):
                    col_idx = current_cols[i]
                    try:
                        matrix[row_idx][col_idx] = float(val)
                    except ValueError:
                        pass
    
    return np.array([row[:50] for row in matrix]) if matrix else None

# ──────────────────────── Main Calculation ────────────────────────
def run_orca(input_file, output_file, nprocs=8):
    """Run ORCA calculation"""
    
    cmd = [ORCA_PATH, str(input_file)]
    
    print(f"  Running ORCA: {input_file.name}")
    start = time.time()
    
    with open(output_file, 'w') as out:
        result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, 
                               cwd=input_file.parent)
    
    elapsed = time.time() - start
    print(f"  Completed in {elapsed/60:.1f} min")
    
    if result.returncode != 0:
        print(f"  WARNING: ORCA exited with code {result.returncode}")
        return False
    
    return True

def calculate_molecule(mol_name, phase="gas", nprocs=8):
    """Run complete ADIABATIC calculation for one molecule
    
    Workflow:
      1. Optimize S1 geometry starting from S0 geometry
      2. Optimize T1 geometry starting from T1 geometry  
      3. Run TD-DFT + SOC at optimized S1 geometry
      4. Calculate adiabatic ΔE_ST = E(S1_opt) - E(T1_opt)
    """
    
    print(f"\n{'='*70}")
    print(f"  {mol_name} ({phase}) - ADIABATIC ΔE_ST")
    print(f"{'='*70}")
    
    mol_dir = RESULTS_DIR / mol_name / phase
    mol_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if adiabatic calculation already done
    summary_adiabatic_file = mol_dir / "summary_adiabatic.json"
    if summary_adiabatic_file.exists():
        print(f"  ✅ Adiabatic calculation already complete. Loading results...")
        with open(summary_adiabatic_file) as f:
            return json.load(f)
    
    # Check if vertical results exist (just for info, won't touch them)
    summary_vertical_file = mol_dir / "summary_vertical.json"
    summary_old_file = mol_dir / "summary.json"
    if summary_vertical_file.exists():
        print(f"  ℹ️  Vertical results found in: summary_vertical.json (will be preserved)")
    elif summary_old_file.exists():
        print(f"  ℹ️  Old results found in: summary.json (will be preserved)")
    
    print(f"  → Running NEW adiabatic calculation (results → summary_adiabatic.json)")
    
    # Load initial geometries
    s0_xyz = GEOM_DIR / phase / f"{mol_name}_{phase}_S0_finalOpt.xtbopt.xyz"
    t1_xyz = GEOM_DIR / phase / f"{mol_name}_{phase}_T1_finalOpt.xtbopt.xyz"
    
    if not s0_xyz.exists():
        print(f"  ERROR: S0 geometry not found: {s0_xyz}")
        return None
    
    if not t1_xyz.exists():
        print(f"  ERROR: T1 geometry not found: {t1_xyz}")
        return None
    
    # Copy geometries to inputs dir
    s0_xyz_local = INPUTS_DIR / s0_xyz.name
    t1_xyz_local = INPUTS_DIR / t1_xyz.name
    shutil.copy(s0_xyz, s0_xyz_local)
    shutil.copy(t1_xyz, t1_xyz_local)
    
    results = {
        "molecule": mol_name,
        "phase": phase,
        "functional": "CAM-B3LYP",
        "basis": "def2-TZVP",
        "dispersion": "D3BJ",
        "calculation_type": "adiabatic",
        "timestamp": datetime.now().isoformat()
    }
    
    # ═════════════════════════════════════════════════════════════
    # Step 1: Optimize S1 geometry (starting from S0)
    # ═════════════════════════════════════════════════════════════
    print("\n  Step 1: S1 geometry optimization (from S0 geometry)")
    s1_opt_out = mol_dir / f"{mol_name}_{phase}_S1_opt.out"
    s1_opt_xyz = mol_dir / f"{mol_name}_{phase}_S1_opt.xyz"
    
    if not s1_opt_out.exists():
        s1_opt_inp = generate_opt_input(mol_name, phase, s0_xyz_local, state="S1", nprocs=nprocs)
        s1_opt_inp_file = INPUTS_DIR / f"{mol_name}_{phase}_S1_opt.inp"
        
        with open(s1_opt_inp_file, 'w') as f:
            f.write(s1_opt_inp)
        
        if not run_orca(s1_opt_inp_file, s1_opt_out, nprocs):
            results["status"] = "failed"
            results["error"] = "S1 optimization failed"
            with open(summary_adiabatic_file, 'w') as f:
                json.dump(results, f, indent=2)
            return results
        
        # Extract optimized geometry
        opt_geom = parse_optimized_geometry(s1_opt_out)
        if opt_geom and opt_geom.exists():
            shutil.copy(opt_geom, s1_opt_xyz)
            print(f"  S1 optimization completed → {s1_opt_xyz.name}")
        else:
            print(f"  WARNING: Could not find optimized S1 geometry")
    else:
        print(f"  S1 optimization already done, skipping...")
    
    # ═════════════════════════════════════════════════════════════
    # Step 2: Optimize T1 geometry (starting from T1)
    # ═════════════════════════════════════════════════════════════
    print("\n  Step 2: T1 geometry optimization (from T1 geometry)")
    t1_opt_out = mol_dir / f"{mol_name}_{phase}_T1_opt.out"
    t1_opt_xyz = mol_dir / f"{mol_name}_{phase}_T1_opt.xyz"
    
    if not t1_opt_out.exists():
        t1_opt_inp = generate_opt_input(mol_name, phase, t1_xyz_local, state="T1", nprocs=nprocs)
        t1_opt_inp_file = INPUTS_DIR / f"{mol_name}_{phase}_T1_opt.inp"
        
        with open(t1_opt_inp_file, 'w') as f:
            f.write(t1_opt_inp)
        
        if not run_orca(t1_opt_inp_file, t1_opt_out, nprocs):
            results["status"] = "failed"
            results["error"] = "T1 optimization failed"
            with open(summary_adiabatic_file, 'w') as f:
                json.dump(results, f, indent=2)
            return results
        
        # Extract optimized geometry
        opt_geom = parse_optimized_geometry(t1_opt_out)
        if opt_geom and opt_geom.exists():
            shutil.copy(opt_geom, t1_opt_xyz)
            print(f"  T1 optimization completed → {t1_opt_xyz.name}")
        else:
            print(f"  WARNING: Could not find optimized T1 geometry")
    else:
        print(f"  T1 optimization already done, skipping...")
    
    # ═════════════════════════════════════════════════════════════
    # Step 3: TD-DFT + SOC at optimized S1 geometry
    # ═════════════════════════════════════════════════════════════
    print("\n  Step 3: TD-DFT + SOC at optimized S1 geometry")
    
    if not s1_opt_xyz.exists():
        results["status"] = "failed"
        results["error"] = "S1 optimized geometry not found"
        with open(summary_adiabatic_file, 'w') as f:
            json.dump(results, f, indent=2)
        return results
    
    # Copy optimized S1 geometry to inputs
    s1_opt_xyz_local = INPUTS_DIR / s1_opt_xyz.name
    shutil.copy(s1_opt_xyz, s1_opt_xyz_local)
    
    tddft_inp = generate_tddft_input(mol_name, phase, s1_opt_xyz_local, nprocs)
    tddft_inp_file = INPUTS_DIR / f"{mol_name}_{phase}_tddft_S1opt.inp"
    tddft_out_file = mol_dir / f"{mol_name}_{phase}_tddft_S1opt.out"
    
    with open(tddft_inp_file, 'w') as f:
        f.write(tddft_inp)
    
    if not run_orca(tddft_inp_file, tddft_out_file, nprocs):
        results["status"] = "failed"
        results["error"] = "TD-DFT calculation at S1 geometry failed"
        with open(summary_adiabatic_file, 'w') as f:
            json.dump(results, f, indent=2)
        return results
    
    # ═════════════════════════════════════════════════════════════
    # Step 4: Parse energies and calculate adiabatic ΔE_ST
    # ═════════════════════════════════════════════════════════════
    print("\n  Step 4: Extracting energies and calculating adiabatic ΔE_ST")
    
    # Parse S1 energy from optimization
    s1_energy_au = parse_orca_energy(s1_opt_out)
    
    # Parse T1 energy from optimization
    t1_energy_au = parse_orca_energy(t1_opt_out)
    
    # Parse TD-DFT states and SOC
    singlets, triplets, soc = parse_tddft_states(tddft_out_file)
    
    if s1_energy_au is not None and t1_energy_au is not None:
        # Calculate adiabatic ΔE_ST from optimized energies
        delta_est_au = s1_energy_au - t1_energy_au
        delta_est_ev = delta_est_au * AU_TO_EV
        
        results["S1_optimized_au"] = round(s1_energy_au, 8)
        results["T1_optimized_au"] = round(t1_energy_au, 8)
        results["Delta_EST_eV"] = round(delta_est_ev, 4)
        
        print(f"  S1 (optimized) = {s1_energy_au:.8f} au")
        print(f"  T1 (optimized) = {t1_energy_au:.8f} au")
        print(f"  ΔE_ST (adiabatic) = {delta_est_ev:.4f} eV")
        
        # Also store vertical excitation energies from TD-DFT for reference
        if singlets and triplets:
            results["S1_vertical_eV"] = round(singlets[0], 4)
            results["T1_vertical_eV"] = round(triplets[0], 4)
            results["all_singlets_eV"] = [round(x, 4) for x in singlets]
            results["all_triplets_eV"] = [round(x, 4) for x in triplets]
            print(f"  S1 (vertical at S1 geom) = {singlets[0]:.4f} eV")
            print(f"  T1 (vertical at S1 geom) = {triplets[0]:.4f} eV")
    else:
        print("  WARNING: Could not extract optimized energies")
        results["status"] = "failed"
        results["error"] = "Energy extraction failed"
    
    if soc:
        results.update(soc)
        print(f"  SOC_total = {soc['SOC_total_cm1']:.4f} cm⁻¹")
        
        # Calculate k_RISC using adiabatic ΔE_ST
        if "Delta_EST_eV" in results:
            krisc = calculate_krisc(soc['SOC_total_cm1'], abs(results["Delta_EST_eV"]))
            results["k_RISC_s-1"] = krisc
            results["log10_k_RISC"] = round(np.log10(max(krisc, 1e-30)), 2)
            print(f"  k_RISC = {krisc:.2e} s⁻¹ (log₁₀ = {results['log10_k_RISC']:.2f})")
    else:
        print("  WARNING: SOC not found in output")
    
    results["status"] = "success"
    
    # Save results to ADIABATIC file (not summary.json)
    with open(summary_adiabatic_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  ✅ Adiabatic results saved to {summary_adiabatic_file.name}")
    
    return results

def calculate_krisc(soc_cm1, delta_est_eV, lambda_eV=0.1, T_K=298.0):
    """Calculate k_RISC using Marcus formula"""
    kT = KB_EV * T_K
    soc_eV = soc_cm1 * CM1_TO_EV
    prefactor = (2 * np.pi / HBAR_EVS) * soc_eV**2 / np.sqrt(4 * np.pi * lambda_eV * kT)
    exponent = -(delta_est_eV + lambda_eV)**2 / (4 * lambda_eV * kT)
    return prefactor * np.exp(exponent)

# ──────────────────────── Main ────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ORCA TADF calculations")
    parser.add_argument("--molecules", nargs="+", default=ALL_MOLECULES,
                       help="Molecules to calculate")
    parser.add_argument("--phases", nargs="+", default=["gas", "toluene"],
                       help="Phases to calculate")
    parser.add_argument("--nprocs", type=int, default=8,
                       help="Number of processors")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: only first molecule")
    args = parser.parse_args()
    
    if args.test:
        molecules = [args.molecules[0]]
        print(f"\n*** TEST MODE: Only calculating {molecules[0]} ***\n")
    else:
        molecules = args.molecules
    
    print(f"\n{'='*70}")
    print(f"  ORCA TADF Calculations")
    print(f"  CAM-B3LYP/def2-TZVP + D3BJ + SOC")
    print(f"{'='*70}")
    print(f"  Molecules: {len(molecules)}")
    print(f"  Phases: {', '.join(args.phases)}")
    print(f"  Processors: {args.nprocs}")
    print(f"  ORCA: {ORCA_PATH}")
    print(f"{'='*70}\n")
    
    all_results = []
    
    for mol in molecules:
        for phase in args.phases:
            result = calculate_molecule(mol, phase, args.nprocs)
            if result:
                all_results.append(result)
    
    # Save summary CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_file = RESULTS_DIR / "all_results_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n{'='*70}")
        print(f"  All results saved to {csv_file}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

