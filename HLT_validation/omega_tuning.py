#!/usr/bin/env python3
"""
Optimal ω tuning for LC-ωPBE functional.

This script performs iterative optimization of the range-separation parameter ω
for LC-ωPBE functional following the IP-tuning criterion:
    J = |ε_HOMO(ω) + IP(ω)|

Where IP is calculated as E(N-1) - E(N) (vertical ionization potential).

Usage:
    python omega_tuning.py <molecule_name> <xyz_file>

Example:
    python omega_tuning.py DMAC-DPS /path/to/DMAC-DPS.xyz
"""

import os
import sys
import subprocess
import re
import numpy as np
from scipy.optimize import minimize_scalar

# Configuration
ORCA_PATH = "/home/tchapet/orca-6-1-0/bin/orca"
NPROCS = 8
MEMORY_MB = 20000
BASIS = "def2-SVP"  # Use smaller basis for tuning, def2-TZVP for production

def create_orca_input(xyz_file, omega, charge, mult, output_name, basis=BASIS):
    """Create ORCA input file for LC-ωPBE calculation."""

    # Read xyz content
    with open(xyz_file, 'r') as f:
        xyz_content = f.read()

    input_content = f"""# LC-wPBE calculation with omega = {omega:.4f}
# Charge = {charge}, Multiplicity = {mult}

! LC-PBE {basis} def2/J RIJCOSX TightSCF
! PAL{NPROCS}

%maxcore {MEMORY_MB // NPROCS}

%method
    RangeSepMu {omega:.6f}
end

%scf
    MaxIter 300
    ConvForced true
end

* xyzfile {charge} {mult} {xyz_file}
"""

    input_file = f"{output_name}.inp"
    with open(input_file, 'w') as f:
        f.write(input_content)

    return input_file


def run_orca(input_file):
    """Run ORCA calculation and return output file path."""
    output_file = input_file.replace('.inp', '.out')

    cmd = f"{ORCA_PATH} {input_file} > {output_file} 2>&1"
    subprocess.run(cmd, shell=True)

    return output_file


def extract_homo_energy(output_file):
    """Extract HOMO energy from ORCA output (in Hartree).

    Finds the last doubly occupied orbital in the ORBITAL ENERGIES section.
    """
    homo_energy = None

    with open(output_file, 'r') as f:
        lines = f.readlines()

    # Find the ORBITAL ENERGIES section and extract HOMO
    in_orbital_section = False
    for line in lines:
        if 'ORBITAL ENERGIES' in line:
            in_orbital_section = True
            continue

        if in_orbital_section:
            # Match lines like: "  94   2.0000      -0.256757        -6.9867"
            match = re.match(r'\s+(\d+)\s+2\.0000\s+(-?\d+\.\d+)', line)
            if match:
                homo_energy = float(match.group(2))
            # Stop when we hit unoccupied orbitals
            elif re.match(r'\s+\d+\s+0\.0000\s+', line):
                break

    return homo_energy  # Returns in Hartree


def extract_total_energy(output_file):
    """Extract total energy from ORCA output (in Hartree)."""
    with open(output_file, 'r') as f:
        content = f.read()

    # Look for final single point energy
    pattern = r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)"
    match = re.search(pattern, content)

    if match:
        return float(match.group(1))
    return None


def calculate_ip(xyz_file, omega, workdir):
    """Calculate vertical IP at given omega."""

    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    # Neutral molecule (N electrons)
    neutral_inp = create_orca_input(xyz_file, omega, 0, 1, "neutral")
    neutral_out = run_orca(neutral_inp)
    E_N = extract_total_energy(neutral_out)
    homo = extract_homo_energy(neutral_out)

    # Cation (N-1 electrons)
    cation_inp = create_orca_input(xyz_file, omega, 1, 2, "cation")
    cation_out = run_orca(cation_inp)
    E_N1 = extract_total_energy(cation_out)

    if E_N is None or E_N1 is None:
        return None, None

    IP = E_N1 - E_N  # In Hartree (should be positive)

    return homo, IP


def objective_function(omega, xyz_file, base_workdir, results_log):
    """Objective function: |ε_HOMO + IP|"""

    workdir = os.path.join(base_workdir, f"omega_{omega:.4f}")
    homo, ip = calculate_ip(xyz_file, omega, workdir)

    if homo is None or ip is None:
        return 1e10  # Large penalty for failed calculations

    # J = |ε_HOMO + IP| (both in Hartree)
    # Note: ε_HOMO should be negative, IP should be positive
    # At optimal ω: ε_HOMO ≈ -IP
    J = abs(homo + ip)

    # Log results
    with open(results_log, 'a') as f:
        f.write(f"{omega:.4f}\t{homo:.6f}\t{ip:.6f}\t{J:.6f}\n")

    print(f"  ω = {omega:.4f}: ε_HOMO = {homo:.4f} Ha, IP = {ip:.4f} Ha, J = {J:.6f}")

    return J


def tune_omega(molecule_name, xyz_file, output_dir):
    """
    Perform ω tuning for a molecule.

    Returns optimal ω value.
    """

    print(f"\n{'='*60}")
    print(f"Optimal ω Tuning for {molecule_name}")
    print(f"{'='*60}")

    # Create working directory
    workdir = os.path.join(output_dir, molecule_name)
    os.makedirs(workdir, exist_ok=True)

    # Results log
    results_log = os.path.join(workdir, "tuning_results.txt")
    with open(results_log, 'w') as f:
        f.write("omega\tHOMO(Ha)\tIP(Ha)\tJ\n")

    # Initial grid search
    print("\nPhase 1: Grid search...")
    omega_values = np.arange(0.10, 0.32, 0.02)
    J_values = []

    for omega in omega_values:
        J = objective_function(omega, xyz_file, workdir, results_log)
        J_values.append(J)

    # Find best omega from grid
    best_idx = np.argmin(J_values)
    omega_init = omega_values[best_idx]
    print(f"\nBest from grid: ω = {omega_init:.4f}")

    # Fine optimization using golden section search
    print("\nPhase 2: Fine optimization...")
    omega_low = max(0.05, omega_init - 0.03)
    omega_high = min(0.40, omega_init + 0.03)

    result = minimize_scalar(
        lambda w: objective_function(w, xyz_file, workdir, results_log),
        bounds=(omega_low, omega_high),
        method='bounded',
        options={'xatol': 0.001}
    )

    optimal_omega = result.x

    # Save optimal omega
    optimal_file = os.path.join(workdir, "optimal_omega.txt")
    with open(optimal_file, 'w') as f:
        f.write(f"Molecule: {molecule_name}\n")
        f.write(f"Optimal omega: {optimal_omega:.4f} bohr^-1\n")
        f.write(f"Final J value: {result.fun:.6f}\n")

    print(f"\n{'='*60}")
    print(f"OPTIMAL ω = {optimal_omega:.4f} bohr⁻¹")
    print(f"{'='*60}")

    return optimal_omega


def create_tddft_input(xyz_file, omega, molecule_name, output_dir):
    """Create TD-DFT input with optimal omega."""

    workdir = os.path.join(output_dir, molecule_name)
    os.makedirs(workdir, exist_ok=True)

    input_content = f"""# TD-DFT calculation with optimally-tuned LC-wPBE
# Optimal omega = {omega:.4f} bohr^-1

! LC-PBE def2-TZVP def2/J RIJCOSX TightSCF
! PAL{NPROCS}

%maxcore {MEMORY_MB // NPROCS}

%method
    RangeSepMu {omega:.6f}
end

%tddft
    NRoots 10
    IRoot 1
    TDA false
    MaxDim 100
end

%scf
    MaxIter 300
end

* xyzfile 0 1 {xyz_file}
"""

    input_file = os.path.join(workdir, f"{molecule_name}_TDDFT.inp")
    with open(input_file, 'w') as f:
        f.write(input_content)

    print(f"Created TD-DFT input: {input_file}")
    return input_file


def main():
    if len(sys.argv) < 3:
        print("Usage: python omega_tuning.py <molecule_name> <xyz_file>")
        print("Example: python omega_tuning.py DMAC-DPS /path/to/DMAC-DPS.xyz")
        sys.exit(1)

    molecule_name = sys.argv[1]
    xyz_file = os.path.abspath(sys.argv[2])

    if not os.path.exists(xyz_file):
        print(f"Error: XYZ file not found: {xyz_file}")
        sys.exit(1)

    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Perform omega tuning
    optimal_omega = tune_omega(molecule_name, xyz_file, output_dir)

    # Create TD-DFT input with optimal omega
    tddft_input = create_tddft_input(xyz_file, optimal_omega, molecule_name, output_dir)

    print(f"\nNext step: Run TD-DFT calculation:")
    print(f"  {ORCA_PATH} {tddft_input}")


if __name__ == "__main__":
    main()
