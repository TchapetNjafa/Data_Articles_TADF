# PySCF ΔROKS Validation (LC-ωPBE/def2-SVP)

This folder contains scripts and results for **vertical** ΔE_ST validation using PySCF with the ΔROKS (delta-Restricted Open-shell Kohn-Sham) approach and a fixed optimal range-separation parameter ω.

## Purpose

Provides higher-level DFT validation (LC-ωPBE/def2-SVP) of the sTD-DFT-xTB predictions for BACN, a molecule with large ΔE_ST that serves as a stress-test for the semi-empirical method.

## Method

- **Functional**: LC-ωPBE (range-separated hybrid)
- **Basis set**: def2-SVP
- **Range-separation parameter**: ω = 0.16 bohr⁻¹ (fixed, from IP-tuning)
- **Approach**: ΔROKS — energy difference between the lowest S1 and T1 states at the S0 geometry (vertical excitations)
- **Solvent**: gas phase and toluene (ddCOSMO implicit solvent)
- **Software**: PySCF (Python-based quantum chemistry)

## Results

| Molecule | Solvent | ω (bohr⁻¹) | ΔE_ST adiabatic (eV) | ΔE_ST vertical (eV) |
|----------|---------|------------|----------------------|---------------------|
| BACN | gas | 0.160 | 0.824 | 0.665 |
| BACN | toluene | 0.160 | 0.784 | 0.666 |

For comparison, the sTD-DFT-xTB vertical ΔE_ST for BACN is **0.991 eV** (gas) and **0.921 eV** (toluene).

## Directory Contents

```
pyscf_validation/
├── pyscf_delta_roks_results.csv          # ΔROKS results for BACN
├── optimized_pyscf_fixed_omega.py        # Main PySCF ΔROKS driver
├── utils_pyscf.py                        # Checkpointing, tmpdir, HDF5 utilities
└── run_pyscf_validation.sh               # Shell launch script
```

## Usage

### Environment Setup

```bash
# Required environment variables
export PYSCF_TMPDIR=./pyscf_tmp
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### Run ΔROKS Calculation

```bash
# Run for a single molecule
python optimized_pyscf_fixed_omega.py \
  --molecule BACN \
  --xyz path/to/BACN.xyz \
  --omega 0.16 \
  --solvent gas

# Run via shell script (supports gas + toluene in parallel)
bash run_pyscf_validation.sh
```

### Results Format

The output CSV (`pyscf_delta_roks_results.csv`) contains columns:
- `molecule`, `solvent`, `omega_bohr-1`, `omega_reference`
- `e_s0_eV`, `e_s1_opt_eV`, `e_t1_opt_eV` — total energies
- `e_s1_vert_eV`, `e_t1_vert_eV` — vertical excited state energies
- `excitation_s1_opt_eV`, `excitation_t1_opt_eV` — adiabatic excitation energies
- `excitation_s1_vert_eV`, `excitation_t1_vert_eV` — vertical excitation energies
- `delta_est_opt_eV`, `delta_est_vert_eV` — adiabatic and vertical ΔE_ST
- `basis`, `functional`

## Notes

- The fixed ω = 0.16 bohr⁻¹ was chosen from IP-tuning on similar donor-acceptor molecules
- PySCF calculations are memory-intensive; 8–16 GB RAM recommended per job
- The `utils_pyscf.py` module implements HDF5-based checkpointing to allow restart of interrupted calculations

## Citation

If you use this data, please cite Article 3 of this repository series (see main README.md).

---
*Last updated: March 2026*
