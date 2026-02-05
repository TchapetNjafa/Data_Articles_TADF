# TADF Calculation Scripts

This directory contains the computational pipeline scripts used to calculate TADF (Thermally Activated Delayed Fluorescence) properties for the 747-molecule benchmark study.

## Overview

The pipeline implements the workflow described in the article "Validation of Semi-Empirical xTB Methods for High-Throughput Screening of TADF Emitters" (DOI: 10.1021/acs.jcim.5c02978).

## Scripts

### Main Pipeline

- **`tadf_calculation_pipeline.py`** - Main orchestration script that runs the complete workflow
  - Reads SMILES from input CSV
  - Generates initial structures
  - Performs geometry optimization and conformer search
  - Calculates excitation energies
  - Extracts results to CSV

### Core Modules

- **`rdkitGen_Mol.py`** - Generates initial 3D molecular structures from SMILES using RDKit
  - Uses MMFF94s force field for initial optimization
  - Falls back to UFF if MMFF94s fails
  - Outputs .xyz files

- **`geo_Opt.py`** - Geometry optimization and conformer search
  - Pre-optimization with xTB (GFN2-xTB)
  - Conformer search with CREST
  - Final optimization with xTB
  - Handles both S0 (ground state) and T1 (triplet) states

- **`excitationEner_Calc.py`** - Calculates vertical excitation energies
  - Uses sTDA (simplified Tamm-Dancoff approximation)
  - Uses sTD-DFT (simplified TD-DFT)
  - Calculates S0→S1 and S0→T1 transitions

- **`data_Extract.py`** - Extracts calculated properties from log files
  - HOMO-LUMO gap
  - Excitation energies (vertical and relaxed)
  - Singlet-triplet gap (ΔE_ST)
  - Oscillator strengths
  - Emission/absorption wavelengths
  - TADF efficiency metrics

### Analysis Scripts

- **`DataAnalysis_Article.ipynb`** - Jupyter notebook for data analysis and figure generation
  - RMSD calculations between geometries
  - Electron density analysis
  - HOMO-LUMO overlap calculations
  - Statistical analysis and visualization

- **`DataAnalysis_ArticleScript.py`** - Python script version of the notebook (for batch processing)

### Input Data

- **`unique_subsidiary_database.csv`** - Input database containing:
  - Compound names
  - SMILES strings
  - Experimental data (when available)

## Dependencies

### Required Software

- **xTB** (v6.7.0 or later) - Extended tight-binding program
- **CREST** (v3.0.0 or later) - Conformer-rotamer ensemble sampling tool
- **sTDA** - Simplified TD-DFT program
- **Multiwfn** (optional) - For wavefunction analysis

### Python Packages

```bash
pip install rdkit pandas numpy scipy matplotlib seaborn pyscf
```

## Usage

### Basic Usage

Run the complete pipeline on the default input:

```bash
python tadf_calculation_pipeline.py
```

### Custom Input

Modify the script to use custom input/output:

```python
from tadf_calculation_pipeline import semiEmpi_tadf

semiEmpi_tadf(
    input_csv='your_molecules.csv',
    output_dir='./results',
    output_csv='your_results'
)
```

### Input CSV Format

The input CSV should contain at minimum:
- `compound.names` - Molecule name(s) (can be list)
- `compound.SMILES` - SMILES string

Example:
```csv
compound.names,compound.SMILES
"['DMAC-DPS']",CC1(C)c2ccccc2N(c2ccc(S(=O)(=O)c3ccc(N4c5ccccc5C(C)(C)c5ccccc54)cc3)cc2)c2ccccc21
```

## Workflow Details

### Step 1: Structure Generation (RDKit)
- Converts SMILES to 3D coordinates
- MMFF94s force field optimization
- Canonical orientation

### Step 2: Geometry Optimization (xTB + CREST)
For both gas phase and toluene solvent:
- **S0 state:**
  1. Pre-optimization (xTB GFN2)
  2. Conformer search (CREST)
  3. Final optimization (xTB GFN2)

- **T1 state:**
  1. Pre-optimization from S0 geometry (xTB GFN2 with UHF=2)
  2. Conformer search (CREST with UHF=2)
  3. Final optimization (xTB GFN2 with UHF=2)

### Step 3: Excitation Energy Calculation (sTDA)
From optimized S0 geometry:
- S0→S1 transition (sTDA and sTD-DFT)
- S0→T1 transition (sTDA and sTD-DFT)

### Step 4: Data Extraction
Extracts and calculates:
- Vertical excitation energies
- Relaxation energies
- Stokes shifts
- Oscillator strengths
- Emission wavelengths
- TADF efficiency parameters

## Output Files

### Per-Molecule Outputs

For each molecule in `{output_dir}/{phase}/{molecule_name}/`:

**Geometry files:**
- `{name}_{phase}_S0_preOpt.xtbopt.xyz` - Pre-optimized S0
- `{name}_{phase}_S0_crest.xyz` - Best conformer S0
- `{name}_{phase}_S0_finalOpt.xtbopt.xyz` - Final optimized S0
- `{name}_{phase}_T1_finalOpt.xtbopt.xyz` - Final optimized T1

**Calculation logs:**
- `{name}_{phase}_S0_finalOpt.log` - xTB optimization log
- `{name}_{phase}_S0_crest.log` - CREST conformer search log
- `{name}_{phase}_S0S1_stda.log` - sTDA S0→S1 log
- `{name}_{phase}_S0T1_stda.log` - sTDA S0→T1 log

**Wavefunction files:**
- `{name}_{phase}_S0_finalOpt.molden` - Molden format wavefunction

**Excitation data:**
- `{name}_{phase}_S0S1_stda.dat` - sTDA excitation data
- `{name}_{phase}_S0S1_stddft.dat` - sTD-DFT excitation data

### Summary Output

- **`{output_csv}.csv`** - Complete results table with all calculated properties

## Computational Cost

Approximate timing per molecule (on 8 cores):
- Structure generation: < 1 second
- S0 optimization: 5-30 minutes
- T1 optimization: 10-60 minutes
- Excitation calculations: 1-5 minutes
- **Total: ~20-100 minutes per molecule**

For 747 molecules: ~250-1250 CPU hours

## Notes

- The pipeline automatically handles failed calculations and continues
- Solvation effects (toluene) use GBSA implicit solvent model
- CPU usage is automatically adjusted (uses total CPUs - 8)
- Temporary files are cleaned up after each calculation

## Troubleshooting

### Common Issues

1. **CREST fails for T1 state**
   - Script automatically retries without UHF flag
   - Check if molecule has unusual electronic structure

2. **sTDA produces no output**
   - Check if xtb4stda completed successfully
   - Verify wfn.xtb file exists

3. **Memory issues**
   - Reduce number of parallel cores
   - Process molecules in smaller batches

## Citation

If you use these scripts, please cite:

```bibtex
@article{tchapet2025validation,
  title={Validation of Semi-Empirical xTB Methods for High-Throughput Screening of TADF Emitters: A 747-Molecule Benchmark Study},
  author={Jean-Pierre Tchapet Njafa, Elvira Vanelle Kameni Tcheuffa, Aissatou Maghame
Foumkpou, and Serge Guy Nana Engo},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  doi={10.1021/acs.jcim.5c02978}
}
```

## License

MIT License - Free to use, modify, and distribute with attribution.

All computational tools used (xTB, CREST, Multiwfn, sTDA, RDKit) are freely available under their respective open-source licenses.

## Contact

[jean-pierre.tchapet@facsciences-uy1.cm]
