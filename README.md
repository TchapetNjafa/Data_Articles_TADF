# TADF Emitters Computational Data Repository [![DOI](https://zenodo.org/badge/1082469447.svg)](https://doi.org/10.5281/zenodo.17436069)

This repository contains computational data and results used in two related research articles on thermally activated delayed fluorescence (TADF) emitters.

## Publications

This dataset supports the following publications:

1. **Validation of Semi-Empirical xTB Methods for High-Throughput Screening of TADF Emitters: A 747-Molecule Benchmark Study**

2. **Data-Driven Design Rules for TADF Emitters from a High-Throughput Screening of 747 Molecules**

## Repository Contents

### CSV Data Files

- **`Data_AllGas_results.csv`** - Physical properties and computational results for all 747 molecules calculated in the gas phase
- **`Data_AllTol_results.csv`** - Physical properties and computational results for all 747 molecules calculated in toluene solvent
- **`tadf_architecture_analysis.csv`** - Molecular architecture classification and structural analysis

### Computational Output Files

- **`Data_calculation_747Mol/`** - Directory containing raw computational output files from the high-throughput screening calculations and molecular structures
  - `gas/` - Gas phase calculation outputs (747 molecules)
  - `toluene/` - Toluene solvent calculation outputs (747 molecules)
  - `RDKit/` - Molecular structure files
  - `nto_orbital_overlap_747mol.csv` - NTO orbital overlap data

### Machine Learning Reproducibility

- **`ML_reproducibility/`** - Complete code and data for reproducing ML/Active Learning results

### High-Level Theory Validation (OT-LC-PBE)

- **`HLT_validation/`** - OT-LC-PBE validation calculations for benchmarking xTB methods
  - `omega_tuning.py` - Automated IP-tuning script for range-separation parameter
  - `extract_hlt_results.py` - Results extraction from ORCA output files
  - `run_hlt_calculations.sh` - Batch calculation script
  - `otlcpbe_validation_results.csv` - Summary of validation results
  - `results/` - Per-molecule results (BACN, DMAC-TRZ, 4CzIPN)

## Computational Methods

All calculations were performed using the following computational tools:

- **xTB** - Extended tight-binding semi-empirical quantum chemistry method
- **CREST** - Conformer-Rotamer Ensemble Sampling Tool for automated conformational searches
- **Multiwfn** - Wavefunction analysis program for post-processing and property calculations
- **sTDA/sTD-DFT** - Simplified time-dependent DFT for excited state calculations

### Machine Learning Tools

- **scikit-learn** - Random Forest and Gaussian Process regression models
- **SHAP** - SHapley Additive exPlanations for model interpretability
- **matplotlib** - Figure generation

## Dataset Overview

This dataset comprises computational results for **747 TADF emitter molecules**, providing comprehensive data for:
- Method validation and benchmarking
- High-throughput screening analysis
- Data-driven design rule extraction

## Usage

Researchers can use this data to:
- Reproduce the results presented in the associated publications
- Develop machine learning models for TADF emitter prediction
- Validate alternative computational methods
- Design new TADF emitter molecules

## Machine Learning Reproducibility Guide

### Directory Structure

```
ML_reproducibility/
├── scripts/
│   ├── data_processing/           # Feature engineering scripts
│   │   ├── build_features_747mol.py
│   │   ├── extract_stda_features_747mol.py
│   │   ├── compute_ct_descriptors_747mol.py
│   │   ├── merge_ct_features_747mol.py
│   │   └── run_hole_electron_analysis.sh
│   └── experiments/               # ML/AL experiment scripts
│       ├── ml_pipeline_747mol.py
│       ├── al_experiment_747mol.py
│       ├── ml_with_ct_features_747mol.py
│       ├── al_experiment_with_ct.py
│       ├── advanced_ml_pipeline.py
│       ├── advanced_al_experiment.py
│       ├── generate_figures_747mol.py
│       ├── generate_al_figures.py
│       ├── benchmark_validation.py
│       ├── identify_candidates.py
│       ├── interpret_model_shap.py
│       └── *_template.py          # Template scripts for reference
├── features/                      # Pre-computed feature tables
│   ├── combined_features_747mol.csv
│   ├── combined_features_747mol_with_ct.csv
│   ├── combined_features_747mol_full_ct.csv
│   ├── stda_features_747mol.csv
│   └── ct_descriptors_747mol.csv
├── results/                       # Model outputs
│   ├── ml_results_747mol.json
│   ├── al_results_747mol.json
│   ├── advanced_ml_results.json
│   ├── advanced_al_results.json
│   ├── predictions_747mol.csv
│   └── *.log                      # Experiment logs
└── figures/                       # Generated figures
    ├── parity_plot_747mol.{pdf,png,pgf}
    ├── feature_importance_747mol.{pdf,png,pgf}
    ├── al_learning_curves_747mol.{pdf,png,pgf}
    ├── al_summary_ct.{pdf,png,pgf}
    ├── al_acquisition_comparison.{pdf,png,pgf}
    └── hlt_validation.{pdf,png,pgf}
```

### Feature Description

The ML models use the following features extracted from sTDA/sTD-DFT calculations:

| Feature | Description | Unit |
|---------|-------------|------|
| `S1_energy_eV` | Singlet S1 excitation energy | eV |
| `T1_energy_eV` | Triplet T1 excitation energy | eV |
| `Delta_E_ST_eV` | Singlet-triplet gap (S1-T1) | eV |
| `S1_osc_strength` | S1 oscillator strength | - |
| `HOMO_LUMO_gap_eV` | HOMO-LUMO energy gap | eV |
| `S1_overlap` | NTO hole-electron overlap for S1 | - |
| `T1_overlap` | NTO hole-electron overlap for T1 | - |
| `lambda_ct_*` | Charge-transfer reorganization energy | eV |
| `d_h_*` | Hole-electron distance | Å |
| `t_index_*` | t-index (CT character) | - |
| `sr_*` | Sr index (spatial overlap) | - |

### Running the ML Pipeline

#### Prerequisites

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pandas scikit-learn matplotlib shap
```

#### Step 1: Feature Extraction (Optional - features are pre-computed)

```bash
cd ML_reproducibility/scripts/data_processing

# Extract sTDA features from log files
python extract_stda_features_747mol.py

# Compute CT descriptors
python compute_ct_descriptors_747mol.py

# Merge all features
python build_features_747mol.py
```

#### Step 2: Train ML Models

```bash
cd ML_reproducibility/scripts/experiments

# Train Random Forest model
python ml_pipeline_747mol.py --target Delta_E_ST_eV --model rf

# Train with CT features
python ml_with_ct_features_747mol.py
```

#### Step 3: Run Active Learning Experiments

```bash
# Basic AL experiment
python al_experiment_747mol.py --model gpr --batch-size 5

# Advanced AL with multiple acquisition strategies
python advanced_al_experiment.py
```

#### Step 4: Generate Figures

```bash
python generate_figures_747mol.py
python generate_al_figures.py
```

### Model Performance Summary

| Model | Target | R² | MAE (eV) | RMSE (eV) |
|-------|--------|-----|----------|-----------|
| Random Forest | ΔE_ST | 0.94 | 0.026 | 0.035 |
| Gaussian Process | ΔE_ST | 0.88 | 0.031 | 0.042 |

### Feature Importance (SHAP Analysis)

| Feature Category | Importance (%) |
|-----------------|----------------|
| Energy features | 88.5 |
| NTO overlap features | 8.0 |
| Oscillator strength | 3.5 |

## OT-LC-PBE High-Level Theory Validation

To validate the xTB-based protocol against high-level DFT methods, we performed optimally-tuned LC-PBE (OT-LC-PBE) calculations on three representative molecules using ORCA 6.1.0.

### Methodology

- **Functional**: LC-PBE (range-separated hybrid)
- **Basis set**: def2-TZVP
- **IP-tuning criterion**: J = |ε_HOMO(ω) + IP(ω)| → 0
- **TD-DFT**: Full TD-DFT (not TDA), 10 roots per spin

### Validation Results

| Molecule | ω (bohr⁻¹) | OT-LC-PBE | sTD-DFT-xTB | Exp. |
|----------|------------|-----------|-------------|------|
| BACN | 0.181 | 0.81 eV | 0.46 eV | -- |
| DMAC-TRZ | 0.185 | 0.17 eV | 0.08 eV | 0.05 eV |
| 4CzIPN | 0.147 | 0.19 eV | 0.08 eV | 0.08 eV |

### Key Findings

1. **ΔE_ST trends are reproduced**: Both methods correctly identify BACN as having the largest gap
2. **xTB closer to experiment**: For molecules with experimental data, sTD-DFT-xTB shows better agreement than vertical OT-LC-PBE
3. **Vertical vs adiabatic**: OT-LC-PBE gives vertical excitations; the larger values compared to experiment are expected

### Running HLT Validation

```bash
cd HLT_validation

# Run omega tuning for a molecule
python omega_tuning.py MOLECULE_NAME path/to/geometry.xyz

# Extract results from ORCA output
python extract_hlt_results.py
```

## Citation

If you use this data in your research, please cite the associated publications:

```
[Citation details to be added upon publication]
```

## License

[chosen license here]

## Contact

[To add]

---

*Last updated: 2025-12-02*
