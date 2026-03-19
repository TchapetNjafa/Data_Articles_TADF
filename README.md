# TADF Emitters Computational Data Repository [![DOI](https://zenodo.org/badge/1082469447.svg)](https://doi.org/10.5281/zenodo.17436069)

This repository contains computational data, results, and manuscripts for two related research articles on thermally activated delayed fluorescence (TADF) emitters, based on high-throughput computational screening of 747 experimentally known molecules.

## Publications

This repository supports the following publications:

### Article 1: xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark

**Status:** ✅ Published  
**Journal:** Journal of Chemical Information and Modeling (2026)  
**DOI:** [10.1021/acs.jcim.5c02978](https://doi.org/10.1021/acs.jcim.5c02978)  
**arXiv:** [2511.00922](https://arxiv.org/abs/2511.00922)

This article validates semi-empirical sTDA-xTB and sTD-DFT-xTB methods for high-throughput screening of TADF emitters using 747 experimentally characterized molecules—the largest benchmark to date. The framework achieves >99% computational cost reduction versus TD-DFT while maintaining strong internal consistency and reasonable agreement with experimental data.

**Manuscript Location:** `ARTICLEs_TADF/Archive1_xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark/`

### Article 2: Data-Driven Design Guidelines for TADF Emitters from a High-Throughput Screening of 747 Molecules

**Status:** 📝 Awaiting Publication (Accepted, revisions in progress)  
**arXiv:** [2511.11606](https://arxiv.org/abs/2511.11606)

This article leverages the validated computational workflow from Article 1 to extract quantitative design guidelines for TADF emitters. Through systematic analysis of molecular architecture, geometry, and electronic structure, it identifies 127 high-performance candidates and establishes structure-property relationships to guide future TADF development.

**Manuscript Location:** `ARTICLEs_TADF/Article2_Data-Driven Design Guidelines for TADF-Emitters from a High-Throughput Screening of 747 Molecules/`

### Relationship Between Articles

These two articles form a cohesive research program:
1. **Article 1** establishes and validates the computational methodology (xTB-based high-throughput screening)
2. **Article 2** applies this validated methodology to extract design rules and identify promising candidates

Both articles analyze the same dataset of 747 TADF molecules and share computational infrastructure.

## Repository Contents

### Manuscripts and Supporting Materials

- **`ARTICLEs_TADF/`** - Complete manuscripts, supporting information, and figures for both articles
  - `Archive1_xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark/` - Published article (Article 1)
  - `Article2_Data-Driven Design Guidelines for TADF-Emitters from a High-Throughput Screening of 747 Molecules/` - Under review (Article 2)
  - See `ARTICLEs_TADF/README.md` for detailed information about each article

### CSV Data Files

- **`Data_AllGas_results.csv`** - Physical properties and computational results for all 747 molecules calculated in the gas phase
- **`Data_AllTol_results.csv`** - Physical properties and computational results for all 747 molecules calculated in toluene solvent
- **`tadf_architecture_analysis.csv`** - Molecular architecture classification and structural analysis

### Computational Scripts

- **`Scripts/`** - Complete computational pipeline for reproducing calculations
  - `tadf_calculation_pipeline.py` - Main orchestration script
  - `rdkitGen_Mol.py` - Initial structure generation from SMILES
  - `geo_Opt.py` - Geometry optimization with xTB and CREST
  - `excitationEner_Calc.py` - Excitation energy calculations with sTDA
  - `data_Extract.py` - Results extraction and processing
  - `DataAnalysis_Article.ipynb` - Data analysis and visualization notebook
  - `unique_subsidiary_database.csv` - Input database with SMILES
  - See `Scripts/README.md` for detailed documentation

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

- **xTB** (v6.7.0+) - Extended tight-binding semi-empirical quantum chemistry method
- **CREST** (v3.0.0+) - Conformer-Rotamer Ensemble Sampling Tool for automated conformational searches
- **Multiwfn** - Wavefunction analysis program for post-processing and property calculations
- **sTDA/sTD-DFT** - Simplified time-dependent DFT for excited state calculations
- **RDKit** - Cheminformatics toolkit for initial structure generation

### Computational Workflow

The complete computational pipeline is available in the `Scripts/` directory:

1. **Structure Generation**: SMILES → 3D coordinates (RDKit with MMFF94s)
2. **Geometry Optimization**: Pre-opt → Conformer search (CREST) → Final opt (xTB GFN2)
3. **Excitation Calculations**: sTDA and sTD-DFT for S0→S1 and S0→T1 transitions
4. **Property Extraction**: Automated extraction of all TADF-relevant properties

See `Scripts/README.md` for detailed usage instructions.

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

### Running Calculations

To reproduce calculations or run on new molecules:

```bash
cd Scripts/
python tadf_calculation_pipeline.py
```

See `Scripts/README.md` for detailed instructions and customization options.

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

If you use this data or scripts in your research, please cite:

```bibtex
@article{tchapet2026validation,
  title={xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark},
  author={Tchapet Njafa, Jean-Pierre and Kameni Tcheuffa, Elvira Vanelle and Foumkpou, Aissatou Maghame and Nana Engo, Serge Guy},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  doi={10.1021/acs.jcim.5c02978}
}
```

**For Article 2 (Design Guidelines):**
```bibtex
@article{tchapet2026design,
  title={Data-Driven Design Guidelines for TADF Emitters from a High-Throughput Screening of 747 Molecules},
  author={Tchapet Njafa, Jean-Pierre and Kameni Tcheuffa, Elvira Vanelle and Foumkpou, Aissatou Maghame and Nana Engo, Serge Guy},
  journal={Awaiting publication},
  year={2026},
  note={arXiv:2511.11606}
}
```

## License

This work is licensed under the MIT License - see below for details.

The computational tools used (xTB, CREST, Multiwfn, sTDA) are freely available under their respective licenses.

### MIT License

```
MIT License

Copyright (c) 2026 [University of Yaoundé 1]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact

[jean-pierre.tchapet@facsciences-uy1.cm]

---

*Last updated: 2026-02-05*
