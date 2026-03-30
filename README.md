# TADF Emitters Computational Data Repository [![DOI](https://zenodo.org/badge/1082469447.svg)](https://doi.org/10.5281/zenodo.17436069)

This repository contains computational data, scripts, and manuscripts for three related research articles on thermally activated delayed fluorescence (TADF) emitters, based on high-throughput computational screening of 747 experimentally known molecules.

> All code and data tested with **Python 3.12**, **scikit-learn 1.7.2**, **RDKit 2024.03**, **SHAP 0.45.0**, available under **CC-BY-4.0** license.

---

## Publications

### Article 1: xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark

**Status:** ✅ Published  
**Journal:** Journal of Chemical Information and Modeling (2026)  
**DOI:** [10.1021/acs.jcim.5c02978](https://doi.org/10.1021/acs.jcim.5c02978)  
**arXiv:** [2511.00922](https://arxiv.org/abs/2511.00922)

Validates semi-empirical sTDA-xTB and sTD-DFT-xTB methods for high-throughput screening of TADF emitters using 747 experimentally characterized molecules—the largest benchmark to date. The framework achieves >99% computational cost reduction versus TD-DFT while maintaining strong internal consistency and reasonable agreement with experimental data.

**Manuscript Location:** `ARTICLEs_TADF/Archive1_xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark/`

---

### Article 2: Data-Driven Design Guidelines for TADF Emitters from a High-Throughput Screening of 747 Molecules

**Status:** ✅ Published  
**Journal:** Journal of Chemical Information and Modeling (2026)  
**DOI:** [10.1021/acs.jcim.5c03068](https://doi.org/10.1021/acs.jcim.5c03068)  
**arXiv:** [2511.11606](https://arxiv.org/abs/2511.11606)

Leverages the validated computational workflow from Article 1 to extract quantitative design guidelines for TADF emitters. Through systematic analysis of molecular architecture, geometry, and electronic structure, it identifies 127 high-performance candidates and establishes structure-property relationships.

**Manuscript Location:** `ARTICLEs_TADF/Article2_Data-Driven Design Guidelines for TADF-Emitters from a High-Throughput Screening of 747 Molecules/`

---

### Article 3: Machine Learning Enables TADF Emitter Discovery for Sustainable Agriculture and Off-Grid Lighting in Resource-Limited Settings

**Status:** 📝 Submitted to Energy & Environmental Science (March 2026)  
**Authors:** Jean-Pierre Tchapet Njafa, Steve Cabrel Teguia Kouam, Patrick Mvoto Kongo, Serge Guy Nana Engo

This article develops a physics-informed ML/Active Learning (AL) framework that predicts ΔE_ST for 736 TADF molecules (2,943 configurations: 2 methods × 2 environments) using charge-transfer descriptors extracted from Natural Transition Orbitals (NTOs) and sTDA/sTD-DFT-xTB computations. The workflow targets sustainable agriculture (indoor horticulture) and off-grid lighting (human-centric lighting) applications in resource-limited settings.

#### Key Results

| Metric | Value |
|--------|-------|
| **Best ML model** | SVR (RBF kernel) |
| **Prediction accuracy (test set)** | MAE = 0.024 eV, R² = 0.960 (589 samples) |
| **Cross-validation** | MAE = 0.025 eV, R² = 0.956 |
| **Feature set** | 38 descriptors (5 categories: energy, CT, NTO, geometry, method) |
| **SHAP feature importance** | Energy: 92%, CT: 6.6% (S_he^T1 = 1.5%), NTO: 0.7%, Other: 0.7% |
| **CAM-B3LYP validation** | MAE = 0.029 eV, R² = 0.93 on 735 molecules |
| **Adiabatic validation** | MAE = 0.195 eV (14 molecules, 28 calculations) |
| **Applicability domain** | 83.4% in-domain (MAE_indomain = 0.017 eV) |
| **Active learning** | Hybrid (uncertainty × diversity): 6.8% sample reduction, Cohen's d = 0.75 |
| **Horticulture candidates** | 20 molecules (PPE > 0.60, 79% easy synthesis SA ≤ 3) |
| **Human-centric lighting candidates** | 4 molecules (MER optimal range, SA 2.36–3.59) |

#### Top Priority Experimental Targets

| Rank | Molecule | Application | SA Score | λ (nm) | ΔE_ST (eV) | Reason |
|------|----------|-------------|----------|--------|------------|--------|
| 1 | **6AcBIQ** | Horticulture | 2.44 | 658 | 0.086 | Easy synthesis + red emission |
| 2 | **dpmb** | Human-centric | 2.36 | 487 | 0.082 | Easiest synthesis + optimal MER |
| 3 | **PxPYM** | Horticulture | 2.47 | 633 | 0.169 | Easy synthesis + high PPE |
| 4 | **AcHPM** | Horticulture | 2.62 | 658 | 0.067 | Ultra-small gap + red |
| 5 | **CBZ** | Human-centric | 2.84 | 485 | 0.159 | Easy synthesis + good MER |

#### Active Learning: 6 Acquisition Functions Compared

| Strategy | Final MAE (eV) | vs. Random |
|----------|---------------|------------|
| **Hybrid (uncertainty × diversity)** | **0.069** | **+4.9%** ✅ best |
| Diversity sampling | 0.069 | +4.6% |
| Random (baseline) | 0.072 | — |
| Uncertainty sampling (US) | >0.072 | worse |
| UCB | 0.094 | −29.7% ❌ catastrophic |

Hybrid AL achieves MAE = 0.050 eV with only 40% of data (1,177 samples), 2.5× data reduction.

#### Model Comparison (5-fold CV)

| Model | CV MAE (eV) | CV R² | Test MAE (eV) | Test R² |
|-------|-------------|-------|---------------|---------|
| **SVR (RBF)** | **0.0251** | **0.9564** | **0.024** | **0.960** |
| Gradient Boosting | 0.027 | 0.946 | — | — |
| Random Forest | 0.035 | 0.918 | — | — |
| Ablation: CT-only (no energies) | 0.099 | 0.619 | — | — |
| Ablation: energy-only | 0.006 | 0.999 | — | — |

> Note: the energy-only model (R²=0.999) confirms feature leakage (ΔE_ST = E_S1 − E_T1); the CT-only model (R²=0.619) proves genuine chemical insight from NTO descriptors.

**Data Location:** All ML reproducibility data is in `ML_reproducibility/` folder (see below for structure)

---

### Relationship Between Articles

1. **Article 1** establishes the xTB/sTDA high-throughput screening protocol (747 molecules)
2. **Article 2** extracts quantitative design rules and identifies 127 high-performance candidates
3. **Article 3** builds an ML/AL framework integrating NTO-based CT descriptors for targeted discovery in non-display applications

All three articles share the same 747-molecule dataset and computational infrastructure.

---

## Repository Contents

### CSV Data Files

- **`Data_AllGas_results.csv`** — Physical properties and computational results for all 747 molecules (gas phase)
- **`Data_AllTol_results.csv`** — Physical properties and computational results for all 747 molecules (toluene solvent)
- **`tadf_architecture_analysis.csv`** — Molecular architecture classification (D-A, D-A-D, MR, TSCT, etc.)

### Computational Scripts

- **`Scripts/`** — Complete xTB/sTDA calculation pipeline (see `Scripts/README.md`)
  - `tadf_calculation_pipeline.py` — Main orchestration script
  - `rdkitGen_Mol.py` — Initial 3D structure generation from SMILES (MMFF94s)
  - `geo_Opt.py` — Geometry optimization (xTB GFN2 + CREST conformational search)
  - `excitationEner_Calc.py` — Excited state calculations (sTDA/sTD-DFT-xTB)
  - `data_Extract.py` — Automated extraction of all TADF-relevant properties
  - `DataAnalysis_Article.ipynb` — Exploratory data analysis notebook

### Computational Output Files

- **`Data_calculation_747Mol/`** — Raw xTB/sTDA output files for all 747 molecules
  - `gas/` — 747 gas-phase calculation folders (each contains: xTB geometry optimization, CREST conformer search, sTDA/sTD-DFT-xTB excited states)
  - `toluene/` — 747 toluene-phase calculation folders (GBSA solvation for ground state, COSMO for excited states)
  - `RDKit/` — Initial 3D structures generated from SMILES (MMFF94s force field)
  - `nto_orbital_overlap_747mol.csv` — NTO overlap integrals computed with Multiwfn (hole-electron spatial overlap for S1 and T1 states)
  - `multiwfn_analysis.log` — Multiwfn batch processing log
  - `DMAC-DPS_gas.tar.gz`, `DMAC-DPS_toluene.tar.gz` — Example complete calculation archives

### Machine Learning Reproducibility Package (Article 3)

- **`ML_reproducibility/`** — Complete ML/AL workflow for Article 3
  - **`features/`** — Curated descriptor tables
    - `combined_features_747mol.csv` — Main feature table (38 descriptors × 2,943 samples)
    - `combined_features_747mol_with_ct.csv` — Extended version with all CT descriptors
    - `combined_features_747mol_full_ct.csv` — Full CT descriptor set (includes all Multiwfn outputs)
    - `ct_descriptors_747mol.csv` — Charge-transfer descriptors only (S_he, Ω_CT, Λ_D, Λ_A, Δr, etc.)
    - `stda_features_747mol.csv` — sTDA/sTD-DFT-xTB energy features (E_S1, E_T1, ΔE_ST, f_osc, τ)
  
  - **`scripts/data_processing/`** — Feature extraction pipeline
    - `build_features_747mol.py` — Merge all features into unified CSV
    - `extract_stda_features_747mol.py` — Parse sTDA/sTD-DFT log files for energies
    - `compute_ct_descriptors_747mol.py` — Extract CT descriptors from Multiwfn outputs
    - `merge_ct_features_747mol.py` — Merge CT descriptors with energy features
    - `run_hole_electron_analysis.sh` — Batch Multiwfn NTO analysis script
  
  - **`scripts/experiments/`** — ML training and analysis
    - `advanced_ml_pipeline.py` — Main ML training script (SVR, GB, RF with 5-fold CV)
    - `advanced_al_experiment.py` — Active learning experiments (6 acquisition functions)
    - `model_benchmarking.py` — Model comparison and performance metrics
    - `interpret_model_shap.py` — SHAP feature importance analysis
    - `generate_advanced_figures.py` — Generate manuscript figures (parity plots, feature importance)
    - `generate_al_figures.py` — Generate AL learning curves and acquisition comparisons
    - `identify_candidates.py` — Screen for horticulture/human-centric lighting candidates
    - `applicability_domain.py` — Compute applicability domain (leverage-based)
    - `ablation_study.py` — Feature ablation experiments (energy-only, CT-only, etc.)
    - `error_by_architecture.py` — Analyze prediction errors by molecular architecture
    - `spectral_application_analysis.py` — Compute PPE and MER for spectral applications
    - `learning_curve_analysis.py` — Generate learning curves for sample efficiency
    - `feature_correlation.py` — Compute feature correlation matrices
    - `descriptor_sensitivity.py` — Sensitivity analysis for CT descriptors
    - `generate_manuscript_tables.py` — Generate LaTeX tables for manuscript
    - `ml_pipeline_template.py` — Template for custom ML experiments
    - `active_learning_template.py` — Template for custom AL experiments
  
  - **`results/`** — Trained models and predictions
    - `advanced_ml_results.json` — Best model performance (SVR: MAE=0.0251 eV, R²=0.9564)
    - `advanced_predictions.json` — Predictions for all 2,943 samples
    - `predictions_747mol.csv` — Predictions in CSV format
    - `ml_results_747mol.json` — Model comparison results
    - `advanced_al_results.json` — Active learning experiment results
    - `al_results_747mol.json` — AL acquisition function comparison
    - `advanced_ml.log` — ML training log
    - `advanced_al.log` — AL experiment log
    - `al_experiment.log` — AL acquisition function log
  
  - **`figures/`** — Manuscript-ready plots (PDF/PNG/PGF formats)
    - `parity_plot_747mol.*` — Predicted vs actual ΔE_ST
    - `feature_importance_747mol.*` — SHAP feature importance (Energy: 92%, CT: 6.6%)
    - `al_learning_curves_747mol.*` — AL learning curves (6 acquisition functions)
    - `al_acquisition_comparison.*` — AL acquisition function comparison
    - `enhanced_ml_comparison.*` — Model comparison (SVR vs GB vs RF)
    - `vertical_vs_adiabatic_comparison.*` — Adiabatic validation (MAE=0.195 eV)
    - `al_summary_ct.*` — AL summary with CT descriptor importance
  - `RDKit/` — Molecular structure files (SDF/XYZ)
  - `nto_orbital_overlap_747mol.csv` — NTO hole-electron overlap data

### Machine Learning Reproducibility (Article 3)

- **`ML_reproducibility/`** — Complete code and data for ML/AL results (see section below)

### Adiabatic Validation (Article 3)

- **`adiabatic_validation/`** — ORCA-based adiabatic ΔE_ST validation (14 molecules, 28 calculations)
  - `vertical_vs_adiabatic_comparison.csv` — Full comparison table (MAE = 0.195 eV, R² = 0.704)
  - `adiabatic_results.json` — Extracted S1/T1 adiabatic energies per molecule and phase
  - `vertical_vs_adiabatic_comparison.pdf/png` — 2-panel comparison figure
  - `extract_orca_results.py` — Script to parse ORCA output files
  - `orca_tadf_calc.py` — ORCA calculation driver
  - `xyz_files/gas/`, `xyz_files/toluene/` — xTB-optimized S0/T1 geometries
  - See `adiabatic_validation/README.md` for full methodology and results table

### PySCF ΔROKS Validation (Article 3)

- **`pyscf_validation/`** — PySCF ΔROKS LC-ωPBE/def2-SVP vertical validation (BACN, ω = 0.16 bohr⁻¹)
  - `pyscf_delta_roks_results.csv` — Results for BACN in gas and toluene
  - `optimized_pyscf_fixed_omega.py` — PySCF ΔROKS driver
  - `utils_pyscf.py` — HDF5 checkpoint utilities
  - `run_pyscf_validation.sh` — Shell launch script
  - See `pyscf_validation/README.md` for full methodology

### Manuscripts

- **`ARTICLEs_TADF/`** — Full manuscript sources and supporting information for all articles

---

## Computational Methods

### Geometry and Ground-State Properties
- **xTB (GFN2-xTB, v6.7.0+)** — Semi-empirical tight-binding DFT for geometry optimization and ground-state properties
- **CREST (v3.0.0+)** — Conformer-Rotamer Ensemble Sampling (automated conformational search)
- **RDKit** — SMILES → 3D structure generation (MMFF94s force field)

### Excited-State Properties
- **sTDA-xTB / sTD-DFT-xTB** — Simplified TD-DFT for S1/T1 excitation energies at GFN2-xTB geometry
- **Solvation**: GBSA (ground state) / COSMO (excited state), toluene (ε = 2.38)

### NTO and CT Descriptor Analysis
- **Multiwfn (v3.8)** — NTO analysis, hole-electron spatial overlap (S_he), CT number (Ω_CT), donor/acceptor localization (Λ_D, Λ_A), hole-particle distance (Δr)
- **29 descriptors** in 5 categories: energy features, NTO overlaps, CT descriptors, oscillator strength, proxy RespA features

### ML and Active Learning
- **scikit-learn (v1.7.2)** — SVR (best model), Gradient Boosting, Random Forest
- **SHAP (v0.45.0)** — SHapley Additive exPlanations for feature importance
- 5-fold cross-validation on 2,943 configurations (80/20 train-test split)

### High-Level Validation
- **ORCA (v6.1.0)** — Adiabatic S1/T1 geometry optimizations for 14 molecules; CAM-B3LYP/def2-TZVP SOC matrix elements for 6 molecules
- **PySCF** — ΔROKS LC-ωPBE/def2-SVP for BACN (ω = 0.16 bohr⁻¹, gas + toluene)
- **CAM-B3LYP/def2-TZVP** — Benchmark ΔE_ST for 735 molecules (MAE = 0.029 eV vs xTB)

---

## Machine Learning Reproducibility Guide (Article 3)

### Directory Structure

```
ML_reproducibility/
├── scripts/
│   ├── data_processing/           # Feature engineering
│   │   ├── build_features_747mol.py          # Merge all features into one CSV
│   │   ├── extract_stda_features_747mol.py   # Parse sTDA/sTD-DFT log files
│   │   ├── compute_ct_descriptors_747mol.py  # Extract CT from NTO molden files
│   │   ├── merge_ct_features_747mol.py       # Combine energy + CT features
│   │   └── run_hole_electron_analysis.sh     # Batch Multiwfn NTO analysis
│   └── experiments/               # ML/AL experiments
│       ├── ml_pipeline_747mol.py              # Baseline RF/GPR ML pipeline
│       ├── advanced_ml_pipeline.py            # SVR + GB + RF + NN benchmark
│       ├── al_experiment_747mol.py            # Basic uncertainty AL
│       ├── advanced_al_experiment.py          # All 6 acquisition functions
│       ├── ablation_study.py                  # Feature ablation (Models A/C/D)
│       ├── applicability_domain.py            # k-NN distance + leverage AD
│       ├── model_benchmarking.py              # Systematic model comparison
│       ├── learning_curve_analysis.py         # Budget vs accuracy trade-off
│       ├── error_by_architecture.py           # ANOVA stratified by D-A/MR/etc.
│       ├── feature_correlation.py             # Pearson r heatmaps for 29 features
│       ├── descriptor_sensitivity.py          # Input sensitivity analysis
│       ├── spectral_application_analysis.py   # PPE/MER screening (horticulture/lighting)
│       ├── generate_manuscript_figures.py     # Reproduce all manuscript figures
│       ├── generate_manuscript_tables.py      # Generate LaTeX tables
│       ├── identify_candidates.py             # Multi-objective candidate filtering
│       ├── interpret_model_shap.py            # Reproduce SHAP importance plots
│       └── benchmark_validation.py            # Parity/correlation plots
├── features/                      # Pre-computed feature tables
│   ├── combined_features_747mol.csv           # 8-feature table (energy + NTO)
│   ├── combined_features_747mol_with_ct.csv   # + partial CT descriptors
│   ├── combined_features_747mol_full_ct.csv   # Full 29-feature table
│   ├── stda_features_747mol.csv               # Raw sTDA excitation energies
│   └── ct_descriptors_747mol.csv              # CT/Multiwfn descriptors
├── results/                       # Model outputs and logs
│   ├── ml_results_747mol.json                 # Baseline RF results
│   ├── advanced_ml_results.json               # SVR/GB/RF/NN/GPR comparison
│   ├── al_results_747mol.json                 # Basic AL results
│   ├── advanced_al_results.json               # All 6 acquisition function results
│   ├── advanced_predictions.json              # Per-molecule predictions
│   └── predictions_747mol.csv                 # Numerical predictions (CSV)
└── figures/                       # Generated figures
    ├── parity_plot_747mol.{pdf,png}           # SVR parity plot (R²=0.960)
    ├── feature_importance_747mol.{pdf,png}    # SHAP importance (29 features)
    ├── al_learning_curves_747mol.{pdf,png}    # Learning curves vs random
    ├── al_acquisition_comparison.{pdf,png}    # 6-strategy AL comparison
    ├── enhanced_ml_comparison.{pdf,png}       # Multi-model performance
    └── vertical_vs_adiabatic_comparison.{pdf,png}  # Adiabatic validation
```

### Feature Description (29-Descriptor Set)

| Category | Features | SHAP Importance |
|----------|----------|-----------------|
| Energy | E_S1, E_T1, HOMO-LUMO gap | 57% (E_T1=31%, E_S1=24%) |
| CT descriptors | S_he^S1, S_he^T1, Ω_CT, Λ_D, Λ_A, Δr (S1+T1+diffs) | 34% (S_he^T1=21%) |
| Oscillator strength | f_S1 | 8% |
| NTO overlap | S_NTO^S1, S_NTO^T1 | 1% |
| Proxy RespA | ΔS_NTO, \|ΔS_NTO\|, products, log-transforms | <1% |

### Reproducing the ML Results

#### Step 1: Install dependencies

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy pandas scikit-learn==1.7.2 shap matplotlib joblib rdkit-pypi
```

#### Step 2: Feature extraction (optional — features pre-computed)

```bash
cd ML_reproducibility/scripts/data_processing
python extract_stda_features_747mol.py     # Parse sTDA log files
python compute_ct_descriptors_747mol.py    # Extract CT from NTO molden files  
python build_features_747mol.py            # Merge all → combined_features_747mol_full_ct.csv
```

#### Step 3: Train SVR (best model)

```bash
cd ML_reproducibility/scripts/experiments
python advanced_ml_pipeline.py             # SVR + GB + RF + NN benchmark
```

Expected output: `results/advanced_ml_results.json`
- SVR: MAE = 0.025 eV, R² = 0.956 (5-fold CV)

#### Step 4: Run Active Learning (all 6 strategies)

```bash
python advanced_al_experiment.py           # 6 acquisition functions × 10 seeds
```

#### Step 5: Spectral application screening

```bash
python spectral_application_analysis.py   # PPE (horticulture) + MER (lighting)
```

Expected output: 20 horticulture candidates (PPE > 0.60), 4 human-centric lighting candidates (optimal MER).

#### Step 6: Reproduce manuscript figures

```bash
python generate_manuscript_figures.py      # All 6+ manuscript figures
```

---

## Adiabatic Validation Summary (Article 3)

Vertical ΔE_ST (at S0 geometry) vs. adiabatic ΔE_ST (at S1/T1 optimized geometries) for 14 representative molecules in gas and toluene:

| Metric | Value |
|--------|-------|
| Mean absolute error | 0.195 eV |
| Systematic offset (intercept) | +0.070 eV |
| R² | 0.704 |
| Linear fit slope | 1.002 |
| Best agreement | APPT-PXZ gas (0.019 eV) — rigid molecule |
| Worst agreement | 2CzTPE gas (0.548 eV) — flexible propeller structure |

**Physical interpretation**: Rigid D-A molecules with locked torsional angles show excellent agreement (<0.1 eV). Flexible molecules with large excited-state geometric relaxation show larger deviations (0.3–0.5 eV). The vertical approximation is valid for screening (ranking preserved), with adiabatic corrections required for quantitative k_RISC prediction.

See `adiabatic_validation/README.md` for full data and methodology.

---

## Dataset Overview

**747 TADF emitter molecules** spanning:
- Donor-Acceptor (D-A)
- Donor-Acceptor-Donor (D-A-D)
- Multi-resonance TADF (MR-TADF)
- Through-Space CT (TSCT)
- and other architectures

**Training set**: 736 molecules (11 excluded for quality control: missing CT descriptors or method discrepancies)  
**Configurations**: 2,943 (2 calculation methods × 2 environments: gas + toluene)  
**Features**: 29 descriptors per configuration

---

## Citation

If you use this data or scripts, please cite:

```bibtex
@article{tchapet2026validation,
  title={xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark},
  author={Tchapet Njafa, Jean-Pierre and Kameni Tcheuffa, Elvira Vanelle and Foumkpou, Aissatou Maghame and Nana Engo, Serge Guy},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  doi={10.1021/acs.jcim.5c02978}
}

@article{tchapet2026design,
  title={Data-Driven Design Guidelines for TADF Emitters from a High-Throughput Screening of 747 Molecules},
  author={Tchapet Njafa, Jean-Pierre and Kameni Tcheuffa, Elvira Vanelle and Foumkpou, Aissatou Maghame and Nana Engo, Serge Guy},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  doi={10.1021/acs.jcim.5c03068}
}

@article{tchapet2026ml,
  title={Machine Learning Enables TADF Emitter Discovery for Sustainable Agriculture and Off-Grid Lighting in Resource-Limited Settings},
  author={Tchapet Njafa, Jean-Pierre and Teguia Kouam, Steve Cabrel and Mvoto Kongo, Patrick and Nana Engo, Serge Guy},
  journal={Submitted},
  year={2026},
  note={Zenodo: 10.5281/zenodo.17436069}
}
```

---

## License

Data and scripts are available under **CC-BY-4.0** (see `LICENSE`).

Computational tools used: xTB, CREST, sTDA, Multiwfn, RDKit, PySCF are freely available under their respective licenses.

---

## Contact

jean-pierre.tchapet@facsciences-uy1.cm

*Department of Physics, Faculty of Science, University of Yaounde I, Cameroon*

---

*Last updated: March 2026*
