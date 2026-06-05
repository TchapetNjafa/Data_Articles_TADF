# TADF Emitters Computational Data Repository [![DOI](https://zenodo.org/badge/1082469447.svg)](https://doi.org/10.5281/zenodo.17436069)

This repository contains computational data, scripts, and manuscripts for three
related research articles on thermally activated delayed fluorescence (TADF)
emitters, based on high-throughput computational screening of 747 experimentally
known molecules, extended to a 22,194-molecule virtual library for Article 3.

> All code and data tested with **Python 3.12**, **scikit-learn 1.7.2**,
> **RDKit 2024.03**, **SHAP 0.45.0**, available under **CC-BY-4.0** license.

---

## Publications

### Article 1: xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark
**Status:** ✅ Published
**Journal:** Journal of Chemical Information and Modeling (2026)
**DOI:** [10.1021/acs.jcim.5c02978](https://doi.org/10.1021/acs.jcim.5c02978)
**arXiv:** [2511.00922](https://arxiv.org/abs/2511.00922)

Validates semi-empirical sTDA-xTB and sTD-DFT-xTB methods for high-throughput
screening of TADF emitters using 747 experimentally characterized molecules —
the largest benchmark to date. The framework achieves >99% computational cost
reduction versus TD-DFT while maintaining strong internal consistency and
reasonable agreement with experimental data.

**Manuscript location:** `ARTICLEs_TADF/Archive1_xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark/`

---

### Article 2: Data-Driven Design Guidelines for TADF Emitters from a High-Throughput Screening of 747 Molecules
**Status:** ✅ Published
**Journal:** Journal of Chemical Information and Modeling (2026)
**DOI:** [10.1021/acs.jcim.5c03068](https://doi.org/10.1021/acs.jcim.5c03068)
**arXiv:** [2511.11606](https://arxiv.org/abs/2511.11606)

Leverages the validated computational workflow from Article 1 to extract
quantitative design guidelines for TADF emitters. Through systematic analysis
of molecular architecture, geometry, and electronic structure, it identifies
127 high-performance candidates and establishes structure-property relationships.

**Manuscript location:** `ARTICLEs_TADF/Article2_Data-Driven Design Guidelines for TADF-Emitters from a High-Throughput Screening of 747 Molecules/`

---

### Article 3: Accelerated Discovery of TADF Emitters via Physics-Informed Machine Learning and Multi-Fidelity Active Learning
**Status:** 📝 Submitted to *Digital Discovery*, RSC (June 2026)
**Authors:** Jean-Pierre Tchapet Njafa, Steve Cabrel Teguia Kouam, Patrick Sorrel
Mvoto Kongo, Panebei Samafou, Serge Guy Nana Engo
**Zenodo DOI:** [10.5281/zenodo.14241084](https://doi.org/10.5281/zenodo.14241084)

Develops a physics-informed multi-fidelity active learning pipeline for
large-scale TADF emitter discovery. NTO-based spatial descriptors combined with
SVR models predict ΔE_ST with high accuracy. Active learning achieves a 27.7-fold
computational reduction over exhaustive TD-DFT. Multi-objective Pareto optimization
improves discovery efficiency by 45.5%. A virtual library expansion from 747 to
22,194 molecules (29.7×) demonstrates industrial-scale applicability, yielding a
nested efficiency exceeding 1000×. Top candidates (DMAC-DPS, PXZ-NAI) exceed
>15% predicted EQE for horticultural and human-centric lighting applications.

#### Key Results (Article 3)

| Metric | Value |
|--------|-------|
| Best ML model | SVR (RBF kernel) |
| Cross-validation accuracy | MAE = 0.025 eV, R² = 0.956 |
| Test set accuracy | MAE = 0.024 eV, R² = 0.960 |
| Training set size | 640 experimental TADF molecules |
| Feature set | 29 descriptors (energy, NTO, CT, oscillator strength) |
| SHAP top features | E_T1: 48%, E_S1: 43%, NTO/CT spatial: 8% (SVR KernelSHAP, Config F) |
| Active learning saving | **27.7× fewer DFT evaluations** (27 vs. 747) |
| Multi-objective efficiency | **+45.5%** Pareto solutions vs. random sampling |
| Virtual library size | **22,194 molecules** (29.7× scale-up from 747) |
| Top candidates identified | **500** (TADF score ≥ 0.9), 8 fully validated |
| Nested multi-fidelity efficiency | **>1000×** vs. exhaustive TD-DFT on 22,194 mol. |
| Best candidate (blue emitter) | DMAC-DPS — 425 nm, Soret overlap Ω = 0.99, >15% EQE |
| Best candidate (red emitter) | PXZ-NAI — 629 nm, Q_y overlap Ω = 0.58, >15% EQE |
| Adiabatic validation | MAE = 0.195 eV (14 molecules, rigid structures: <0.1 eV) |

#### Model Comparison (5-fold CV)

| Model | CV MAE (eV) | CV R² | Notes |
|-------|-------------|-------|-------|
| **SVR (RBF)** | **0.025** | **0.956** | Best overall — used in manuscript |
| Gradient Boosting | 0.027 | 0.946 | |
| Random Forest | 0.035 | 0.918 | |
| Ablation: CT-only (no energies) | 0.099 | 0.619 | Proves NTO descriptors encode genuine physics |
| Ablation: energy-only | 0.006 | 0.999 | Feature leakage (ΔE_ST ≈ E_S1 − E_T1) |

#### Active Learning Summary

| Strategy | Final MAE (eV) | vs. Random |
|----------|---------------|------------|
| **Lower-confidence bound (LCB)** | **0.069** | **best** |
| Diversity sampling | 0.069 | comparable |
| Random (baseline) | 0.072 | — |
| Uncertainty sampling | >0.072 | worse |
| Upper-confidence bound (UCB) | 0.094 | −29.7% |

LCB active learning reaches Spearman ρ = 0.95 with only 27 TD-DFT evaluations
(3.6% of the 747-molecule space) — a **27.7-fold reduction** vs. random sampling
(>200 evaluations needed).

---

### Relationship Between Articles

1. **Article 1** establishes the xTB/sTDA high-throughput screening protocol (747 molecules)
2. **Article 2** extracts quantitative design rules and identifies 127 high-performance candidates
3. **Article 3** builds a physics-informed ML/AL framework for multi-objective discovery
   and scales it to a 22,194-molecule virtual library

All three articles share the same 747-molecule core dataset and computational
infrastructure. Article 3 extends this with the virtual library expansion
(see `virtual_library_expansion/`) and the Digital Discovery manuscript figures
(see `ML_reproducibility/figures/`).

---

## Repository Contents

### Root-level files

- **`interactive_umap.html`** — Interactive UMAP visualization of the 747-molecule
  chemical space (hover for molecule details: name, ΔE_ST, architecture). Open in
  any modern web browser. Referenced in the Article 3 Data Availability statement
  (DOI: 10.5281/zenodo.14241084).
- **`Data_AllGas_results.csv`** — Physical properties and computational results for
  all 747 molecules (gas phase)
- **`Data_AllTol_results.csv`** — Physical properties and computational results for
  all 747 molecules (toluene solvent)
- **`SMILES_747mol.csv`** — SMILES strings for all 747 molecules
- **`tadf_architecture_analysis.csv`** — Molecular architecture classification
  (D-A, D-A-D, MR, TSCT, etc.)
- **`DMAC-DPSGas_results.csv`**, **`DMAC-DPSTol_results.csv`** — Full results for the
  top candidate DMAC-DPS (gas + toluene phases)

### `virtual_library_expansion/` — 22,194-molecule virtual library (Article 3, REQ-1)

The 29.7× scale-up of the TADF screening library from 747 to 22,194 molecules.
See `virtual_library_expansion/README.md` for full documentation.

- **`data/expanded_library_final.csv`** — Complete 22,194-molecule library with
  RDKit properties (MW, LogP, rings, heteroatoms, etc.)
- **`data/top_candidates_for_qc.csv`** — 500 top candidates (TADF score ≥ 0.9,
  mean predicted ΔE_ST = 0.130 ± 0.005 eV)
- **`data/ml_predictions.csv`** — Heuristic ΔE_ST predictions for all 22,194 molecules
- **`data/pubchem_tadf_filtered.csv`** — Intermediate SMARTS-filtered PubChem dataset
- **`analysis/`** — 5 manuscript figures (PDF+PNG) + 3 tables + summary statistics
- **`scripts/`** — Complete reproducible pipeline (5 scripts: download → filter →
  build → score → analyze)

### `Scripts/` — xTB/sTDA calculation pipeline (Articles 1–2)

Complete high-throughput screening pipeline. See `Scripts/README.md`.

- `tadf_calculation_pipeline.py` — Main orchestration script
- `rdkitGen_Mol.py` — 3D structure generation from SMILES (MMFF94s)
- `geo_Opt.py` — Geometry optimization (xTB GFN2 + CREST)
- `excitationEner_Calc.py` — Excited-state calculations (sTDA/sTD-DFT-xTB)
- `data_Extract.py` — Automated extraction of TADF-relevant properties
- `DataAnalysis_Article.ipynb` — Exploratory data analysis notebook

### `Data_calculation_747Mol/` — Raw xTB/sTDA outputs

Raw calculation outputs for all 747 molecules.

- `gas/` — 747 gas-phase calculation folders
- `toluene/` — 747 toluene-phase calculation folders
- `RDKit/` — Initial 3D structures from SMILES
- `nto_orbital_overlap_747mol.csv` — NTO hole-electron overlap integrals (Multiwfn)
- `DMAC-DPS_gas.tar.gz`, `DMAC-DPS_toluene.tar.gz` — Example complete calculation archives

### `ML_reproducibility/` — Complete ML/AL package (Article 3)

Full ML training, active learning, SHAP analysis, and figure generation for
Article 3. See the **Machine Learning Reproducibility Guide** section below.

**`features/`** — Pre-computed descriptor tables:
- `combined_features_747mol.csv` — Main feature table (38 descriptors × 2,943 samples)
- `combined_features_747mol_with_ct.csv` — Extended with all CT descriptors
- `combined_features_747mol_full_ct.csv` — Full CT descriptor set
- `ct_descriptors_747mol.csv` — Charge-transfer descriptors (S_he, Ω_CT, Λ_D, Λ_A, Δr)
- `stda_features_747mol.csv` — sTDA/sTD-DFT-xTB energy features

**`results/`** — Model outputs and SHAP values:
- `advanced_ml_results.json` — SVR/GB/RF model comparison (8 features)
- `advanced_predictions.json` — Per-molecule predictions (all 2,943 samples)
- `predictions_747mol.csv` — Predictions in CSV format
- `advanced_al_results.json` — Active learning results (all 6 acquisition functions)
- `shap_svr_modelC_configF_article3.csv` — **Authoritative SHAP** (SVR KernelSHAP, Config F, 29 feat)
- `shap_svr_modelA_energy_only.csv` — Model A ablation SHAP (energy-only)
- `shap_svr_modelD_ct_only.csv` — Model D ablation SHAP (CT-only, R²=0.619)
- `ablation_summary_svr_article3.json` — Model A/C/D performance comparison
- `shap_svr_article3_correction_note.json` — Correction record (RF Gini → SVR KernelSHAP)

**`figures/`** — All manuscript-ready figures (PDF+PNG):

*Article 3 Digital Discovery figures (new):*
- `fig1_model_performance.{pdf,png}` — Model comparison + parity plot + SHAP
- `fig1c_shap_beeswarm.{pdf,png}` — SHAP beeswarm plot (all features)
- `fig1c_model_c_shap.{pdf,png}` — Model C SHAP summary
- `fig2_active_learning.{pdf,png}` — Active learning convergence (27.7× reduction)
- `fig3_multi_objective.{pdf,png}` — Multi-objective Pareto optimization (45.5% gain)
- `fig4_device_predictions.{pdf,png}` — SOC + Marcus device predictions (>15% EQE)
- `fig5_pipeline.{pdf,png}` — Full multi-fidelity pipeline overview (22,194 mol.)
- `fig_si_calibration.{pdf,png}` — GPR uncertainty calibration (ECE = 0.301)
- `fig_si_supplementary_distributions.{pdf,png}` — Supplementary property distributions

*Legacy figures (Articles 1–2, preserved for reference):*
- `parity_plot_747mol.{pdf,png,pgf}` — SVR parity plot
- `feature_importance_747mol.{pdf,png,pgf}` — Original RF Gini importance (preserved)
- `feature_importance_747mol_v2_article3.{pdf,png}` — Corrected SVR KernelSHAP figure
- `al_learning_curves_747mol.{pdf,png,pgf}` — AL learning curves
- `al_acquisition_comparison.{pdf,png,pgf}` — 6-strategy AL comparison
- `enhanced_ml_comparison.{pdf,png}` — Multi-model performance comparison
- `vertical_vs_adiabatic_comparison.{pdf,png}` — Adiabatic validation

**`scripts/`** — Training and analysis scripts (see ML Reproducibility Guide below)

### `adiabatic_validation/` — ORCA adiabatic validation (Article 3)

- `vertical_vs_adiabatic_comparison.csv` — Full comparison (MAE = 0.195 eV, R² = 0.704)
- `adiabatic_results.json` — S1/T1 adiabatic energies per molecule and phase
- `vertical_vs_adiabatic_comparison.{pdf,png}` — 2-panel comparison figure
- `extract_orca_results.py` — ORCA output parser
- `orca_tadf_calc.py` — ORCA calculation driver
- `xyz_files/gas/`, `xyz_files/toluene/` — xTB-optimized S0/T1 geometries

### `pyscf_validation/` — PySCF ΔROKS validation (Article 3)

- `pyscf_delta_roks_results.csv` — Results for BACN (gas + toluene, ω = 0.16 bohr⁻¹)
- `optimized_pyscf_fixed_omega.py` — PySCF ΔROKS driver
- See `pyscf_validation/README.md` for full methodology

### `ARTICLEs_TADF/` — Full manuscript sources

Manuscript LaTeX sources and supporting information for Articles 1–2.

---

## Computational Methods

### Geometry and Ground-State Properties
- **xTB (GFN2-xTB, v6.7.0+)** — Semi-empirical tight-binding DFT for geometry
  optimization and ground-state properties
- **CREST (v3.0.0+)** — Conformer-Rotamer Ensemble Sampling
- **RDKit** — SMILES → 3D structure generation (MMFF94s force field)

### Excited-State Properties
- **sTDA-xTB / sTD-DFT-xTB** — Simplified TD-DFT for S1/T1 excitation energies
- **Solvation:** GBSA (ground state) / COSMO (excited state), toluene (ε = 2.38)

### NTO and CT Descriptor Analysis
- **Multiwfn (v3.8)** — NTO analysis, hole-electron overlap (S_he), CT number (Ω_CT),
  donor/acceptor localization (Λ_D, Λ_A), centroid distance (Δr)
- **29 descriptors** in 5 categories: energy, NTO overlaps, CT descriptors,
  oscillator strength, proxy RespA

### ML and Active Learning
- **scikit-learn (v1.7.2)** — SVR (best model), Gradient Boosting, Random Forest
- **SHAP (v0.45.0)** — KernelSHAP for SVR feature importance
- 5-fold cross-validation on 2,943 configurations (80/20 train-test split)

### High-Level Validation
- **ORCA (v6.1.0)** — Adiabatic S1/T1 optimization (14 molecules); CAM-B3LYP/def2-TZVP
  SOC calculations (8 top candidates)
- **PySCF** — ΔROKS LC-ωPBE/def2-SVP for BACN (ω = 0.16 bohr⁻¹)
- **Gaussian 16 (B3LYP/6-31G(d))** — TD-DFT benchmark (27 molecules)

---

## Machine Learning Reproducibility Guide (Article 3)

### Directory Structure

```
ML_reproducibility/
├── scripts/
│   ├── data_processing/
│   │   ├── build_features_747mol.py           # Merge all features into one CSV
│   │   ├── extract_stda_features_747mol.py    # Parse sTDA/sTD-DFT log files
│   │   ├── compute_ct_descriptors_747mol.py   # Extract CT from NTO molden files
│   │   ├── merge_ct_features_747mol.py        # Combine energy + CT features
│   │   └── run_hole_electron_analysis.sh      # Batch Multiwfn NTO analysis
│   └── experiments/
│       ├── advanced_ml_pipeline.py            # SVR + GB + RF benchmark
│       ├── advanced_al_experiment.py          # All 6 acquisition functions
│       ├── run_svr_shap_verification.py       # SVR KernelSHAP (Config F, 29 feat)
│       ├── ablation_study.py                  # Feature ablation (Models A/C/D)
│       ├── applicability_domain.py            # k-NN distance + leverage AD
│       ├── spectral_application_analysis.py   # Spectral overlap screening
│       ├── identify_candidates.py             # Multi-objective candidate filtering
│       ├── generate_advanced_figures.py       # Manuscript figures
│       └── interpret_model_shap.py            # SHAP importance plots
├── features/                                  # Pre-computed descriptor tables
├── results/                                   # Model outputs and SHAP values
└── figures/                                   # All manuscript figures
```

### Feature Description (29-Descriptor Set, Config F)

| Category | Features | SHAP Importance |
|----------|----------|----------------|
| Energy | E_S1, E_T1, HOMO-LUMO gap | 92% (E_T1=48%, E_S1=43%) |
| NTO/CT descriptors | S_he^S1, S_he^T1, Ω_CT, Λ_D, Λ_A, Δr (S1+T1+diffs) | 8% |
| Oscillator strength | f_S1 | 0.5% |
| NTO overlap | S_NTO^S1, S_NTO^T1 | <1% |
| Proxy RespA | ΔS_NTO, products, log-transforms | <1% |

> **SHAP note:** Values above are from SVR KernelSHAP, Config F (29 features,
> 5-fold CV, seeds 42/123/777). Earlier README versions cited RF Gini importances.
> Those values are superseded. See `results/shap_svr_article3_correction_note.json`.

### Reproducing the ML Results

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install numpy pandas scikit-learn==1.7.2 shap matplotlib joblib rdkit-pypi

# 2. Train SVR (reproduces Table 1 of manuscript)
cd ML_reproducibility/scripts/experiments
python advanced_ml_pipeline.py
# -> results/advanced_ml_results.json  (SVR: MAE=0.025 eV, R²=0.956)

# 3. Reproduce Article 3 authoritative SHAP values (Config F)
python run_svr_shap_verification.py
# -> results/shap_svr_modelC_configF_article3.csv

# 4. Run active learning (all 6 strategies)
python advanced_al_experiment.py
# -> results/advanced_al_results.json

# 5. Reproduce manuscript figures
python generate_advanced_figures.py
```

---

## Adiabatic Validation Summary (Article 3)

Vertical ΔE_ST (at S0 geometry) vs. adiabatic ΔE_ST (at S1/T1 optimized geometries)
for 14 representative molecules in gas and toluene:

| Metric | Value |
|--------|-------|
| Mean absolute error | 0.195 eV |
| Systematic offset | +0.070 eV |
| R² | 0.704 |
| Linear fit slope | 1.002 |
| Best agreement | APPT-PXZ gas (0.019 eV) — rigid molecule |
| Worst agreement | 2CzTPE gas (0.548 eV) — flexible propeller structure |

**Interpretation:** Rigid D-A molecules show excellent agreement (<0.1 eV). Flexible
molecules with large geometric relaxation show larger deviations (0.3–0.5 eV). The
vertical approximation is valid for ranking purposes; adiabatic corrections are
needed for quantitative k_RISC prediction.

---

## Dataset Overview

**Core dataset:** 747 TADF emitter molecules (Articles 1–2)
spanning D-A, D-A-D, MR-TADF, TSCT, and other architectures.
**Training set:** 640 molecules (used in Article 3 ML models)
**Configurations:** 2,943 (2 calculation methods × 2 environments: gas + toluene)
**Features:** 29 descriptors per configuration (Config F)

**Virtual library expansion:** 22,194 molecules (Article 3, REQ-1)
Source: PubChem CID-SMILES (January 2026), filtered for TADF-relevant D-A structures.
Top 500 candidates selected by heuristic TADF scoring (score ≥ 0.9).

---

## Interactive Visualization

**`interactive_umap.html`** — Open in any web browser to explore the 747-molecule
chemical space as a UMAP scatter plot. Hover over any point to see:
- Molecule name and SMILES
- ΔE_ST (gas + toluene)
- Molecular architecture
- TADF score

This file is part of the Article 3 open-data package (DOI: 10.5281/zenodo.14241084).

---

## Citation

If you use this data or scripts, please cite the relevant article(s):

```bibtex
@article{tchapet2026benchmark,
  title   = {xTB-Based High-Throughput Screening of TADF Emitters: 747-Molecule Benchmark},
  author  = {Tchapet Njafa, Jean-Pierre and Kameni Tcheuffa, Elvira Vanelle and
             Foumkpou, Aissatou Maghame and Nana Engo, Serge Guy},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2026},
  doi     = {10.1021/acs.jcim.5c02978}
}

@article{tchapet2026design,
  title   = {Data-Driven Design Guidelines for TADF Emitters from a High-Throughput
             Screening of 747 Molecules},
  author  = {Tchapet Njafa, Jean-Pierre and Kameni Tcheuffa, Elvira Vanelle and
             Foumkpou, Aissatou Maghame and Nana Engo, Serge Guy},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2026},
  doi     = {10.1021/acs.jcim.5c03068}
}

@article{tchapet2026ml,
  title   = {Accelerated Discovery of TADF Emitters via Physics-Informed Machine
             Learning and Multi-Fidelity Active Learning},
  author  = {Tchapet Njafa, Jean-Pierre and Teguia Kouam, Steve Cabrel and
             Mvoto Kongo, Patrick Sorrel and Samafou, Panebei and
             Nana Engo, Serge Guy},
  journal = {Digital Discovery},
  year    = {2026},
  doi     = {10.5281/zenodo.14241084}
}
```

---

## License

Data and scripts are available under **CC-BY-4.0** (see `LICENSE`).
Computational tools used: xTB, CREST, sTDA, Multiwfn, RDKit, PySCF are freely
available under their respective licenses.

---

## Contact

jean-pierre.tchapet@facsciences-uy1.cm
*Department of Physics, Faculty of Science, University of Yaoundé I, Cameroon*

---

*Last updated: June 2026 — Added virtual library expansion (22,194 molecules,
REQ-1), interactive UMAP, and Digital Discovery manuscript figures. Full Article 3
README rewrite: removed Chemistry of Materials preparation notes, updated to
Digital Discovery submission.*
