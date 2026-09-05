# TADF Emitters Computational Data Repository [![DOI](https://zenodo.org/badge/1082469447.svg)](https://doi.org/10.5281/zenodo.17436069)

This repository contains computational data, scripts, and manuscripts for three
related research articles on thermally activated delayed fluorescence (TADF)
emitters, based on high-throughput computational screening of 747 experimentally
known molecules, extended to a 22,194-molecule virtual library for Article 3.

> Article 3 code and data tested with **Python 3.12**, **scikit-learn 1.8.0**,
> **RDKit 2024.03+**, **SHAP 0.50.0**, available under **CC-BY-4.0** license.

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

### Article 3: How Far Can Structure-Based Machine Learning Predict Experimental Singlet–Triplet Gaps? An Honest Benchmark for TADF Emitter Triage
**Status:** 📝 Under submission to *RSC Advances* (September 2026)
**Prior submission:** *Digital Discovery*, RSC — declined 27 Aug 2026; the referee
reports were addressed and the manuscript substantially revised (see below)
**Authors:** Jean-Pierre Tchapet Njafa, Steve Cabrel Teguia Kouam, Patrick Sorrel
Mvoto Kongo, Panebei Samafou, Serge Guy Nana Engo
**Zenodo DOI:** [10.5281/zenodo.17436069](https://doi.org/10.5281/zenodo.17436069)

A scaffold-validated, deliberately honest benchmark of how accurately cheap
structure-based machine learning can predict the **experimental** singlet–triplet
gap (ΔE_ST). On 231 donor–acceptor molecules spanning 212 Bemis–Murcko scaffolds,
random forests built from the 2D structure alone (Morgan fingerprints, RDKit
descriptors) predict the measured ΔE_ST with MAE ≈ 0.096 eV — matching
physics-informed NTO descriptors that require a semi-empirical excited-state
calculation. The two agree to within **0.017 eV** on paired folds, so the
excited-state step buys no *accuracy*; it does, however, buy *ranking* that
survives a change of source laboratory (Spearman ρ 0.35 vs 0.16 under a
source-paper split, Δρ = 0.19, CI excluding zero). Accuracy is bounded by the
label: repeat reports of the same compound scatter by 0.072 eV RMS and the median
actually regressed on is known only to 0.049 eV. A 609-core-hour range-separated
TD-DFT reference does not beat the structure-only model. The model is a modest
triage filter (1.2–1.4× enrichment over random). **No ranked candidate shortlist
is deposited:** on the 22,194-molecule filtered library the conformal intervals
are ~7.6× the predicted gap and every one of the 100 lowest-predicted intervals
spans a negative gap, so the procedure refuses molecule-level selection.

> ⚠️ **Supersedes an earlier, withdrawn version.** A prior preprint of this work
> (Zenodo 10.5281/zenodo.14241084) reported an SVR model at MAE 0.025 eV / R² 0.956,
> a 27.7× active-learning speed-up, multi-objective Pareto optimisation, and
> device-level EQE / k_RISC predictions. On review those results were found to rest
> on **feature–target leakage** (regressing ΔE_ST on its own constituent S₁/T₁
> energies, ΔE_ST ≡ E_{S₁}−E_{T₁}) and on molecule-independent device estimates, and
> have been **withdrawn**. The honest benchmark below replaces them; the retired
> active-learning / device scripts and figures have been removed from this deposit.

#### Key Results (Article 3, honest benchmark)

| Quantity | Value |
|----------|-------|
| Dataset | 231 donor–acceptor molecules / 212 Bemis–Murcko scaffolds |
| Target | experimental ΔE_ST (median of literature reports) |
| Best accuracy | MAE = 0.091 eV (Morgan) / 0.096 eV (NTO); R² ≈ 0.25–0.31 |
| Ranking | Spearman ρ ≈ 0.26–0.36; mean-value baseline MAE = 0.107 eV |
| Validation | scaffold GroupKFold + label-permutation test (p = 0.001) + Butina cluster-CV (MAE 0.099–0.104) |
| Capacity test | RF best; HistGBR (0.105), MLP (0.169), SVR (0.120), ElasticNet (0.102) do **not** beat it |
| Morgan vs NTO | equivalent to within **0.017 eV** (paired, identical folds, Wilcoxon p = 0.66) |
| Ranking under protocol transfer | source-paper split: NTO ρ = 0.35 vs Morgan 0.16, **Δρ = 0.19** (CI 0.04–0.34) |
| Label bound | single-report scatter 0.072 eV RMS (molecule-weighted); **precision of the median target 0.049 eV** |
| TD-DFT reference | wB97X-D4/def2-SVP TDA, 231/231, **609 core-hours**; raw gap MAE 0.654 eV, R² = −16.3 (worst predictor); as a feature 0.091 → 0.087 eV |
| Functional dependence | B3LYP vs CAM-B3LYP up to 0.58 eV — bounds a **computed** reference, not this regression; not combined into one floor |
| Vertical vs adiabatic | MAE 0.195 eV, largely additive (slope 1.00, ρ 0.71) but residual RMS 0.245 eV, n = 14 |
| Triplet manifold | T₁–T₂ for 14 emitters (CAM-B3LYP); sTDA-vs-CAM agreement unresolved (ρ = 0.43, p = 0.40, n = 6) |
| Triage | 1.2–1.4× enrichment on held-out labelled data; **no ranked shortlist deposited** (conformal intervals refuse molecule-level selection) |
| Condition-controlled test | source-paper proxy (no solvent metadata exists): **null**, Δ advantage 0.005 eV, CI −0.014 to 0.035 |
| Circularity / leakage | quantified: linear model given the constituent energies recovers the target exactly (CV R² = 1.000) vs 0.36 without; a random forest shows only 0.59, understating the leak |
| Estimator selection | nested CV reproduces the reported MAE exactly; selection optimism 0.000 eV |

Headline numbers trace to `ML_reproducibility/data/final_model_metrics.json` and the
other JSON/CSV outputs in `ML_reproducibility/data/`, regenerated by the scripts in
`ML_reproducibility/code/`.

**Provenance.** `ML_reproducibility/analysis-ledger.md` records, for every number above,
the method, uncertainty, written interpretation and caveats — including entries that
*withdraw* earlier claims. Notably: **A-008** retracts a label-quality result after a
constant predictor was shown to reproduce it; **A-008b** restores a figure that an
earlier internal check had wrongly rejected; **A-010** corrects run-cost figures that were
a power-law projection quoted as a measurement (604 → 609 core-hours, median 18.2 → 16.1
min). Superseded entries are in `ML_reproducibility/analysis-ledger-archive/`.

**A note on the enrichment files.** Three deposited files report enrichment under
different but individually correct conventions: `enrichment_curve.json` is the Morgan
model by top-*fraction*, `multi_criteria_triage.json` is the NTO model by top-fraction,
and `triage_metrics.json` is the Morgan model by top-*N molecules* (10/20/30/50).
Comparing across them without matching the convention will appear to show a
contradiction; it is not one.

---

### Relationship Between Articles

1. **Article 1** establishes the xTB/sTDA high-throughput screening protocol (747 molecules)
2. **Article 2** extracts quantitative design rules and identifies 127 high-performance candidates
3. **Article 3** asks, with scaffold-validated rigour, how far cheap structural
   features can predict the *experimental* ΔE_ST, and reports the limits openly

All three articles share the same 747-molecule core dataset and computational
infrastructure. Article 3 evaluates the 231-molecule experimentally-labelled subset,
extends the chemical space with the virtual library (see `virtual_library_expansion/`),
and deposits its honest analysis code and outputs in `ML_reproducibility/code/` and
`ML_reproducibility/data/`. The manuscript (PDF + sources) is in
`ARTICLEs_TADF/Article3_TADF-Emitter-Triage-Honest-Benchmark/`.

---

## Repository Contents

### Root-level files

- **`interactive_umap.html`** — Interactive UMAP visualization of the 747-molecule
  chemical space (hover for molecule details: name, ΔE_ST, architecture). Open in
  any modern web browser. Referenced in the Article 3 Data Availability statement
  (DOI: 10.5281/zenodo.17436069).
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

### `ML_reproducibility/` — Honest ML benchmark package (Article 3)

Analysis code and verified outputs for the structure-based ΔE_ST benchmark. Every
headline number traces to a file in `data/`, regenerated by a script in `code/`.

**`code/`** — Analysis scripts:
- `_dataset.py` — shared loader defining the 231-molecule / 212-scaffold set
- `finalize_model.py` — headline random forest, metrics, and Fig 1
- `compute_baselines.py`, `r2_ci_power_analysis.py` — baselines, bootstrap CIs, power analysis
- `scaffold_split_analysis.py`, `cluster_cv_analysis.py` — split rigour (label-permutation + Butina cluster-CV)
- `capacity_and_noise_ceiling.py`, `dimensionality_check.py` — capacity and over-parameterisation tests
- `permutation_importance_check.py` — SHAP cross-check
- `extract_triplet_manifold.py` — T₁/T₂ extraction from ORCA outputs
- `adiabatic_validation_analysis.py` — vertical vs adiabatic gap
- `enrichment_curve.py`, `triage_evaluation.py`, `build_shortlist.py` — triage + conformal intervals
  (`build_shortlist.py` is retained because it produces the reported *negative* result: the
  intervals are ~7.6× the predicted gap, so no ranked list is released)
- `create_dft_benchmark_table.py`, `uq_calibration.py`, `si_supplementary_data.py`, `interactive_umap.py` — SI artifacts

**`data/`** — Verified model outputs:
- `final_model_metrics.json` — headline MAE/R²/ρ and feature-set comparison
- `scaffold_split_analysis.json`, `cluster_cv.json` — validation rigour
- `ceiling_tests.json`, `dim_check.json` — capacity / dimensionality
- `triplet_manifold.csv` (+ `triplet_manifold_stats.json`) — T₁–T₂ manifold (14 emitters)
- `vertical_vs_adiabatic_comparison.csv`, `adiabatic_validation_stats.json` — adiabatic check
- `enrichment_curve.json`, `triage_metrics.json` — triage curves (no shortlist is deposited)
- `shap_*`, `perm_importance.json`, `dft_benchmark_*` — supporting analyses

**`features/`** — Pre-computed descriptor tables (model inputs):
- `combined_features_747mol_full_ct.csv` — full NTO/CT descriptor set (the model's feature source)
- `ct_descriptors_747mol.csv`, `stda_features_747mol.csv`, `combined_features_747mol.csv`

**`scripts/data_processing/`** — feature-building scripts (sTDA parsing, CT/NTO extraction, feature merge).

> ⚠️ The active-learning, multi-objective, device-prediction and SVR-Model-C
> scripts, result files and figures from the earlier (withdrawn) version have been
> **removed** from this deposit; their claims do not appear in the current manuscript.

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

### Machine Learning
- **scikit-learn (v1.8.0)** — random forest (headline model), with gradient boosting,
  MLP, SVR and ElasticNet evaluated as capacity controls (none beats the RF)
- **SHAP (v0.50.0)** — exact TreeSHAP for random-forest feature importance,
  cross-checked by model-agnostic permutation importance
- Bemis–Murcko scaffold GroupKFold (k = 5) on 231 experimentally-labelled molecules,
  with label-permutation and Butina cluster-CV robustness checks

### High-Level Validation
- **ORCA (v6.1.0)** — Adiabatic S1/T1 optimization (14 molecules); CAM-B3LYP/def2-TZVP
  SOC calculations (8 top candidates)
- **PySCF** — ΔROKS LC-ωPBE/def2-SVP for BACN (ω = 0.16 bohr⁻¹)

---

## Machine Learning Reproducibility Guide (Article 3)

### Directory Structure

```
ML_reproducibility/
├── code/                                     # Honest analysis scripts
│   ├── _dataset.py                           # Shared 231-molecule / 212-scaffold loader
│   ├── finalize_model.py                     # Headline RF, metrics, Fig 1
│   ├── compute_baselines.py                  # Mean/median/permutation baselines + CIs
│   ├── r2_ci_power_analysis.py               # R² bootstrap CI + correlation power
│   ├── scaffold_split_analysis.py            # Label-permutation test, scaffold overlap
│   ├── cluster_cv_analysis.py                # Butina cluster cross-validation
│   ├── capacity_and_noise_ceiling.py         # RF vs GBR/MLP/SVR/EN; clean-label test
│   ├── dimensionality_check.py               # Morgan 512/1024/2048 + regularised refs
│   ├── permutation_importance_check.py       # SHAP cross-check
│   ├── extract_triplet_manifold.py           # T1/T2 from ORCA outputs
│   ├── adiabatic_validation_analysis.py      # Vertical vs adiabatic gap
│   ├── enrichment_curve.py                   # Triage precision curve + conformal
│   ├── triage_evaluation.py                  # Enrichment / precision@k
│   └── build_shortlist.py                    # Conformal intervals (no list released)
├── data/                                     # Verified outputs (every headline number)
├── features/                                 # Pre-computed descriptor tables (inputs)
└── scripts/data_processing/                  # Feature-building (sTDA parse, CT/NTO, merge)
```

### Feature set (35 NTO spatial descriptors, energy scalars excluded)

The model is trained on 35 natural-transition-orbital spatial descriptors
(hole–electron overlap S_he, CT number, donor/acceptor weights, separation Δr,
localisation and composites). The S₁/T₁ **energy scalars are deliberately excluded**:
including them makes the task circular (ΔE_ST ≡ E_{S₁}−E_{T₁}). Structure-only Morgan
fingerprints and RDKit descriptors match this NTO set, so the excited-state step adds
no accuracy. Exact TreeSHAP attributes ~60% of the weight to the top five descriptors,
led by S_he(S₁); the ranking is reproduced by permutation importance.

### Reproducing the ML results

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install numpy pandas scikit-learn shap rdkit matplotlib scipy

# 2. Headline model + metrics (MAE 0.096 eV, R² 0.25, rho 0.36)
cd ML_reproducibility/code
python finalize_model.py            # -> ../data/final_model_metrics.json + Fig 1

# 3. Validation rigour
python scaffold_split_analysis.py   # permutation test, p = 0.001
python cluster_cv_analysis.py       # Butina cluster-CV (MAE 0.099-0.104)

# 4. Ceiling tests (capacity + dimensionality)
python capacity_and_noise_ceiling.py
python dimensionality_check.py

# 5. Triplet manifold, triage curve, conformal intervals
python extract_triplet_manifold.py
python enrichment_curve.py
python build_shortlist.py           # conformal intervals; no ranked list is deposited
```

> Scripts use paths as published; the feature matrix is in
> `ML_reproducibility/features/combined_features_747mol_full_ct.csv`. Re-run from the
> manuscript repository for exact path resolution.

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
**Article 3 benchmark set:** 231 molecules with an experimentally-reported ΔE_ST
matched to a structure, spanning 212 Bemis–Murcko scaffolds.
**Features:** 35 NTO spatial descriptors (energy scalars excluded), plus structure-only
Morgan-fingerprint and RDKit baselines.

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

This file is part of the Article 3 open-data package (DOI: 10.5281/zenodo.17436069).

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
  title   = {How Far Can Structure-Based Machine Learning Predict Experimental
             Singlet--Triplet Gaps? An Honest Benchmark for TADF Emitter Triage},
  author  = {Tchapet Njafa, Jean-Pierre and Teguia Kouam, Steve Cabrel and
             Mvoto Kongo, Patrick Sorrel and Samafou, Panebei and
             Nana Engo, Serge Guy},
  journal = {RSC Advances},
  year    = {2026},
  doi     = {10.5281/zenodo.17436069}
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
REQ-1), interactive UMAP, and Article 3 manuscript figures. Full Article 3
README rewrite: removed Chemistry of Materials preparation notes, updated to
RSC Advances submission.*
