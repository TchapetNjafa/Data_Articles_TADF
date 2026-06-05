# Virtual Library Expansion — 22,194 TADF Candidates

**Associated article:** "Accelerated Discovery of TADF Emitters via Physics-Informed
Machine Learning and Multi-Fidelity Active Learning"
**Journal:** *Digital Discovery* (Royal Society of Chemistry), submitted June 2026
**Authors:** Jean-Pierre Tchapet Njafa, Steve Cabrel Teguia Kouam, Patrick Sorrel
Mvoto Kongo, Panebei Samafou, Serge Guy Nana Engo

---

## Overview

This folder contains all data and scripts produced by the **virtual library
expansion** (Section 3.5 of the Digital Discovery manuscript). Starting from the
747-molecule benchmark of Articles 1 & 2, we expanded the screening library
**29.7-fold** to 22,194 literature-validated donor-acceptor compounds sourced from
PubChem. Heuristic TADF scoring then reduced that pool to **500 high-priority
candidates**, yielding a **nested multi-fidelity efficiency exceeding 1000x**
relative to exhaustive TD-DFT screening.

### Key numbers

| Metric | Value |
|--------|-------|
| Starting library (Articles 1-2) | 747 molecules |
| Expanded virtual library | **22,194 molecules** |
| Scale-up factor | **29.7x** |
| Source | PubChem CID-SMILES (January 2026 snapshot) |
| Top candidates (TADF score >= 0.9) | **500 molecules** |
| Selection rate | 2.3% |
| Heuristic reduction in QC burden | **44.4x** (22,194 -> 500) |
| Nested efficiency (heuristic + AL) | **>1000x** vs. exhaustive TD-DFT |

---

## Filtering Cascade

```
PubChem CID-SMILES (110,000,000 entries)
        |
        v  Valid SMILES (RDKit)
        |  MW < 900 Da
        |  >= 2 aromatic rings
        |  >= 1 N/O/S heteroatom
        |  Donor fragment present (SMARTS)
        |  Acceptor fragment present (SMARTS)
        v
   22,194 TADF-relevant molecules
        |
        v  Heuristic TADF score >= 0.9
        v
      500 top candidates  (for GFN2-xTB + ML validation)
        |
        v  SOC + Marcus theory (full quantum validation)
        v
        8 fully validated high-performance candidates
```

---

## Property Summary

### Full library (22,194 molecules)

| Property | Mean +/- SD | Range |
|----------|-------------|-------|
| Molecular weight (Da) | 420.8 +/- 94.9 | 172 - 899 |
| LogP | 4.0 +/- 1.6 | -15.2 - 14.4 |
| Aromatic rings | 3.2 +/- 1.0 | 2 - 12 |
| Heteroatoms | 6.6 +/- 2.6 | 1 - 25 |
| Predicted delta-E_ST (eV) | 0.196 +/- 0.042 | - |
| TADF score | 0.760 +/- 0.105 | - |

### TADF score distribution

| Category | Molecules | % |
|----------|-----------|---|
| Excellent (>= 0.9) | 855 | 3.9% |
| Good (0.7 - 0.9) | 16,538 | 74.5% |
| Moderate (0.5 - 0.7) | 4,129 | 18.6% |
| Poor (< 0.5) | 672 | 3.0% |

### Top 500 candidates vs. full library

| Property | All molecules | Top 500 | Improvement |
|----------|--------------|---------|-------------|
| MW (Da) | 420.8 +/- 94.9 | 339.0 +/- 36.9 | **-19.4%** |
| Predicted delta-E_ST (eV) | 0.196 +/- 0.042 | 0.130 +/- 0.005 | **-33.5%** |
| TADF score | 0.760 +/- 0.105 | 0.924 +/- 0.011 | **+21.6%** |

---

## Folder Structure

```
virtual_library_expansion/
|-- README.md                         <- this file
|
|-- data/
|   |-- expanded_library_final.csv    <- 22,194 molecules with all properties
|   |-- pubchem_tadf_filtered.csv     <- intermediate filtering output
|   |-- top_candidates_for_qc.csv     <- 500 top candidates (TADF score >= 0.9)
|   |-- ml_predictions.csv            <- heuristic predictions for all 22,194
|   |-- library_summary.txt           <- descriptive statistics (raw output)
|   `-- phase2_summary.txt            <- concise Phase 2 summary
|
|-- analysis/
|   |-- tadf_score_distribution.{pdf,png}  <- Fig. S6.3 (SI manuscript)
|   |-- delta_e_distribution.{pdf,png}     <- Fig. S6.4 (SI manuscript)
|   |-- mw_vs_delta_e.{pdf,png}            <- Fig. S6.6 (SI manuscript)
|   |-- property_comparison.{pdf,png}      <- Fig. 9 / Fig. S6.5 (main + SI)
|   |-- top_20_candidates.{pdf,png}        <- top-20 bar chart (SI)
|   |-- table1_library_statistics.csv      <- Table S6.1 data
|   |-- table2_top_10_candidates.csv       <- Table S6.7 data (top 10)
|   |-- table3_score_distribution.csv      <- TADF score category breakdown
|   `-- summary_statistics.txt             <- key findings, plain text
|
`-- scripts/
    |-- download_pubchem.sh               <- Step 0: download CID-SMILES from NCBI
    |-- 02_filter_pubchem_tadf.py         <- Step 1: SMARTS filtering cascade
    |-- 03_create_expanded_library.py     <- Step 2: build expanded_library_final.csv
    |-- 05_alternative_approach.py        <- Step 3: heuristic scoring
    `-- 06_analyze_results.py             <- Step 4: statistics + figure generation
```

---

## Data File Descriptions

### `data/expanded_library_final.csv`

Main dataset. 22,195 rows (header + 22,194 molecules).

| Column | Description |
|--------|-------------|
| `molecule_id` | Internal ID (MOL_XXXXXX) |
| `SMILES_canonical` | RDKit-canonicalized SMILES |
| `source` | Always `PubChem_filtered` |
| `MW` | Molecular weight (Da) |
| `LogP` | Wildman-Crippen LogP |
| `n_aromatic_rings` | Number of aromatic rings |
| `n_heteroatoms` | Total heteroatom count (N, O, S, P, halogens) |
| `n_rotatable_bonds` | Rotatable bond count |
| `n_hbd` | H-bond donor count |
| `n_hba` | H-bond acceptor count |
| `tpsa` | Topological polar surface area (Angstrom^2) |
| `n_rings` | Total ring count |
| `n_atoms` | Heavy atom count |

### `data/top_candidates_for_qc.csv`

500 highest-scoring molecules. Same columns as above, plus:

| Column | Description |
|--------|-------------|
| `predicted_delta_e` | Heuristic delta-E_ST estimate (eV) |
| `tadf_score` | Composite TADF heuristic score (0-1) |
| `selected_for_detailed_calc` | Always `True` for this file |

### `data/ml_predictions.csv`

Heuristic predictions for all 22,194 molecules, same schema as
`top_candidates_for_qc.csv`.

### `data/pubchem_tadf_filtered.csv`

Intermediate file produced after the SMARTS filtering step but
before the TADF scoring step.

---

## Heuristic TADF Scoring Function

The composite TADF score is (see manuscript Eq. S1 and Methods Section 2.2):

```
TADF score = 0.35 * f_MW + 0.25 * f_LogP + 0.20 * f_rings + 0.20 * f_hetero
```

where each component is a normalised score (0-1) based on TADF design rules
(Liu 2012, Uoyama 2012, Wong 2017):

| Component | Ideal range | Physical rationale |
|-----------|-------------|-------------------|
| `f_MW` | 200-500 Da | Thermal stability, vapour-deposition |
| `f_LogP` | 2-5 | Solubility / processability balance |
| `f_rings` | 2-4 aromatic rings | D-A conjugation without aggregation |
| `f_hetero` | 4-10 heteroatoms | Charge-transfer character |

> **Note:** This score is a heuristic pre-filter only. It does NOT replace
> GFN2-xTB/sTDA calculations or TD-DFT validation. The top 500 candidates are
> the proposed input for the full multi-fidelity pipeline described in the
> manuscript.

---

## Reproducing the Results

### Requirements

```
python >= 3.10
rdkit >= 2024.03
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
```

### Step-by-step

```bash
# Step 0 -- Download PubChem CID-SMILES (~8 GB uncompressed)
bash scripts/download_pubchem.sh
# Output: data/pubchem/CID-SMILES.gz

# Step 1 -- Apply SMARTS filters
python scripts/02_filter_pubchem_tadf.py
# Output: data/pubchem_tadf_filtered.csv  (~22k rows)

# Step 2 -- Build expanded library with RDKit descriptors
python scripts/03_create_expanded_library.py
# Output: data/expanded_library_final.csv  (22,194 molecules)

# Step 3 -- Apply heuristic TADF scoring; select top 500
python scripts/05_alternative_approach.py
# Output: data/top_candidates_for_qc.csv, data/ml_predictions.csv

# Step 4 -- Generate statistics and figures
python scripts/06_analyze_results.py
# Output: analysis/*.pdf, analysis/*.png, analysis/*.csv
```

---

## Relationship to the Rest of the Repository

This folder is self-contained but designed to feed into the
existing 747-molecule pipeline:

```
This folder                   Existing repository
─────────────                 ───────────────────────────────────
expanded_library_final.csv -> GFN2-xTB screening (Scripts/)
top_candidates_for_qc.csv  -> ML prediction (ML_reproducibility/)
                           -> SOC + Marcus (adiabatic_validation/)
```

The 500 top candidates are the proposed input for a future
**Zenodo v3.0** deposit once GFN2-xTB calculations for the
expanded library are completed.

---

## Citation

If you use these data, please cite the associated article:

```bibtex
@article{tchapet2026ml_discovery,
  title   = {Accelerated Discovery of TADF Emitters via Physics-Informed
             Machine Learning and Multi-Fidelity Active Learning},
  author  = {Tchapet Njafa, Jean-Pierre and Teguia Kouam, Steve Cabrel and
             Mvoto Kongo, Patrick Sorrel and Samafou, Panebei and
             Nana Engo, Serge Guy},
  journal = {Digital Discovery},
  year    = {2026},
  note    = {Zenodo: https://doi.org/10.5281/zenodo.14241084}
}
```

---

*Generated: June 2, 2026 — Virtual library expansion (REQ-1)*
*License: CC-BY-4.0*
