# Adiabatic Validation of sTD-DFT-xTB Vertical Approximation

This folder contains data and scripts for the adiabatic validation study performed in **Article 3** (ML/AL prediction of ΔE_ST for TADF emitters). It benchmarks vertical sTD-DFT-xTB excitation energies against adiabatic singlet-triplet gaps obtained from ORCA geometry optimizations of the excited states.

## Purpose

The vertical approximation computes ΔE_ST at the ground-state geometry, which is computationally 60× faster than the full adiabatic treatment. This study quantifies the systematic error introduced by this approximation across 14 representative TADF molecules in both gas phase and toluene solvent.

## Key Results

| Metric | Value |
|--------|-------|
| Mean absolute deviation (MAE) | 0.195 eV |
| R² (vertical vs adiabatic) | 0.704 |
| Slope of linear fit | 1.002 |
| Intercept | +0.070 eV |
| Min deviation | 0.019 eV (APPT-PXZ, gas) |
| Max deviation | 0.548 eV (2CzTPE, gas) |

**Conclusion**: The vertical approximation introduces a systematic +0.2 eV offset, acceptable for high-throughput screening where molecular ranking matters more than absolute accuracy.

## Molecules

14 TADF molecules validated in gas and toluene (28 calculations total):
`DMAC-DPS`, `DMAC-TRZ`, `4CzIPN`, `PXZ-NAI`, `TPA-APy`, `BACN`, `BMZ-TZ`, `2CzPN`, `APPT-PXZ`, `2PXZP`, `ACRSA`, `ACRFLCN`, `2CzTPE`, `BBPA`

## Directory Structure

```
adiabatic_validation/
├── vertical_vs_adiabatic_comparison.csv   # Full comparison table (28 rows)
├── adiabatic_results.json                 # Extracted adiabatic energies (S1, T1)
├── vertical_vs_adiabatic_comparison.pdf   # Comparison figure (2 panels)
├── vertical_vs_adiabatic_comparison.png   # Same figure (PNG format)
├── extract_orca_results.py                # Script to parse ORCA .out files
├── orca_tadf_calc.py                      # ORCA calculation driver
└── xyz_files/
    ├── gas/                               # xTB-optimized S0/T1 geometries (gas)
    └── toluene/                           # xTB-optimized S0/T1 geometries (toluene)
```

## Methodology

1. **Ground-state geometries** were optimized using xTB (GFN2-xTB) + CREST (conformational search), the same workflow used for the 747-molecule screening.
2. **T1 geometry optimizations** were performed with ORCA 6.1.0 at the same GFN2-xTB level from the xTB-optimized S0 geometry as starting point.
3. **Adiabatic ΔE_ST** = E(S1 at S1-opt geometry) − E(T1 at T1-opt geometry), computed as single-point sTD-DFT-xTB calculations on the respective optimized geometries.
4. **Comparison** against the vertical ΔE_ST from the main 747-molecule screening.

## Usage

```bash
# Extract adiabatic energies from ORCA output files
python extract_orca_results.py --orca-dir path/to/orca_outputs/ --output adiabatic_results.json

# Run ORCA calculations for a new molecule
python orca_tadf_calc.py --molecule MOLECULE_NAME --xyz path/to/geometry.xyz
```

## Vertical vs Adiabatic ΔE_ST Table

| Molecule | Phase | Vertical (eV) | Adiabatic (eV) | Difference (eV) |
|----------|-------|---------------|----------------|-----------------|
| DMAC-DPS | gas   | 0.188 | 0.385 | +0.197 |
| DMAC-DPS | toluene | 0.207 | 0.558 | +0.351 |
| DMAC-TRZ | gas   | 0.261 | 0.442 | +0.181 |
| DMAC-TRZ | toluene | 0.271 | 0.638 | +0.367 |
| 4CzIPN   | gas   | 0.361 | 0.402 | +0.041 |
| 4CzIPN   | toluene | 0.335 | 0.436 | +0.101 |
| PXZ-NAI  | gas   | 0.355 | 0.243 | −0.112 |
| PXZ-NAI  | toluene | 0.360 | 0.491 | +0.131 |
| TPA-APy  | gas   | 0.876 | 0.753 | −0.123 |
| TPA-APy  | toluene | 0.765 | 0.895 | +0.130 |
| BACN     | gas   | 0.991 | 1.129 | +0.138 |
| BACN     | toluene | 0.921 | 1.063 | +0.142 |
| BMZ-TZ   | gas   | 0.938 | 0.410 | −0.528 |
| BMZ-TZ   | toluene | 0.933 | 0.700 | −0.233 |
| 2CzPN    | gas   | 0.613 | 0.560 | −0.053 |
| 2CzPN    | toluene | 0.621 | 0.652 | +0.031 |
| APPT-PXZ | gas   | 0.350 | 0.331 | −0.019 |
| 2PXZP    | gas   | 0.467 | 0.384 | −0.083 |
| 2PXZP    | toluene | 0.485 | 0.639 | +0.154 |
| ACRSA    | gas   | 0.447 | 0.204 | −0.243 |
| ACRSA    | toluene | 0.372 | 0.677 | +0.305 |
| ACRFLCN  | gas   | 0.590 | 0.302 | −0.288 |
| ACRFLCN  | toluene | 0.548 | 0.576 | +0.028 |
| 2CzTPE   | gas   | 1.143 | 1.691 | +0.548 |
| 2CzTPE   | toluene | 1.042 | 1.535 | +0.493 |
| BBPA     | gas   | 1.521 | 1.641 | +0.120 |
| BBPA     | toluene | 1.433 | 1.562 | +0.129 |

## Citation

If you use this data, please cite Article 3 of this repository series (see main README.md).

---
*Last updated: March 2026*
