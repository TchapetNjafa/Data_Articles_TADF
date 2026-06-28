#!/usr/bin/env python3
"""
adiabatic_validation_analysis.py
================================
Quantify the vertical (ground-state-geometry) approximation used for the NTO
descriptors against a full adiabatic singlet-triplet gap.

Source data: data/vertical_vs_adiabatic_comparison.csv — byte-identical copy of
orca_package/results/vertical_vs_adiabatic_comparison.csv (this repo's own ORCA
calculation package). Vertical = sTDA-xTB at the ground-state geometry;
adiabatic = CAM-B3LYP/def2-TZVP (D3BJ, RIJCOSX) with excited-state geometry
optimisation in ORCA, 14 representative TADF emitters in gas and toluene.

Reports MAE, RMSE, R^2, linear fit (slope/intercept) and the worst case, to
establish that the vertical step is a systematic, bounded offset (rank-preserving)
rather than random error.

Output: data/adiabatic_validation_stats.json (+ console summary)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent
df = pd.read_csv(ROOT/'data'/'vertical_vs_adiabatic_comparison.csv')
d = df.dropna(subset=['Delta_EST_vertical_eV', 'Delta_EST_adiabatic_eV']).copy()
v = d['Delta_EST_vertical_eV'].to_numpy(float)
a = d['Delta_EST_adiabatic_eV'].to_numpy(float)

mae = float(np.mean(np.abs(v - a)))
rmse = float(np.sqrt(np.mean((v - a)**2)))
bias = float(np.mean(a - v))                      # adiabatic - vertical
r2 = float(pearsonr(v, a)[0]**2)
rho = float(spearmanr(v, a).correlation)          # rank preservation
slope, intercept = np.polyfit(v, a, 1)
imax = int(np.argmax(np.abs(v - a)))

out = dict(
    level_vertical='sTDA-xTB (ground-state geometry)',
    level_adiabatic='CAM-B3LYP/def2-TZVP, D3BJ, excited-state optimisation (ORCA)',
    n_points=int(len(d)), n_molecules=int(d.molecule.nunique()),
    phases=sorted(d.phase.unique().tolist()),
    mae_eV=round(mae, 3), rmse_eV=round(rmse, 3),
    mean_bias_adiabatic_minus_vertical_eV=round(bias, 3),
    pearson_r2=round(r2, 3), spearman_rho=round(rho, 3),
    linear_fit=dict(slope=round(float(slope), 3), intercept_eV=round(float(intercept), 3)),
    max_abs_dev_eV=round(float(np.abs(v - a)[imax]), 3),
    max_abs_dev_molecule=f"{d.iloc[imax].molecule} ({d.iloc[imax].phase})")
(ROOT/'data'/'adiabatic_validation_stats.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print('\nSaved -> data/adiabatic_validation_stats.json')
