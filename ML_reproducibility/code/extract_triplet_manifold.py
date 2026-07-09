#!/usr/bin/env python3
"""
extract_triplet_manifold.py  (Task R7)
======================================
Extract the triplet manifold (T1, T2) from the repo's own ORCA TD-DFT outputs to
answer the "ignores T1-T2 splitting" critique with real data.

Each orca_package/results/<MOL>/<PHASE>/<MOL>_<PHASE>_tddft.out has TDA blocks:
  TD-DFT/TDA EXCITED STATES (SINGLETS)   -> "Mult 1" STATE lines
  TD-DFT/TDA EXCITED STATES (TRIPLETS)   -> "Mult 3" STATE lines
S1 = lowest Mult-1 state; T1,T2 = two lowest Mult-3 states (CAM-B3LYP/def2-TZVP,
vertical). delta_EST = S1-T1; T1_T2_gap = T2-T1.

Anchor (ACRSA toluene): S1=3.746, T1=3.374, T2=3.500 -> dEST=0.372, T1_T2=0.126.

Output: data/triplet_manifold.csv + data/triplet_manifold_stats.json
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).parent.parent
STATE = re.compile(r'STATE\s+\d+:.*?([\d.]+)\s*eV.*?Mult\s+(\d)')

rows = []
for out in sorted((ROOT/'orca_package'/'results').glob('*/*/*_tddft.out')):
    mol = out.parent.parent.name
    phase = out.parent.name
    singlets, triplets = [], []
    for line in out.read_text(errors='ignore').splitlines():
        m = STATE.search(line)
        if not m:
            continue
        ev, mult = float(m.group(1)), int(m.group(2))
        (singlets if mult == 1 else triplets if mult == 3 else []).append(ev)
    if not singlets or len(triplets) < 2:
        continue
    singlets.sort(); triplets.sort()
    s1, t1, t2 = singlets[0], triplets[0], triplets[1]
    rows.append(dict(molecule=mol, phase=phase,
                     S1_eV=round(s1, 3), T1_eV=round(t1, 3), T2_eV=round(t2, 3),
                     delta_EST_eV=round(s1 - t1, 3), T1_T2_gap_eV=round(t2 - t1, 3)))

df = pd.DataFrame(rows).sort_values(['molecule', 'phase']).reset_index(drop=True)
df.to_csv(ROOT/'data'/'triplet_manifold.csv', index=False)

g = df.T1_T2_gap_eV.to_numpy(float)
stats = dict(
    n_files=int(len(df)), n_molecules=int(df.molecule.nunique()),
    level='CAM-B3LYP/def2-TZVP TDA, vertical (ORCA)',
    T1_T2_gap_eV=dict(mean=round(float(g.mean()), 3), median=round(float(np.median(g)), 3),
                      min=round(float(g.min()), 3), max=round(float(g.max()), 3)),
    n_small_gap_lt_0p3=int((g < 0.3).sum()),
    n_total=int(len(g)))
(ROOT/'data'/'triplet_manifold_stats.json').write_text(json.dumps(stats, indent=2))

print(df.to_string(index=False))
print('\n' + json.dumps(stats, indent=2))
print('\nSaved -> data/triplet_manifold.csv + data/triplet_manifold_stats.json')
