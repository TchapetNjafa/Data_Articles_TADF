#!/usr/bin/env python3
"""
solvent_analysis.py  (Task D3)
==============================
Rebut "the consensus gap (median over mixed conditions) represents no physical reality".

The extracted corpus does NOT record solvent/phase per measurement (the `phase` and
`atmosphere` fields are empty for all 1490 EST rows). We therefore cannot stratify by
solvent; instead we quantify label heterogeneity directly via the spread among
INDEPENDENT literature reports for the same molecule -- the empirical footprint of the
uncontrolled measurement conditions (solvent, host, method) that Olivier et al. (2018)
identify as the dominant obstacle to TADF modelling.

Computes, over molecules with >=2 reports:
  - distribution of intra-molecule report spread (max-min) and std
  - fraction whose spread exceeds the model MAE (~0.096 eV)
  - how the consensus (median) reduces variance vs picking a single random report

Output: data/solvent_heterogeneity.json (+ console summary)
"""
import json, ast, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).parent.parent
MODEL_MAE = 0.096  # headline NTO scaffold-CV MAE (data/final_model_metrics.json)

exp = pd.read_csv(ROOT/'MCA_submission/tables/experimental_delta_est_extracted.csv')
exp = exp[exp['specifier'].astype(str).str.contains('EST', case=False, na=False)]
nm = lambda x: ([str(n).strip() for n in ast.literal_eval(x)]
                if str(x).startswith('[') else [str(x)])
def vv(v):
    try:
        p = ast.literal_eval(v); return float(p[0]) if isinstance(p, list) else float(p)
    except Exception:
        try: return float(re.findall(r'[-\d.]+', str(v))[0])
        except Exception: return np.nan
exp['nl'] = exp['compound.names'].map(nm); exp['v'] = exp['standard_value'].map(vv)
exp = exp[np.isfinite(exp.v) & exp.v.between(-0.3, 1.5)]

# restrict to the 231 benchmark molecules (present in the feature set)
feat = pd.read_csv(ROOT/'CALCULATIONS-MADE/data_processing/combined_features_747mol_full_ct.csv')
NTOcol = ['S1_S_he', 'Delta_E_ST_eV']
feat = feat[(feat.environment == 'gas') & (feat.method == 'stda')].dropna(subset=NTOcol)
fm = set(feat.molecule.astype(str))

reports = {}
for _, r in exp.iterrows():
    for n in r['nl']:
        if n in fm:
            reports.setdefault(n, []).append(r.v)

# confirm phase metadata really is absent
phase_present = int(exp['phase'].notna().sum())

multi = {k: np.array(v) for k, v in reports.items() if len(v) >= 2}
spreads = np.array([v.max() - v.min() for v in multi.values()])   # range (max-min)
stds = np.array([v.std(ddof=1) for v in multi.values()])          # SD; matches provenance

# variance reduction from median consensus: expected squared error of a single random
# report about its molecule's median, averaged over multi-report molecules
single_report_rmse = float(np.sqrt(np.mean([
    np.mean((v - np.median(v))**2) for v in multi.values()])))

out = dict(
    n_est_rows=int(len(exp)),
    phase_metadata_rows_present=phase_present,           # == 0  (no solvent recorded)
    n_benchmark_molecules=len(reports),
    n_multi_report_molecules=len(multi),
    intra_molecule_range_eV=dict(   # max - min across a molecule's reports
        mean=round(float(spreads.mean()), 3),
        median=round(float(np.median(spreads)), 3),
        p90=round(float(np.percentile(spreads, 90)), 3),
        max=round(float(spreads.max()), 3),
        frac_exceeding_model_MAE=round(float(np.mean(spreads > MODEL_MAE)), 3)),
    intra_molecule_std_eV=dict(     # SD; max should match si_corpus_provenance (0.564)
        mean=round(float(stds.mean()), 3),
        median=round(float(np.median(stds)), 3),
        max=round(float(stds.max()), 3)),
    single_report_rmse_about_median_eV=round(single_report_rmse, 3),
    model_mae_eV=MODEL_MAE,
    note=("phase/atmosphere unrecorded in corpus; heterogeneity quantified via "
          "inter-report spread, the empirical proxy for uncontrolled solvent/host/method"))

(ROOT/'data'/'solvent_heterogeneity.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print('\nSaved -> data/solvent_heterogeneity.json')
