#!/usr/bin/env python3
"""
build_shortlist.py  (Task R10)
==============================
Answer "you refuse to give a ranked shortlist => zero actionable value" by providing
a top-N shortlist of the enumerated library ranked by predicted experimental DeltaE_ST
WITH split-conformal prediction intervals, so each candidate carries its uncertainty.
Honesty is preserved by the intervals, not by withholding the list.

Model: structure-only Morgan-FP RF trained on the 231-molecule benchmark.
Conformal: split-conformal absolute-residual quantile (90%).
NO device/EQE/k_RISC columns (withdrawn narrative) are produced.

Output: data/shortlist_top100.csv + data/shortlist_meta.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402
RDLogger.DisableLog('rdApp.*')

ROOT = Path(__file__).parent.parent
LIB = ROOT/'REQ1-implementation'/'data'/'expanded_library_final.csv'
ALPHA = 0.10
N_TOP = 100

# ---- conformal half-width from a calibration split of the benchmark ----------
ds = load_dataset()
y = ds.y
Xb, _ = make_features(ds, 'Morgan')
Xtr, Xcal, ytr, ycal = train_test_split(Xb, y, test_size=0.4, random_state=0)
m_cal = rf().fit(Xtr, ytr)
resid = np.abs(ycal - m_cal.predict(Xcal))
half = float(np.quantile(resid, 1 - ALPHA))

# ---- final model on all benchmark data --------------------------------------
model = rf().fit(Xb, y)

# ---- featurize library + predict --------------------------------------------
lib = pd.read_csv(LIB)
lib = lib.loc[:, ~lib.columns.duplicated()]          # drop duplicated header cols
smi_col = 'SMILES_canonical'
mols = [Chem.MolFromSmiles(str(s)) for s in lib[smi_col]]
ok = [i for i, mm in enumerate(mols) if mm is not None]
fp = np.vstack([np.array(AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, nBits=2048), dtype=float)
                for i in ok])
pred = model.predict(fp)
res = lib.iloc[ok].copy()
res['pred_delta_EST_eV'] = np.round(pred, 3)
res['PI_low_eV'] = np.round(pred - half, 3)
res['PI_high_eV'] = np.round(pred + half, 3)
res = res.sort_values('pred_delta_EST_eV').head(N_TOP)

cols = ['molecule_id', smi_col, 'pred_delta_EST_eV', 'PI_low_eV', 'PI_high_eV', 'MW', 'n_aromatic_rings']
cols = [c for c in cols if c in res.columns]
res[cols].to_csv(ROOT/'data'/'shortlist_top100.csv', index=False)

meta = dict(
    model='Morgan-FP random forest, structure only, trained on 231 benchmark molecules',
    conformal=f'split-conformal {int((1-ALPHA)*100)}% interval, half-width {round(half,3)} eV '
              f'(empirical coverage ~0.89, see data/enrichment_curve.json)',
    library_size=int(len(lib)), library_featurised=int(len(ok)),
    n_shortlist=int(len(res)),
    top_pred_range_eV=[float(res.pred_delta_EST_eV.min()), float(res.pred_delta_EST_eV.max())],
    caveat='Predictions carry +/- {:.2f} eV (90% PI). The list prioritises; it does not '
           'guarantee any individual candidate. No device-level quantity is predicted.'.format(half))
(ROOT/'data'/'shortlist_meta.json').write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
print('\nTop-5 preview:')
print(res[cols].head().to_string(index=False))
print('\nSaved -> data/shortlist_top100.csv + data/shortlist_meta.json')
