#!/usr/bin/env python3
"""
dimensionality_check.py  (Task R2, prong b)
===========================================
Answer "2048-bit Morgan on 231 molecules is over-parameterised; R^2 CI dips to -0.19".

Sweep Morgan fingerprint width (512/1024/2048) and add a low-dimensional, regularised
reference (ElasticNet on RDKit descriptors). Report MAE and the R^2 95% bootstrap CI
under scaffold CV. If lower dimensionality / regularisation lifts the negative R^2 tail,
the critique is mitigated; if not, the negative tail is intrinsic to variance-explained
on a low-signal target (the honest reading).

Output: data/dim_check.json (+ console)
"""
import json, sys
from pathlib import Path
import numpy as np
from rdkit.Chem import AllChem
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

ROOT = Path(__file__).parent.parent
ds = load_dataset()
y = ds.y
scaf = ds.scaf

def boot_r2_ci(yt, yp, n=2000):
    rng = np.random.default_rng(0); s = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        try: s.append(r2_score(yt[i], yp[i]))
        except Exception: pass
    return [round(float(np.percentile(s, 2.5)), 3), round(float(np.percentile(s, 97.5)), 3)]

def evals(X, model=None):
    model = model or rf()
    yp = cross_val_predict(model, X, y, cv=GroupKFold(5), groups=scaf)
    return dict(MAE=round(mean_absolute_error(y, yp), 4),
                R2=round(r2_score(y, yp), 4), R2_CI=boot_r2_ci(y, yp))

out = {}
for nb in (512, 1024, 2048):
    M = np.vstack([np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nb), dtype=float)
                   for m in ds.mols])
    out[f'Morgan_{nb}_RF'] = evals(M)

Xr, _ = make_features(ds, 'RDKit')
out['RDKit_ElasticNet'] = evals(Xr, make_pipeline(StandardScaler(), ElasticNetCV(cv=5, random_state=0)))
Xn, _ = make_features(ds, 'NTO')
out['NTO_ElasticNet'] = evals(Xn, make_pipeline(StandardScaler(), ElasticNetCV(cv=5, random_state=0)))

# does any setting lift the lower R^2 CI to >= 0 ?
out['_summary'] = dict(
    note='lower R2 CI >= 0 means the model is distinguishable from the mean at 95%',
    settings_with_nonneg_R2_CI=[k for k, v in out.items()
                                if isinstance(v, dict) and 'R2_CI' in v and v['R2_CI'][0] >= 0])
(ROOT/'data'/'dim_check.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print('\nSaved -> data/dim_check.json')
