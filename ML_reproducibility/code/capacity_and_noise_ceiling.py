#!/usr/bin/env python3
"""
capacity_and_noise_ceiling.py  (Task R8)
========================================
Test whether the ~0.096 eV accuracy floor is a real ceiling or an excuse.

(a) CAPACITY test: under the SAME Bemis-Murcko scaffold GroupKFold, compare the
    headline random forest against higher-/different-capacity learners
    (HistGradientBoosting, MLP, SVR, ElasticNet). If none beats the RF MAE, model
    capacity is NOT the limit.

(b) CLEAN-LABEL test: restrict to molecules whose experimental label is internally
    consistent (single report, or multi-report inter-report SD <= SD_THRESHOLD, a
    threshold fixed BEFORE looking at any MAE). If MAE drops on clean labels, the
    ceiling is label noise, not model capacity.

Output: data/ceiling_tests.json (+ console)
"""
import json, ast, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _dataset import load_dataset, make_features  # noqa: E402

ROOT = Path(__file__).parent.parent
SD_THRESHOLD = 0.05   # eV — fixed BEFORE seeing results (pre-registered)

ds = load_dataset()
y = ds.y
scaf = ds.scaf
X, _ = make_features(ds, 'NTO')
gkf = GroupKFold(5)

def evalu(model, Xm, ym, groups):
    yp = cross_val_predict(model, Xm, ym, cv=GroupKFold(5), groups=groups)
    return dict(MAE=round(mean_absolute_error(ym, yp), 4),
                R2=round(r2_score(ym, yp), 4),
                rho=round(float(spearmanr(yp, ym).correlation), 3))

models = {
    'RandomForest (headline)': RandomForestRegressor(400, random_state=0, n_jobs=-1),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=0),
    'MLP (100,50)': make_pipeline(StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=0)),
    'SVR (RBF)': make_pipeline(StandardScaler(), SVR(C=1.0, gamma='scale')),
    'ElasticNetCV': make_pipeline(StandardScaler(), ElasticNetCV(cv=5, random_state=0)),
}
capacity = {name: evalu(m, X, y, scaf) for name, m in models.items()}
rf_mae = capacity['RandomForest (headline)']['MAE']
beats_rf = [n for n, r in capacity.items() if r['MAE'] < rf_mae - 1e-9 and 'headline' not in n]

# ---- per-molecule inter-report SD (same corpus QC as _dataset) -------------
exp = pd.read_csv(ROOT/'MCA_submission/tables/experimental_delta_est_extracted.csv')
exp = exp[exp['specifier'].astype(str).str.contains('EST', case=False, na=False)]
nm = lambda x: ([str(n).strip() for n in ast.literal_eval(x)] if str(x).startswith('[') else [str(x)])
def vv(v):
    try:
        p = ast.literal_eval(v); return float(p[0]) if isinstance(p, list) else float(p)
    except Exception:
        try: return float(re.findall(r'[-\d.]+', str(v))[0])
        except Exception: return np.nan
exp['nl'] = exp['compound.names'].map(nm); exp['v'] = exp['standard_value'].map(vv)
exp = exp[np.isfinite(exp.v) & exp.v.between(-0.3, 1.5)]
reports = {}
for _, r in exp.iterrows():
    for n in r['nl']:
        reports.setdefault(n, []).append(r.v)
sd = {k: (np.std(v, ddof=1) if len(v) >= 2 else 0.0) for k, v in reports.items()}

mol = ds.df.molecule.astype(str).values
keep = np.array([sd.get(m, 0.0) <= SD_THRESHOLD for m in mol])
clean = evalu(RandomForestRegressor(400, random_state=0, n_jobs=-1),
              X[keep], y[keep], scaf[keep])

out = dict(
    capacity_test=dict(
        note='same scaffold GroupKFold(5), NTO features, 231 molecules',
        results=capacity, rf_mae=rf_mae, models_beating_rf=beats_rf,
        verdict=('no model beats RF -> capacity is not the limit' if not beats_rf
                 else 'a higher-capacity model beats RF -> ESCALATE, ceiling claim wrong')),
    clean_label_test=dict(
        sd_threshold_eV=SD_THRESHOLD, n_clean=int(keep.sum()), n_full=int(len(y)),
        clean_subset=clean, full_set_rf=capacity['RandomForest (headline)'],
        verdict=('MAE drops on clean labels -> ceiling is label noise'
                 if clean['MAE'] < rf_mae else 'no improvement on clean labels')))
(ROOT/'data'/'ceiling_tests.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print('\nSaved -> data/ceiling_tests.json')
