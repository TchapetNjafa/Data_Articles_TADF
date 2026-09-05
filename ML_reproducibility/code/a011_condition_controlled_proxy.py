"""
a011_condition_controlled_proxy.py
==================================
Digital Discovery Referee 3, point 2, asked us to "test or discuss model performance on a
strictly solvent- and condition-controlled subset rather than attributing model failure
entirely to external data noise".

A strictly controlled subset cannot be built: `phase` and `atmosphere` are non-null for
0/1520 rows of the source table (ledger A-009), and `record_method` holds text-parser names.
The nearest available proxy is the SOURCE PAPER. Molecules whose reported values all come
from a single DOI were measured under one laboratory's conditions; molecules with values
from two or more DOIs certainly mix conditions. If uncontrolled conditions are what limits
accuracy, error should be lower on the single-source group.

CONTROL (the lesson of ledger A-008): any such subset comparison must be run against a
feature-free predictor fitted within the same subset. A-006 T5 was withdrawn precisely
because a constant predictor reproduced the apparent effect — the strata differed in the
dispersion of the target, not in label quality. That control is included here.

Output: data/a011_condition_controlled.json
"""
import ast, json, pathlib, re, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
SEED, K, NBOOT = 0, 5, 2000
rng = np.random.default_rng(SEED)

ds = load_dataset(); y, g = ds.y, ds.scaf
Xm, _ = make_features(ds, 'Morgan')
Xn, _ = make_features(ds, 'NTO')

def oof(X):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=K).split(X, y, g):
        p[te] = rf().fit(X[tr], y[tr]).predict(X[te])
    return p
pm, pn = oof(Xm), oof(Xn)

# ---- reconstruct per-molecule DOI multiplicity (same loader as ref3_p1p2p3.py) ----
exp = pd.read_csv(ROOT/'MCA_submission/tables/experimental_delta_est_extracted.csv')
exp = exp[exp['specifier'].astype(str).str.contains('EST', case=False, na=False)].copy()
def nm(x): return ([str(n).strip() for n in ast.literal_eval(x)] if str(x).startswith('[') else [str(x)])
def vv(v):
    try:
        q = ast.literal_eval(v); return float(q[0]) if isinstance(q, list) else float(q)
    except Exception:
        try: return float(re.findall(r'[-\d.]+', str(v))[0])
        except Exception: return np.nan
exp['nl'] = exp['compound.names'].map(nm); exp['v'] = exp['standard_value'].map(vv)
exp = exp[np.isfinite(exp.v) & exp.v.between(-0.3, 1.5)]
mols = set(ds.df.molecule.astype(str))
rows = [(n, r.v, r.doi) for _, r in exp.iterrows() for n in r['nl'] if n in mols]
long = pd.DataFrame(rows, columns=['molecule', 'v', 'doi'])
agg = long.groupby('molecule').agg(n_reports=('v','size'), n_doi=('doi','nunique'))
ndoi = agg.n_doi.reindex(ds.df.molecule.astype(str)).values
nrep = agg.n_reports.reindex(ds.df.molecule.astype(str)).fillna(1).values

single_src = np.nan_to_num(ndoi, nan=1) <= 1          # one laboratory -> internally consistent
multi_src  = np.nan_to_num(ndoi, nan=1) >= 2          # >=2 laboratories -> conditions mixed

def block(mask, name):
    if mask.sum() < 5: return None
    yv = y[mask]
    const = float(np.abs(yv - np.median(yv)).mean())   # feature-free predictor, same subset
    return dict(name=name, n=int(mask.sum()),
                morgan_MAE=round(float(mean_absolute_error(yv, pm[mask])), 4),
                nto_MAE=round(float(mean_absolute_error(yv, pn[mask])), 4),
                constant_predictor_MAE=round(const, 4),
                target_SD=round(float(yv.std()), 4),
                morgan_advantage_over_constant=round(const - float(mean_absolute_error(yv, pm[mask])), 4))

S, M = block(single_src, 'single-source (one DOI)'), block(multi_src, 'multi-source (>=2 DOIs)')

# bootstrap the difference in the model's *advantage over its own constant*, not raw MAE
def adv(mask):
    yv = y[mask]; return np.abs(yv-np.median(yv)).mean() - np.abs(yv-pm[mask]).mean()
i1 = np.where(single_src)[0]; i2 = np.where(multi_src)[0]
d = []
for _ in range(NBOOT):
    a = rng.choice(i1, len(i1), True); b = rng.choice(i2, len(i2), True)
    d.append((np.abs(y[a]-np.median(y[a])).mean()-np.abs(y[a]-pm[a]).mean()) -
             (np.abs(y[b]-np.median(y[b])).mean()-np.abs(y[b]-pm[b]).mean()))
lo, hi = np.percentile(d, [2.5, 97.5])

out = dict(
    meta=dict(script='code/a011_condition_controlled_proxy.py', seed=SEED, n_splits=K,
              n_bootstrap=NBOOT, cv='GroupKFold on Bemis-Murcko scaffolds (as A-001/A-006/A-007)'),
    why_proxy=('phase and atmosphere are non-null for 0/1520 source rows (A-009), so a strictly '
               'solvent-controlled subset cannot be constructed; source paper is the closest proxy '
               'for a single set of measurement conditions.'),
    single_source=S, multi_source=M,
    difference_in_advantage_over_constant=dict(
        delta=round(adv(single_src)-adv(multi_src), 4), CI=[round(float(lo),4), round(float(hi),4)],
        excludes_zero=bool(lo > 0 or hi < 0)),
    note=('Comparing each group against its OWN constant predictor controls for differing target '
          'dispersion between groups — the artefact that invalidated A-006 T5 (see A-008).'))
(ROOT/'data'/'a011_condition_controlled.json').write_text(json.dumps(out, indent=1))

for b in (S, M):
    print(f"  {b['name']:26s} n={b['n']:3d}  Morgan {b['morgan_MAE']:.4f}  NTO {b['nto_MAE']:.4f}  "
          f"constant {b['constant_predictor_MAE']:.4f}  targetSD {b['target_SD']:.4f}  "
          f"advantage {b['morgan_advantage_over_constant']:+.4f}")
dd = out['difference_in_advantage_over_constant']
print(f"\n  difference in advantage-over-constant (single - multi) = {dd['delta']:+.4f} eV"
      f"  CI [{dd['CI'][0]:+.4f}, {dd['CI'][1]:+.4f}]  excludes zero: {dd['excludes_zero']}")
print('\nwrote data/a011_condition_controlled.json')
