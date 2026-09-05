"""
ref1_gap_baseline.py
====================
Answers Referee 1 (DD-ART-07-2026-000486, rejected 27-Aug-2026):

    "Particularly strange is the refusal of the authors to use the computed gap as a
     descriptor while they have used other computed quantity (that cost as much as the
     gap) as descriptors. I expect that the semiempirical computed gap is much better
     than the ML gap based on structural parameters alone."

Head-to-head, same 231-molecule set, same scaffold GroupKFold folds:

  B0  mean baseline                       (no information)
  B1  raw GFN2-xTB/sTDA computed gap      (zero-parameter physics predictor)
  B2  linear-calibrated computed gap      (a*gap + b, fitted in-fold -> no leakage)
  M1  RF on Morgan fingerprints           (structure only, published headline)
  M2  RF on NTO descriptors               (semi-empirical, energy-free)
  M3  RF on NTO + computed gap            (does the gap ADD to features?)
  M4  RF on Morgan + computed gap         (cheap structure + one physics number)

Metrics: MAE, RMSE, R^2, Pearson r, Spearman rho, all with 95% bootstrap CI.
Output: data/ref1_gap_baseline.json  (+ console table)
"""
import json, pathlib, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _dataset import load_dataset, make_features, rf   # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
N_BOOT, SEED, N_SPLITS = 2000, 0, 5
rng = np.random.default_rng(SEED)


def boot_ci(fn, y, p, n=N_BOOT):
    idx = np.arange(len(y))
    vals = [fn(y[s], p[s]) for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def metrics(y, p, label):
    r = stats.pearsonr(y, p)
    rho = stats.spearmanr(y, p)
    return dict(
        model=label, n=int(len(y)),
        MAE=float(mean_absolute_error(y, p)), MAE_CI=boot_ci(mean_absolute_error, y, p),
        RMSE=rmse(y, p), RMSE_CI=boot_ci(rmse, y, p),
        R2=float(r2_score(y, p)), R2_CI=boot_ci(r2_score, y, p),
        pearson_r=float(r.statistic), pearson_p=float(r.pvalue),
        spearman_rho=float(rho.statistic), spearman_p=float(rho.pvalue),
    )


def oof_rf(X, y, groups, cv):
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        m = rf(random_state=SEED).fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return p


def oof_linear_gap(g, y, groups, cv):
    """In-fold linear calibration of the computed gap: no test-set leakage."""
    p = np.zeros(len(y))
    for tr, te in cv.split(g.reshape(-1, 1), y, groups):
        a, b = np.polyfit(g[tr], y[tr], 1)
        p[te] = a * g[te] + b
    return p


def oof_mean(y, groups, cv):
    p = np.zeros(len(y))
    for tr, te in cv.split(y.reshape(-1, 1), y, groups):
        p[te] = y[tr].mean()
    return p


ds = load_dataset()
y = ds.y
gap = ds.df['Delta_E_ST_eV'].values.astype(float)
groups = ds.scaf
cv = GroupKFold(n_splits=N_SPLITS)

Xm, _ = make_features(ds, 'Morgan')
Xn, _ = make_features(ds, 'NTO')
Xng = np.hstack([Xn, gap.reshape(-1, 1)])
Xmg = np.hstack([Xm, gap.reshape(-1, 1)])

res = [
    metrics(y, oof_mean(y, groups, cv), 'B0 mean baseline'),
    metrics(y, gap, 'B1 raw sTDA computed gap'),
    metrics(y, oof_linear_gap(gap, y, groups, cv), 'B2 calibrated computed gap (a*gap+b, in-fold)'),
    metrics(y, oof_rf(Xm, y, groups, cv), 'M1 RF Morgan (structure only)'),
    metrics(y, oof_rf(Xn, y, groups, cv), 'M2 RF NTO (energy-free semi-empirical)'),
    metrics(y, oof_rf(Xng, y, groups, cv), 'M3 RF NTO + computed gap'),
    metrics(y, oof_rf(Xmg, y, groups, cv), 'M4 RF Morgan + computed gap'),
]

# descriptive: how well does raw theory track experiment (referee claims r>0.95)?
extra = dict(
    n_molecules=int(len(y)),
    n_scaffolds=int(len(set(groups))),
    exp_gap_mean=float(y.mean()), exp_gap_sd=float(y.std(ddof=1)),
    comp_gap_mean=float(gap.mean()), comp_gap_sd=float(gap.std(ddof=1)),
    signed_bias_comp_minus_exp=float(np.mean(gap - y)),
    referee1_claim='semi-empirical computed gap should beat structure-only ML; exp-vs-theory r>0.95',
)

out = dict(meta=dict(script='code/ref1_gap_baseline.py', seed=SEED, n_splits=N_SPLITS,
                     n_bootstrap=N_BOOT, cv='GroupKFold on Bemis-Murcko scaffolds'),
           dataset=extra, results=res)
(ROOT / 'data' / 'ref1_gap_baseline.json').write_text(json.dumps(out, indent=2))

hdr = f"{'model':46s} {'MAE':>6s} {'RMSE':>6s} {'R2':>7s} {'r':>6s} {'rho':>6s}"
print(f"n={len(y)} molecules, {len(set(groups))} scaffolds\n{hdr}\n{'-'*len(hdr)}")
for r in res:
    print(f"{r['model']:46s} {r['MAE']:6.3f} {r['RMSE']:6.3f} {r['R2']:7.3f} "
          f"{r['pearson_r']:6.3f} {r['spearman_rho']:6.3f}")
print(f"\nsigned bias (comp - exp) = {extra['signed_bias_comp_minus_exp']:+.3f} eV")
print("wrote data/ref1_gap_baseline.json")
