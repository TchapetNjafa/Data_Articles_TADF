"""
ref3_p1p2p3.py
==============
Answers Referee 3 points 1-3 of the Digital Discovery rejection.

P1  "R^2 CI spans into negative territory => model indistinguishable from a mean predictor."
    -> Paired bootstrap of the MAE DIFFERENCE (model - mean baseline) on identical folds.
       A paired test on the metric we actually claim is the right instrument; R^2 on a
       low-variance target with n=231 is not.

P2  "Labels aggregate unrecorded experimental conditions => uncurated consensus target."
    -> (a) Direct measurement of label noise from multi-report molecules (sd across reports).
       (b) DOI-grouped CV: each source paper is one group. Testing on unseen papers = testing
           on unseen measurement protocols. Strictly harder than scaffold CV.
       (c) Single-report vs multi-report subset performance.

P3  "212/231 scaffolds are singletons => GroupKFold == random split."
    -> Head-to-head scaffold-CV vs random-CV vs DOI-CV on identical models, plus the
       exact singleton fraction and fold-composition statistics.

Output: data/ref3_p1p2p3.json
"""
import ast, json, pathlib, re, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
SEED, N_SPLITS, N_BOOT = 0, 5, 2000
rng = np.random.default_rng(SEED)

ds = load_dataset()
y, groups_scaf = ds.y, ds.scaf
Xm, _ = make_features(ds, 'Morgan')
Xn, _ = make_features(ds, 'NTO')


def oof(X, y, cv, groups=None):
    p = np.zeros(len(y))
    for tr, te in (cv.split(X, y, groups) if groups is not None else cv.split(X)):
        p[te] = rf(random_state=SEED).fit(X[tr], y[tr]).predict(X[te])
    return p


def oof_mean(y, cv, groups=None):
    p = np.zeros(len(y))
    for tr, te in (cv.split(y.reshape(-1, 1), y, groups) if groups is not None
                   else cv.split(y.reshape(-1, 1))):
        p[te] = y[tr].mean()
    return p


# ---------- P1: paired bootstrap of MAE difference ----------
def paired_mae_delta(y, p_model, p_base, n=N_BOOT):
    """Bootstrap the paired difference MAE(base) - MAE(model). Positive = model better."""
    idx = np.arange(len(y))
    d = [mean_absolute_error(y[s], p_base[s]) - mean_absolute_error(y[s], p_model[s])
         for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n))]
    d = np.array(d)
    return dict(delta_MAE=float(mean_absolute_error(y, p_base) - mean_absolute_error(y, p_model)),
                CI=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                frac_bootstrap_model_better=float((d > 0).mean()))


cv_scaf = GroupKFold(n_splits=N_SPLITS)
p_morgan = oof(Xm, y, cv_scaf, groups_scaf)
p_nto = oof(Xn, y, cv_scaf, groups_scaf)
p_base = oof_mean(y, cv_scaf, groups_scaf)

# Wilcoxon on paired absolute errors — distribution-free, no CI-of-R2 problem
w_morgan = stats.wilcoxon(np.abs(y - p_base), np.abs(y - p_morgan))
w_nto = stats.wilcoxon(np.abs(y - p_base), np.abs(y - p_nto))

P1 = dict(
    morgan_vs_mean=paired_mae_delta(y, p_morgan, p_base),
    nto_vs_mean=paired_mae_delta(y, p_nto, p_base),
    wilcoxon_morgan=dict(stat=float(w_morgan.statistic), p=float(w_morgan.pvalue)),
    wilcoxon_nto=dict(stat=float(w_nto.statistic), p=float(w_nto.pvalue)),
    note='paired on identical folds; positive delta_MAE = model beats mean baseline',
)

# ---------- P2a: label noise from multi-report molecules ----------
exp = pd.read_csv(ROOT / 'MCA_submission/tables/experimental_delta_est_extracted.csv')
exp = exp[exp['specifier'].astype(str).str.contains('EST', case=False, na=False)].copy()


def nm(x):
    return ([str(n).strip() for n in ast.literal_eval(x)] if str(x).startswith('[') else [str(x)])


def vv(v):
    try:
        p = ast.literal_eval(v)
        return float(p[0]) if isinstance(p, list) else float(p)
    except Exception:
        try:
            return float(re.findall(r'[-\d.]+', str(v))[0])
        except Exception:
            return np.nan


exp['nl'] = exp['compound.names'].map(nm)
exp['v'] = exp['standard_value'].map(vv)
exp = exp[np.isfinite(exp.v) & exp.v.between(-0.3, 1.5)]

mols = set(ds.df.molecule.astype(str))
rows = [(n, r.v, r.doi) for _, r in exp.iterrows() for n in r['nl'] if n in mols]
long = pd.DataFrame(rows, columns=['molecule', 'v', 'doi'])
agg = long.groupby('molecule').agg(n_reports=('v', 'size'), sd=('v', 'std'),
                                   spread=('v', lambda s: s.max() - s.min()),
                                   n_doi=('doi', 'nunique'))
multi = agg[agg.n_reports > 1]
P2a = dict(
    n_molecules_matched=int(len(agg)),
    n_single_report=int((agg.n_reports == 1).sum()),
    n_multi_report=int(len(multi)),
    sd_median=float(multi.sd.median()), sd_mean=float(multi.sd.mean()),
    sd_max=float(multi.sd.max()),
    rms_sd=float(np.sqrt(np.mean(multi.sd.dropna() ** 2))),
    spread_median=float(multi.spread.median()), spread_max=float(multi.spread.max()),
    frac_multi_with_zero_sd=float((multi.sd == 0).mean()),
    note='sd across independent literature reports of the SAME molecule = label noise',
)

# ---------- P2b: DOI-grouped CV (unseen source paper = unseen protocol) ----------
primary_doi = long.groupby('molecule').doi.agg(lambda s: s.value_counts().index[0])
doi_map = ds.df.molecule.astype(str).map(primary_doi)
mask = doi_map.notna().values
g_doi = doi_map[mask].values
yd = y[mask]
Xmd, Xnd = Xm[mask], Xn[mask]
n_doi_groups = len(set(g_doi))
k_doi = min(N_SPLITS, n_doi_groups)
cvd = GroupKFold(n_splits=k_doi)
pm_d = oof(Xmd, yd, cvd, g_doi)
pn_d = oof(Xnd, yd, cvd, g_doi)
pb_d = oof_mean(yd, cvd, g_doi)
P2b = dict(
    n=int(len(yd)), n_doi_groups=int(n_doi_groups), n_splits=int(k_doi),
    morgan=dict(MAE=float(mean_absolute_error(yd, pm_d)), R2=float(r2_score(yd, pm_d)),
                rho=float(stats.spearmanr(yd, pm_d).statistic)),
    nto=dict(MAE=float(mean_absolute_error(yd, pn_d)), R2=float(r2_score(yd, pn_d)),
             rho=float(stats.spearmanr(yd, pn_d).statistic)),
    mean_baseline=dict(MAE=float(mean_absolute_error(yd, pb_d)), R2=float(r2_score(yd, pb_d))),
    morgan_vs_mean=paired_mae_delta(yd, pm_d, pb_d),
)

# ---------- P2c: single- vs multi-report subsets ----------
nrep = ds.df.molecule.astype(str).map(agg.n_reports).fillna(1).values
P2c = {}
for lab, sel in [('single_report', nrep == 1), ('multi_report', nrep > 1)]:
    if sel.sum() < 30:
        P2c[lab] = dict(n=int(sel.sum()), note='too few for CV')
        continue
    cv_s = GroupKFold(n_splits=N_SPLITS)
    ps = oof(Xm[sel], y[sel], cv_s, groups_scaf[sel])
    pbs = oof_mean(y[sel], cv_s, groups_scaf[sel])
    P2c[lab] = dict(n=int(sel.sum()),
                    morgan_MAE=float(mean_absolute_error(y[sel], ps)),
                    morgan_R2=float(r2_score(y[sel], ps)),
                    morgan_rho=float(stats.spearmanr(y[sel], ps).statistic),
                    mean_baseline_MAE=float(mean_absolute_error(y[sel], pbs)),
                    delta=paired_mae_delta(y[sel], ps, pbs))

# ---------- P3: scaffold vs random vs DOI CV ----------
vc = pd.Series(groups_scaf).value_counts()
p_rand = oof(Xm, y, KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED))
P3 = dict(
    n_scaffolds=int(len(vc)), n_singleton=int((vc == 1).sum()),
    frac_singleton=float((vc == 1).sum() / len(vc)),
    max_scaffold_size=int(vc.max()),
    molecules_per_scaffold=float(len(y) / len(vc)),
    morgan_scaffold_CV=dict(MAE=float(mean_absolute_error(y, p_morgan)), R2=float(r2_score(y, p_morgan))),
    morgan_random_CV=dict(MAE=float(mean_absolute_error(y, p_rand)), R2=float(r2_score(y, p_rand))),
    morgan_doi_CV=dict(MAE=P2b['morgan']['MAE'], R2=P2b['morgan']['R2']),
    note='if scaffold CV == random CV the referee is right that scaffold split adds nothing; '
         'DOI CV is the strictly harder split that does carry information',
)

out = dict(meta=dict(script='code/ref3_p1p2p3.py', seed=SEED, n_splits=N_SPLITS,
                     n_bootstrap=N_BOOT, rf='400 trees, random_state=0'),
           P1_paired_vs_baseline=P1, P2a_label_noise=P2a, P2b_doi_cv=P2b,
           P2c_report_subsets=P2c, P3_split_comparison=P3)
(ROOT / 'data' / 'ref3_p1p2p3.json').write_text(json.dumps(out, indent=2))

print('P1 paired MAE delta (baseline - model), positive = model wins')
for k in ('morgan_vs_mean', 'nto_vs_mean'):
    d = P1[k]
    print(f"  {k:16s} {d['delta_MAE']:+.4f} eV  CI [{d['CI'][0]:+.4f},{d['CI'][1]:+.4f}]  "
          f"P(better)={d['frac_bootstrap_model_better']:.3f}")
print(f"  wilcoxon morgan p={P1['wilcoxon_morgan']['p']:.2e}  nto p={P1['wilcoxon_nto']['p']:.2e}")
print(f"\nP2a label noise: n_multi={P2a['n_multi_report']} sd_median={P2a['sd_median']:.3f} "
      f"rms_sd={P2a['rms_sd']:.3f} max={P2a['sd_max']:.3f} eV  zero-sd frac={P2a['frac_multi_with_zero_sd']:.2f}")
print(f"\nP2b DOI-grouped CV (n={P2b['n']}, {P2b['n_doi_groups']} papers, k={P2b['n_splits']}):")
for k in ('morgan', 'nto', 'mean_baseline'):
    v = P2b[k]
    print(f"  {k:14s} MAE {v['MAE']:.3f}  R2 {v['R2']:+.3f}" + (f"  rho {v['rho']:+.3f}" if 'rho' in v else ''))
print(f"  morgan vs mean: {P2b['morgan_vs_mean']['delta_MAE']:+.4f} eV "
      f"CI [{P2b['morgan_vs_mean']['CI'][0]:+.4f},{P2b['morgan_vs_mean']['CI'][1]:+.4f}]")
print('\nP2c subsets:')
for k, v in P2c.items():
    if 'morgan_MAE' in v:
        print(f"  {k:14s} n={v['n']:3d} MAE {v['morgan_MAE']:.3f} vs base {v['mean_baseline_MAE']:.3f} "
              f"R2 {v['morgan_R2']:+.3f} rho {v['morgan_rho']:+.3f} delta {v['delta']['delta_MAE']:+.4f}")
    else:
        print(f"  {k:14s} n={v['n']} {v['note']}")
print(f"\nP3 splits: {P3['n_singleton']}/{P3['n_scaffolds']} singleton "
      f"({P3['frac_singleton']:.1%}), {P3['molecules_per_scaffold']:.2f} mol/scaffold")
for k in ('morgan_scaffold_CV', 'morgan_random_CV', 'morgan_doi_CV'):
    print(f"  {k:20s} MAE {P3[k]['MAE']:.3f}  R2 {P3[k]['R2']:+.3f}")
print('\nwrote data/ref3_p1p2p3.json')
