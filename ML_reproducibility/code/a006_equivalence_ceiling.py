"""
a006_equivalence_ceiling.py
===========================
L3 review re-entry to L1. Answers four findings raised by the independent review
passes in outputs/critical-reviews/ that the existing ledger cannot settle.

T1  "The excited-state step adds nothing" is an EQUIVALENCE claim supported only by a
    failure to reject, while paired tests are used everywhere else (A-005 resolves
    0.0032 eV with this same instrument).
    -> Paired Morgan-vs-NTO bootstrap on IDENTICAL scaffold folds + Wilcoxon, and the
       smallest symmetric margin at which a TOST-style equivalence claim holds.

T2  Under the DOI (source-paper) split the ledger A-002 P2b records NTO rho = 0.346 vs
    Morgan 0.155 -- the one place the semi-empirical features outrank fingerprints, and
    the manuscript reports only the Morgan row while Methods declares rho primary.
    -> Is that rho advantage statistically resolved, or noise? Paired bootstrap of
       delta-rho on identical DOI folds.

T3  Triage enrichment takes mutually inconsistent values across three deposited files
    (precision@10% = 0.565 / 0.600 / 0.652).
    -> Recompute from ONE definition on the same OOF predictions; report the deltas.

T4  data/ceiling_decomposition.json asserts an RSS floor of 0.128 eV and states the model
    "cannot penetrate below the noise floor" while the model MAE is 0.0956 eV -- below it.
    Terms B and C are properties of a COMPUTED reference, not of the experimental label.
    -> Audit the arithmetic and compute the defensible quantity: the precision of the
       median target itself.

Reuses code/_dataset.py and the fold construction of code/ref3_p1p2p3.py verbatim so
every number is on the same folds as A-001/A-002/A-005.

Output: data/a006_equivalence_ceiling.json
"""
import ast, json, pathlib, re, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
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
    for tr, te in cv.split(X, y, groups):
        m = rf().fit(X[tr], y[tr]); p[te] = m.predict(X[te])
    return p


def oof_mean(y, cv, groups=None):
    p = np.zeros(len(y))
    for tr, te in cv.split(np.zeros((len(y), 1)), y, groups):
        p[te] = y[tr].mean()
    return p


def paired_delta(y, p_a, p_b, n=N_BOOT):
    """Bootstrap MAE(b) - MAE(a). Positive = a better than b. Paired on molecules."""
    ea, eb = np.abs(y - p_a), np.abs(y - p_b)
    obs = eb.mean() - ea.mean()
    idx = rng.integers(0, len(y), size=(n, len(y)))
    d = eb[idx].mean(1) - ea[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta_MAE=float(obs), CI=[float(lo), float(hi)],
                frac_bootstrap_a_better=float((d > 0).mean()))


# ---------- T1: paired Morgan vs NTO, scaffold folds ----------
cv = GroupKFold(n_splits=N_SPLITS)
pm = oof(Xm, y, cv, groups_scaf)
pn = oof(Xn, y, cv, groups_scaf)
pb = oof_mean(y, cv, groups_scaf)

nto_vs_morgan = paired_delta(y, pn, pm)          # positive = NTO better
w = stats.wilcoxon(np.abs(y - pn), np.abs(y - pm))
lo, hi = nto_vs_morgan['CI']
equiv_margin = float(max(abs(lo), abs(hi)))

T1 = dict(
    morgan=dict(MAE=float(mean_absolute_error(y, pm)), R2=float(r2_score(y, pm)),
                rho=float(stats.spearmanr(y, pm).statistic)),
    nto=dict(MAE=float(mean_absolute_error(y, pn)), R2=float(r2_score(y, pn)),
             rho=float(stats.spearmanr(y, pn).statistic)),
    mean_baseline_MAE=float(mean_absolute_error(y, pb)),
    nto_minus_morgan=nto_vs_morgan,
    wilcoxon=dict(stat=float(w.statistic), p=float(w.pvalue)),
    smallest_equivalence_margin_eV=equiv_margin,
    note=('positive delta_MAE = NTO better. TOST: the two feature sets are equivalent at '
          'any margin >= smallest_equivalence_margin_eV, and NOT distinguishable below it.'),
)

# ---------- rebuild the DOI groups exactly as ref3_p1p2p3.py does ----------
exp = pd.read_csv(ROOT / 'MCA_submission/tables/experimental_delta_est_extracted.csv')
exp = exp[exp['specifier'].astype(str).str.contains('EST', case=False, na=False)].copy()


def nm(x):
    # verbatim from code/ref3_p1p2p3.py:93-94 -- case-preserving, matches ds.df.molecule
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
                                   n_doi=('doi', 'nunique'))

primary_doi = long.groupby('molecule').doi.agg(lambda s: s.value_counts().index[0])
doi_map = ds.df.molecule.astype(str).map(primary_doi)
mask = doi_map.notna().values
g_doi = doi_map[mask].values
yd = y[mask]
Xmd, Xnd = Xm[mask], Xn[mask]
cvd = GroupKFold(n_splits=min(N_SPLITS, len(set(g_doi))))
pm_d, pn_d = oof(Xmd, yd, cvd, g_doi), oof(Xnd, yd, cvd, g_doi)

# ---------- T2: is the NTO rho advantage under the DOI split resolved? ----------
rho_m = float(stats.spearmanr(yd, pm_d).statistic)
rho_n = float(stats.spearmanr(yd, pn_d).statistic)
idx = rng.integers(0, len(yd), size=(N_BOOT, len(yd)))
drho = np.array([stats.spearmanr(yd[i], pn_d[i]).statistic -
                 stats.spearmanr(yd[i], pm_d[i]).statistic for i in idx])
dlo, dhi = np.percentile(drho, [2.5, 97.5])
T2 = dict(
    n=int(len(yd)), n_doi_groups=int(len(set(g_doi))),
    morgan=dict(MAE=float(mean_absolute_error(yd, pm_d)), R2=float(r2_score(yd, pm_d)), rho=rho_m),
    nto=dict(MAE=float(mean_absolute_error(yd, pn_d)), R2=float(r2_score(yd, pn_d)), rho=rho_n),
    delta_rho_nto_minus_morgan=float(rho_n - rho_m),
    delta_rho_CI=[float(dlo), float(dhi)],
    delta_rho_excludes_zero=bool(dlo > 0 or dhi < 0),
    nto_minus_morgan_MAE=paired_delta(yd, pn_d, pm_d),
    note='paired bootstrap on identical DOI folds; resamples molecules, recomputes both rho',
)

# ---------- T3: canonical enrichment, one definition ----------
THRESH, GOOD = 0.1, None
good = (y < THRESH).astype(int)
base = float(good.mean())


def enrich(pred, fracs=(0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)):
    order = np.argsort(pred)          # ascending predicted gap = most promising first
    out = []
    for f in fracs:
        k = max(1, int(round(f * len(pred))))
        sel = order[:k]
        prec = float(good[sel].mean())
        out.append(dict(fraction=f, k=int(k), precision=round(prec, 4),
                        enrichment=round(prec / base, 4)))
    return out


T3 = dict(threshold_eV=THRESH, base_rate=round(base, 4), n=int(len(y)),
          morgan_curve=enrich(pm), nto_curve=enrich(pn),
          definition=('good = experimental gap < 0.1 eV; rank ascending by predicted gap; '
                      'precision = fraction good in the top-k; enrichment = precision / base_rate; '
                      'scaffold-GroupKFold out-of-fold predictions, seed 0'))

dep = {}
for f in ['enrichment_curve.json', 'triage_metrics.json', 'multi_criteria_triage.json']:
    p = ROOT / 'data' / f
    if p.exists():
        dep[f] = json.loads(p.read_text())
T3['deposited_values_at_top10pct'] = {
    'enrichment_curve.json': next((c for c in dep.get('enrichment_curve.json', {}).get('curve', [])
                                   if abs(c['fraction'] - 0.1) < 1e-9), None),
    'triage_metrics.json': {k: v for k, v in dep.get('triage_metrics.json', {}).items()
                            if k in ('precision_at_10', 'enrichment_at_10', 'precision_at_20',
                                     'enrichment_at_20', 'precision_at_50', 'enrichment_at_50')},
    'multi_criteria_triage.json': next((c for c in dep.get('multi_criteria_triage.json', {})
                                        .get('single_criterion_curve', [])
                                        if abs(c['fraction'] - 0.1) < 1e-9), None),
}

# ---------- T4: ceiling decomposition audit ----------
cd = json.loads((ROOT / 'data' / 'ceiling_decomposition.json').read_text())
A, B, C = cd['A_inter_lab_rms_eV'], cd['B_vertical_adiabatic_mae_eV'], cd['C_functional_spread_mean_eV']
rss_abc = float(np.sqrt(A ** 2 + B ** 2 + C ** 2))
multi = agg[agg.n_reports > 1]
# precision of the per-molecule MEDIAN label (this is what the model is trained on)
sem = (multi.sd / np.sqrt(multi.n_reports)).dropna()
rms_sem = float(np.sqrt(np.mean(sem ** 2)))
T4 = dict(
    stored_rss_floor_eV=cd['rss_floor_eV'],
    stored_rf_mae_eV=cd['rf_mae_eV'],
    model_is_below_stored_floor=bool(cd['rf_mae_eV'] < cd['rss_floor_eV']),
    recomputed_rss_of_A_B_C_eV=round(rss_abc, 4),
    components=dict(A_inter_report_rms_sd=A, B_vertical_vs_adiabatic_MAE=B,
                    C_functional_spread_mean=C),
    category_error=('B and C are dispersions of COMPUTED references (sTDA-vs-adiabatic, '
                    'B3LYP-vs-CAM-B3LYP). They bound agreement between levels of theory, '
                    'not the attainable error of a model regressed on EXPERIMENTAL labels. '
                    'Only A is a property of the label.'),
    defensible_label_precision=dict(
        rms_standard_error_of_median_eV=round(rms_sem, 4),
        n_multi_report=int(len(multi)),
        n_single_report=int((agg.n_reports == 1).sum()),
        model_MAE_eV=round(float(mean_absolute_error(y, pn)), 4),
        note=('RMS standard error of the per-molecule median label, over molecules with '
              'replicates. This is the precision of the quantity actually regressed on. '
              'Single-report molecules have no replicate and contribute no estimate.'),
    ),
)

# ---------- T5: the clean-label test, done properly ----------
# Review pass 2 (H2): the published "SD <= 0.05 eV" subset used sd.fillna(0), so the 129
# molecules with NO replicate were scored as perfectly clean and only ~23 were discarded.
# The test that actually bears on the ceiling thesis is: conditional on label quality,
# is the model's error lower where the label is known to be reproducible?
sd_by_mol = agg.sd.reindex(ds.df.molecule.astype(str)).values
nrep = agg.n_reports.reindex(ds.df.molecule.astype(str)).fillna(1).values
abs_err_nto = np.abs(y - pn)

has_rep = nrep > 1
clean = has_rep & (sd_by_mol <= 0.05)
noisy = has_rep & (sd_by_mol > 0.05)
as_published = np.nan_to_num(sd_by_mol, nan=0.0) <= 0.05   # the fillna(0) version

def sub(m):
    return dict(n=int(m.sum()), MAE=float(abs_err_nto[m].mean()) if m.sum() else None)

T5 = dict(
    as_published_subset=dict(**sub(as_published),
                             n_singletons_counted_as_clean=int((as_published & ~has_rep).sum()),
                             n_actually_discarded=int((~as_published).sum()),
                             note='sd.fillna(0) <= 0.05 -- molecules with no replicate are scored clean'),
    replicated_clean=dict(**sub(clean), note='n_reports > 1 AND sd <= 0.05 eV'),
    replicated_noisy=dict(**sub(noisy), note='n_reports > 1 AND sd > 0.05 eV'),
    singletons=dict(**sub(~has_rep), note='no replicate: label precision unmeasurable'),
    full_set_MAE=float(abs_err_nto.mean()),
    interpretation_test=('If label noise is the binding constraint, MAE on replicated_clean '
                         'must be materially below MAE on replicated_noisy. Compare them.'),
)
if clean.sum() and noisy.sum():
    d = paired_delta(y[noisy], pn[noisy], y[noisy] * 0 + y[noisy].mean(), n=1)  # placeholder guard
    idx_c = rng.integers(0, int(clean.sum()), size=(N_BOOT, int(clean.sum())))
    idx_n = rng.integers(0, int(noisy.sum()), size=(N_BOOT, int(noisy.sum())))
    ec, en = abs_err_nto[clean], abs_err_nto[noisy]
    diff = en[idx_n].mean(1) - ec[idx_c].mean(1)     # positive = noisy labels hurt
    lo5, hi5 = np.percentile(diff, [2.5, 97.5])
    T5['noisy_minus_clean_MAE'] = dict(delta=float(en.mean() - ec.mean()),
                                       CI=[float(lo5), float(hi5)],
                                       excludes_zero=bool(lo5 > 0 or hi5 < 0),
                                       note='unpaired bootstrap; positive = higher error where the label is noisier')

out = dict(
    meta=dict(script='code/a006_equivalence_ceiling.py', seed=SEED, n_splits=N_SPLITS,
              n_bootstrap=N_BOOT, rf='400 trees, random_state=0',
              cv='GroupKFold on Bemis-Murcko scaffolds (identical to A-001/A-005)'),
    T1_morgan_vs_nto_equivalence=T1,
    T2_doi_split_rho=T2,
    T3_enrichment_reconciliation=T3,
    T4_ceiling_audit=T4,
    T5_clean_label_test=T5,
)
(ROOT / 'data' / 'a006_equivalence_ceiling.json').write_text(json.dumps(out, indent=1))

print('=== T1  Morgan vs NTO, scaffold folds (paired, identical folds) ===')
print(f"  Morgan MAE {T1['morgan']['MAE']:.4f}  rho {T1['morgan']['rho']:.3f}")
print(f"  NTO    MAE {T1['nto']['MAE']:.4f}  rho {T1['nto']['rho']:.3f}")
d = T1['nto_minus_morgan']
print(f"  delta MAE (Morgan-NTO) = {d['delta_MAE']:+.4f} eV  CI [{d['CI'][0]:+.4f}, {d['CI'][1]:+.4f}]"
      f"  Wilcoxon p={T1['wilcoxon']['p']:.3f}")
print(f"  -> smallest defensible equivalence margin: {equiv_margin:.4f} eV")
print()
print('=== T2  DOI split: is the NTO rho advantage real? ===')
print(f"  Morgan rho {T2['morgan']['rho']:.3f}   NTO rho {T2['nto']['rho']:.3f}"
      f"   delta {T2['delta_rho_nto_minus_morgan']:+.3f}")
print(f"  delta_rho CI [{T2['delta_rho_CI'][0]:+.3f}, {T2['delta_rho_CI'][1]:+.3f}]"
      f"  excludes zero: {T2['delta_rho_excludes_zero']}")
print()
print('=== T3  enrichment, one definition (base rate %.3f) ===' % base)
for c in T3['morgan_curve']:
    print(f"  Morgan top{c['fraction']*100:5.1f}%  k={c['k']:3d}  precision {c['precision']:.3f}"
          f"  enrichment {c['enrichment']:.2f}")
print('  deposited values at top-10%:', json.dumps(T3['deposited_values_at_top10pct']['triage_metrics.json']))
print()
print('=== T4  ceiling audit ===')
print(f"  stored RSS floor {cd['rss_floor_eV']} eV vs model MAE {cd['rf_mae_eV']} eV"
      f"  -> model below its own floor: {T4['model_is_below_stored_floor']}")
print(f"  RSS of the three stated components recomputes to {rss_abc:.4f} eV, not {cd['rss_floor_eV']}")
print(f"  defensible label precision (RMS SEM of the median): {rms_sem:.4f} eV"
      f"  vs model MAE {T4['defensible_label_precision']['model_MAE_eV']} eV")
print()
print('=== T5  clean-label test, done properly ===')
print(f"  as published (sd.fillna(0)<=0.05): n={T5['as_published_subset']['n']}, "
      f"MAE {T5['as_published_subset']['MAE']:.4f}  "
      f"({T5['as_published_subset']['n_singletons_counted_as_clean']} singletons counted clean, "
      f"only {T5['as_published_subset']['n_actually_discarded']} discarded)")
print(f"  replicated & clean (sd<=0.05): n={T5['replicated_clean']['n']}, MAE {T5['replicated_clean']['MAE']}")
print(f"  replicated & noisy (sd> 0.05): n={T5['replicated_noisy']['n']}, MAE {T5['replicated_noisy']['MAE']}")
print(f"  singletons (no replicate):     n={T5['singletons']['n']}, MAE {T5['singletons']['MAE']}")
if 'noisy_minus_clean_MAE' in T5:
    z = T5['noisy_minus_clean_MAE']
    print(f"  noisy - clean = {z['delta']:+.4f} eV  CI [{z['CI'][0]:+.4f}, {z['CI'][1]:+.4f}]  excludes zero: {z['excludes_zero']}")
print()
print('wrote data/a006_equivalence_ceiling.json')
