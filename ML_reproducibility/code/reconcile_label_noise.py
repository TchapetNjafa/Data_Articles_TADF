"""
reconcile_label_noise.py
========================
Reconciles the two label-noise numbers now in circulation:
  0.072 eV  = data/solvent_heterogeneity.json  "single_report_rmse_about_median_eV"
  0.086 eV  = data/ref3_p1p2p3.json            P2a "rms_sd"  (RMS of per-molecule SD, ddof=1)

Both are computed on the SAME 102 multi-report molecules of the 231-molecule benchmark.
They are different estimators of the same underlying spread, not conflicting measurements.
This script recomputes every candidate definition side by side and identifies which one
belongs in the manuscript's accuracy-ceiling argument.

Output: data/label_noise_reconciled.json
"""
import ast, json, pathlib, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _dataset import load_dataset  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
ds = load_dataset()
mols = set(ds.df.molecule.astype(str))

exp = pd.read_csv(ROOT / 'MCA_submission/tables/experimental_delta_est_extracted.csv')
exp = exp[exp['specifier'].astype(str).str.contains('EST', case=False, na=False)].copy()
nm = lambda x: ([str(n).strip() for n in ast.literal_eval(x)] if str(x).startswith('[') else [str(x)])
def vv(v):
    try:
        p = ast.literal_eval(v); return float(p[0]) if isinstance(p, list) else float(p)
    except Exception:
        try: return float(re.findall(r'[-\d.]+', str(v))[0])
        except Exception: return np.nan
exp['nl'] = exp['compound.names'].map(nm); exp['v'] = exp['standard_value'].map(vv)
exp = exp[np.isfinite(exp.v) & exp.v.between(-0.3, 1.5)]

groups = {}
for _, r in exp.iterrows():
    for n in r['nl']:
        if n in mols:
            groups.setdefault(n, []).append(r.v)
multi = {k: np.asarray(v) for k, v in groups.items() if len(v) > 1}
n_rep = np.array([len(v) for v in multi.values()])

sd_ddof1 = np.array([v.std(ddof=1) for v in multi.values()])
sd_ddof0 = np.array([v.std(ddof=0) for v in multi.values()])
dev_median = np.concatenate([v - np.median(v) for v in multi.values()])
dev_mean = np.concatenate([v - v.mean() for v in multi.values()])
sem_median = sd_ddof1 / np.sqrt(n_rep)          # uncertainty OF THE LABEL we regress on
spread = np.array([v.max() - v.min() for v in multi.values()])

out = dict(
    n_multi_report_molecules=int(len(multi)),
    n_reports_distribution={int(k): int(v) for k, v in zip(*np.unique(n_rep, return_counts=True))},
    estimators={
        'rms_of_per_molecule_SD_ddof1': round(float(np.sqrt(np.mean(sd_ddof1 ** 2))), 4),
        'rms_of_per_molecule_SD_ddof0': round(float(np.sqrt(np.mean(sd_ddof0 ** 2))), 4),
        'rms_single_report_about_median': round(float(np.sqrt(np.mean(dev_median ** 2))), 4),
        'rms_single_report_about_mean': round(float(np.sqrt(np.mean(dev_mean ** 2))), 4),
        'rms_SEM_of_median_label': round(float(np.sqrt(np.mean(sem_median ** 2))), 4),
        'mean_abs_dev_about_median': round(float(np.mean(np.abs(dev_median))), 4),
        'median_of_per_molecule_SD': round(float(np.median(sd_ddof1)), 4),
        'max_per_molecule_SD': round(float(sd_ddof1.max()), 4),
        'median_spread': round(float(np.median(spread)), 4),
        'max_spread': round(float(spread.max()), 4),
        'frac_zero_spread': round(float((spread == 0).mean()), 4),
    },
    ratio_check=dict(
        sd_ddof1_over_about_median=round(float(np.sqrt(np.mean(sd_ddof1 ** 2)) /
                                              np.sqrt(np.mean(dev_median ** 2))), 4),
        expected_ratio_if_all_n2='sqrt(2) = 1.4142',
        note='ddof=1 SD inflates by sqrt(n/(n-1)) relative to the RMS deviation about the centre; '
             'for n=2 that factor is sqrt(2). The observed ratio sits below sqrt(2) because some '
             'molecules have n=3,4,8.'
    ),
    recommendation=dict(
        for_accuracy_ceiling='rms_SEM_of_median_label',
        reason='the regression target IS the per-molecule median, so the floor on predicting it is '
               'the uncertainty of that median, not the spread of individual reports',
        for_describing_corpus_heterogeneity='rms_single_report_about_median (0.072 eV, already published)',
        do_not_use='rms_of_per_molecule_SD_ddof1 alone — it answers neither question and is the '
                   'largest of the three, so quoting it looks like inflation'
    ),
)
(ROOT / 'data' / 'label_noise_reconciled.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
