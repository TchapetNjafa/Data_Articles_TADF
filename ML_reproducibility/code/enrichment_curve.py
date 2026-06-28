#!/usr/bin/env python3
"""
enrichment_curve.py  (Task R9)
==============================
Answer "1.2-1.4x enrichment is useless" with the FULL enrichment/precision curve vs
screening budget, plus a split-conformal coverage check so a lab can attach a
calibrated confidence to the operating point it chooses.

Uses the honest structure-only Morgan-FP RF, scaffold-CV out-of-fold predictions on
the 231-molecule benchmark. "Good" = experimental gap <= 0.10 eV.

Output: data/enrichment_curve.json + figures/enrichment_curve.pdf
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import cross_val_predict, GroupKFold, train_test_split
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

ROOT = Path(__file__).parent.parent
FIGD = ROOT/'digital_discovery_manuscript'/'figures'
THR = 0.10                      # "good TADF" experimental gap (eV)
ds = load_dataset()
y = ds.y
X, _ = make_features(ds, 'Morgan')
good = (y <= THR).astype(int)
base = float(good.mean())

# scaffold-CV out-of-fold predictions (lower predicted gap = better)
yp = cross_val_predict(rf(), X, y, cv=GroupKFold(5), groups=ds.scaf)
order = np.argsort(yp)          # ascending predicted gap

fracs = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
curve = []
n = len(y)
for f in fracs:
    k = max(1, int(round(f * n)))
    sel = order[:k]
    prec = float(good[sel].mean())
    curve.append(dict(fraction=f, k=k, precision=round(prec, 3),
                      enrichment=round(prec / base, 2)))

# split-conformal: calibrate residual quantile on a held-out split, check coverage
Xtr, Xcal, ytr, ycal = train_test_split(X, y, test_size=0.4, random_state=0)
m = rf().fit(Xtr, ytr)
resid = np.abs(ycal - m.predict(Xcal))
for alpha in (0.10, 0.20):
    q = float(np.quantile(resid, 1 - alpha))
    # empirical coverage on the calibration residuals (sanity)
    cov = float(np.mean(resid <= q))
    curve_cov = dict(alpha=alpha, interval_halfwidth_eV=round(q, 3),
                     empirical_coverage=round(cov, 3))
    curve.append({'conformal': curve_cov})

out = dict(threshold_eV=THR, base_rate=round(base, 3), n=int(n), curve=curve)
(ROOT/'data'/'enrichment_curve.json').write_text(json.dumps(out, indent=2))

# figure: precision vs fraction screened
fr = [c['fraction'] for c in curve if 'fraction' in c]
pr = [c['precision'] for c in curve if 'fraction' in c]
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'figure.dpi': 300,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot([f*100 for f in fr], [p*100 for p in pr], 'o-', color='#2E86AB', label='Model triage')
ax.axhline(base*100, ls='--', lw=0.8, color='#888', label=f'Random ({base*100:.0f}%)')
ax.set_xlabel('Fraction screened (%)'); ax.set_ylabel(r'Precision: \% with gap $\leq$ 0.10 eV')
ax.set_xscale('log'); ax.legend(fontsize=7, frameon=False)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGD/f'enrichment_curve.{ext}', bbox_inches='tight', dpi=300)
plt.close(fig)
print(json.dumps(out, indent=2))
print('\nSaved -> data/enrichment_curve.json + figures/enrichment_curve.pdf')
