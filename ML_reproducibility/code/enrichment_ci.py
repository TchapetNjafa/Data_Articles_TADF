#!/usr/bin/env python3
"""
enrichment_ci.py (audit item 10.3, AUDIT_REPORT.md 2026-09-12)
================================================================
Bootstrap CI on the triage/enrichment precision numbers already stated in the main
text (results_rsc_advances.tex, "Modest triage utility" / "Quantifying triage
utility" paragraphs): precision at top-10, top-50 molecules and top-20% fraction,
threshold <=0.10 eV, structure-only Morgan-FP RF, scaffold-CV out-of-fold
predictions -- reproduces enrichment_curve.py's point estimates, adds a
nonparametric bootstrap CI the audit asked for.

Resamples molecules with replacement over the fixed out-of-fold predictions/labels
(the ranking is re-derived within each resample; the model itself is not refit --
standard for bootstrapping a metric of an already cross-validated prediction set,
consistent with the paired-CI method used elsewhere in this repo, e.g. A-002/A-006).

Output: data/enrichment_ci.json
"""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import cross_val_predict, GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

ROOT = Path(__file__).parent.parent
THR = 0.10
N_BOOT = 5000
RNG = np.random.default_rng(0)

ds = load_dataset()
y = ds.y
X, _ = make_features(ds, 'Morgan')
good = (y <= THR).astype(int)
base = float(good.mean())
n = len(y)

yp = cross_val_predict(rf(), X, y, cv=GroupKFold(5), groups=ds.scaf)
order = np.argsort(yp)

ks = {'top10': 10, 'top20': 20, 'top20pct': max(1, round(0.20 * n)), 'top50': 50}

point = {}
for name, k in ks.items():
    sel = order[:k]
    prec = float(good[sel].mean())
    point[name] = dict(k=int(k), precision=round(prec, 3),
                        enrichment=round(prec / base, 3))

boot = {name: [] for name in ks}
idx_all = np.arange(n)
for _ in range(N_BOOT):
    samp = RNG.choice(idx_all, size=n, replace=True)
    yp_s, good_s = yp[samp], good[samp]
    order_s = np.argsort(yp_s)
    base_s = good_s.mean()
    if base_s == 0:
        continue
    for name, k in ks.items():
        kk = min(k, n)
        prec_s = good_s[order_s[:kk]].mean()
        boot[name].append(prec_s / base_s)

ci = {}
for name in ks:
    arr = np.array(boot[name])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    ci[name] = dict(enrichment_ci95=[round(float(lo), 2), round(float(hi), 2)],
                     n_boot=int(len(arr)))

out = dict(threshold_eV=THR, base_rate=round(base, 4), n=int(n),
           point=point, bootstrap_ci=ci)
(ROOT/'data'/'enrichment_ci.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
