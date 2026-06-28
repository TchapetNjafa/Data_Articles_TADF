#!/usr/bin/env python3
"""
shap_bootstrap_stability.py  (Task E1)
======================================
Rebut "interpreting SHAP for a rho=0.36 model is reading tea leaves".

Refit the headline RF (NTO features) on N_BOOT bootstrap resamples of the 231-molecule
set; for each fit compute exact TreeExplainer SHAP and record the top-k features by
mean|SHAP|. Report how often the canonical (full-data) top-k features reappear --
a high reappearance frequency means the attribution is a stable signal, not noise.

Output: data/shap_bootstrap_stability.json (+ console summary)
"""
import json, sys
from pathlib import Path
from collections import Counter
import numpy as np
import shap

sys.path.insert(0, str(Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

N_BOOT = 200
TOPK = 5
SEED = 0
ROOT = Path(__file__).parent.parent

ds = load_dataset()
y = ds.y
X, names = make_features(ds, 'NTO')
names = np.array(names)
n = len(y)
print(f'Loaded {n} molecules; {X.shape[1]} NTO features')

def top_features(model, Xm, k):
    sv = shap.TreeExplainer(model).shap_values(Xm)
    mabs = np.abs(sv).mean(0)
    order = np.argsort(mabs)[::-1]
    return order[:k], mabs

# ---- canonical full-data ranking -------------------------------------------
m_full = rf().fit(X, y)
canon_idx, canon_mabs = top_features(m_full, X, TOPK)
canon_total = canon_mabs.sum()
canon_top = list(names[canon_idx])
canon_share = float(canon_mabs[canon_idx].sum() / canon_total)
print('Canonical top-5:', canon_top)

# ---- bootstrap resamples ----------------------------------------------------
rng = np.random.default_rng(SEED)
freq = Counter()              # how often each feature appears in bootstrap top-k
canon_hits = np.zeros(N_BOOT)  # size of overlap with canonical top-k each round
for b in range(N_BOOT):
    idx = rng.integers(0, n, n)
    mb = rf(random_state=b).fit(X[idx], y[idx])
    bi, _ = top_features(mb, X[idx], TOPK)
    bf = set(names[bi])
    for f in bf:
        freq[f] += 1
    canon_hits[b] = len(bf & set(canon_top))
    if (b + 1) % 50 == 0:
        print(f'  bootstrap {b+1}/{N_BOOT}  mean canon-overlap '
              f'{canon_hits[:b+1].mean():.2f}/{TOPK}')

per_feature_stability = {f: round(freq[f] / N_BOOT, 3) for f in canon_top}
mean_overlap = float(canon_hits.mean())
# fraction of bootstraps where >=4 of the canonical top-5 are recovered
frac_ge4 = float(np.mean(canon_hits >= 4))

out = dict(
    n_molecules=int(n), n_features=int(X.shape[1]),
    n_bootstrap=N_BOOT, topk=TOPK,
    canonical_top5=canon_top,
    canonical_top5_attribution_share=round(canon_share, 3),
    canonical_top1=canon_top[0],
    canonical_top1_share=round(float(canon_mabs[canon_idx][0] / canon_total), 3),
    per_feature_bootstrap_stability=per_feature_stability,
    mean_top5_overlap=round(mean_overlap, 2),
    frac_bootstrap_recovering_ge4_of_top5=round(frac_ge4, 3))

(ROOT/'data'/'shap_bootstrap_stability.json').write_text(json.dumps(out, indent=2))
print('\n' + json.dumps(out, indent=2))
print('\nSaved -> data/shap_bootstrap_stability.json')
