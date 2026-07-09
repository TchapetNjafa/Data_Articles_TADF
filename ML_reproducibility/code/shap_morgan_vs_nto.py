#!/usr/bin/env python3
"""
shap_morgan_vs_nto.py  (Task C1)
================================
Reframe "NTO descriptors match Morgan FP => physics features are wasted compute"
as an informative null result: equal accuracy, but the NTO model is mechanistically
INTERPRETABLE whereas the Morgan model attributes to anonymous hash bits.

Computes exact TreeExplainer SHAP for the NTO and Morgan headline RFs, records the
top-10 mean|SHAP| features of each, and renders a side-by-side comparison figure.

Outputs:
  data/shap_morgan_vs_nto.json
  digital_discovery_manuscript/figures/shap_morgan_vs_nto.pdf/png
"""
import json, sys
from pathlib import Path
import numpy as np
import shap
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

ROOT = Path(__file__).parent.parent
FIGD = ROOT/'digital_discovery_manuscript'/'figures'
LABELS = {'T1_S_he': r'$S_{he}(T_1)$', 'S1_S_he': r'$S_{he}(S_1)$',
          'Delta_S_he': r'$\Delta S_{he}$', 'Abs_Delta_S_he': r'$|\Delta S_{he}|$',
          'S1_CT_number': r'CT$(S_1)$', 'T1_CT_number': r'CT$(T_1)$',
          'S1_osc_strength': r'$f_{S_1}$', 'S_NTO_product': r'$S_{NTO}^{prod}$',
          'S1_Delta_r': r'$\Delta r(S_1)$', 'T1_Delta_r': r'$\Delta r(T_1)$',
          'Char_diff_squared': r'$Char^2_{diff}$', 'S1_overlap': r'$S_1$ ovlp',
          'T1_overlap': r'$T_1$ ovlp', 'S_NTO_sum': r'$S_{NTO}^{sum}$',
          'S_NTO_ratio': r'$S_{NTO}^{ratio}$'}

ds = load_dataset()
y = ds.y

def top_shap(name, k=10):
    X, names = make_features(ds, name)
    names = np.array(names)
    m = rf().fit(X, y)
    sv = shap.TreeExplainer(m).shap_values(X)
    mabs = np.abs(sv).mean(0)
    order = np.argsort(mabs)[::-1][:k]
    total = mabs.sum()
    return [(str(names[i]), round(float(mabs[i]), 5)) for i in order], float(total)

nto_top, nto_total = top_shap('NTO')
morgan_top, morgan_total = top_shap('Morgan')

# how concentrated is attribution: share of top-10
nto_share = round(sum(v for _, v in nto_top) / nto_total, 3)
morgan_share = round(sum(v for _, v in morgan_top) / morgan_total, 3)
# fraction of Morgan top-10 that map to a chemically nameable group (none: they are hash bits)
morgan_named = 0

out = dict(
    nto_top10=[{'feature': f, 'mean_abs_shap_eV': v} for f, v in nto_top],
    morgan_top10=[{'feature': f, 'mean_abs_shap_eV': v} for f, v in morgan_top],
    nto_top10_attribution_share=nto_share,
    morgan_top10_attribution_share=morgan_share,
    nto_features_interpretable=len(nto_top),
    morgan_features_interpretable=morgan_named,
    note='Equal CV-MAE (NTO 0.096 / Morgan 0.091) but NTO attributions are named '
         'physical descriptors; Morgan attributions are anonymous hashed substructure bits.')
(ROOT/'data'/'shap_morgan_vs_nto.json').write_text(json.dumps(out, indent=2))

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'figure.dpi': 300,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.4))
nl = [LABELS.get(f, f) for f, _ in nto_top][::-1]
a1.barh(range(len(nto_top)), [v for _, v in nto_top][::-1], color='#F18F01')
a1.set_yticks(range(len(nto_top))); a1.set_yticklabels(nl, fontsize=8)
a1.set_xlabel(r'mean $|$SHAP$|$ (eV)')
a1.set_title('A) NTO model (named descriptors)', loc='left', fontweight='bold', fontsize=9)
ml = [f for f, _ in morgan_top][::-1]
a2.barh(range(len(morgan_top)), [v for _, v in morgan_top][::-1], color='#6C757D')
a2.set_yticks(range(len(morgan_top))); a2.set_yticklabels(ml, fontsize=8)
a2.set_xlabel(r'mean $|$SHAP$|$ (eV)')
a2.set_title('B) Morgan model (anonymous bits)', loc='left', fontweight='bold', fontsize=9)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGD/f'shap_morgan_vs_nto.{ext}', bbox_inches='tight', dpi=300)
plt.close(fig)
print(json.dumps(out, indent=2))
print('\nSaved -> data/shap_morgan_vs_nto.json + figures/shap_morgan_vs_nto.pdf')
