#!/usr/bin/env python3
"""
learning_curve_analysis.py  (Task B2)
=====================================
Address "231 molecules is embarrassingly small".

Scaffold-CV learning curve for the headline NTO random forest: train MAE and
cross-validated MAE as a function of training-set size. If the CV-MAE plateaus before
the full set, the bottleneck is label noise rather than training-set size in the
current regime; if it is still falling, more data would help (an honest future direction).

Outputs:
  data/learning_curve.json
  digital_discovery_manuscript/figures/learning_curve.pdf/png
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import learning_curve, GroupKFold
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

ROOT = Path(__file__).parent.parent
FIGD = ROOT/'digital_discovery_manuscript'/'figures'

ds = load_dataset()
y = ds.y
X, _ = make_features(ds, 'NTO')
print(f'Loaded {len(y)} molecules')

train_sizes, train_scores, test_scores = learning_curve(
    rf(), X, y, groups=ds.scaf, cv=GroupKFold(5),
    train_sizes=np.linspace(0.25, 1.0, 7),
    scoring='neg_mean_absolute_error', shuffle=False, n_jobs=-1)

train_mae = -train_scores.mean(1)
test_mae = -test_scores.mean(1)
test_sd = test_scores.std(1)

# plateau diagnostic: relative change in CV-MAE over the last third of the curve
tail = test_mae[len(test_mae)//2:]
rel_tail_change = float(abs(tail[-1] - tail[0]) / tail[0])

out = dict(
    n_molecules=int(len(y)),
    train_sizes=[int(s) for s in train_sizes],
    train_mae=[round(float(v), 4) for v in train_mae],
    cv_mae=[round(float(v), 4) for v in test_mae],
    cv_mae_sd=[round(float(v), 4) for v in test_sd],
    cv_mae_relative_change_second_half=round(rel_tail_change, 3),
    plateau=bool(rel_tail_change < 0.05))
(ROOT/'data'/'learning_curve.json').write_text(json.dumps(out, indent=2))

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'figure.dpi': 300,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot(train_sizes, train_mae, 'o-', color='#2E86AB', label='Train MAE')
ax.plot(train_sizes, test_mae, 's-', color='#F18F01', label='Scaffold-CV MAE')
ax.fill_between(train_sizes, test_mae-test_sd, test_mae+test_sd, color='#F18F01', alpha=0.15)
ax.axhline(0.107, ls='--', lw=0.8, color='#888', label='Mean baseline (0.107)')
ax.set_xlabel('Training-set size'); ax.set_ylabel(r'MAE (eV)')
ax.legend(fontsize=7, frameon=False)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGD/f'learning_curve.{ext}', bbox_inches='tight', dpi=300)
plt.close(fig)
print(json.dumps(out, indent=2))
print('\nSaved -> data/learning_curve.json + figures/learning_curve.pdf')
