#!/usr/bin/env python3
"""
retrain_svr_shap.py
====================
Re-trains SVR Model C on the full NTO feature matrix and generates
per-molecule SHAP values required for the beeswarm plot (REQ-5).

Outputs
-------
data/shap_Model_C_per_molecule.csv   — per-molecule SHAP matrix (N × n_features)
data/svr_model_C_metrics.json        — R², MAE, feature list
digital_discovery_manuscript/figures/fig1_model_performance.pdf/png  — updated Fig 1C

Usage
-----
    source /home/tchapet/VirtualEnv/bin/activate
    python code/retrain_svr_shap.py
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'CALCULATIONS-MADE' / 'data_processing'
OUT_DATA = ROOT / 'data'
OUT_FIG  = ROOT / 'digital_discovery_manuscript' / 'figures'
OUT_DATA.mkdir(exist_ok=True)
OUT_FIG.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'text.usetex': False, 'mathtext.fontset': 'dejavuserif',
})
BLUE, ORANGE, GREEN, RED = '#2E86AB', '#F18F01', '#2DC653', '#C73E1D'

# ── Model C feature set (spatial + energetic) ─────────────────────────────────
FEATURES_C = [
    'S1_energy_eV', 'T1_energy_eV',
    'S1_S_he', 'T1_S_he', 'Delta_S_he', 'Abs_Delta_S_he',
    'S1_CT_number', 'T1_CT_number', 'Delta_CT_number', 'Abs_Delta_CT_number',
    'S1_Lambda_D', 'S1_Lambda_A', 'T1_Lambda_D', 'T1_Lambda_A',
    'Delta_Lambda_D', 'Abs_Delta_Lambda_D', 'Delta_Lambda_A', 'Abs_Delta_Lambda_A',
    'S1_Delta_r', 'T1_Delta_r', 'Delta_Delta_r', 'Abs_Delta_Delta_r',
    'S1_hole_on_A', 'S1_particle_on_D', 'T1_hole_on_A', 'T1_particle_on_D',
    'Char_diff_squared', 'S_NTO_sum', 'S_NTO_product', 'S_NTO_ratio',
    'S1_osc_strength', 'HOMO_LUMO_gap_eV',
    'Delta_S_NTO', 'Abs_Delta_S_NTO',
    'Log_Abs_S1', 'Log_Abs_T1',
    'S1_overlap', 'T1_overlap',
]

TARGET = 'Delta_E_ST_eV'

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading feature matrix...')
df = pd.read_csv(DATA_DIR / 'combined_features_747mol_full_ct.csv')
df = df[(df['environment'] == 'gas') & (df['method'] == 'stda')].copy()
df = df.dropna(subset=FEATURES_C + [TARGET])
print(f'  Dataset: {len(df)} molecules × {len(FEATURES_C)} features')

X = df[FEATURES_C].values
y = df[TARGET].values
mol_ids = df['molecule'].values

# ── Train/test split (80/20, same seed as original) ───────────────────────────
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)), test_size=0.2, random_state=42
)

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
X_all_s   = scaler.transform(X)

# ── Train SVR Model C ─────────────────────────────────────────────────────────
print('Training SVR Model C (RBF kernel)...')
svr = SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.01)
svr.fit(X_train_s, y_train)

y_pred_test = svr.predict(X_test_s)
r2  = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
print(f'  Test R² = {r2:.3f}  |  MAE = {mae:.3f} eV')

# Save metrics
metrics = {'R2': round(r2, 4), 'MAE': round(mae, 4),
           'n_train': len(X_train), 'n_test': len(X_test),
           'features': FEATURES_C}
with open(OUT_DATA / 'svr_model_C_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ── SHAP — KernelExplainer on full dataset ────────────────────────────────────
print('Computing per-molecule SHAP values (KernelExplainer)...')
print('  This may take 5–15 minutes...')

# Use a background summary (k-means of training set) for speed
background = shap.kmeans(X_train_s, 50)
explainer  = shap.KernelExplainer(svr.predict, background)
shap_values = explainer.shap_values(X_all_s, nsamples=200, silent=True)

print(f'  SHAP matrix shape: {shap_values.shape}')

# Save per-molecule SHAP values
shap_df = pd.DataFrame(shap_values, columns=FEATURES_C)
shap_df.insert(0, 'molecule', mol_ids)
shap_df.insert(1, 'Delta_E_ST_eV', y)
shap_df.to_csv(OUT_DATA / 'shap_Model_C_per_molecule.csv', index=False)
print(f'  Saved: data/shap_Model_C_per_molecule.csv')

# Also save mean |SHAP| for reference
mean_abs = np.abs(shap_values).mean(axis=0)
mean_df = pd.DataFrame({'feature': FEATURES_C, 'mean_abs_shap': mean_abs})
mean_df = mean_df.sort_values('mean_abs_shap', ascending=False)
mean_df.to_csv(OUT_DATA / 'shap_Model_C_means_updated.csv', index=False)

# ── Figure 1C: SHAP Beeswarm plot ─────────────────────────────────────────────
print('Generating SHAP beeswarm plot...')

# Select top 10 features by mean |SHAP|
top10_idx = np.argsort(mean_abs)[::-1][:10]
top10_features = [FEATURES_C[i] for i in top10_idx]
top10_shap     = shap_values[:, top10_idx]
top10_X        = X[:, top10_idx]

# Pretty feature labels
LABEL_MAP = {
    'T1_energy_eV':    r'$E_{T_1}$ (eV)',
    'S1_energy_eV':    r'$E_{S_1}$ (eV)',
    'T1_S_he':         r'$S_{he}(T_1)$',
    'S1_S_he':         r'$S_{he}(S_1)$',
    'Delta_S_he':      r'$\Delta S_{he}$',
    'Abs_Delta_S_he':  r'$|\Delta S_{he}|$',
    'T1_CT_number':    r'CT$(T_1)$',
    'S1_CT_number':    r'CT$(S_1)$',
    'T1_Lambda_A':     r'$\Lambda_A(T_1)$',
    'S1_Lambda_A':     r'$\Lambda_A(S_1)$',
    'T1_Lambda_D':     r'$\Lambda_D(T_1)$',
    'S1_Lambda_D':     r'$\Lambda_D(S_1)$',
    'T1_Delta_r':      r'$\Delta r(T_1)$',
    'S1_Delta_r':      r'$\Delta r(S_1)$',
    'Char_diff_squared': r'$\mathrm{Char}^2_\mathrm{diff}$',
    'S_NTO_sum':       r'$S_\mathrm{NTO}^\mathrm{sum}$',
    'S_NTO_product':   r'$S_\mathrm{NTO}^\mathrm{prod}$',
    'S_NTO_ratio':     r'$S_\mathrm{NTO}^\mathrm{ratio}$',
    'S1_osc_strength': r'$f_{S_1}$',
    'HOMO_LUMO_gap_eV': r'$\Delta E_\mathrm{HL}$ (eV)',
    'Delta_S_NTO':     r'$\Delta S_\mathrm{NTO}$',
    'Abs_Delta_S_NTO': r'$|\Delta S_\mathrm{NTO}|$',
    'Log_Abs_S1':      r'$\log|S_1|$',
    'Log_Abs_T1':      r'$\log|T_1|$',
    'S1_overlap':      r'$S_1$ overlap',
    'T1_overlap':      r'$T_1$ overlap',
    'Delta_CT_number': r'$\Delta$CT',
    'Abs_Delta_CT_number': r'$|\Delta$CT$|$',
    'Delta_Lambda_D':  r'$\Delta\Lambda_D$',
    'Abs_Delta_Lambda_D': r'$|\Delta\Lambda_D|$',
    'Delta_Lambda_A':  r'$\Delta\Lambda_A$',
    'Abs_Delta_Lambda_A': r'$|\Delta\Lambda_A|$',
    'Delta_Delta_r':   r'$\Delta\Delta r$',
    'Abs_Delta_Delta_r': r'$|\Delta\Delta r|$',
    'S1_hole_on_A':    r'hole$_A(S_1)$',
    'S1_particle_on_D': r'part$_D(S_1)$',
    'T1_hole_on_A':    r'hole$_A(T_1)$',
    'T1_particle_on_D': r'part$_D(T_1)$',
}
labels = [LABEL_MAP.get(f, f) for f in top10_features]

fig, ax = plt.subplots(figsize=(5.5, 3.8))

# Beeswarm: jitter dots along y-axis
rng = np.random.default_rng(42)
cmap = plt.cm.RdBu_r

for i, (feat_idx, feat_name) in enumerate(zip(range(len(top10_features)), top10_features)):
    sv   = top10_shap[:, feat_idx]
    fv   = top10_X[:, feat_idx]
    y_i  = len(top10_features) - 1 - i  # top feature at top

    # Normalize feature values for color
    fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)

    # Jitter
    jitter = rng.uniform(-0.3, 0.3, size=len(sv))

    sc = ax.scatter(sv, y_i + jitter, c=fv_norm, cmap=cmap,
                    s=6, alpha=0.6, linewidths=0, vmin=0, vmax=1)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Feature value\n(low → high)', fontsize=7)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['Low', 'Mid', 'High'], fontsize=7)

# Axes
ax.axvline(0, color='#888888', lw=0.8, ls='--')
ax.set_yticks(range(len(top10_features)))
ax.set_yticklabels(labels[::-1], fontsize=8)
ax.set_xlabel(r'SHAP value (impact on $\Delta E_{ST}$ prediction, eV)', fontsize=8)
ax.set_title(r'C) SHAP Beeswarm — Model C', fontweight='bold', loc='left', fontsize=9)

# Annotation: key finding
ax.annotate(r'High $S_{he}(T_1)$ → larger $\Delta E_{ST}$',
            xy=(0.02, 0.97), xycoords='axes fraction',
            fontsize=7, color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                      edgecolor='#CCCCCC', alpha=0.9),
            va='top', ha='left')

plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(OUT_FIG / f'fig1c_shap_beeswarm.{ext}',
                bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'  Saved: digital_discovery_manuscript/figures/fig1c_shap_beeswarm.pdf/png')

print('\n✅ REQ-5a complete: per-molecule SHAP values + beeswarm plot generated')
print(f'   R² = {r2:.3f}, MAE = {mae:.3f} eV')
