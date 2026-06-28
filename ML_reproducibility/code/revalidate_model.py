#!/usr/bin/env python3
"""
revalidate_model.py  —  Honest re-validation of SVR Model C (Task T1.1, CRITICAL260622 #1)

Answers, with leakage-free protocol:
  1. Reproduce the committed single-split number (sanity vs svr_model_C_metrics.json).
  2. Repeated K-fold CV (5x5) -> honest mean +/- SD for R2 and MAE.
  3. GroupKFold by molecule (no molecule in both train and test).
  4. Leakage / triviality probe: target Delta_E_ST = E_S1 - E_T1, and BOTH energies
     are features. Compare the SVR against the trivial physical baseline
     y_hat = S1_energy_eV - T1_energy_eV  (no learning at all).
  5. Spatial-only (Model D) CV, to test the 'beyond energy arithmetic' claim honestly.

Writes: data/model_revalidation_metrics.json  (single source of truth for the manuscript)
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (train_test_split, RepeatedKFold,
                                     GroupKFold, cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'CALCULATIONS-MADE' / 'data_processing'
OUT = ROOT / 'data'

FEATURES_C = [
    'S1_energy_eV','T1_energy_eV','S1_S_he','T1_S_he','Delta_S_he','Abs_Delta_S_he',
    'S1_CT_number','T1_CT_number','Delta_CT_number','Abs_Delta_CT_number',
    'S1_Lambda_D','S1_Lambda_A','T1_Lambda_D','T1_Lambda_A','Delta_Lambda_D',
    'Abs_Delta_Lambda_D','Delta_Lambda_A','Abs_Delta_Lambda_A','S1_Delta_r','T1_Delta_r',
    'Delta_Delta_r','Abs_Delta_Delta_r','S1_hole_on_A','S1_particle_on_D','T1_hole_on_A',
    'T1_particle_on_D','Char_diff_squared','S_NTO_sum','S_NTO_product','S_NTO_ratio',
    'S1_osc_strength','HOMO_LUMO_gap_eV','Delta_S_NTO','Abs_Delta_S_NTO','Log_Abs_S1',
    'Log_Abs_T1','S1_overlap','T1_overlap']
# Model D = spatial / NTO only (energies and direct-arithmetic terms removed)
FEATURES_D = [f for f in FEATURES_C if f not in
              ('S1_energy_eV','T1_energy_eV','HOMO_LUMO_gap_eV')]
TARGET = 'Delta_E_ST_eV'

df = pd.read_csv(DATA_DIR / 'combined_features_747mol_full_ct.csv')
df = df[(df['environment']=='gas') & (df['method']=='stda')].copy()
df = df.dropna(subset=FEATURES_C+[TARGET])
print(f'Dataset: {len(df)} molecules, {df["molecule"].nunique()} unique ids')

X = df[FEATURES_C].values
y = df[TARGET].values
groups = df['molecule'].values

def svr_pipe():
    return make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.01))

res = {}

# 1. Reproduce committed single split (random_state=42)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
m = svr_pipe().fit(Xtr,ytr); p = m.predict(Xte)
res['single_split_42'] = {'R2': round(r2_score(yte,p),4), 'MAE': round(mean_absolute_error(yte,p),4),
                          'n_test': len(yte)}

# 2. Repeated 5-fold CV x5 (random splits) — honest dispersion
rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
r2s, maes = [], []
for tr,te in rkf.split(X):
    mm = svr_pipe().fit(X[tr],y[tr]); pp = mm.predict(X[te])
    r2s.append(r2_score(y[te],pp)); maes.append(mean_absolute_error(y[te],pp))
res['repeated_cv_5x5'] = {'R2_mean': round(np.mean(r2s),4),'R2_sd': round(np.std(r2s),4),
                          'MAE_mean': round(np.mean(maes),4),'MAE_sd': round(np.std(maes),4)}

# 3. GroupKFold by molecule (each id appears ~once here, but enforce anyway)
gkf = GroupKFold(n_splits=5)
pg = cross_val_predict(svr_pipe(), X, y, cv=gkf, groups=groups)
res['group_kfold_5'] = {'R2': round(r2_score(y,pg),4),'MAE': round(mean_absolute_error(y,pg),4)}

# 4. Triviality probe: trivial physical baseline y_hat = E_S1 - E_T1
s1 = df['S1_energy_eV'].values; t1 = df['T1_energy_eV'].values
y_triv = s1 - t1
res['target_is_S1_minus_T1'] = {
    'corr(y, S1-T1)': round(float(np.corrcoef(y, y_triv)[0,1]),4),
    'MAE(y, S1-T1)': round(float(mean_absolute_error(y, y_triv)),4),
    'R2(y, S1-T1)': round(float(r2_score(y, y_triv)),4),
    'note': 'If MAE~0 the target is arithmetically the energy gap; SVR has little to learn.'}

# 5. Model D (spatial only) repeated CV
Xd = df[FEATURES_D].values
r2d, maed = [], []
for tr,te in RepeatedKFold(n_splits=5,n_repeats=5,random_state=0).split(Xd):
    mm = svr_pipe().fit(Xd[tr],y[tr]); pp=mm.predict(Xd[te])
    r2d.append(r2_score(y[te],pp)); maed.append(mean_absolute_error(y[te],pp))
res['model_D_spatial_only_cv'] = {'R2_mean': round(np.mean(r2d),4),'R2_sd': round(np.std(r2d),4),
                                  'MAE_mean': round(np.mean(maed),4),'MAE_sd': round(np.std(maed),4),
                                  'n_features': len(FEATURES_D)}

(OUT/'model_revalidation_metrics.json').write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
print('\nSaved -> data/model_revalidation_metrics.json')
