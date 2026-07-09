#!/usr/bin/env python3
"""
calibration_model.py — Honest sTDA->experimental DeltaE_ST calibration (Task T1.1 reframed)

Reframing (user decision, 2026-06-22): the sTDA gap (S1-T1) is a CHEAP, BIASED estimator
of the experimental singlet-triplet gap. The ML task is to learn a correction toward the
experimental value -- a NON-circular task (target = experimental DeltaE_ST, not the
arithmetic feature difference).

Dataset: NTO features + sTDA gap  joined (by molecule name) to literature-extracted
experimental DeltaE_ST (MCA_submission/tables/experimental_delta_est_extracted.csv).

Reports, with leakage-free repeated CV:
  - Baseline B0 : raw sTDA gap vs experimental         (no learning)
  - Baseline B1 : global constant shift (mean residual) (trivial calibration)
  - Model  M    : SVR correction predicting experimental from features
Writes data/calibration_metrics.json (single source of truth) + a parity figure.
"""
import json, ast, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'data'
FEATURES = [
    'S1_energy_eV','T1_energy_eV','S1_S_he','T1_S_he','Delta_S_he','Abs_Delta_S_he',
    'S1_CT_number','T1_CT_number','Delta_CT_number','Abs_Delta_CT_number',
    'S1_Lambda_D','S1_Lambda_A','T1_Lambda_D','T1_Lambda_A','Delta_Lambda_D',
    'Abs_Delta_Lambda_D','Delta_Lambda_A','Abs_Delta_Lambda_A','S1_Delta_r','T1_Delta_r',
    'Delta_Delta_r','Abs_Delta_Delta_r','S1_hole_on_A','S1_particle_on_D','T1_hole_on_A',
    'T1_particle_on_D','Char_diff_squared','S_NTO_sum','S_NTO_product','S_NTO_ratio',
    'S1_osc_strength','HOMO_LUMO_gap_eV','Delta_S_NTO','Abs_Delta_S_NTO','Log_Abs_S1',
    'Log_Abs_T1','S1_overlap','T1_overlap']

# ---- features (gas/stda) ----
feat = pd.read_csv(ROOT/'CALCULATIONS-MADE/data_processing/combined_features_747mol_full_ct.csv')
feat = feat[(feat.environment=='gas')&(feat.method=='stda')].dropna(subset=FEATURES+['Delta_E_ST_eV'])
feat = feat.rename(columns={'Delta_E_ST_eV':'stda_gap'})

# ---- experimental DeltaE_ST (literature) ----
exp = pd.read_csv(ROOT/'MCA_submission/tables/experimental_delta_est_extracted.csv')
def names(x):
    try: return [str(n).strip() for n in ast.literal_eval(x)]
    except: return [str(x)]
def val(v):
    try:
        p=ast.literal_eval(v); return float(p[0]) if isinstance(p,list) else float(p)
    except:
        try: return float(re.findall(r'[-\d.]+',str(v))[0])
        except: return np.nan
exp['names_l']=exp['compound.names'].map(names)
exp['val_ev']=exp['value'].map(val)
mev = exp['units'].astype(str).str.contains(r'10\^-3', regex=True)
exp.loc[mev,'val_ev']=exp.loc[mev,'val_ev']/1000.0
featmols=set(feat.molecule.astype(str))
rows=[(nm,r.val_ev) for _,r in exp.iterrows() for nm in r['names_l']
      if nm in featmols and np.isfinite(r.val_ev) and -0.5<r.val_ev<2.0]
em=pd.DataFrame(rows,columns=['molecule','exp_est']).groupby('molecule').exp_est.mean().reset_index()

df = feat.merge(em, on='molecule', how='inner')
print(f'Calibration set: {len(df)} molecules with experimental DeltaE_ST')

X = df[FEATURES].values
y = df['exp_est'].values          # TARGET = experimental gap (non-circular)
stda = df['stda_gap'].values      # cheap estimator

def cv_scores(predict_fn):
    rkf=RepeatedKFold(n_splits=5,n_repeats=5,random_state=0); r2s,maes=[],[]
    for tr,te in rkf.split(X):
        yp=predict_fn(tr,te); r2s.append(r2_score(y[te],yp)); maes.append(mean_absolute_error(y[te],yp))
    return dict(R2_mean=round(np.mean(r2s),4),R2_sd=round(np.std(r2s),4),
               MAE_mean=round(np.mean(maes),4),MAE_sd=round(np.std(maes),4))

res={'n_molecules':int(len(df)),
     'exp_range_eV':[round(float(y.min()),3),round(float(y.max()),3)],
     'exp_mean_eV':round(float(y.mean()),3)}
# B0: raw sTDA gap (no fit)
res['B0_raw_stda_vs_exp']={'R2':round(r2_score(y,stda),4),'MAE':round(mean_absolute_error(y,stda),4),
                           'mean_bias_eV':round(float(np.mean(stda-y)),4)}
# B1: constant shift learned on train (trivial calibration)
def b1(tr,te):
    shift=np.mean(y[tr]-stda[tr]); return stda[te]+shift
res['B1_constant_shift']=cv_scores(b1)
# M: SVR predicting experimental from features
def m(tr,te):
    p=make_pipeline(StandardScaler(),SVR(kernel='rbf',C=10,gamma='scale',epsilon=0.01)).fit(X[tr],y[tr])
    return p.predict(X[te])
res['M_svr_calibration']=cv_scores(m)

(OUT/'calibration_metrics.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
print('\nSaved -> data/calibration_metrics.json')
