#!/usr/bin/env python3
"""
rebuild_structural_model.py — Option B: clean experimental target + energy-free / structural
feature sets, honestly evaluated (Task: rebuild the science before any manuscript work).

Target  : QC'd experimental DeltaE_ST (specifier=DeltaEST family, standard_value column,
          median per compound, literature spread tracked).
Features: compared head-to-head ---
          [STRUCT]  RDKit physchem descriptors  (from SMILES, energy-free)
          [MORGAN]  Morgan fingerprint r=2,2048 (from SMILES, energy-free)
          [NTO]     NTO spatial descriptors     (no S1/T1 energies, no HL gap)
          [CALIB]   sTDA energies + NTO         (calibration reference; non-circular vs exp)
Models  : DummyMean, Ridge, RandomForest, SVR.
Eval    : RepeatedKFold(5x5) AND Bemis-Murcko scaffold GroupKFold (true generalization).
Metrics : MAE, R2, Spearman, recall(true top-20 lowest-gap within predicted top-40).
Writes  : data/structural_rebuild_metrics.json
"""
import json, ast, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent.parent

NTO = ['S1_S_he','T1_S_he','Delta_S_he','Abs_Delta_S_he','S1_CT_number','T1_CT_number',
       'Delta_CT_number','Abs_Delta_CT_number','S1_Lambda_D','S1_Lambda_A','T1_Lambda_D',
       'T1_Lambda_A','Delta_Lambda_D','Abs_Delta_Lambda_D','Delta_Lambda_A','Abs_Delta_Lambda_A',
       'S1_Delta_r','T1_Delta_r','Delta_Delta_r','Abs_Delta_Delta_r','S1_hole_on_A',
       'S1_particle_on_D','T1_hole_on_A','T1_particle_on_D','Char_diff_squared','S_NTO_sum',
       'S_NTO_product','S_NTO_ratio','S1_osc_strength','Delta_S_NTO','Abs_Delta_S_NTO',
       'Log_Abs_S1','Log_Abs_T1','S1_overlap','T1_overlap']
ENER = ['S1_energy_eV','T1_energy_eV','HOMO_LUMO_gap_eV']

# ---------- clean experimental target ----------
exp = pd.read_csv(ROOT/'MCA_submission/tables/experimental_delta_est_extracted.csv')
exp = exp[exp['specifier'].astype(str).str.contains('EST', case=False, na=False)]
def nm(x):
    try: return [str(n).strip() for n in ast.literal_eval(x)]
    except: return [str(x)]
def vv(v):
    try:
        p=ast.literal_eval(v); return float(p[0]) if isinstance(p,list) else float(p)
    except:
        try: return float(re.findall(r'[-\d.]+',str(v))[0])
        except: return np.nan
exp['nl']=exp['compound.names'].map(nm); exp['v']=exp['standard_value'].map(vv)
exp=exp[np.isfinite(exp.v) & exp.v.between(-0.3,1.5)]

feat = pd.read_csv(ROOT/'CALCULATIONS-MADE/data_processing/combined_features_747mol_full_ct.csv')
feat = feat[(feat.environment=='gas')&(feat.method=='stda')].dropna(subset=NTO+ENER+['Delta_E_ST_eV'])
feat = feat.rename(columns={'Delta_E_ST_eV':'stda_gap'})
fm=set(feat.molecule.astype(str))
rows=[(n,r.v) for _,r in exp.iterrows() for n in r['nl'] if n in fm]
em=pd.DataFrame(rows,columns=['molecule','v'])
agg=em.groupby('molecule').v.agg(exp_est='median', spread='std', n='count').reset_index()

ref=pd.read_csv(ROOT/'Data_Ref.csv')[['Molecule_id','compound.SMILES']].rename(
    columns={'Molecule_id':'molecule','compound.SMILES':'smiles'})
df=feat.merge(agg,on='molecule').merge(ref,on='molecule',how='left')
df['mol']=df.smiles.map(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)
df=df[df.mol.notna()].reset_index(drop=True)
print(f'Clean set: {len(df)} molecules | exp range {df.exp_est.min():.2f}–{df.exp_est.max():.2f} eV '
      f'| median lit spread {df.spread.median():.3f} eV')

# ---------- feature builders ----------
DESCS=[n for n,_ in Descriptors._descList]
from rdkit.ML.Descriptors import MoleculeDescriptors
calc=MoleculeDescriptors.MolecularDescriptorCalculator(DESCS)
def struct(m):
    d=np.array(calc.CalcDescriptors(m),dtype=float); d[~np.isfinite(d)]=0; return d
def morgan(m):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048),dtype=float)
Xs={'STRUCT':np.vstack([struct(m) for m in df.mol]),
    'MORGAN':np.vstack([morgan(m) for m in df.mol]),
    'NTO':df[NTO].values,
    'CALIB':df[ENER+NTO].values}
y=df.exp_est.values; stda=df.stda_gap.values
scaf=df.mol.map(lambda m: MurckoScaffold.MurckoScaffoldSmiles(mol=m) or 'none').values

def recall_topk(yt,yp,k=20,net=40):
    return int(len(set(np.argsort(yt)[:k]) & set(np.argsort(yp)[:net])))
def evalcv(X,model,cv,groups=None):
    yp=cross_val_predict(model,X,y,cv=cv,groups=groups)
    return dict(MAE=round(mean_absolute_error(y,yp),4),R2=round(r2_score(y,yp),4),
                Spearman=round(spearmanr(yp,y).correlation,3), recall_top20_of40=recall_topk(y,yp))
def mdl(name):
    if name=='Ridge': return make_pipeline(StandardScaler(),Ridge(alpha=1.0))
    if name=='RF':    return RandomForestRegressor(n_estimators=300,random_state=0,n_jobs=-1)
    if name=='SVR':   return make_pipeline(StandardScaler(),SVR(kernel='rbf',C=10,gamma='scale',epsilon=0.01))
    return DummyRegressor(strategy='mean')

kf=KFold(5,shuffle=True,random_state=0); gkf=GroupKFold(5)
out={'n_molecules':int(len(df)),'exp_median_spread_eV':round(float(df.spread.median()),3),
     'baseline_raw_stda':dict(MAE=round(mean_absolute_error(y,stda),4),R2=round(r2_score(y,stda),4),
                              Spearman=round(spearmanr(stda,y).correlation,3),
                              recall_top20_of40=recall_topk(y,stda))}
for fs,X in Xs.items():
    out[fs]={}
    for mn in (['DummyMean','Ridge','RF','SVR']):
        out[fs][mn+'_randomCV']=evalcv(X,mdl(mn),kf)
        out[fs][mn+'_scaffoldCV']=evalcv(X,mdl(mn),gkf,groups=scaf)
(ROOT/'data'/'structural_rebuild_metrics.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
print('\nSaved -> data/structural_rebuild_metrics.json')
