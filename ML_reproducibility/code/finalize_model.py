#!/usr/bin/env python3
"""
finalize_model.py — Lock the honest repositioned model (option 1) and regenerate Fig 1.

Headline model: RandomForest on energy-free NTO spatial descriptors (physics-informed,
non-circular). Target = QC'd experimental DeltaE_ST. Evaluated with Bemis-Murcko scaffold
GroupKFold (true generalization) + bootstrap 95% CI. TreeExplainer SHAP (exact for RF).

Outputs:
  data/final_model_metrics.json            — headline metrics + feature-set comparison
  data/final_model_scaffold_predictions.csv
  digital_discovery_manuscript/figures/fig1_model_performance.pdf/png   (parity + SHAP)
"""
import json, ast, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent.parent
OUTD = ROOT/'data'; OUTF = ROOT/'digital_discovery_manuscript'/'figures'

NTO = ['S1_S_he','T1_S_he','Delta_S_he','Abs_Delta_S_he','S1_CT_number','T1_CT_number',
       'Delta_CT_number','Abs_Delta_CT_number','S1_Lambda_D','S1_Lambda_A','T1_Lambda_D',
       'T1_Lambda_A','Delta_Lambda_D','Abs_Delta_Lambda_D','Delta_Lambda_A','Abs_Delta_Lambda_A',
       'S1_Delta_r','T1_Delta_r','Delta_Delta_r','Abs_Delta_Delta_r','S1_hole_on_A',
       'S1_particle_on_D','T1_hole_on_A','T1_particle_on_D','Char_diff_squared','S_NTO_sum',
       'S_NTO_product','S_NTO_ratio','S1_osc_strength','Delta_S_NTO','Abs_Delta_S_NTO',
       'Log_Abs_S1','Log_Abs_T1','S1_overlap','T1_overlap']
LABELS={'T1_S_he':r'$S_{he}(T_1)$','S1_S_he':r'$S_{he}(S_1)$','Delta_S_he':r'$\Delta S_{he}$',
        'Abs_Delta_S_he':r'$|\Delta S_{he}|$','S1_CT_number':r'CT$(S_1)$','T1_CT_number':r'CT$(T_1)$',
        'S1_osc_strength':r'$f_{S_1}$','S_NTO_product':r'$S_{NTO}^{prod}$','S1_Delta_r':r'$\Delta r(S_1)$',
        'T1_Delta_r':r'$\Delta r(T_1)$','Char_diff_squared':r'$Char^2_{diff}$','S1_overlap':r'$S_1$ ovlp',
        'T1_overlap':r'$T_1$ ovlp','S_NTO_sum':r'$S_{NTO}^{sum}$','S_NTO_ratio':r'$S_{NTO}^{ratio}$'}

# ---- clean experimental target (same QC as rebuild) ----
exp = pd.read_csv(ROOT/'MCA_submission/tables/experimental_delta_est_extracted.csv')
exp = exp[exp['specifier'].astype(str).str.contains('EST',case=False,na=False)]
nm=lambda x:( [str(n).strip() for n in ast.literal_eval(x)] if str(x).startswith('[') else [str(x)])
def vv(v):
    try: p=ast.literal_eval(v); return float(p[0]) if isinstance(p,list) else float(p)
    except:
        try: return float(re.findall(r'[-\d.]+',str(v))[0])
        except: return np.nan
exp['nl']=exp['compound.names'].map(nm); exp['v']=exp['standard_value'].map(vv)
exp=exp[np.isfinite(exp.v)&exp.v.between(-0.3,1.5)]
feat=pd.read_csv(ROOT/'CALCULATIONS-MADE/data_processing/combined_features_747mol_full_ct.csv')
feat=feat[(feat.environment=='gas')&(feat.method=='stda')].dropna(subset=NTO+['Delta_E_ST_eV'])
fm=set(feat.molecule.astype(str))
rows=[(n,r.v) for _,r in exp.iterrows() for n in r['nl'] if n in fm]
em=pd.DataFrame(rows,columns=['molecule','v']).groupby('molecule').v.median().reset_index().rename(columns={'v':'exp_est'})
ref=pd.read_csv(ROOT/'Data_Ref.csv')[['Molecule_id','compound.SMILES']].rename(columns={'Molecule_id':'molecule','compound.SMILES':'smiles'})
df=feat.merge(em,on='molecule').merge(ref,on='molecule',how='left')
df['m']=df.smiles.map(lambda s:Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)
df=df[df.m.notna()].reset_index(drop=True)
y=df.exp_est.values
scaf=df.m.map(lambda m:MurckoScaffold.MurckoScaffoldSmiles(mol=m) or 'none').values
print(f'Final set: {len(df)} molecules, {len(set(scaf))} scaffolds')

# ---- feature-set comparison (for SI) ----
desc_names=[n for n,_ in Descriptors._descList]
calc=MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
def feats(name):
    if name=='NTO': return df[NTO].values, NTO
    if name=='RDKit':
        M=np.vstack([np.nan_to_num(np.array(calc.CalcDescriptors(m),dtype=float),nan=0,posinf=0,neginf=0) for m in df.m]); return M, desc_names
    if name=='Morgan':
        M=np.vstack([np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048),dtype=float) for m in df.m]); return M, [f'fp{i}' for i in range(2048)]
def rf(): return RandomForestRegressor(n_estimators=400,random_state=0,n_jobs=-1)
gkf=GroupKFold(5)
def boot_ci(yt,yp,fn,n=2000):
    rng=np.random.default_rng(0); s=[]
    for _ in range(n):
        i=rng.integers(0,len(yt),len(yt))
        try: s.append(fn(yt[i],yp[i]))
        except: pass
    return round(float(np.percentile(s,2.5)),3), round(float(np.percentile(s,97.5)),3)
comp={}
preds={}
for fsname in ['NTO','RDKit','Morgan']:
    X,_=feats(fsname)
    yp=cross_val_predict(rf(),X,y,cv=gkf,groups=scaf)
    preds[fsname]=yp
    comp[fsname]=dict(MAE=round(mean_absolute_error(y,yp),4),
                      MAE_CI=boot_ci(y,yp,mean_absolute_error),
                      R2=round(r2_score(y,yp),4), R2_CI=boot_ci(y,yp,r2_score),
                      Spearman=round(spearmanr(yp,y).correlation,3))
HEAD='NTO'
metrics=dict(headline_model='RandomForest on energy-free NTO spatial descriptors',
             target='experimental Delta_E_ST (median of literature values)',
             n_molecules=int(len(df)), n_scaffolds=int(len(set(scaf))),
             validation='Bemis-Murcko scaffold GroupKFold (5)',
             headline=comp[HEAD], feature_set_comparison=comp,
             dummy_mean_MAE=round(mean_absolute_error(y,np.full_like(y,y.mean())),4),
             raw_stda_MAE=round(mean_absolute_error(y,df['Delta_E_ST_eV'].values),4))
(OUTD/'final_model_metrics.json').write_text(json.dumps(metrics,indent=2))
pd.DataFrame({'molecule':df.molecule,'exp_est':y,'pred_'+HEAD:preds[HEAD]}).to_csv(
    OUTD/'final_model_scaffold_predictions.csv',index=False)

# ---- SHAP (TreeExplainer, exact) on headline model fit to all data ----
Xh,_=feats(HEAD)
m_full=rf().fit(Xh,y)
sv=shap.TreeExplainer(m_full).shap_values(Xh)
mean_abs=np.abs(sv).mean(0); order=np.argsort(mean_abs)[::-1][:10]

# ---- Figure 1: parity (A) + SHAP bar (B) ----
plt.rcParams.update({'font.family':'serif','font.size':9,'figure.dpi':300,
                     'axes.spines.top':False,'axes.spines.right':False})
fig,(axA,axB)=plt.subplots(1,2,figsize=(8.2,3.6))
yp=preds[HEAD]
axA.scatter(y,yp,s=14,alpha=0.55,color='#2E86AB',linewidths=0)
lim=[min(y.min(),yp.min())-0.05,max(y.max(),yp.max())+0.05]
axA.plot(lim,lim,'--',color='#888',lw=0.8); axA.set_xlim(lim); axA.set_ylim(lim)
axA.set_xlabel(r'Experimental $\Delta E_{ST}$ (eV)'); axA.set_ylabel(r'Predicted $\Delta E_{ST}$ (eV)')
axA.set_title('A) Scaffold-CV parity', loc='left', fontweight='bold', fontsize=9)
axA.text(0.04,0.92,f"MAE = {comp[HEAD]['MAE']:.3f} eV\n$R^2$ = {comp[HEAD]['R2']:.2f}\n"+
         rf"$\rho$ = {comp[HEAD]['Spearman']:.2f}",transform=axA.transAxes,va='top',fontsize=8,
         bbox=dict(boxstyle='round,pad=0.3',fc='#F5F5F5',ec='#CCC'))
feats_top=[LABELS.get(NTO[i],NTO[i]) for i in order][::-1]
axB.barh(range(len(order)),mean_abs[order][::-1],color='#F18F01')
axB.set_yticks(range(len(order))); axB.set_yticklabels(feats_top,fontsize=8)
axB.set_xlabel(r'mean $|$SHAP$|$ (eV)'); axB.set_title('B) Feature importance (RF / SHAP)',loc='left',fontweight='bold',fontsize=9)
plt.tight_layout()
for ext in ('pdf','png'): fig.savefig(OUTF/f'fig1_model_performance.{ext}',bbox_inches='tight',dpi=300)
plt.close(fig)
print(json.dumps(metrics,indent=2))
print('\nSaved final_model_metrics.json + fig1_model_performance.pdf/png')
