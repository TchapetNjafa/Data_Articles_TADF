#!/usr/bin/env python3
"""
triage_evaluation.py — Does the structure-only model actually enrich? (review #2 F2/F3)

Honest triage metrics for the Morgan-fingerprint random forest (the TRUE structure-only
model: computed from SMILES alone, no xTB/sTDA). Out-of-fold predictions under
Bemis-Murcko scaffold GroupKFold, target = experimental DeltaE_ST.

Reports: precision@k and enrichment factor for finding low-gap emitters
(ΔE_ST ≤ 0.10 eV = "good TADF"), vs random selection. Writes data/triage_metrics.json.
"""
import json, ast, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, GroupKFold
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent.parent

# rebuild the 231-molecule set (same QC as finalize_model.py)
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
feat=feat[(feat.environment=='gas')&(feat.method=='stda')]
fm=set(feat.molecule.astype(str))
rows=[(n,r.v) for _,r in exp.iterrows() for n in r['nl'] if n in fm]
em=pd.DataFrame(rows,columns=['molecule','v']).groupby('molecule').v.median().reset_index().rename(columns={'v':'exp'})
ref=pd.read_csv(ROOT/'Data_Ref.csv')[['Molecule_id','compound.SMILES']].rename(columns={'Molecule_id':'molecule','compound.SMILES':'smiles'})
df=em.merge(ref,on='molecule',how='left')
df['m']=df.smiles.map(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)
df=df[df.m.notna()].reset_index(drop=True)
y=df.exp.values
X=np.vstack([np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048),dtype=float) for m in df.m])
scaf=df.m.map(lambda m:MurckoScaffold.MurckoScaffoldSmiles(mol=m) or 'none').values
yp=cross_val_predict(RandomForestRegressor(400,random_state=0,n_jobs=-1),X,y,cv=GroupKFold(5),groups=scaf)

THR=0.10  # "good TADF" experimental gap
good=(y<=THR); base_rate=good.mean()
res={'n':int(len(y)),'threshold_eV':THR,'base_rate_good':round(float(base_rate),3),
     'spearman':round(float(spearmanr(yp,y).correlation),3)}
order=np.argsort(yp)  # model says lowest-gap first
for k in (10,20,30,50):
    sel=order[:k]
    prec=good[sel].mean()
    res[f'precision_at_{k}']=round(float(prec),3)
    res[f'enrichment_at_{k}']=round(float(prec/base_rate),2) if base_rate>0 else None
# ---- shortlist projection (Task C3): expected hit rate of a top-5% shortlist ----
# Library size and base rate (gap<0.2 eV) used in the main-text triage paragraph.
from scipy.stats import binomtest
LIB_N = 22194; SHORTLIST = round(0.05*LIB_N)   # top 5% ~ 1100
base02 = 0.494                                 # base rate of gap<=0.10 eV (114/231, matches precision target)
shortlist = {}
for ef in (1.2, 1.4):
    p = base02*ef; k = round(p*SHORTLIST)
    ci = binomtest(k, SHORTLIST).proportion_ci(0.95)
    shortlist[f'enrich_{ef}x'] = dict(
        hit_rate=round(p,3), ci=[round(ci.low,3), round(ci.high,3)],
        extra_hits_vs_random=int(k-round(base02*SHORTLIST)))
res['shortlist_projection']=dict(library_n=LIB_N, shortlist_n=SHORTLIST,
                                 base_rate_gap_lt_0p2=base02, bands=shortlist)
res['note']=('precision@k = fraction of model top-k with exp gap<=0.10 eV; '
             'enrichment = precision@k / base rate (1.0 = no better than random). '
             'shortlist_projection applies the 1.2-1.4x enrichment band to a top-5% '
             'shortlist of the enumerated library at a 0.494 base rate (gap<=0.10 eV).')
(ROOT/'data'/'triage_metrics.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
