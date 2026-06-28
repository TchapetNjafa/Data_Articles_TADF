#!/usr/bin/env python3
"""
si_supplementary_data.py — generate SI tables for the honest model (F7/F10).
Outputs:
  data/si_nto_shap_means.csv     — per-feature mean|SHAP| for the NTO-only RF (honest model)
  data/si_corpus_provenance.json — experimental-corpus matching/dedup statistics
"""
import json, ast, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
import shap
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent.parent

NTO = ['S1_S_he','T1_S_he','Delta_S_he','Abs_Delta_S_he','S1_CT_number','T1_CT_number',
       'Delta_CT_number','Abs_Delta_CT_number','S1_Lambda_D','S1_Lambda_A','T1_Lambda_D',
       'T1_Lambda_A','Delta_Lambda_D','Abs_Delta_Lambda_D','Delta_Lambda_A','Abs_Delta_Lambda_A',
       'S1_Delta_r','T1_Delta_r','Delta_Delta_r','Abs_Delta_Delta_r','S1_hole_on_A',
       'S1_particle_on_D','T1_hole_on_A','T1_particle_on_D','Char_diff_squared','S_NTO_sum',
       'S_NTO_product','S_NTO_ratio','S1_osc_strength','Delta_S_NTO','Abs_Delta_S_NTO',
       'Log_Abs_S1','Log_Abs_T1','S1_overlap','T1_overlap']

# ---- experimental corpus with full provenance accounting ----
exp = pd.read_csv(ROOT/'MCA_submission/tables/experimental_delta_est_extracted.csv')
prov = {'raw_rows': int(len(exp))}
exp_est = exp[exp['specifier'].astype(str).str.contains('EST',case=False,na=False)].copy()
prov['rows_specifier_EST'] = int(len(exp_est))
nm=lambda x:( [str(n).strip() for n in ast.literal_eval(x)] if str(x).startswith('[') else [str(x)])
def vv(v):
    try: p=ast.literal_eval(v); return float(p[0]) if isinstance(p,list) else float(p)
    except:
        try: return float(re.findall(r'[-\d.]+',str(v))[0])
        except: return np.nan
exp_est['nl']=exp_est['compound.names'].map(nm); exp_est['v']=exp_est['standard_value'].map(vv)
n_parse_fail = int(exp_est['v'].isna().sum())
exp_est=exp_est[np.isfinite(exp_est.v)]
prov['rows_value_parsed'] = int(len(exp_est))
prov['rows_value_unparseable'] = n_parse_fail
in_range = exp_est[exp_est.v.between(-0.3,1.5)]
prov['rows_in_physical_range'] = int(len(in_range))
prov['rows_dropped_out_of_range'] = int(len(exp_est)-len(in_range))

feat=pd.read_csv(ROOT/'CALCULATIONS-MADE/data_processing/combined_features_747mol_full_ct.csv')
feat=feat[(feat.environment=='gas')&(feat.method=='stda')].dropna(subset=NTO+['Delta_E_ST_eV'])
fm=set(feat.molecule.astype(str))
rows=[(n,r.v) for _,r in in_range.iterrows() for n in r['nl'] if n in fm]
m=pd.DataFrame(rows,columns=['molecule','exp'])
prov['name_value_matches'] = int(len(m))
prov['unique_molecules_matched'] = int(m.molecule.nunique())
g=m.groupby('molecule').exp
spread=g.std().dropna()
prov['molecules_with_multiple_reports'] = int((g.count()>1).sum())
prov['median_inter_report_spread_eV'] = round(float(spread.median()),3)
prov['max_inter_report_spread_eV'] = round(float(spread.max()),3)
em=g.median().reset_index()

ref=pd.read_csv(ROOT/'Data_Ref.csv')[['Molecule_id','compound.SMILES']].rename(columns={'Molecule_id':'molecule','compound.SMILES':'smiles'})
df=feat.merge(em,on='molecule').merge(ref,on='molecule',how='left')
df['mol']=df.smiles.map(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)
n_no_smiles=int(df.mol.isna().sum())
df=df[df.mol.notna()].reset_index(drop=True)
prov['molecules_dropped_no_valid_smiles'] = n_no_smiles
prov['final_benchmark_n'] = int(len(df))
prov['final_scaffolds'] = int(df.mol.map(lambda mm: MurckoScaffold.MurckoScaffoldSmiles(mol=mm) or 'none').nunique())
prov['matching_rule'] = ('compound.names exploded; exact string match to feature-set molecule ids; '
                         'specifier must contain "EST"; standard_value in [-0.3,1.5] eV; '
                         'per-molecule median over reports; SMILES from Data_Ref.csv (RDKit-parseable).')
(ROOT/'data'/'si_corpus_provenance.json').write_text(json.dumps(prov,indent=2))

# ---- per-feature SHAP for the honest NTO-only model ----
X=df[NTO].values; y=df.exp.values
m_full=RandomForestRegressor(400,random_state=0,n_jobs=-1).fit(X,y)
sv=shap.TreeExplainer(m_full).shap_values(X)
sh=pd.DataFrame({'feature':NTO,'mean_abs_shap_eV':np.abs(sv).mean(0)}).sort_values('mean_abs_shap_eV',ascending=False)
sh.to_csv(ROOT/'data'/'si_nto_shap_means.csv',index=False)
print(json.dumps(prov,indent=2))
print('\nTop-12 NTO SHAP:'); print(sh.head(12).to_string(index=False))
print('\nSaved si_corpus_provenance.json + si_nto_shap_means.csv')
