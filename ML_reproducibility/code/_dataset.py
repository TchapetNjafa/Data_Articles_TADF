#!/usr/bin/env python3
"""
_dataset.py — single source of truth for the honest-benchmark dataset.

Extracted verbatim from finalize_model.py so every revision script (B1, B2, C1,
E1, D3) trains on the IDENTICAL 231-molecule / 212-scaffold set. Do not change the
loading/QC logic here without re-running finalize_model.py and updating the headline
numbers in the manuscript + qa_score.py.

Provides:
  load_dataset() -> Namespace(df, y, scaf, smiles, mols, NTO)
  make_features(ds, name) -> (X, feature_names)   name in {'NTO','RDKit','Morgan'}
  rf()                    -> a fresh canonical RandomForestRegressor (400 trees)
"""
import ast, re, warnings
from types import SimpleNamespace
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor

RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent.parent

NTO = ['S1_S_he','T1_S_he','Delta_S_he','Abs_Delta_S_he','S1_CT_number','T1_CT_number',
       'Delta_CT_number','Abs_Delta_CT_number','S1_Lambda_D','S1_Lambda_A','T1_Lambda_D',
       'T1_Lambda_A','Delta_Lambda_D','Abs_Delta_Lambda_D','Delta_Lambda_A','Abs_Delta_Lambda_A',
       'S1_Delta_r','T1_Delta_r','Delta_Delta_r','Abs_Delta_Delta_r','S1_hole_on_A',
       'S1_particle_on_D','T1_hole_on_A','T1_particle_on_D','Char_diff_squared','S_NTO_sum',
       'S_NTO_product','S_NTO_ratio','S1_osc_strength','Delta_S_NTO','Abs_Delta_S_NTO',
       'Log_Abs_S1','Log_Abs_T1','S1_overlap','T1_overlap']


def load_dataset():
    exp = pd.read_csv(ROOT/'MCA_submission/tables/experimental_delta_est_extracted.csv')
    exp = exp[exp['specifier'].astype(str).str.contains('EST', case=False, na=False)]
    nm = lambda x: ([str(n).strip() for n in ast.literal_eval(x)]
                    if str(x).startswith('[') else [str(x)])
    def vv(v):
        try:
            p = ast.literal_eval(v); return float(p[0]) if isinstance(p, list) else float(p)
        except Exception:
            try: return float(re.findall(r'[-\d.]+', str(v))[0])
            except Exception: return np.nan
    exp['nl'] = exp['compound.names'].map(nm); exp['v'] = exp['standard_value'].map(vv)
    exp = exp[np.isfinite(exp.v) & exp.v.between(-0.3, 1.5)]
    feat = pd.read_csv(ROOT/'CALCULATIONS-MADE/data_processing/combined_features_747mol_full_ct.csv')
    feat = feat[(feat.environment == 'gas') & (feat.method == 'stda')].dropna(subset=NTO+['Delta_E_ST_eV'])
    fm = set(feat.molecule.astype(str))
    rows = [(n, r.v) for _, r in exp.iterrows() for n in r['nl'] if n in fm]
    em = (pd.DataFrame(rows, columns=['molecule', 'v']).groupby('molecule').v.median()
          .reset_index().rename(columns={'v': 'exp_est'}))
    ref = (pd.read_csv(ROOT/'Data_Ref.csv')[['Molecule_id', 'compound.SMILES']]
           .rename(columns={'Molecule_id': 'molecule', 'compound.SMILES': 'smiles'}))
    df = feat.merge(em, on='molecule').merge(ref, on='molecule', how='left')
    df['m'] = df.smiles.map(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)
    df = df[df.m.notna()].reset_index(drop=True)
    y = df.exp_est.values
    scaf = df.m.map(lambda m: MurckoScaffold.MurckoScaffoldSmiles(mol=m) or 'none').values
    return SimpleNamespace(df=df, y=y, scaf=scaf, smiles=df.smiles.values,
                           mols=list(df.m.values), NTO=NTO)


_DESC = [n for n, _ in Descriptors._descList]
_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC)


def make_features(ds, name):
    if name == 'NTO':
        return ds.df[NTO].values, list(NTO)
    if name == 'RDKit':
        M = np.vstack([np.nan_to_num(np.array(_CALC.CalcDescriptors(m), dtype=float),
                                     nan=0, posinf=0, neginf=0) for m in ds.mols])
        return M, list(_DESC)
    if name == 'Morgan':
        M = np.vstack([np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), dtype=float)
                       for m in ds.mols])
        return M, [f'fp{i}' for i in range(2048)]
    raise ValueError(name)


def rf(n_estimators=400, random_state=0):
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)


if __name__ == '__main__':
    ds = load_dataset()
    print(f'Loaded {len(ds.df)} molecules, {len(set(ds.scaf))} scaffolds')
