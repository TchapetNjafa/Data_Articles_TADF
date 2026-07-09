#!/usr/bin/env python3
"""
scaffold_split_analysis.py  (Task B1)
=====================================
Rebut "scaffold split is statistically indistinguishable from a random split".

Three measurements on the canonical 231-molecule set (shared loader _dataset.py):

  1. Train/test SCAFFOLD OVERLAP per fold:
       - scaffold GroupKFold  -> 0 shared scaffolds by construction
       - random KFold         -> some molecules share a scaffold across the split
  2. Train/test NEAREST-NEIGHBOUR Tanimoto similarity (Morgan FP, r=2, 2048 bit):
       for every test molecule, max Tanimoto to any TRAIN molecule. Scaffold split
       yields lower NN similarity => test molecules are genuinely less similar to
       training (a harder, leakage-free evaluation) than under random split.
  3. PERMUTATION-LABEL CV test: shuffle y 1000x, recompute scaffold-CV MAE under the
       null. The true CV-MAE sits in the left tail => the model learns real signal,
       not scaffold-group artefacts.

Output: data/scaffold_split_analysis.json  (+ console summary)
"""
import json, sys
from pathlib import Path
from collections import Counter
import numpy as np
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

N_SPLITS = 5
N_PERM = 1000
SEED = 0
ROOT = Path(__file__).parent.parent

ds = load_dataset()
y = ds.y
scaf = ds.scaf
X, _ = make_features(ds, 'NTO')
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in ds.mols]
n = len(y)
print(f'Loaded {n} molecules, {len(set(scaf))} scaffolds')

# ---- 1. scaffold overlap + NN similarity per fold --------------------------
def overlap_stats(splitter, groups=None):
    shared, nn_sims = [], []
    splits = splitter.split(X, y, groups) if groups is not None else splitter.split(X)
    for tr, te in splits:
        s_tr, s_te = set(scaf[tr]), set(scaf[te])
        shared.append(len(s_tr & s_te))
        for j in te:
            sims = DataStructs.BulkTanimotoSimilarity(fps[j], [fps[i] for i in tr])
            nn_sims.append(max(sims))
    return shared, np.array(nn_sims)

gkf = GroupKFold(N_SPLITS)
rkf = KFold(N_SPLITS, shuffle=True, random_state=SEED)
sh_scaf, nn_scaf = overlap_stats(gkf, groups=scaf)
sh_rand, nn_rand = overlap_stats(rkf)

sc = Counter(scaf)
singleton_frac = sum(1 for v in sc.values() if v == 1) / len(sc)

# ---- 2. permutation-label CV test ------------------------------------------
def scaffold_cv_mae(target, n_estimators=400, seed=0):
    yp = cross_val_predict(rf(n_estimators=n_estimators, random_state=seed),
                           X, target, cv=GroupKFold(N_SPLITS), groups=scaf)
    return mean_absolute_error(target, yp)

true_mae = scaffold_cv_mae(y, n_estimators=400)
rng = np.random.default_rng(SEED)
null = np.empty(N_PERM)
for k in range(N_PERM):
    null[k] = scaffold_cv_mae(rng.permutation(y), n_estimators=200, seed=0)
    if (k + 1) % 100 == 0:
        print(f'  permutation {k+1}/{N_PERM}  null-MAE so far '
              f'{null[:k+1].mean():.4f} +/- {null[:k+1].std():.4f}')

p_left = float((np.sum(null <= true_mae) + 1) / (N_PERM + 1))

out = dict(
    n_molecules=int(n), n_scaffolds=int(len(set(scaf))),
    singleton_scaffold_fraction=round(singleton_frac, 3),
    scaffold_split=dict(
        shared_scaffolds_per_fold=[int(x) for x in sh_scaf],
        nn_tanimoto_mean=round(float(nn_scaf.mean()), 3),
        nn_tanimoto_median=round(float(np.median(nn_scaf)), 3)),
    random_split=dict(
        shared_scaffolds_per_fold=[int(x) for x in sh_rand],
        nn_tanimoto_mean=round(float(nn_rand.mean()), 3),
        nn_tanimoto_median=round(float(np.median(nn_rand)), 3)),
    nn_similarity_delta=round(float(nn_rand.mean() - nn_scaf.mean()), 3),
    permutation_test=dict(
        n_permutations=N_PERM,
        true_cv_mae=round(float(true_mae), 4),
        null_mae_mean=round(float(null.mean()), 4),
        null_mae_std=round(float(null.std()), 4),
        null_mae_ci=[round(float(np.percentile(null, 2.5)), 4),
                     round(float(np.percentile(null, 97.5)), 4)],
        p_value_left_tail=round(p_left, 4)))

(ROOT/'data'/'scaffold_split_analysis.json').write_text(json.dumps(out, indent=2))
print('\n' + json.dumps(out, indent=2))
print('\nSaved -> data/scaffold_split_analysis.json')
