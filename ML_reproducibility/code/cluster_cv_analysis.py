#!/usr/bin/env python3
"""
cluster_cv_analysis.py  (Task R1)
=================================
Answer "scaffold GroupKFold == random split because scaffolds are singletons".

Build NON-singleton groups by Butina clustering on Morgan-FP Tanimoto distance, then
evaluate the headline RF under cluster-grouped CV (clusters kept whole within a fold)
across a cutoff scan, and compare with scaffold-grouped CV and a plain random split.
If MAE rises as clusters coarsen (more dissimilar held-out chemistry), that is the
honest generalisation gap; if it stays flat, the data simply has few near-duplicates.

Output: data/cluster_cv.json (+ console)
"""
import json, sys
from pathlib import Path
import numpy as np
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import cross_val_predict, GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

ROOT = Path(__file__).parent.parent
ds = load_dataset()
y = ds.y
X, _ = make_features(ds, 'NTO')
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in ds.mols]
n = len(y)

# Butina clustering: distance matrix (1 - Tanimoto), lower triangle
dists = []
for i in range(1, n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1.0 - s for s in sims])

def cluster_labels(cutoff):
    cl = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    lab = np.empty(n, int)
    for cid, members in enumerate(cl):
        for m in members:
            lab[m] = cid
    return lab, len(cl)

def cv_metrics(groups):
    yp = cross_val_predict(rf(), X, y, cv=GroupKFold(5), groups=groups)
    return dict(MAE=round(mean_absolute_error(y, yp), 4),
                R2=round(r2_score(y, yp), 4),
                rho=round(float(spearmanr(yp, y).correlation), 3))

out = {'n_molecules': int(n), 'cutoff_scan': {}}
for cutoff in (0.3, 0.4, 0.5, 0.6):           # distance cutoff (higher = coarser clusters)
    lab, ncl = cluster_labels(cutoff)
    sizes = np.bincount(lab)
    out['cutoff_scan'][f'{cutoff}'] = dict(
        n_clusters=int(ncl),
        singleton_fraction=round(float((sizes == 1).mean()), 3),
        largest_cluster=int(sizes.max()),
        **cv_metrics(lab))

# references
yp_rand = cross_val_predict(rf(), X, y, cv=KFold(5, shuffle=True, random_state=0))
out['scaffold_cv'] = cv_metrics(ds.scaf)
out['random_cv'] = dict(MAE=round(mean_absolute_error(y, yp_rand), 4),
                        R2=round(r2_score(y, yp_rand), 4),
                        rho=round(float(spearmanr(yp_rand, y).correlation), 3))
(ROOT/'data'/'cluster_cv.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print('\nSaved -> data/cluster_cv.json')
