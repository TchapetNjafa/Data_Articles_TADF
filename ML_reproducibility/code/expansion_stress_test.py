#!/usr/bin/env python3
"""
expansion_stress_test.py — F1-C generalisation stress-test (canonical, traceable).

Merges the 231 curated training molecules with the F1-B expanded set and measures how
scaffold-CV accuracy responds as the corpus grows and as label confidence is tightened.
The result is a NEGATIVE, honest finding: expansion does not improve structure-only
ΔE_ST prediction — it exposes the experimental-scatter + chemical-diversity ceiling.

Writes data/expansion_stress_test.json with every number cited in the manuscript.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold, cross_val_predict
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from _dataset import load_dataset, make_features, rf, NTO  # noqa: E402
# Prefer the relocated deliverables in data/; fall back to the TASK-F1-B working dir.
TASK = ROOT / "data" if (ROOT / "data" / "expanded_features_nto_clean.csv").exists() \
    else ROOT / "TASK-F1-B" / "data"


def scaffold(s):
    m = Chem.MolFromSmiles(str(s))
    return (MurckoScaffold.MurckoScaffoldSmiles(mol=m) or "none") if m else "none"

def ik(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToInchiKey(m) if m else None

def cv(X, y, groups):
    oof = cross_val_predict(rf(), X, y, cv=GroupKFold(5), groups=groups, n_jobs=-1)
    return {"n": int(len(y)),
            "mae_eV": round(float(np.mean(np.abs(oof - y))), 4),
            "r2": round(float(1 - np.sum((oof - y)**2) / np.sum((y - y.mean())**2)), 4),
            "spearman_rho": round(float(stats.spearmanr(oof, y)[0]), 4)}


def main():
    ds = load_dataset()
    X_orig, _ = make_features(ds, "NTO")
    y_orig = np.asarray(ds.y, float)
    scaf_orig = np.asarray(ds.scaf, object)
    g_orig = pd.factorize(scaf_orig)[0]
    train_ik = {ik(s) for s in ds.smiles}

    clean = pd.read_csv(TASK / "expanded_features_nto_clean.csv")
    res = pd.read_csv(TASK / "expanded_smiles_resolved.csv")
    todo = res[res["already_featurised"] != True].reset_index(drop=True)  # noqa: E712
    todo["mol_id"] = [f"mol_{i:05d}" for i in range(len(todo))]
    clean = clean.merge(todo[["mol_id", "exp_est", "n_reports", "sd"]], on="mol_id", how="left")

    n_notarget = int(clean["exp_est"].isna().sum())
    clean = clean.dropna(subset=["exp_est"]).copy()
    clean["_ik"] = clean["SMILES"].map(ik)
    dup_train = int(clean["_ik"].isin(train_ik).sum())
    clean = clean[~clean["_ik"].isin(train_ik)]
    n_before = len(clean); clean = clean.drop_duplicates("_ik"); dup_self = n_before - len(clean)

    def expanded(mask):
        sub = clean[mask]
        Xn = sub[NTO].to_numpy(float); yn = sub["exp_est"].to_numpy(float)
        sn = np.asarray([scaffold(s) for s in sub["SMILES"]], object)
        X = np.vstack([X_orig, Xn]); y = np.concatenate([y_orig, yn])
        g = pd.factorize(np.concatenate([scaf_orig, sn]))[0]
        m = cv(X, y, g); m["n_new"] = int(len(sub)); return m

    out = {
        "feature_set": "NTO(35)", "model": "RF(400)", "cv": "scaffold GroupKFold(5)",
        "drop_breakdown": {"no_target": n_notarget, "dup_of_training": dup_train,
                           "dup_within_new": dup_self, "usable_new": int(len(clean))},
        "baseline_231": cv(X_orig, y_orig, g_orig),
        "expanded_all": expanded(clean["exp_est"].notna()),
        "expanded_multireport": expanded(clean["n_reports"] >= 2),
        "note": ("Expansion to a 3x larger, more diverse experimental corpus does not "
                 "improve structure-only DeltaE_ST prediction. Multi-report labels "
                 "(n_reports>=2) partially recover accuracy (label noise) but stay below "
                 "the curated baseline (chemical diversity). The inter-report SD filter is "
                 "confounded: single-report entries have sd=0 by construction, so n_reports "
                 "is the meaningful confidence signal."),
    }
    (ROOT / "data" / "expansion_stress_test.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
