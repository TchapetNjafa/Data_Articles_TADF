#!/usr/bin/env python3
"""
oscillator_strength_model.py — F5-A (PLAN260630) — CORRECTED
=============================================================
Build a binary classifier for the oscillator strength filter using the real
_dataset.py interface (returns a SimpleNamespace, not positional tuple).

S1_osc_strength is feature index 28 in the NTO feature set.

Outputs
-------
data/oscillator_strength_filter.json  — threshold, accuracy, precision, recall

Usage
-----
    python code/oscillator_strength_model.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from _dataset import load_dataset, make_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

OSC_THRESHOLD = 0.01  # minimum S1 oscillator strength to be considered "bright"


def main():
    ds = load_dataset()
    X_nto, feat_names = make_features(ds, "NTO")
    y = ds.y
    scaf = ds.scaf

    if "S1_osc_strength" not in feat_names:
        print(f"ERROR: S1_osc_strength not in NTO features: {feat_names}")
        (DATA / "oscillator_strength_filter.json").write_text(
            json.dumps({"error": "S1_osc_strength not in feature set"}, indent=2))
        return

    # Extract oscillator strength values
    osc_idx = feat_names.index("S1_osc_strength")
    osc_values = X_nto[:, osc_idx]
    y_osc = (osc_values >= OSC_THRESHOLD).astype(int)

    frac_bright = float(y_osc.mean())
    print(f"Dataset: {len(y)} molecules")
    print(f"S1 oscillator strength ≥ {OSC_THRESHOLD}: {y_osc.sum()} molecules "
          f"({100*frac_bright:.1f}% 'bright')")
    print(f"S1_osc_strength range: {osc_values.min():.4f} – {osc_values.max():.4f}")

    if y_osc.sum() < 5 or y_osc.sum() > len(y_osc) - 5:
        print("WARNING: nearly all molecules in one class — classifier trivial")
        result = {
            "threshold_osc": OSC_THRESHOLD,
            "frac_bright_in_corpus": round(frac_bright, 4),
            "note": "Nearly all molecules bright — filter has minimal discriminatory power at this threshold",
            "scaffold_cv_accuracy": None,
        }
        (DATA / "oscillator_strength_filter.json").write_text(json.dumps(result, indent=2))
        return

    # Scaffold GroupKFold: encode scaffold strings as integers
    enc = LabelEncoder()
    groups = enc.fit_transform(scaf)

    # Train classifier (exclude oscillator strength itself from features — it IS the target)
    feat_no_osc = np.delete(X_nto, osc_idx, axis=1)
    feat_names_no_osc = [f for f in feat_names if f != "S1_osc_strength"]

    clf = RandomForestClassifier(
        n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    gkf = GroupKFold(n_splits=5)
    oof_preds = cross_val_predict(clf, feat_no_osc, y_osc, cv=gkf, groups=groups)

    accuracy = float(np.mean(oof_preds == y_osc))
    report = classification_report(y_osc, oof_preds, output_dict=True, zero_division=0)

    print(f"\nScaffold-CV oscillator classifier:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision (bright): {report.get('1', {}).get('precision', 0):.3f}")
    print(f"  Recall (bright):    {report.get('1', {}).get('recall', 0):.3f}")

    # Distribution by quintile
    quintiles = np.percentile(osc_values, [20, 40, 60, 80])
    print(f"\n  Osc strength quintiles: {[round(q, 4) for q in quintiles]}")

    result = {
        "threshold_osc": OSC_THRESHOLD,
        "frac_bright_in_corpus": round(frac_bright, 4),
        "n_bright": int(y_osc.sum()),
        "n_total": int(len(y_osc)),
        "osc_strength_stats": {
            "min": round(float(osc_values.min()), 4),
            "mean": round(float(osc_values.mean()), 4),
            "median": round(float(np.median(osc_values)), 4),
            "max": round(float(osc_values.max()), 4),
            "p20": round(float(quintiles[0]), 4),
            "p80": round(float(quintiles[3]), 4),
        },
        "scaffold_cv_accuracy": round(accuracy, 4),
        "precision_bright": round(report.get("1", {}).get("precision", 0.0), 4),
        "recall_bright": round(report.get("1", {}).get("recall", 0.0), 4),
        "note": (
            f"Binary oscillator-strength filter (S1_osc ≥ {OSC_THRESHOLD}). "
            f"{100*frac_bright:.0f}% of corpus is 'bright'. Used as Filter 2 in multi_criteria_triage.py. "
            f"Scaffold-CV accuracy={accuracy:.3f}."
        ),
    }

    out = DATA / "oscillator_strength_filter.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
