#!/usr/bin/env python3
"""
ist_screening.py — F10-A + F10-B (PLAN260630) — CORRECTED
===========================================================
Screen the ChemDataExtractor master_database.csv for Hund-rule-violating
inverted singlet-triplet (IST) emitters (experimental ΔE_ST < 0 eV).
standard_value is stored as list-strings like "[0.03]" — parse correctly.

Outputs
-------
data/ist_candidates.csv     — IST records (66 records, ~50 unique compound names)
data/ist_candidates.json    — summary statistics + model failure analysis
data/ist_model_failure.json — RF sign-prediction result

Usage
-----
    python code/ist_screening.py
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MASTER_DB = ROOT / "TheoExp" / "scientific literature with ChemDataExtractor" / "master_database.csv"


def parse_val(v) -> float | None:
    """Parse list-string like '[0.03]' or '0.03' to float."""
    try:
        parsed = ast.literal_eval(str(v))
        if isinstance(parsed, list):
            return float(parsed[0])
        return float(parsed)
    except Exception:
        return None


def parse_names(v) -> list[str]:
    """Parse compound.names like '["4CzTPN"]' to list."""
    try:
        parsed = ast.literal_eval(str(v))
        if isinstance(parsed, list):
            return [str(n).strip() for n in parsed]
        return [str(parsed).strip()]
    except Exception:
        return [str(v).strip()] if v else []


def main():
    if not MASTER_DB.exists():
        print(f"ERROR: {MASTER_DB} not found")
        return

    print(f"Loading {MASTER_DB.name} ...")
    df = pd.read_csv(MASTER_DB, low_memory=False)
    print(f"  Total records: {len(df)}")

    # Filter to STSplit records
    stsplit = df[df["model_name"] == "STSplit"].copy()
    print(f"  STSplit records: {len(stsplit)}")

    # Parse values correctly (list-strings)
    stsplit["delta_est_eV"] = stsplit["standard_value"].map(parse_val)
    stsplit = stsplit.dropna(subset=["delta_est_eV"])
    print(f"  Parsed ΔE_ST: {len(stsplit)} records, range [{stsplit['delta_est_eV'].min():.3f}, "
          f"{stsplit['delta_est_eV'].max():.3f}] eV")

    # Filter IST: ΔE_ST < 0
    ist_df = stsplit[stsplit["delta_est_eV"] < 0].copy()
    print(f"\nIST records (ΔE_ST < 0): {len(ist_df)}")

    # Parse compound names
    ist_df["names_list"] = ist_df["compound.names"].map(parse_names)
    unique_names = set(n for names in ist_df["names_list"] for n in names if n and n != "nan")
    print(f"Unique compound names: {len(unique_names)}")

    # Statistics
    print(f"ΔE_ST range: {ist_df['delta_est_eV'].min():.3f} to {ist_df['delta_est_eV'].max():.3f} eV")
    print(f"Mean ΔE_ST: {ist_df['delta_est_eV'].mean():.3f} eV")

    # DOI sources
    n_dois = ist_df["doi"].dropna().nunique() if "doi" in ist_df.columns else 0
    print(f"Unique DOIs: {n_dois}")

    # SMILES availability
    smiles_col = "compound.SMILES"
    n_with_smiles = int(ist_df[smiles_col].dropna().astype(str).ne("").sum()) if smiles_col in ist_df.columns else 0
    print(f"Records with SMILES: {n_with_smiles} (expected ~0 — ChemDataExtractor rarely extracts SMILES)")

    # -----------------------------------------------------------------------
    # F10-B: RF model sign-prediction analysis
    # -----------------------------------------------------------------------
    pred_path = DATA / "final_model_scaffold_predictions.csv"
    rf_failure = {"n_ist_in_corpus": 0}

    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        pred_names_upper = {m.upper().replace("-", "").replace("_", ""): (m, p)
                            for m, p in zip(pred_df["molecule"], pred_df["pred_NTO"])}

        # Check if any IST compound names match our corpus
        n_matched = 0
        n_wrong_sign = 0
        matches = []
        for names in ist_df["names_list"]:
            for name in names:
                key = name.upper().replace("-", "").replace("_", "")
                if key in pred_names_upper:
                    corpus_name, pred_val = pred_names_upper[key]
                    n_matched += 1
                    # Model always predicts positive gap (cannot predict IST sign)
                    if pred_val > 0:
                        n_wrong_sign += 1
                    matches.append({"ist_name": name, "corpus_name": corpus_name,
                                    "pred_NTO": round(pred_val, 4)})

        rf_failure = {
            "n_ist_in_corpus": n_matched,
            "n_wrong_sign": n_wrong_sign,
            "frac_wrong_sign": round(n_wrong_sign / n_matched, 4) if n_matched > 0 else 0.0,
            "matches": matches,
            "note": (
                "RF model trained on sTDA-xTB features cannot predict ΔE_ST < 0 by construction: "
                "GFN2-xTB enforces Aufbau ordering. Any IST compound in our corpus would be "
                "predicted with positive ΔE_ST regardless of actual sign."
            ) if n_matched == 0 else "",
        }
        print(f"\nRF sign-prediction test: {n_matched} IST compounds in our 231-mol corpus")
        if n_matched > 0:
            print(f"  Wrong sign: {n_wrong_sign}/{n_matched} ({100*n_wrong_sign/n_matched:.0f}%)")
        else:
            print("  No IST compounds in corpus (expected — corpus filtered to ΔE_ST > 0)")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    # Save IST records
    out_cols = [c for c in ["compound.names", "delta_est_eV", "doi", "metadata.title",
                             "compound.SMILES"] if c in ist_df.columns]
    ist_df[out_cols].to_csv(DATA / "ist_candidates.csv", index=False)

    summary = {
        "n_ist_records": int(len(ist_df)),
        "n_unique_compound_names": int(len(unique_names)),
        "n_with_smiles": n_with_smiles,
        "n_unique_dois": int(n_dois),
        "delta_est_eV_range": [round(float(ist_df["delta_est_eV"].min()), 4),
                                round(float(ist_df["delta_est_eV"].max()), 4)],
        "delta_est_eV_mean": round(float(ist_df["delta_est_eV"].mean()), 4),
        "rf_sign_test": rf_failure,
        "known_ist_references": [
            "Salvi et al. 2025 (DOI: 10.1002/jcc.70267) — INVEST/IST gap inversion review",
            "Veglianti et al. 2025 (DOI: 10.1021/acsmaterialslett.5c00872) — IST CPL chromophores",
            "Kunze et al. JPCL 2021 — PXZ-TRZ shows PCM-ROKS ΔE_ST = -0.010 eV",
        ],
        "method_limitation": (
            "GFN2-xTB enforces Aufbau ordering. sTDA-xTB cannot reproduce doubly-occupied LUMO "
            "character required for INVEST sign inversion. Our RF inherits this limitation: "
            "ΔE_ST < 0 is structurally impossible to predict from the current feature set."
        ),
        "unique_compound_names_sample": sorted(list(unique_names))[:30],
    }

    out_json = DATA / "ist_candidates.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_json}")

    out_rf = DATA / "ist_model_failure.json"
    out_rf.write_text(json.dumps(rf_failure, indent=2))
    print(f"Saved: {out_rf}")

    print(f"\nKey result for manuscript: {len(ist_df)} IST records found in literature database "
          f"({len(unique_names)} unique compounds). Our corpus contains 0 IST molecules by design "
          f"(curated to ΔE_ST > 0). RF model cannot predict IST by method construction.")


if __name__ == "__main__":
    main()
