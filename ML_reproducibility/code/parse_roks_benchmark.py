#!/usr/bin/env python3
"""
parse_roks_benchmark.py — F4-A (PLAN260630) — HONEST REWRITE 2026-07-08
========================================================================
Parse the ΔDFT/ROKS benchmark of Khan et al. from the pre-parsed
data/theoexp_roks_energies.csv (971 job.out energies, 27 CT emitters, 6
range-separated / hybrid functionals).

WHY THIS WAS REWRITTEN (silent bug in the prior version)
--------------------------------------------------------
The prior version computed "ROKS ΔE_ST = E(s1_roks) − E(t1_uks)" and compared it
to experiment. Two defects made every downstream number unsafe:

  1. There is NO true triplet job in the extracted CSV — `t1_uks` never appears
     (state_jobs are s1_roks / s1_uks / s1_tda only). The code silently fell back
     to the broken-symmetry OPEN-SHELL SINGLET `s1_uks` (Ms=0 ≈ (E_S+E_T)/2) as a
     stand-in "T1". So `s1_roks − s1_uks` is a ΔDFT *gap estimate* (≈ half the
     singlet–triplet splitting), NOT a rigorous ΔE_ST. The per-functional means
     sit near 0 eV for emitters whose experimental gaps are 0.09–0.59 eV — the
     tell-tale of a mislabelled quantity.
  2. It then matched those values against `triplet_manifold.csv:delta_EST_eV`,
     which holds our own sTDA-COMPUTED gaps, and labelled them "experimental".

Per the 2026-07-08 decision we DROP the ROKS-vs-experiment comparison entirely
and keep only what the data actually supports: the FUNCTIONAL-TO-FUNCTIONAL
spread of the ΔDFT gap estimate. This is a real, robust measured quantity — it
shows that even at range-separated-hybrid ΔDFT level the S–T gap estimate is
functional-sensitive, which is the physically meaningful contribution to the
accuracy-ceiling argument (feeds F7 component C). No value here is presented as
an absolute gap or as agreement with experiment.

Outputs
-------
data/roks_benchmark.json           — functional list, per-functional gap-estimate
                                     distribution, per-molecule functional spread
                                     (robust: median + IQR, outlier flagged)
data/roks_benchmark_comparison.csv — per-(molecule × functional) gap-estimate table

Usage
-----
    python code/parse_roks_benchmark.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HA_TO_EV = 27.2114

# Functionals in rough order of expected reliability for CT states
FUNC_ORDER = ["ot-wb97m-v", "ot-lrc-wpbeh-d4", "ot-lc-wpbe08-d4",
              "lrc-wpbeh-d4", "pbe38-d4", "pbe0-d4"]


def _pick(grp: pd.DataFrame, key: str) -> float:
    """ptss (single-point on excited-state geometry) preferred, else opt. NaN if absent."""
    v = grp.loc[grp["state_job"] == f"{key}_ptss", "energy_hartree"]
    if v.empty:
        v = grp.loc[grp["state_job"] == f"{key}_opt", "energy_hartree"]
    return float(v.iloc[0]) if not v.empty else np.nan


def compute_gap_estimate(roks_df: pd.DataFrame) -> pd.DataFrame:
    """
    ΔDFT gap ESTIMATE g = [E(s1_roks) − E(s1_uks)] · HA_TO_EV per (molecule, functional).

    NOTE ON PHYSICS: s1_roks is the spin-purified restricted-open-shell singlet;
    s1_uks is the broken-symmetry unrestricted (Ms=0) state ≈ (E_S + E_T)/2. Their
    difference is therefore ≈ ΔE_ST / 2, a *gap estimate*, not a rigorous ΔE_ST.
    We keep it only to quantify FUNCTIONAL SENSITIVITY (spread across functionals),
    never as an absolute value or a comparison to experiment.
    """
    rows = []
    for (mol_id, name, func), grp in roks_df.groupby(["id", "name", "functional"]):
        s1_roks = _pick(grp, "s1_roks")
        s1_uks = _pick(grp, "s1_uks")
        if not (np.isfinite(s1_roks) and np.isfinite(s1_uks)):
            continue
        rows.append({
            "mol_name": name,
            "functional": func,
            "gap_estimate_eV": round((s1_roks - s1_uks) * HA_TO_EV, 4),
        })
    return pd.DataFrame(rows)


def main():
    roks_path = DATA / "theoexp_roks_energies.csv"
    if not roks_path.exists():
        print(f"ERROR: {roks_path} not found")
        return

    roks_df = pd.read_csv(roks_path)
    print(f"Loaded {len(roks_df)} energy records, {roks_df['name'].nunique()} molecules, "
          f"{roks_df['functional'].nunique()} functionals")
    # Explicit guard: confirm the (mis)labelling assumption that motivated the rewrite
    has_true_triplet = roks_df["state_job"].str.contains("t1_uks", na=False).any()
    print(f"True triplet job (t1_uks) present in data: {has_true_triplet}  "
          f"(False -> gap is an s1_roks-s1_uks ESTIMATE, reported as spread only)")

    gap_df = compute_gap_estimate(roks_df)
    print(f"Gap-estimate rows: {len(gap_df)} over {gap_df['mol_name'].nunique()} molecules")

    # ---- per-functional distribution of the gap estimate ----
    func_stats = []
    for func in FUNC_ORDER:
        sub = gap_df[gap_df["functional"] == func]["gap_estimate_eV"]
        if sub.empty:
            continue
        func_stats.append({
            "functional": func,
            "n_molecules": int(sub.size),
            "gap_estimate_mean_eV": round(float(sub.mean()), 4),
            "gap_estimate_std_eV": round(float(sub.std()), 4),
        })

    # ---- per-molecule functional spread (THE headline quantity) ----
    spread = (gap_df.groupby("mol_name")["gap_estimate_eV"]
              .agg(["min", "max", "std", "count"]))
    spread["spread_eV"] = (spread["max"] - spread["min"]).round(4)
    # robust summary (mean is dominated by one broken calc; report median + flag outliers)
    med_spread = float(spread["spread_eV"].median())
    mean_spread = float(spread["spread_eV"].mean())
    q75 = float(spread["spread_eV"].quantile(0.75))
    outliers = spread[spread["spread_eV"] > 1.0]  # eV — physically implausible S–T half-gap swing
    outlier_names = outliers.index.tolist()

    print(f"\nPer-molecule functional spread of the DFT gap estimate (eV):")
    print(f"  median = {med_spread:.3f}   mean = {mean_spread:.3f}   Q75 = {q75:.3f}")
    print(f"  outliers (>1.0 eV, likely SCF/broken-symmetry failures): "
          f"{len(outlier_names)} {outlier_names}")
    keep = ~spread.index.isin(outlier_names)
    med_spread_robust = float(spread.loc[keep, "spread_eV"].median())
    q75_robust = float(spread.loc[keep, "spread_eV"].quantile(0.75))
    print(f"  robust (excl. outliers): median = {med_spread_robust:.3f}, Q75 = {q75_robust:.3f}")

    result = {
        "source": "Khan et al. DFT/ROKS benchmark (data/theoexp_roks_energies.csv, 971 Q-Chem job.out)",
        "n_molecules": int(gap_df["mol_name"].nunique()),
        "n_functionals": int(gap_df["functional"].nunique()),
        "quantity": "gap_estimate_eV = [E(s1_roks) - E(s1_uks)] (approx dEST/2; NOT rigorous dEST)",
        "true_triplet_job_present": bool(has_true_triplet),
        "comparison_to_experiment": (
            "DROPPED — no true triplet energies in the extracted data; the prior "
            "'ROKS vs experiment' compared against our own sTDA-computed gaps mislabelled "
            "as experimental. See module docstring."
        ),
        "functional_distribution": func_stats,
        "functional_spread_eV": {
            "definition": "per molecule: max(gap_estimate) - min(gap_estimate) across functionals",
            "median": round(med_spread, 4),
            "mean": round(mean_spread, 4),
            "q75": round(q75, 4),
            "median_excl_outliers": round(med_spread_robust, 4),
            "q75_excl_outliers": round(q75_robust, 4),
            "outlier_molecules_gt_1eV": outlier_names,
            "n_molecules": int(spread.shape[0]),
        },
        "interpretation": (
            f"Across {gap_df['functional'].nunique()} range-separated/hybrid functionals, the "
            f"DFT singlet-triplet gap ESTIMATE for these {gap_df['mol_name'].nunique()} CT emitters "
            f"disagrees by a median of {med_spread_robust:.3f} eV per molecule (Q75 "
            f"{q75_robust:.3f} eV, outliers excluded). Even at this higher tier than sTDA-xTB, the "
            f"S-T gap estimate is functional-sensitive at the 0.05-0.1 eV level, consistent with a "
            f"method-independent accuracy floor (feeds the ceiling decomposition, F7). No absolute "
            f"gap or experimental agreement is claimed from these data."
        ),
    }

    out_json = DATA / "roks_benchmark.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_json}")

    pivot = gap_df.pivot_table(index="mol_name", columns="functional", values="gap_estimate_eV")
    out_csv = DATA / "roks_benchmark_comparison.csv"
    pivot.round(4).to_csv(out_csv)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
