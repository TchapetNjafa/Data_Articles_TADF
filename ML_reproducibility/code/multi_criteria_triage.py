#!/usr/bin/env python3
"""
multi_criteria_triage.py — F5-B (PLAN260630)
=============================================
Multi-criteria triage combining: predicted ΔE_ST, oscillator strength proxy,
and T1–T2 gap filter (from F6). Computes enrichment vs single-criterion baseline.

Outputs
-------
data/multi_criteria_triage.json     — enrichment metrics for single vs multi criteria
data/shortlist_top100_multi.csv     — top-100 candidates passing all three filters
figures/enrichment_comparison.pdf   — comparison figure

Usage
-----
    python code/multi_criteria_triage.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _dataset import load_dataset, make_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "digital_discovery_manuscript" / "figures"

# Thresholds
DELTA_EST_THRESHOLD = 0.10  # eV — "good" TADF candidate
OSC_THRESHOLD = 0.01         # min oscillator strength (not dark)
T1T2_THRESHOLD = 0.10        # eV — manifold safety margin


# ---------------------------------------------------------------------------
# Load predictions and T1–T2 data
# ---------------------------------------------------------------------------
def load_data():
    pred_path = DATA / "final_model_scaffold_predictions.csv"
    t1t2_path = DATA / "t1_t2_full.csv"

    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found — run the model-fitting step first; "
            "refusing to fabricate placeholder predictions.")
    pred_df = pd.read_csv(pred_path)

    t1t2_df = None
    if t1t2_path.exists():
        t1t2_df = pd.read_csv(t1t2_path)
        # Use gas phase
        t1t2_df = t1t2_df[t1t2_df.get("phase", pd.Series("gas")) == "gas"] if "phase" in t1t2_df.columns else t1t2_df

    return pred_df, t1t2_df


# ---------------------------------------------------------------------------
# Compute enrichment metrics
# ---------------------------------------------------------------------------
def compute_enrichment(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, base_rate: float,
                        fractions: list[float] = None) -> list[dict]:
    if fractions is None:
        fractions = [0.05, 0.10, 0.20, 0.30, 0.50]
    n = len(y_true)
    order = np.argsort(y_pred)  # ascending: low predicted gap first
    y_sorted = y_true[order]

    results = []
    for frac in fractions:
        k = max(1, int(np.round(frac * n)))
        top_k = y_sorted[:k]
        precision = float(np.mean(top_k <= threshold))
        enrichment = precision / base_rate if base_rate > 0 else 0.0
        results.append({
            "fraction": frac,
            "k": k,
            "precision": round(precision, 4),
            "enrichment": round(enrichment, 4),
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pred_df, t1t2_df = load_data()

    # Identify column names
    y_true_col = next((c for c in pred_df.columns if "true" in c.lower() or "exp" in c.lower()), pred_df.columns[1])
    y_pred_col = next((c for c in pred_df.columns if "pred" in c.lower()), pred_df.columns[2] if len(pred_df.columns) > 2 else pred_df.columns[1])
    mol_col = next((c for c in pred_df.columns if "mol" in c.lower() or "name" in c.lower()), pred_df.columns[0])

    y_true = pred_df[y_true_col].values.astype(float)
    y_pred = pred_df[y_pred_col].values.astype(float)
    base_rate = float(np.mean(y_true <= DELTA_EST_THRESHOLD))
    n = len(y_true)

    print(f"Dataset: {n} molecules, base_rate = {base_rate:.3f}")

    # --- Single-criterion enrichment ---
    single_curve = compute_enrichment(y_true, y_pred, DELTA_EST_THRESHOLD, base_rate)

    # --- Multi-criteria enrichment ---
    # Build composite mask: apply each additional filter
    composite_mask = np.ones(n, dtype=bool)

    # Filter 2: NTO oscillator strength — aligned by molecule name (no silent fallback)
    ds = load_dataset()
    X_nto, feat_names = make_features(ds, "NTO")
    if "S1_osc_strength" not in feat_names:
        raise KeyError(f"S1_osc_strength missing from NTO features: {feat_names}")
    osc_by_mol = dict(zip(map(str, ds.df.molecule),
                          X_nto[:, feat_names.index("S1_osc_strength")]))
    osc = np.array([osc_by_mol.get(str(m), np.nan) for m in pred_df[mol_col].values])
    n_missing = int(np.isnan(osc).sum())
    if n_missing:
        raise ValueError(f"{n_missing} prediction molecules have no NTO oscillator value "
                         "(name mismatch) — refusing to silently pass them")
    composite_mask &= (osc >= OSC_THRESHOLD)
    print(f"  After oscillator filter (>={OSC_THRESHOLD}): {composite_mask.sum()} molecules")

    # Filter 3: T1–T2 gap safety
    if t1t2_df is not None and "T1_T2_gap_eV" in t1t2_df.columns:
        t1t2_by_mol = t1t2_df.groupby(t1t2_df.columns[0])["T1_T2_gap_eV"].first()
        mol_names = pred_df[mol_col].values
        t1t2_vals = np.array([t1t2_by_mol.get(m, np.nan) for m in mol_names])
        t1t2_mask = (t1t2_vals > T1T2_THRESHOLD) | np.isnan(t1t2_vals)  # pass if no data (safe)
        composite_mask &= t1t2_mask
        print(f"  After T1–T2 filter (>{T1T2_THRESHOLD} eV): {composite_mask.sum()} molecules")

    # Apply composite mask to predictions
    y_true_mc = y_true[composite_mask]
    y_pred_mc = y_pred[composite_mask]
    base_rate_mc = float(np.mean(y_true_mc <= DELTA_EST_THRESHOLD)) if len(y_true_mc) > 0 else base_rate

    multi_curve = compute_enrichment(y_true_mc, y_pred_mc, DELTA_EST_THRESHOLD, base_rate,  # use original base_rate
                                     fractions=[0.05, 0.10, 0.20, 0.30, 0.50])

    # --- Top-100 multi-criteria shortlist ---
    order_mc = np.argsort(y_pred_mc)
    k100 = min(100, len(order_mc))
    shortlist_idx = order_mc[:k100]
    orig_idx = np.where(composite_mask)[0][shortlist_idx]
    shortlist_df = pred_df.iloc[orig_idx].copy()
    shortlist_df["composite_rank"] = range(1, k100 + 1)
    shortlist_df.to_csv(DATA / "shortlist_top100_multi.csv", index=False)

    # --- Save results ---
    result = {
        "n_total": n,
        "n_after_composite_filter": int(composite_mask.sum()),
        "base_rate": round(base_rate, 4),
        "delta_est_threshold_eV": DELTA_EST_THRESHOLD,
        "oscillator_threshold": OSC_THRESHOLD,
        "t1t2_threshold_eV": T1T2_THRESHOLD,
        "single_criterion_curve": single_curve,
        "multi_criteria_curve": multi_curve,
    }
    out_json = DATA / "multi_criteria_triage.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"Saved: {out_json}")

    # --- Figure: enrichment comparison ---
    FIGURES.mkdir(parents=True, exist_ok=True)
    fracs_s = [r["fraction"] for r in single_curve]
    enrich_s = [r["enrichment"] for r in single_curve]
    fracs_m = [r["fraction"] for r in multi_curve]
    enrich_m = [r["enrichment"] for r in multi_curve]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fracs_s, enrich_s, "o-", color="#2196F3", linewidth=2, label="Single criterion (ΔE_ST only)")
    ax.plot(fracs_m, enrich_m, "s--", color="#FF9800", linewidth=2, label="Multi-criteria (gap + osc + T₁–T₂)")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.2, label="Random baseline (1.0×)")
    ax.set_xlabel("Fraction of candidates selected", fontsize=11)
    ax.set_ylabel("Enrichment factor", fontsize=11)
    ax.set_title("Triage enrichment: single vs multi-criteria screening", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES / "enrichment_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {FIGURES / 'enrichment_comparison.pdf'}")


if __name__ == "__main__":
    main()
