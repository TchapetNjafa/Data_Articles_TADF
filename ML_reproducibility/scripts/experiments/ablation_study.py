#!/usr/bin/env python3
"""
Ablation Study — Feature Leakage Resolution for Article 3
==========================================================

Addresses Critical Reviewer Concern:
  "The feature set includes both S₁ and T₁ excitation energies, yet the target
   is ΔE_ST = E_S₁ - E_T₁. This creates direct feature leakage."

Two models are trained and compared:
  Model C: Full feature set (all 42 columns) → predict ΔE_ST
  Model D: CT descriptors ONLY (no energy features) → predict ΔE_ST directly
           If Model D achieves meaningful R², CT descriptors have independent
           predictive power beyond the trivial E_S₁ - E_T₁ subtraction.

Note: In this dataset, ΔE_ST ≡ E_S₁ - E_T₁ exactly, so a "delta-learning"
approach (predicting δ = ΔE_ST - (E_S₁-E_T₁)) would trivially yield zero.
Instead we test whether CT descriptors alone can predict ΔE_ST.

Output: Comparison table + SHAP importance plots for each model.
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# Parallelization config — must be imported BEFORE numpy/sklearn
from parallel_config import (MAX_THREADS, GPU_AVAILABLE, NUMBA_AVAILABLE,
                              JOBLIB_AVAILABLE, jit, prange,
                              Parallel, delayed, get_parallel_config)

import numpy as np
import pandas as pd

# ──────────────────────── Configuration ────────────────────────
# All data copied locally into ML-IMPROVEMENT/data/
DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_FILE = DATA_DIR / "csv" / "combined_features_747mol_full_ct.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "ablation_study"

# Feature groups
ENERGY_FEATURES = [
    "S1_energy_eV", "T1_energy_eV", "HOMO_LUMO_gap_eV",
    "S1_osc_strength"
]

CT_FEATURES = [
    "S1_CT_number", "S1_Lambda_D", "S1_Lambda_A",
    "S1_hole_on_A", "S1_particle_on_D", "S1_Delta_r", "S1_S_he",
    "T1_CT_number", "T1_Lambda_D", "T1_Lambda_A",
    "T1_hole_on_A", "T1_particle_on_D", "T1_Delta_r", "T1_S_he",
    "Delta_CT_number", "Abs_Delta_CT_number",
    "Delta_Lambda_D", "Abs_Delta_Lambda_D",
    "Delta_Lambda_A", "Abs_Delta_Lambda_A",
    "Delta_Delta_r", "Abs_Delta_Delta_r",
    "Delta_S_he", "Abs_Delta_S_he"
]

OVERLAP_FEATURES = [
    "S1_overlap", "T1_overlap", "Delta_S_NTO", "Abs_Delta_S_NTO",
    "Char_diff_squared", "S_NTO_sum", "S_NTO_product",
    "Log_Abs_S1", "Log_Abs_T1", "S_NTO_ratio"
]

TARGET_COL = "Delta_E_ST_eV"


def load_data(describe_only=False):
    """Load and prepare the feature dataset."""
    if not FEATURES_FILE.exists():
        print(f"ERROR: Feature file not found: {FEATURES_FILE}")
        sys.exit(1)

    df = pd.read_csv(FEATURES_FILE)
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    if describe_only:
        print("\n--- Numeric columns ---")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in numeric_cols:
            print(f"  {c}: {df[c].notna().sum()} non-null, "
                  f"mean={df[c].mean():.4f}, std={df[c].std():.4f}")
        return None

    # Verify target column
    if TARGET_COL not in df.columns:
        print(f"WARNING: '{TARGET_COL}' not found. Checking alternatives...")
        # Try to compute it
        if "S1_energy_eV" in df.columns and "T1_energy_eV" in df.columns:
            df[TARGET_COL] = df["S1_energy_eV"] - df["T1_energy_eV"]
            print(f"  Computed {TARGET_COL} = S1_energy_eV - T1_energy_eV")
        else:
            print("ERROR: Cannot find or compute target column.")
            sys.exit(1)

    # Drop rows with NaN in essential columns
    essential = [TARGET_COL] + [c for c in ENERGY_FEATURES + CT_FEATURES + OVERLAP_FEATURES
                                 if c in df.columns]
    n_before = len(df)
    df = df.dropna(subset=essential)
    print(f"After dropping NaN: {len(df)} samples (removed {n_before - len(df)})")

    return df


def get_feature_sets(df):
    """Define feature sets for each ablation model."""
    available = set(df.columns)

    # Model C: Full feature set (all numeric except target and identifiers)
    id_cols = {"molecule", "environment", "method"}
    all_numeric = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in id_cols and c != TARGET_COL]
    model_c_features = [c for c in all_numeric if c in available]

    # Model D: CT descriptors ONLY (no energy features)
    model_d_ct_only = [c for c in CT_FEATURES + OVERLAP_FEATURES if c in available]

    # Model A (bonus): Energy features ONLY (trivial baseline)
    model_a_energy = [c for c in ENERGY_FEATURES if c in available]

    return {
        "Model_A_energy_only": model_a_energy,
        "Model_C_full": model_c_features,
        "Model_D_CT_only": model_d_ct_only,
    }


def train_and_evaluate(df, feature_cols, target_col, model_name,
                       n_splits=5, random_state=42):
    """Train SVR model with cross-validation. Returns metrics dict."""
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline

    X = df[feature_cols].values
    y = df[target_col].values

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")
    print(f"  Target: {target_col}")
    print(f"{'='*60}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"))
    ])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(pipe, X, y, cv=kf, n_jobs=min(n_splits, MAX_THREADS))

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print(f"\n  Results ({n_splits}-fold CV):")
    print(f"    MAE  = {mae:.4f} eV")
    print(f"    RMSE = {rmse:.4f} eV")
    print(f"    R²   = {r2:.4f}")

    # Train final model on full data for SHAP
    pipe.fit(X, y)

    return {
        "model_name": model_name,
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "feature_names": feature_cols,
        "mae_eV": round(mae, 4),
        "rmse_eV": round(rmse, 4),
        "r2": round(r2, 4),
        "delta_learning": False,
        "pipeline": pipe,
        "X": X,
        "y": y,
    }


def run_shap_analysis(result, output_dir):
    """Run SHAP analysis on trained model."""
    try:
        import shap
    except ImportError:
        print("  SHAP not installed. Skipping SHAP analysis.")
        print("  Install with: pip install shap")
        return None

    print(f"\n  Running SHAP analysis for {result['model_name']}...")

    pipe = result["pipeline"]
    X = result["X"]
    feature_names = result["feature_names"]

    # Use KernelExplainer for SVR (model-agnostic)
    # Subsample for speed
    n_bg = min(100, len(X))
    bg_idx = np.random.choice(len(X), n_bg, replace=False)
    X_bg = pipe.named_steps["scaler"].transform(X[bg_idx])

    n_explain = min(200, len(X))
    exp_idx = np.random.choice(len(X), n_explain, replace=False)
    X_explain = pipe.named_steps["scaler"].transform(X[exp_idx])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(pipe.named_steps["svr"].predict, X_bg)
        shap_values = explainer.shap_values(X_explain, nsamples=100)

    # Feature importance (mean absolute SHAP)
    importance = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance
    }).sort_values("mean_abs_shap", ascending=False)

    imp_df.to_csv(output_dir / f"shap_{result['model_name']}.csv", index=False)
    print(f"  SHAP results saved to shap_{result['model_name']}.csv")

    # Print top 10
    print(f"\n  Top 10 features by SHAP importance:")
    for i, row in imp_df.head(10).iterrows():
        pct = 100 * row["mean_abs_shap"] / importance.sum()
        print(f"    {row['feature']:30s}  {row['mean_abs_shap']:.4f}  ({pct:.1f}%)")

    return imp_df


def main():
    parser = argparse.ArgumentParser(description="Ablation study for feature leakage resolution")
    parser.add_argument("--describe-only", action="store_true",
                        help="Only describe the dataset, don't train models")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP analysis (faster)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for results")
    args = parser.parse_args()

    df = load_data(describe_only=args.describe_only)
    if df is None:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = get_feature_sets(df)
    results = []

    # ── Model A: Energy-only baseline ──
    res_a = train_and_evaluate(
        df, feature_sets["Model_A_energy_only"], TARGET_COL,
        "Model_A_energy_only"
    )
    results.append(res_a)

    # ── Model C: Full feature set ──
    res_c = train_and_evaluate(
        df, feature_sets["Model_C_full"], TARGET_COL,
        "Model_C_full"
    )
    results.append(res_c)

    # ── Model D: CT descriptors only (no energy features) ──
    res_d = train_and_evaluate(
        df, feature_sets["Model_D_CT_only"], TARGET_COL,
        "Model_D_CT_only"
    )
    results.append(res_d)

    # ── SHAP analysis ──
    if not args.skip_shap:
        for res in results:
            run_shap_analysis(res, output_dir)

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("  ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<30s}  {'Features':>8s}  {'MAE (eV)':>9s}  {'RMSE (eV)':>10s}  {'R²':>6s}")
    print("-" * 70)
    for res in results:
        print(f"  {res['model_name']:<30s}  {res['n_features']:>8d}  "
              f"{res['mae_eV']:>9.4f}  {res['rmse_eV']:>10.4f}  {res['r2']:>6.4f}")
    print("=" * 70)

    print("\n  INTERPRETATION:")
    print("  Model A (energy-only): baseline showing how well E_S1, E_T1 alone")
    print("    can predict ΔE_ST (since ΔE_ST ≡ E_S1-E_T1, this should be near-perfect)")
    print("  Model C (full): current publication model")
    print("  Model D (CT-only): if R² > 0.5, CT descriptors capture REAL physics")
    print("    beyond just the energy subtraction → feature leakage concern mitigated")
    print("  Key metric: Model D R² quantifies the independent value of CT descriptors")

    # Save summary
    par_cfg = get_parallel_config()
    summary = {
        "date": datetime.now().isoformat(),
        "dataset": str(FEATURES_FILE),
        "n_samples": len(df),
        "parallelization": par_cfg,
        "results": [{k: v for k, v in r.items()
                      if k not in ("pipeline", "X", "y")}
                     for r in results]
    }
    with open(output_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {output_dir / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
