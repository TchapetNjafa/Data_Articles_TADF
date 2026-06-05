"""
run_svr_shap_verification.py
============================
Re-runs the SVR ablation study (Models A, C, D) and SHAP analysis on
the 736/2943-sample dataset to verify the feature importance values
reported in the CEJ manuscript.

Reproduces the logic of ablation_study.py without requiring parallel_config.

Models
------
  Model A : Energy-only features (E_S1, E_T1, HOMO-LUMO, f_osc)
  Model C : Full feature set (all 38 molecular descriptors)
  Model D : CT/NTO descriptors ONLY (no excitation energies)

SVR hyperparameters (from ablation_study.py)
--------------------------------------------
  kernel=rbf, C=10.0, epsilon=0.01, gamma=scale
  StandardScaler pre-processing
  5-fold CV (shuffle=True, random_state=42)

SHAP
----
  KernelExplainer on scaled SVR (model-agnostic)
  Background: 100 random samples
  Explain:    200 random samples
  nsamples=100 per explanation

Outputs (written to code/svr_shap_results/)
-------------------------------------------
  ablation_summary.json          model metrics (MAE, RMSE, R²)
  shap_Model_A_energy_only.csv   mean |SHAP| per feature
  shap_Model_C_full.csv
  shap_Model_D_CT_only.csv
  shap_summary_report.txt        human-readable percentage table

Usage
-----
    python code/run_svr_shap_verification.py
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import shap

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
REPO_ROOT    = Path("/home/tchapet/Documents/GitHub/TADF/smiEmpirical-TADF")
DATA_FILE    = (REPO_ROOT / "Public_Results/Result_article1_TADF_xTB"
                / "ML_reproducibility/features/combined_features_747mol_full_ct.csv")
OUT_DIR      = SCRIPT_DIR / "svr_shap_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "Delta_E_ST_eV"
RANDOM_STATE = 42
N_SPLITS     = 5

# ── Feature sets (mirrors ablation_study.py) ─────────────────────────────────
ENERGY_FEATURES  = ["S1_energy_eV", "T1_energy_eV", "HOMO_LUMO_gap_eV", "S1_osc_strength"]
CT_FEATURES      = [
    "S1_CT_number", "S1_Lambda_D", "S1_Lambda_A",
    "S1_hole_on_A", "S1_particle_on_D", "S1_Delta_r", "S1_S_he",
    "T1_CT_number", "T1_Lambda_D", "T1_Lambda_A",
    "T1_hole_on_A", "T1_particle_on_D", "T1_Delta_r", "T1_S_he",
    "Delta_CT_number", "Abs_Delta_CT_number",
    "Delta_Lambda_D", "Abs_Delta_Lambda_D",
    "Delta_Lambda_A", "Abs_Delta_Lambda_A",
    "Delta_Delta_r", "Abs_Delta_Delta_r",
    "Delta_S_he", "Abs_Delta_S_he",
]
OVERLAP_FEATURES = [
    "S1_overlap", "T1_overlap", "Delta_S_NTO", "Abs_Delta_S_NTO",
    "Char_diff_squared", "S_NTO_sum", "S_NTO_product",
    "Log_Abs_S1", "Log_Abs_T1", "S_NTO_ratio",
]

# ── SHAP grouping for percentage summary ─────────────────────────────────────
SHAP_GROUPS = {
    "Energy":      ["S1_energy_eV", "T1_energy_eV", "HOMO_LUMO_gap_eV"],
    "Oscillator":  ["S1_osc_strength"],
    "NTO_CT":      CT_FEATURES + OVERLAP_FEATURES,
}


def load_data() -> pd.DataFrame:
    print(f"Loading data from:\n  {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    if TARGET_COL not in df.columns:
        df[TARGET_COL] = df["S1_energy_eV"] - df["T1_energy_eV"]
        print(f"  Computed {TARGET_COL} = S1 - T1")

    id_cols = {"molecule", "environment", "method"}
    all_needed = set(ENERGY_FEATURES + CT_FEATURES + OVERLAP_FEATURES + [TARGET_COL])
    available  = set(df.columns)
    drop_na_on = list(all_needed & available)
    n_before   = len(df)
    df = df.dropna(subset=drop_na_on)
    print(f"  After dropna: {len(df)} rows (dropped {n_before - len(df)})")
    return df


def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    available = set(df.columns)
    id_cols   = {"molecule", "environment", "method"}

    model_c = [c for c in df.select_dtypes(include=[np.number]).columns
               if c not in id_cols and c != TARGET_COL]
    model_d = [c for c in CT_FEATURES + OVERLAP_FEATURES if c in available]
    model_a = [c for c in ENERGY_FEATURES if c in available]

    return {
        "Model_A_energy_only": model_a,
        "Model_C_full":        model_c,
        "Model_D_CT_only":     model_d,
    }


def train_and_evaluate(df: pd.DataFrame, features: list[str],
                       name: str) -> dict:
    X = df[features].values
    y = df[TARGET_COL].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale")),
    ])
    kf     = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(pipe, X, y, cv=kf)

    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)

    print(f"\n{'─'*55}")
    print(f"  {name}  ({len(features)} features, {len(X)} samples)")
    print(f"  MAE  = {mae:.4f} eV")
    print(f"  RMSE = {rmse:.4f} eV")
    print(f"  R²   = {r2:.4f}")

    pipe.fit(X, y)  # refit on full data for SHAP
    return {"name": name, "n_features": len(features), "n_samples": len(X),
            "features": features, "mae": round(mae, 4),
            "rmse": round(rmse, 4), "r2": round(r2, 4),
            "pipeline": pipe, "X": X, "y": y}


def run_shap(result: dict) -> pd.DataFrame:
    print(f"\n  SHAP → {result['name']} …")
    pipe     = result["pipeline"]
    X        = result["X"]
    features = result["features"]
    scaler   = pipe.named_steps["scaler"]
    svr_pred = pipe.named_steps["svr"].predict

    rng    = np.random.RandomState(RANDOM_STATE)
    n_bg   = min(100, len(X))
    n_exp  = min(200, len(X))
    X_bg   = scaler.transform(X[rng.choice(len(X), n_bg,   replace=False)])
    X_exp  = scaler.transform(X[rng.choice(len(X), n_exp,  replace=False)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer   = shap.KernelExplainer(svr_pred, X_bg)
        shap_values = explainer.shap_values(X_exp, nsamples=100)

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df   = (pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True))

    csv_path = OUT_DIR / f"shap_{result['name']}.csv"
    imp_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.name}")

    total = mean_abs.sum()
    print(f"  Top-10 features (mean |SHAP|):")
    for _, row in imp_df.head(10).iterrows():
        pct = 100 * row["mean_abs_shap"] / total
        print(f"    {row['feature']:30s}  {row['mean_abs_shap']:.5f}  ({pct:.1f}%)")

    return imp_df


def build_summary_report(results: list[dict],
                          shap_results: dict[str, pd.DataFrame]) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append("SVR ABLATION STUDY — VERIFICATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 65)

    # Model performance table
    lines.append("\nMODEL PERFORMANCE (5-fold CV)")
    lines.append(f"  {'Model':<30s}  {'Feat':>4s}  {'MAE (eV)':>9s}  {'R²':>6s}")
    lines.append("  " + "─" * 55)
    for r in results:
        lines.append(f"  {r['name']:<30s}  {r['n_features']:>4d}  "
                     f"{r['mae']:>9.4f}  {r['r2']:>6.4f}")

    # SHAP breakdown for Model C
    if "Model_C_full" in shap_results:
        imp = shap_results["Model_C_full"]
        total = imp["mean_abs_shap"].sum()
        lines.append("\nSHAP FEATURE IMPORTANCE — Model C (full SVR)")
        lines.append(f"  {'Feature':<30s}  {'mean|SHAP|':>10s}  {'%':>6s}")
        lines.append("  " + "─" * 50)
        for _, row in imp.iterrows():
            pct = 100 * row["mean_abs_shap"] / total
            lines.append(f"  {row['feature']:<30s}  "
                         f"{row['mean_abs_shap']:>10.5f}  {pct:>6.1f}%")

        lines.append("\nSHAP CATEGORY TOTALS — Model C")
        lines.append(f"  {'Category':<20s}  {'total |SHAP|':>12s}  {'%':>6s}")
        lines.append("  " + "─" * 43)
        feat_imp = dict(zip(imp["feature"], imp["mean_abs_shap"]))
        for cat, feats in SHAP_GROUPS.items():
            cat_total = sum(feat_imp.get(f, 0.0) for f in feats)
            pct = 100 * cat_total / total
            lines.append(f"  {cat:<20s}  {cat_total:>12.5f}  {pct:>6.1f}%")

        # Individual top features for manuscript comparison
        lines.append("\nMANUSCRIPT COMPARISON (manuscript values → computed values)")
        manuscript = {
            "T1_energy_eV":    40.0,
            "S1_energy_eV":    26.0,
            "S1_osc_strength": 24.0,
            "HOMO_LUMO_gap_eV": 9.0,
            "NTO/CT (total)":   1.0,
        }
        nto_ct_total = sum(feat_imp.get(f, 0.0)
                          for f in CT_FEATURES + OVERLAP_FEATURES)
        computed_ind = {k: 100 * feat_imp.get(k, 0.0) / total
                        for k in list(manuscript.keys())[:-1]}
        computed_ind["NTO/CT (total)"] = 100 * nto_ct_total / total

        lines.append(f"  {'Feature':<22s}  {'Manuscript %':>12s}  {'Computed %':>10s}  {'Δ':>6s}")
        lines.append("  " + "─" * 55)
        for feat, ms_pct in manuscript.items():
            cp_pct = computed_ind[feat]
            delta  = cp_pct - ms_pct
            flag   = "  ← DISCREPANCY" if abs(delta) > 3.0 else ""
            lines.append(f"  {feat:<22s}  {ms_pct:>12.1f}  {cp_pct:>10.1f}  "
                         f"{delta:>+6.1f}{flag}")

    lines.append("\n" + "=" * 65)
    return "\n".join(lines)


def main():
    print("\n" + "=" * 55)
    print("  SVR SHAP VERIFICATION — CEJ manuscript")
    print("=" * 55)

    df = load_data()
    feature_sets = get_feature_sets(df)

    results = []
    for name, feats in feature_sets.items():
        results.append(train_and_evaluate(df, feats, name))

    shap_results = {}
    for r in results:
        shap_results[r["name"]] = run_shap(r)

    report = build_summary_report(results, shap_results)
    print("\n" + report)

    report_path = OUT_DIR / "shap_summary_report.txt"
    report_path.write_text(report)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_file": str(DATA_FILE),
        "models": [
            {"name": r["name"], "n_features": r["n_features"],
             "n_samples": r["n_samples"],
             "mae_eV": r["mae"], "rmse_eV": r["rmse"], "r2": r["r2"]}
            for r in results
        ],
    }
    json_path = OUT_DIR / "ablation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
