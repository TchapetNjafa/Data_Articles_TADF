"""
compute_baselines.py
====================
Purpose
-------
Compute baseline predictors for the NTO-RF model benchmark (Task A1-code).

We evaluate three naive baselines on the out-of-fold (OOF) predictions produced
by scaffold-GroupKFold cross-validation, and compare them against the headline
NTO Random-Forest model.

Baselines
---------
1. **Mean predictor**   – always predicts the OOF training-set mean.
2. **Median predictor** – always predicts the OOF training-set median.
3. **Random-permutation predictor** – shuffles the predicted values 1 000 times
   to build a null distribution of MAE; the 2.5th–97.5th percentile of that
   distribution forms the null CI.

Metrics reported
----------------
* MAE with 95 % bootstrap CI (n_bootstrap = 2 000, scipy.stats.bootstrap).
* For the real model, absolute and relative improvement over the mean baseline.

Outputs
-------
* Console: formatted summary table (LaTeX-ready).
* File:    data/baseline_comparison.json
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = pathlib.Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OOF_CSV = DATA / "final_model_scaffold_predictions.csv"
METRICS_JSON = DATA / "final_model_metrics.json"
OUT_JSON = DATA / "baseline_comparison.json"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 1000
CI_LEVEL = 0.95
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def bootstrap_mae_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """Return (point_mae, ci_low, ci_high) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_maes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_maes[i] = mae(y_true[idx], y_pred[idx])
    alpha = 1.0 - ci
    lo = float(np.percentile(boot_maes, 100 * alpha / 2))
    hi = float(np.percentile(boot_maes, 100 * (1 - alpha / 2)))
    pt = mae(y_true, y_pred)
    return pt, lo, hi


def permutation_null_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = RNG_SEED,
) -> np.ndarray:
    """Shuffle y_pred `n_permutations` times; return array of MAEs."""
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(y_pred)
        null[i] = mae(y_true, shuffled)
    return null


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(OOF_CSV)
    # Columns: molecule, exp_est, pred_NTO
    y_true = df["exp_est"].values.astype(float)
    y_pred = df["pred_NTO"].values.astype(float)

    with open(METRICS_JSON) as fh:
        metrics = json.load(fh)

    model_mae = metrics["headline"]["MAE"]
    model_ci = metrics["headline"]["MAE_CI"]

    print(f"\nLoaded {len(df)} molecules from {OOF_CSV.name}")
    print(f"Headline model MAE = {model_mae:.4f}  CI=[{model_ci[0]:.3f}, {model_ci[1]:.3f}]")

    # ------------------------------------------------------------------
    # 2. Compute baselines
    # ------------------------------------------------------------------
    mean_val = float(np.mean(y_true))
    median_val = float(np.median(y_true))

    # -- Mean baseline
    pred_mean = np.full_like(y_true, mean_val)
    mae_mean, ci_mean_lo, ci_mean_hi = bootstrap_mae_ci(y_true, pred_mean)

    # -- Median baseline
    pred_median = np.full_like(y_true, median_val)
    mae_median, ci_median_lo, ci_median_hi = bootstrap_mae_ci(y_true, pred_median)

    # -- Random-permutation null distribution
    null_maes = permutation_null_mae(y_true, y_pred)
    mae_perm_mean = float(np.mean(null_maes))
    mae_perm_lo = float(np.percentile(null_maes, 2.5))
    mae_perm_hi = float(np.percentile(null_maes, 97.5))

    # ------------------------------------------------------------------
    # 3. Model improvement over mean baseline
    # ------------------------------------------------------------------
    # Absolute improvement and bootstrap CI on the *difference*
    rng = np.random.default_rng(RNG_SEED)
    n = len(y_true)
    diff_boot = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        diff_boot[i] = (
            mae(y_true[idx], pred_mean[idx]) - mae(y_true[idx], y_pred[idx])
        )

    improvement_abs = mae_mean - model_mae
    improvement_lo = float(np.percentile(diff_boot, 2.5))
    improvement_hi = float(np.percentile(diff_boot, 97.5))

    improvement_rel = improvement_abs / mae_mean * 100  # percent

    # ------------------------------------------------------------------
    # 4. Print summary table
    # ------------------------------------------------------------------
    sep = "-" * 78
    print(f"\n{sep}")
    print("BASELINE COMPARISON  (MAE, eV; 95 % CI via percentile bootstrap, n=2000)")
    print(sep)
    hdr = f"{'Predictor':<35} {'MAE':>8}  {'95% CI':^19}  {'vs. mean (abs)':>14}"
    print(hdr)
    print(sep)

    rows = [
        (
            "Mean constant predictor",
            mae_mean,
            ci_mean_lo,
            ci_mean_hi,
            "-",
        ),
        (
            "Median constant predictor",
            mae_median,
            ci_median_lo,
            ci_median_hi,
            f"{mae_median - mae_mean:+.4f}",
        ),
        (
            "Random-permutation null (mean)",
            mae_perm_mean,
            mae_perm_lo,
            mae_perm_hi,
            f"{mae_perm_mean - mae_mean:+.4f}",
        ),
        (
            "NTO-RF model (OOF headline)",
            model_mae,
            model_ci[0],
            model_ci[1],
            f"{model_mae - mae_mean:+.4f}",
        ),
    ]

    for name, point, lo, hi, delta in rows:
        print(
            f"  {name:<33} {point:>8.4f}  [{lo:.4f}, {hi:.4f}]  {delta:>14}"
        )

    print(sep)
    print(
        f"\n  NTO-RF improvement over mean baseline: "
        f"{improvement_abs:.4f} eV  ({improvement_rel:.1f} %)"
    )
    print(
        f"  95 % CI on improvement: [{improvement_lo:.4f}, {improvement_hi:.4f}] eV"
    )

    # ------------------------------------------------------------------
    # 5. LaTeX-ready table rows
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("LaTeX table rows (\\midrule separated):")
    print(sep)
    latex_rows = [
        (
            "Mean predictor",
            mae_mean,
            ci_mean_lo,
            ci_mean_hi,
        ),
        (
            "Median predictor",
            mae_median,
            ci_median_lo,
            ci_median_hi,
        ),
        (
            "Random-permutation null",
            mae_perm_mean,
            mae_perm_lo,
            mae_perm_hi,
        ),
        (
            "NTO-RF (this work)",
            model_mae,
            model_ci[0],
            model_ci[1],
        ),
    ]
    for name, point, lo, hi in latex_rows:
        ci_str = f"[{lo:.3f}, {hi:.3f}]"
        print(f"  {name} & {point:.3f} & {ci_str} \\\\")
    print(sep)

    # ------------------------------------------------------------------
    # 6. Save to JSON
    # ------------------------------------------------------------------
    result = {
        "description": (
            "Baseline comparison for NTO-RF model. "
            "MAE in eV; 95 % CI via percentile bootstrap (n=2000) "
            "on OOF scaffold-GroupKFold predictions."
        ),
        "n_molecules": int(n),
        "baselines": {
            "mean_predictor": {
                "constant_value": round(mean_val, 6),
                "MAE": round(mae_mean, 6),
                "CI_95": [round(ci_mean_lo, 6), round(ci_mean_hi, 6)],
            },
            "median_predictor": {
                "constant_value": round(median_val, 6),
                "MAE": round(mae_median, 6),
                "CI_95": [round(ci_median_lo, 6), round(ci_median_hi, 6)],
            },
            "random_permutation_null": {
                "n_permutations": N_PERMUTATIONS,
                "MAE_mean": round(mae_perm_mean, 6),
                "CI_95_of_null_distribution": [
                    round(mae_perm_lo, 6),
                    round(mae_perm_hi, 6),
                ],
            },
        },
        "model": {
            "name": "NTO-RF (headline, scaffold GroupKFold)",
            "MAE": model_mae,
            "CI_95": model_ci,
        },
        "improvement_over_mean_baseline": {
            "absolute_eV": round(improvement_abs, 6),
            "relative_pct": round(improvement_rel, 2),
            "CI_95_of_difference": [
                round(improvement_lo, 6),
                round(improvement_hi, 6),
            ],
        },
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\nResults saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
