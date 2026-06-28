"""
r2_ci_power_analysis.py
=======================
Purpose
-------
Task A2 — contextualise the (already honestly reported) R^2 confidence intervals
that span zero, by (a) supplying a bootstrap CI for the Spearman rank correlation
rho on the same out-of-fold scaffold-CV predictions, and (b) running a power
analysis for the correlation test so the wide CI is framed as a dataset-size
limitation rather than a model failure.

Methods
-------
* Spearman rho with 95% bootstrap CI (paired resampling of (y_true, y_pred),
  n_bootstrap = 5000, percentile method).
* Pearson r with 95% bootstrap CI (for completeness, matches R^2 sign).
* Power analysis for a correlation test using the Fisher z transform
  (two-sided, alpha = 0.05):
    - achieved power at the observed n to detect the observed rho;
    - required n to reach 80% and 90% power for the observed rho.

Inputs
------
* data/final_model_scaffold_predictions.csv  (molecule, exp_est, pred_NTO)
* data/final_model_metrics.json              (headline R^2, R^2 CI, Spearman)

Outputs
-------
* console summary (LaTeX-ready numbers)
* data/r2_ci_power_analysis.json
"""

import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr

BASE = pathlib.Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OOF_CSV = DATA / "final_model_scaffold_predictions.csv"
METRICS_JSON = DATA / "final_model_metrics.json"
OUT_JSON = DATA / "r2_ci_power_analysis.json"

N_BOOTSTRAP = 5000
ALPHA = 0.05
RNG_SEED = 42


def bootstrap_corr_ci(x, y, method, n_bootstrap=N_BOOTSTRAP, seed=RNG_SEED):
    """Percentile bootstrap CI for a correlation coefficient (paired resample)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if method == "spearman":
            boot[i] = spearmanr(x[idx], y[idx]).statistic
        else:
            boot[i] = pearsonr(x[idx], y[idx]).statistic
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return lo, hi


def corr_power(rho, n, alpha=ALPHA):
    """Achieved power of a two-sided correlation test via Fisher z transform."""
    if n <= 3:
        return float("nan")
    z = np.arctanh(abs(rho))
    se = 1.0 / np.sqrt(n - 3)
    z_alpha = norm.ppf(1 - alpha / 2)
    lam = z / se
    power = (
        1.0
        - norm.cdf(z_alpha - lam)
        + norm.cdf(-z_alpha - lam)
    )
    return float(power)


def required_n(rho, target_power, alpha=ALPHA):
    """Sample size to detect `rho` at `target_power` (two-sided correlation test)."""
    z = np.arctanh(abs(rho))
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(target_power)
    n = ((z_alpha + z_power) / z) ** 2 + 3
    return int(np.ceil(n))


def main():
    df = pd.read_csv(OOF_CSV)
    y_true = df["exp_est"].values.astype(float)
    y_pred = df["pred_NTO"].values.astype(float)
    n = len(df)

    with open(METRICS_JSON) as fh:
        metrics = json.load(fh)
    r2 = metrics["headline"]["R2"]
    r2_ci = metrics["headline"]["R2_CI"]

    # --- correlations + bootstrap CIs ---
    rho = float(spearmanr(y_true, y_pred).statistic)
    rho_p = float(spearmanr(y_true, y_pred).pvalue)
    rho_lo, rho_hi = bootstrap_corr_ci(y_true, y_pred, "spearman")

    r = float(pearsonr(y_true, y_pred).statistic)
    r_lo, r_hi = bootstrap_corr_ci(y_true, y_pred, "pearson")

    # --- power analysis on the Spearman rho ---
    power_at_n = corr_power(rho, n)
    n_80 = required_n(rho, 0.80)
    n_90 = required_n(rho, 0.90)

    sep = "-" * 72
    print(sep)
    print(f"n molecules                : {n}")
    print(f"R^2 (headline)             : {r2:.3f}  CI {r2_ci}")
    print(sep)
    print(f"Spearman rho               : {rho:.3f}  95% CI [{rho_lo:.3f}, {rho_hi:.3f}]")
    print(f"Spearman p-value           : {rho_p:.2e}")
    print(f"Pearson  r                 : {r:.3f}  95% CI [{r_lo:.3f}, {r_hi:.3f}]")
    print(sep)
    print(f"Achieved power at n={n}    : {power_at_n:.3f}  (two-sided, alpha=0.05)")
    print(f"n for 80% power (rho={rho:.2f}): {n_80}")
    print(f"n for 90% power (rho={rho:.2f}): {n_90}")
    print(sep)

    result = {
        "n_molecules": int(n),
        "R2": r2,
        "R2_CI": r2_ci,
        "spearman": {
            "rho": round(rho, 4),
            "CI_95": [round(rho_lo, 4), round(rho_hi, 4)],
            "p_value": rho_p,
        },
        "pearson": {
            "r": round(r, 4),
            "CI_95": [round(r_lo, 4), round(r_hi, 4)],
        },
        "power_analysis": {
            "method": "Fisher z transform, two-sided, alpha=0.05",
            "achieved_power_at_n": round(power_at_n, 4),
            "n_for_80pct_power": n_80,
            "n_for_90pct_power": n_90,
        },
        "n_bootstrap": N_BOOTSTRAP,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
