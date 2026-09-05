"""
ref1_tddft_gap.py
=================
Extends ref1_gap_baseline.py with the TD-DFT reference run (answers Referee 1 fully).

Referee 1 (DD-ART-07-2026-000486, rejected 27-Aug-2026) argued the computed
singlet-triplet gap should beat structure-only ML, and that A-001 only tested a
semi-empirical level (GFN2-xTB/sTDA). This script adds the higher level:

    wB97X-D4/def2-SVP, RIJCOSX, TDA, nroots 8, triplets true, gas-phase vertical
    231/231 molecules usable, 0 failures, 604 core-hours
    source: tddft_server_package/results/tddft_results.csv  (dEST_eV column)

Same 231-molecule set, same Bemis-Murcko scaffold GroupKFold(5) folds as A-001.

New rows on top of A-001's B0-M4:
  B3  raw TD-DFT computed gap              (zero-parameter physics predictor)
  B4  linear-calibrated TD-DFT gap        (a*gap + b, fitted in-fold -> no leakage)
  M5  RF on NTO + TD-DFT gap              (does the TD-DFT gap ADD to features?)
  M6  RF on Morgan + TD-DFT gap
  M7  RF on NTO + sTDA gap + TD-DFT gap   (both physics numbers together)

Plus the three-way agreement table (experimental / sTDA / TD-DFT): MAE, RMSE, R2,
Pearson r, Spearman rho, signed bias.

Output: data/ref1_tddft_gap.json  (+ console table)
"""
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
TDDFT_CSV = ROOT / "tddft_server_package" / "results" / "tddft_results.csv"
TDDFT_LEVEL = "wB97X-D4/def2-SVP RIJCOSX TDA nroots 8 triplets true, gas-phase vertical"
N_BOOT, SEED, N_SPLITS = 2000, 0, 5
rng = np.random.default_rng(SEED)


def boot_ci(fn, y, p, n=N_BOOT):
    idx = np.arange(len(y))
    vals = [fn(y[s], p[s]) for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def metrics(y, p, label):
    r = stats.pearsonr(y, p)
    rho = stats.spearmanr(y, p)
    return dict(
        model=label, n=int(len(y)),
        MAE=float(mean_absolute_error(y, p)), MAE_CI=boot_ci(mean_absolute_error, y, p),
        RMSE=rmse(y, p), RMSE_CI=boot_ci(rmse, y, p),
        R2=float(r2_score(y, p)), R2_CI=boot_ci(r2_score, y, p),
        pearson_r=float(r.statistic), pearson_p=float(r.pvalue),
        spearman_rho=float(rho.statistic), spearman_p=float(rho.pvalue),
    )


def oof_rf(X, y, groups, cv):
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        m = rf(random_state=SEED).fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return p


def oof_linear_gap(g, y, groups, cv):
    """In-fold linear calibration of a computed gap: no test-set leakage."""
    p = np.zeros(len(y))
    for tr, te in cv.split(g.reshape(-1, 1), y, groups):
        a, b = np.polyfit(g[tr], y[tr], 1)
        p[te] = a * g[te] + b
    return p


def oof_mean(y, groups, cv):
    p = np.zeros(len(y))
    for tr, te in cv.split(y.reshape(-1, 1), y, groups):
        p[te] = y[tr].mean()
    return p


ds = load_dataset()
y = ds.y
groups = ds.scaf
cv = GroupKFold(n_splits=N_SPLITS)

# --- merge the TD-DFT gap by molecule name, in dataset row order -------------
td = pd.read_csv(TDDFT_CSV)
merged = ds.df[["molecule"]].merge(
    td[["molecule", "dEST_eV", "S1_eV", "T1_eV", "states_parsed", "terminated_normally"]],
    on="molecule", how="left",
)
assert len(merged) == len(ds.df), "row count changed on merge"
assert merged["dEST_eV"].notna().all(), f"{merged['dEST_eV'].isna().sum()} molecules missing a TD-DFT gap"
assert bool(merged["states_parsed"].all()), "some TD-DFT states unparsed"
assert bool(merged["terminated_normally"].all()), "some ORCA jobs did not terminate normally"
tddft_gap = merged["dEST_eV"].values.astype(float)

stda_gap = ds.df["Delta_E_ST_eV"].values.astype(float)

Xm, _ = make_features(ds, "Morgan")
Xn, _ = make_features(ds, "NTO")
Xn_td = np.hstack([Xn, tddft_gap.reshape(-1, 1)])
Xm_td = np.hstack([Xm, tddft_gap.reshape(-1, 1)])
Xn_both = np.hstack([Xn, stda_gap.reshape(-1, 1), tddft_gap.reshape(-1, 1)])

res = [
    metrics(y, oof_mean(y, groups, cv), "B0 mean baseline"),
    metrics(y, stda_gap, "B1 raw sTDA gap (GFN2-xTB/sTDA)"),
    metrics(y, tddft_gap, "B3 raw TD-DFT gap (wB97X-D4/def2-SVP TDA)"),
    metrics(y, oof_linear_gap(stda_gap, y, groups, cv), "B2 calibrated sTDA gap (in-fold a*g+b)"),
    metrics(y, oof_linear_gap(tddft_gap, y, groups, cv), "B4 calibrated TD-DFT gap (in-fold a*g+b)"),
    metrics(y, oof_rf(Xm, y, groups, cv), "M1 RF Morgan (structure only)"),
    metrics(y, oof_rf(Xn, y, groups, cv), "M2 RF NTO (energy-free semi-empirical)"),
    metrics(y, oof_rf(Xn_td, y, groups, cv), "M5 RF NTO + TD-DFT gap"),
    metrics(y, oof_rf(Xm_td, y, groups, cv), "M6 RF Morgan + TD-DFT gap"),
    metrics(y, oof_rf(Xn_both, y, groups, cv), "M7 RF NTO + sTDA gap + TD-DFT gap"),
]

# --- three-way agreement with experiment (referee claims exp-vs-theory r>0.95) ---
def agree(pred, name):
    r = stats.pearsonr(y, pred)
    rho = stats.spearmanr(y, pred)
    return dict(
        level=name, n=int(len(y)),
        MAE=float(mean_absolute_error(y, pred)), RMSE=rmse(y, pred),
        R2=float(r2_score(y, pred)),
        pearson_r=float(r.statistic), pearson_p=float(r.pvalue),
        spearman_rho=float(rho.statistic),
        signed_bias_pred_minus_exp=float(np.mean(pred - y)),
        pred_mean=float(pred.mean()), pred_sd=float(pred.std(ddof=1)),
    )


three_way = [
    agree(stda_gap, "sTDA GFN2-xTB gas vertical"),
    agree(tddft_gap, f"TD-DFT {TDDFT_LEVEL}"),
]

# --- paired bootstrap of the MAE difference on identical folds ---------------
# positive delta_mae = second model better (lower MAE)
def paired_delta_mae(pred_a, pred_b, n=N_BOOT):
    """ΔMAE = MAE(a) - MAE(b), paired bootstrap CI + P(b better)."""
    ea, eb = np.abs(y - pred_a), np.abs(y - pred_b)
    idx = np.arange(len(y))
    d = float(ea.mean() - eb.mean())
    draws = [float(ea[s].mean() - eb[s].mean())
             for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n))]
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    w = stats.wilcoxon(ea, eb)
    return dict(delta_mae=d, ci=[lo, hi], p_b_better=float(np.mean(np.array(draws) > 0)),
               wilcoxon_p=float(w.pvalue))


oof = dict(
    B0=oof_mean(y, groups, cv), B1=stda_gap, B3=tddft_gap,
    B2=oof_linear_gap(stda_gap, y, groups, cv), B4=oof_linear_gap(tddft_gap, y, groups, cv),
    M1=oof_rf(Xm, y, groups, cv), M2=oof_rf(Xn, y, groups, cv),
    M5=oof_rf(Xn_td, y, groups, cv), M6=oof_rf(Xm_td, y, groups, cv),
)
paired = {
    "M6_vs_M1 (Morgan+TDDFT vs Morgan)": paired_delta_mae(oof["M1"], oof["M6"]),
    "M5_vs_M2 (NTO+TDDFT vs NTO)": paired_delta_mae(oof["M2"], oof["M5"]),
    "B4_vs_B0 (calib TD-DFT gap vs mean)": paired_delta_mae(oof["B0"], oof["B4"]),
    "B4_vs_M1 (calib TD-DFT gap vs Morgan RF)": paired_delta_mae(oof["M1"], oof["B4"]),
    "B3_vs_B1 (raw TD-DFT vs raw sTDA)": paired_delta_mae(oof["B1"], oof["B3"]),
}

# sTDA vs TD-DFT directly (do the two theory levels at least agree with each other?)
r_ss = stats.pearsonr(stda_gap, tddft_gap)
rho_ss = stats.spearmanr(stda_gap, tddft_gap)
theory_theory = dict(
    pearson_r=float(r_ss.statistic), pearson_p=float(r_ss.pvalue),
    spearman_rho=float(rho_ss.statistic),
    mae_tddft_vs_stda=float(mean_absolute_error(stda_gap, tddft_gap)),
    mean_tddft_minus_stda=float(np.mean(tddft_gap - stda_gap)),
)

extra = dict(
    n_molecules=int(len(y)),
    n_scaffolds=int(len(set(groups))),
    exp_gap_mean=float(y.mean()), exp_gap_sd=float(y.std(ddof=1)),
    tddft_level=TDDFT_LEVEL,
    tddft_n_usable=int(len(td)), tddft_n_failed=int(231 - len(td)),
    referee1_claim="higher level of theory (TD-DFT / range-separated) should correlate better and beat structure-only ML",
)

out = dict(
    meta=dict(script="code/ref1_tddft_gap.py", seed=SEED, n_splits=N_SPLITS,
              n_bootstrap=N_BOOT, cv="GroupKFold on Bemis-Murcko scaffolds (identical to A-001)",
              tddft_source=str(TDDFT_CSV.relative_to(ROOT))),
    dataset=extra, three_way_vs_experiment=three_way, theory_vs_theory=theory_theory,
    paired_delta_mae=paired, results=res,
)
(ROOT / "data" / "ref1_tddft_gap.json").write_text(json.dumps(out, indent=2))

hdr = f"{'model':44s} {'MAE':>6s} {'RMSE':>6s} {'R2':>7s} {'r':>6s} {'rho':>6s}"
print(f"n={len(y)} molecules, {len(set(groups))} scaffolds")
print(f"TD-DFT: {TDDFT_LEVEL}\n        {len(td)}/231 usable, 0 failed\n")
print(hdr)
print("-" * len(hdr))
for r in res:
    print(f"{r['model']:44s} {r['MAE']:6.3f} {r['RMSE']:6.3f} {r['R2']:7.3f} "
          f"{r['pearson_r']:6.3f} {r['spearman_rho']:6.3f}")

print("\nthree-way agreement with experiment:")
for a in three_way:
    print(f"  {a['level']:42s} MAE {a['MAE']:.3f}  R2 {a['R2']:+.3f}  r {a['pearson_r']:.3f}  "
          f"rho {a['spearman_rho']:.3f}  bias {a['signed_bias_pred_minus_exp']:+.3f} eV")
print(f"\nsTDA vs TD-DFT: r {theory_theory['pearson_r']:.3f}  rho {theory_theory['spearman_rho']:.3f}  "
      f"mean(TD-DFT - sTDA) {theory_theory['mean_tddft_minus_stda']:+.3f} eV")
print("\npaired ΔMAE (positive = second model better, lower MAE):")
for k, v in paired.items():
    print(f"  {k:42s} ΔMAE {v['delta_mae']:+.4f}  CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]  "
          f"P {v['p_b_better']:.3f}  Wilcoxon p {v['wilcoxon_p']:.3g}")
print("\nwrote data/ref1_tddft_gap.json")
