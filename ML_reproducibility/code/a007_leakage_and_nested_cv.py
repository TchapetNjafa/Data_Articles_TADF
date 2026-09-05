"""
a007_leakage_and_nested_cv.py
=============================
Two L3 findings that no ledger entry could settle.

H-12  The feature-target leakage diagnosis -- contribution (i) of the paper -- cites
      "correlation 1.000, zero MAE" and "in-sample R^2 = 0.87", and elsewhere
      "sub-0.05 eV accuracies". No ledger entry and no file in data/ produces any of
      these. Worse, an IN-SAMPLE R^2 demonstrates overfitting capacity, not leakage, and
      0.87 undercuts the argument it is offered to prove: if the target is exactly a
      difference of two inputs, a leaked model should reach R^2 ~ 1 under CROSS-validation.
      -> Compute the leakage demonstration properly, cross-validated, on the same folds.

H-3   The random forest was chosen over SVR/ridge/GBM/MLP using the same out-of-fold
      predictions used to report performance, and the SI asserts this "carries no
      test-set selection bias". Not tuning hyperparameters does not remove selection
      bias when the ESTIMATOR is picked on the reported folds.
      -> Nested cross-validation: select the learner in an inner loop, score it on a
         held-out outer fold, and compare with the optimistically-selected number.

Output: data/a007_leakage_nested_cv.json
"""
import json, pathlib, sys, warnings
import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
SEED, K = 0, 5
ds = load_dataset()
y_exp, groups = ds.y, ds.scaf
Xn, nto_names = make_features(ds, 'NTO')

# ---------------------------------------------------------------- H-12: leakage
E_S1 = ds.df['S1_energy_eV'].values.astype(float)
E_T1 = ds.df['T1_energy_eV'].values.astype(float)
gap_computed = E_S1 - E_T1                     # the target a leaked study would regress on

ok = np.isfinite(gap_computed) & np.isfinite(Xn).all(1)
Xn_ok, gap_ok = Xn[ok], gap_computed[ok]
g_ok = groups[ok]
X_leak = np.column_stack([Xn_ok, E_S1[ok], E_T1[ok]])   # features CONTAIN the constituents


def oof(X, y, g, model_fn):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=K).split(X, y, g):
        p[te] = model_fn().fit(X[tr], y[tr]).predict(X[te])
    return p


# A random forest is axis-aligned and cannot represent a DIFFERENCE of two continuous
# inputs, so it understates leakage. Report several learners: the leakage argument is
# that the arithmetic is RECOVERABLE, and a linear model recovers it exactly.
from sklearn.linear_model import LinearRegression
LEAK_LEARNERS = {
    'RandomForest': rf,
    'Ridge':        lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    'SVR':          lambda: make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.05)),
    'LinearRegression': LinearRegression,
}
leak_rows, clean_rows = {}, {}
for nm, fn in LEAK_LEARNERS.items():
    pl = oof(X_leak, gap_ok, g_ok, fn)
    pc = oof(Xn_ok,  gap_ok, g_ok, fn)
    leak_rows[nm]  = dict(cv_MAE_eV=round(float(mean_absolute_error(gap_ok, pl)), 6),
                          cv_R2=round(float(r2_score(gap_ok, pl)), 6))
    clean_rows[nm] = dict(cv_MAE_eV=round(float(mean_absolute_error(gap_ok, pc)), 6),
                          cv_R2=round(float(r2_score(gap_ok, pc)), 6))
p_leak = oof(X_leak, gap_ok, g_ok, rf)
p_clean = oof(Xn_ok, gap_ok, g_ok, rf)

H12 = dict(
    target='sTDA computed gap  E_S1 - E_T1  (the quantity a leaked study regresses on)',
    n=int(len(gap_ok)),
    leaked_features=dict(
        description='35 NTO spatial descriptors PLUS the S1 and T1 energy scalars',
        cv_MAE_eV=round(float(mean_absolute_error(gap_ok, p_leak)), 5),
        cv_R2=round(float(r2_score(gap_ok, p_leak)), 5),
        pearson_r=round(float(stats.pearsonr(gap_ok, p_leak).statistic), 5)),
    energy_free_features=dict(
        description='the same 35 NTO spatial descriptors, energies excluded',
        cv_MAE_eV=round(float(mean_absolute_error(gap_ok, p_clean)), 5),
        cv_R2=round(float(r2_score(gap_ok, p_clean)), 5),
        pearson_r=round(float(stats.pearsonr(gap_ok, p_clean).statistic), 5)),
    by_learner_leaked=leak_rows,
    by_learner_energy_free=clean_rows,
    note=('Both cross-validated on the same Bemis-Murcko scaffold GroupKFold(5) as the rest '
          'of the study. The contrast between the two rows IS the leakage demonstration. '
          'Learner choice matters: a random forest is axis-aligned and cannot represent a '
          'difference of two continuous inputs, so it UNDERSTATES the leakage; a linear model '
          'recovers the identity exactly. The leakage claim is about recoverability, so the '
          'linear row is the honest demonstration.'),
)

# ------------------------------------------------------- H-3: nested CV over learners
def m_rf():    return rf()
def m_svr():   return make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.05))
def m_ridge(): return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
def m_gbm():   return GradientBoostingRegressor(random_state=SEED)
def m_mlp():   return make_pipeline(StandardScaler(),
                                    MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000,
                                                 random_state=SEED))
LEARNERS = {'RandomForest': m_rf, 'SVR': m_svr, 'Ridge': m_ridge,
            'GradientBoosting': m_gbm, 'MLP': m_mlp}

# (a) the optimistic protocol actually used: score every learner on the SAME outer folds,
#     then report the best.
flat = {}
for name, fn in LEARNERS.items():
    p = oof(Xn, y_exp, groups, fn)
    flat[name] = round(float(mean_absolute_error(y_exp, p)), 5)
best_flat = min(flat, key=flat.get)

# (b) nested: choose the learner inside each training fold, score on the untouched outer fold
outer = GroupKFold(n_splits=K)
pred_nested = np.zeros(len(y_exp))
chosen = []
for tr, te in outer.split(Xn, y_exp, groups):
    Xtr, ytr, gtr = Xn[tr], y_exp[tr], groups[tr]
    n_inner = min(K, len(set(gtr)))
    scores = {}
    for name, fn in LEARNERS.items():
        pin = np.zeros(len(ytr))
        for itr, ite in GroupKFold(n_splits=n_inner).split(Xtr, ytr, gtr):
            pin[ite] = fn().fit(Xtr[itr], ytr[itr]).predict(Xtr[ite])
        scores[name] = mean_absolute_error(ytr, pin)
    pick = min(scores, key=scores.get)
    chosen.append(pick)
    pred_nested[te] = LEARNERS[pick]().fit(Xtr, ytr).predict(Xn[te])

H3 = dict(
    flat_protocol=dict(
        description='every learner scored on the identical outer folds; the best is reported',
        MAE_by_learner=flat, best_learner=best_flat, reported_MAE_eV=flat[best_flat]),
    nested_protocol=dict(
        description='learner selected inside each training fold, scored on the untouched outer fold',
        MAE_eV=round(float(mean_absolute_error(y_exp, pred_nested)), 5),
        R2=round(float(r2_score(y_exp, pred_nested)), 5),
        learner_chosen_per_fold=chosen),
    selection_optimism_eV=round(float(mean_absolute_error(y_exp, pred_nested) - flat[best_flat]), 5),
    mean_baseline_MAE_eV=round(float(mean_absolute_error(
        y_exp, oof(np.zeros((len(y_exp), 1)), y_exp, groups,
                   lambda: __import__('sklearn.dummy', fromlist=['DummyRegressor']).DummyRegressor()))), 5),
    note='positive selection_optimism = the flat protocol was optimistic by that many eV',
)

out = dict(meta=dict(script='code/a007_leakage_and_nested_cv.py', seed=SEED, n_splits=K,
                     cv='GroupKFold on Bemis-Murcko scaffolds (identical to A-001/A-005/A-006)'),
           H12_leakage_demonstration=H12, H3_nested_cv=H3)
(ROOT/'data'/'a007_leakage_nested_cv.json').write_text(json.dumps(out, indent=1))

print('=== H-12  leakage, CROSS-VALIDATED (target = computed gap E_S1 - E_T1) ===')
L, C = H12['leaked_features'], H12['energy_free_features']
print(f"  {'learner':18s} {'WITH energies':>26s}   {'energies EXCLUDED':>26s}")
for nm in LEAK_LEARNERS:
    a, b = leak_rows[nm], clean_rows[nm]
    print(f"  {nm:18s} MAE {a['cv_MAE_eV']:.6f}  R2 {a['cv_R2']:>9.5f}   "
          f"MAE {b['cv_MAE_eV']:.6f}  R2 {b['cv_R2']:>9.5f}")
print()
print('=== H-3  estimator selection ===')
for k, v in sorted(flat.items(), key=lambda kv: kv[1]):
    print(f"  flat  {k:17s} MAE {v:.5f}")
print(f"  -> flat protocol reports {best_flat} at {flat[best_flat]:.5f} eV")
print(f"  -> nested CV            {H3['nested_protocol']['MAE_eV']:.5f} eV  (picks per fold: {chosen})")
print(f"  -> selection optimism    {H3['selection_optimism_eV']:+.5f} eV")
print()
print('wrote data/a007_leakage_nested_cv.json')
