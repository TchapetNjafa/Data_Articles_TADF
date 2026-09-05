# Analysis Ledger — Structure-Based ML for Experimental ΔE_ST

Append-only. No number reaches the manuscript except through an entry here.

---

### A-001 .. A-003 — ARCHIVED 2026-09-03
Full entries: `outputs/analysis/archive/ledger-A001-A003-2026-09.md` (grep target, do not read to restore context).
- **A-001** sTDA gap head-to-head, 7 predictors, scaffold GroupKFold → `data/ref1_gap_baseline.json`.
  Raw sTDA gap MAE 0.251 / R² −2.22 / bias +0.221; calibrated 0.104; as a feature, no change.
- **A-002** P1 paired baseline test, P2 DOI-grouped CV (173 papers), P3 split geometry → `data/ref3_p1p2p3.json`.
  P2b: Morgan MAE 0.102 / R² 0.184 / ρ 0.155 · NTO 0.099 / 0.142 / **ρ 0.346**; Morgan-vs-mean CI crosses zero.
  P3: 92.9%% singleton scaffolds, scaffold CV ≈ random CV, DOI CV harder.
- **A-003** label-noise reconciliation → `data/label_noise_reconciled.json`.
  Inter-report RMS SD 0.072 eV; **RMS standard error of the median label 0.049 eV** — the number to
  quote for how precisely the target is known. States it does NOT support 'label noise alone explains the error'.

### A-004 — ARCHIVED 2026-09-04
Full entry: `outputs/analysis/archive/ledger-A004-2026-09.md` (grep target).
Referee-3 points 4-7 verdicts. Key numbers still cited: **P4** expansion to 728 raises MAE 0.098->0.124,
R2 0.18->-0.01, rho 0.34->0.16 (caveat i: different structure pipeline, confounded). **P5** every one of the
top-100 conformal intervals spans a negative gap; **mean ratio of interval WIDTH to predicted value = 7.6**
(half-width ratio is 3.8 — do not conflate). **P6** vertical vs adiabatic n=27/14 mol: MAE 0.195, RMSE 0.245,
bias +0.071, r2 0.704, rho 0.711, slope 1.002, intercept 0.070, max dev 0.548. **P7** 66 IST records/61 names,
n_with_smiles=0; 5 IST molecules in corpus.

### A-005 — ARCHIVED 2026-09-04
Full entry: `outputs/analysis/archive/ledger-A005-2026-09.md` (grep target).
TD-DFT reference, wB97X-D4/def2-SVP RIJCOSX TDA, 231/231 converged, 604.5 core-h. Key numbers:
raw TD-DFT gap MAE **0.654** / R² **-16.3** / signed bias **+0.654** (worst predictor in the study);
calibrated in-fold **0.099**; Morgan+gap **0.087** vs Morgan 0.091 (ΔMAE +0.0039, CI [+0.0004,+0.0071],
Wilcoxon p 0.016); NTO+gap 0.092 vs 0.096. Pearson r vs experiment 0.43 (not >0.95). Claim C9.

### A-006 — ARCHIVED 2026-09-05
Full entry: `outputs/analysis/archive/ledger-A006-2026-09.md` (grep target).
T1 Morgan-vs-NTO paired equivalence: dMAE -0.0043, CI [-0.0171,+0.0092], Wilcoxon p 0.657,
**margin 0.017 eV**. T2 DOI split: NTO rho 0.346 vs Morgan 0.155, **drho +0.191 CI [+0.037,+0.344]**
(caveat iv: resamples molecules not DOI groups). T3 enrichment -> superseded by A-006b.
T4 ceiling audit: stored rss_floor 0.1282 > rf_mae 0.0956 (incoherent); defensible label
precision **0.049 eV**. T5 clean-label -> WITHDRAWN by A-008 (constant predictor reproduces it).

### A-006b — T3 RESOLVED: the enrichment files are consistent; the manuscript's quotation is not
- **Date:** 2026-09-03, same beat. **Supersedes A-006 T3** (which called the four values "mutually
  inconsistent" and the base rate irreproducible). That reading was wrong: it used a `< 0.1` threshold
  where every deposited file uses `<= 0.1`. Eleven molecules sit exactly at 0.10 eV.
- **Source:** recomputation on the same scaffold folds, seed 0 (see checkpoint L1 entry).
- **Result — every deposited value reproduces exactly under its own stated convention:**

| file | ranking model | cut convention | status |
|---|---|---|---|
| `data/enrichment_curve.json` | Morgan | top-**fraction** | reproduces row for row |
| `data/multi_criteria_triage.json` (`single_criterion_curve`) | **NTO** | top-fraction | reproduces (0.652 / 1.32 at top-10%) |
| `data/triage_metrics.json` | Morgan | top-**N molecules** (10/20/30/50) | all four points reproduce; ρ 0.306 matches |

  Base rate `y <= 0.1` = **0.4935** = the deposited 0.494.

  Morgan by fraction: top-1% 1.01 · top-2% 1.22 · top-5% 1.35 · top-10% 1.15 · top-20% 1.37 · top-30% 1.32 · top-50% 1.24.
  Morgan by molecule count: top-10 **0.600 / 1.22** · top-20 0.600 / 1.22 · top-30 0.633 / 1.28 · top-50 **0.680 / 1.38**.
  NTO by fraction: top-2% 1.62 · top-10% 1.32 · **top-20% 1.59** · top-30% 1.53.
- **Interpretation.** There is no data defect. The defect is in the manuscript and cover letter, which
  state "precision 0.60 at the top \qty{10}{\percent}, 0.68 at the top \qty{20}{\percent}": 0.600 is the
  top-**10 molecules** figure and 0.680 is the top-**50 molecules** figure, from `triage_metrics.json`,
  and neither is a percentage. Two conventions were mixed and both were relabelled as percentiles.
  Quote one convention and name it. The curve is genuinely non-monotone by fraction (1.35 at top-5%
  dipping to 1.15 at top-10%), so any single "1.2–1.4×" summary must say at which cut it holds.
  Separately, **NTO enriches better than Morgan** (1.59 vs 1.37 at top-20%), which is the triage-side
  counterpart of A-006 T2 and points the same way: the excited-state features help ranking, not MAE.
- **Caveats.** (i) Enrichment is computed on held-out *labelled* molecules only; it says nothing about
  the enumerated library. (ii) The top-1% and top-2% cuts are 2 and 5 molecules — not quotable.
- **Claims:** C13 changes from BLOCKER to VERIFIED, with the operating point named.

### A-006c — Provenance of the 22,194-structure library; review finding A2 partly REFUTED
- **Date:** 2026-09-04. **Trigger:** L3 review pass 3 finding A2, "the enumerated donor--acceptor
  library is not a donor--acceptor library", which recommended deleting the whole section.
- **Source:** `REQ1-implementation/data/expanded_library_final.csv` (`source` column),
  `REQ1-implementation/scripts/02_filter_pubchem_tadf.py` (SMARTS lists),
  `REQ1-implementation/scripts/03_create_expanded_library.py`; RDKit substructure scan.
- **Result.**
  All \num{22194} rows carry `source = PubChem_filtered`; the file is produced by a PubChem
  CID--SMILES substructure screen, **not** by combinatorial enumeration of fragment pairs.
  A donor/acceptor script (`02_enumerate_da_fragments.py`) exists but its output is not in this file.
  Against the script's ACTUAL SMARTS lists (7 donor, 8 acceptor patterns):
  **22194/22194 (100%) of the library, and 100/100 of the top-ranked subset, contain both a donor and
  an acceptor motif.** The filter does exactly what its own code claims.
  Top-100 motif counts — donors: dimethylaniline 85, acridine 13, carbazole 2, diphenylamine 2;
  acceptors: pyridine 59, cyano 25, pyrimidine 12, triazine 3, quinoxaline 2, naphthoquinone 2.
  Composition: 97/100 carry a halogen, 53/100 a stereocentre; enantiomer pairs appear as separate
  entries with identical predicted gaps.
- **Interpretation.**
  Review finding A2 is **refuted in its headline** and **upheld in its particulars**. It tested the
  narrow fragment list printed in Methods (carbazole/phenoxazine/acridine; triazine/benzonitrile/
  sulfone) rather than the broader list in the code, and so measured 4--5/100 where the correct
  figure is 100/100. The library IS a donor--acceptor set by its own definition. Deleting the
  section, as A2 recommended, would have removed sound material on a measurement error.
  What is genuinely wrong is the Methods description, on three counts: (i) "enumerated ...
  combinations" describes construction that did not occur — it is a substructure filter over an
  existing database; (ii) the stated fragment list is wrong and incomplete, and names **sulfone**,
  which is not among the patterns used at all; (iii) the composition was undisclosed, and it matters,
  because the matches lean on weak motifs (dimethylaniline, pyridine) and the set inherits PubChem's
  halogen/stereocentre profile rather than that of a designed emitter series.
- **Caveats.** (i) The upstream `pubchem_tadf_filtered.csv` and the raw PubChem dump are not in the
  deposit; provenance is reconstructed from the scripts and the `source` column, not re-run.
  (ii) 100% donor+acceptor coverage is a statement about SMARTS matching, not about whether these are
  plausible TADF emitters — a bare cyano group counts as an acceptor.
- **Claims:** new **C14** (library is a PubChem substructure filter, 100% donor+acceptor by the code's
  own patterns, weak-motif dominated). Methods/Results/SI corrected 2026-09-04 to match.

### A-007 — Leakage demonstrated cross-validated (H-12); estimator selection nested (H-3)
- **Date:** 2026-09-04. **Trigger:** L3 review pass 1 findings H-12 and H-3, neither of which any
  existing ledger entry could settle. H-12 flagged that the paper's leakage diagnosis — contribution
  (i) — cited "correlation 1.000, zero MAE", "in-sample $R^2 = 0.87$" and "sub-0.05 eV" with **no
  source in `data/` and no ledger entry**, and that an *in-sample* score cannot demonstrate leakage.
- **Source:** `code/a007_leakage_and_nested_cv.py` → `data/a007_leakage_nested_cv.json`.
  Same `_dataset` loader and the same scaffold GroupKFold(5) as A-001/A-005/A-006. seed 0.

**H-12 — leakage, cross-validated.** Target = the sTDA computed gap $E_{S_1}-E_{T_1}$, i.e. the
quantity a leaked study regresses on. Features = the 35 NTO spatial descriptors, with and without
the $S_1$/$T_1$ energy scalars. $n = 231$.

| learner | with energies: MAE / R² | energies excluded: MAE / R² |
|---|---|---|
| LinearRegression | **0.000000 / 1.000000** | 0.103875 / 0.361 |
| Ridge | 0.017658 / 0.981 | 0.101576 / 0.382 |
| RandomForest | 0.080893 / 0.587 | 0.089050 / 0.522 |
| SVR | 0.094789 / 0.411 | 0.109907 / 0.263 |

**H-3 — estimator selection.** Flat protocol (all five learners scored on the identical outer folds,
best reported): RandomForest 0.09563, GradientBoosting 0.09777, SVR 0.11198, Ridge 0.11361,
MLP 0.16056. Nested CV (learner chosen inside each training fold, scored on the untouched outer
fold): **MAE 0.09563 eV**, RandomForest selected in **5/5** outer folds.
**Selection optimism = 0.00000 eV.**

- **Interpretation.**
  (H-12) The leakage claim is now demonstrated properly and is *stronger* than the unsourced figures
  it replaces: under cross-validation a linear model supplied with the two constituent energies
  recovers the target exactly ($R^2 = 1.000$, MAE $= 0$), while the same features without the
  energies reach $R^2 = 0.36$. The retired "in-sample $R^2 = 0.87$" was both unsourced and the wrong
  statistic — an in-sample score shows overfitting capacity, not leakage, and 0.87 undercut the very
  argument it was offered to support. A methodological point worth keeping: leakage severity depends
  on whether the learner can *express* the identity. A random forest is axis-aligned and cannot
  represent a difference of two continuous inputs, so it understates leakage (0.587); this is why a
  tree-based study can carry the defect without the cross-validated score looking suspicious.
  (H-3) The selection concern is valid in principle and empirically nil here: the random forest wins
  in every outer fold, so choosing it on the reported folds costs nothing measurable. The SI sentence
  claiming "no test-set selection bias" was nonetheless the wrong argument (it appealed to untuned
  hyperparameters, not to nested validation) and should cite this entry instead.
- **Caveats.** (i) The exact zero for LinearRegression is arithmetic recovery to floating-point
  precision, not a statistical result; report it as such. (ii) Learner defaults are untuned, so the
  flat-protocol ranking is a comparison of default configurations, not of model families at their
  best. (iii) Nested CV inherits the singleton-scaffold geometry of A-002 P3, so the outer folds are
  no harder than a random split.
- **Claims:** new **C15** (leakage recovers the target exactly under CV with a linear learner,
  $R^2 = 1.000$; energy-free 0.36), **C16** (nested CV reproduces the reported MAE exactly; selection
  optimism 0.000 eV). Supersedes the unsourced "in-sample $R^2 = 0.87$", which must not reappear.

### A-008 — Two control checks that INVALIDATE A-006 T5 and correct the 0.072 eV label figure
- **Date:** 2026-09-04. **Trigger:** L3 beat-2 review (`review-10-critical-thinking-beat2.md`, CRITICAL 1
  and HIGH 4). Both challenged results this project had already written into the manuscript. Both are
  confirmed by direct recomputation, so both of my earlier readings were wrong.
- **Source:** recomputation on the A-006 folds (seed 0, scaffold GroupKFold(5), NTO features);
  `data/label_noise_reconciled.json` read directly.

**(1) The clean-label ordering is reproduced by a feature-free predictor — A-006 T5 is not evidence.**

| subset | n | model MAE | constant-predictor MAE | SD of y in subset |
|---|---|---|---|---|
| replicated, clean (sd $\le$ 0.05) | 79 | 0.0741 | **0.0739** | 0.1008 |
| replicated, noisy (sd $>$ 0.05) | 23 | 0.1062 | **0.1070** | 0.1525 |
| singletons (no replicate) | 129 | 0.1069 | **0.1099** | 0.1952 |

  The random forest ties a constant predictor on every subset. The 0.074 / 0.106 / 0.107 ordering
  tracks the dispersion of the target within each subset (SD 0.10 / 0.15 / 0.20), which is what a
  mean-absolute-error must do, and carries no information about label quality.

**(2) The 0.072 eV label-scatter figure does not reproduce.** `data/label_noise_reconciled.json`
  (n = 102 multi-report molecules) contains: RMS per-molecule SD (ddof=1) **0.0856**; (ddof=0) **0.0701**;
  RMS single report about the median **0.0905**; RMS single report about the mean 0.0875;
  RMS standard error of the median **0.0490**; mean absolute deviation about median 0.0312.
  **0.072 is not among them.** The manuscript attaches 0.072 to the sentence "a single report sits
  X (RMS) from its molecule's median", whose computed value is **0.0905** — a 26 % understatement.

- **Interpretation.**
  (1) A-006 T5 must be withdrawn as support for the label-limited account, and every manuscript
  sentence drawing on it removed. What the control actually shows is broader and worth stating: on
  this task the forest does not beat a constant predictor *within* any label-quality stratum, so the
  0.096 eV headline rests on between-stratum variation. This strengthens the paper's own cautionary
  thesis while destroying the specific argument I had built. The honest residual claim about the
  ceiling is A-006 T4's 0.049 eV precision of the median, which reproduces exactly.
  (2) Quote **0.0905 eV** for single-report scatter about the median, or **0.049 eV** for the precision
  of the median actually regressed on, and name the estimator in the sentence. 0.072 must not appear.
- **Caveats.** (i) The constant-predictor comparison is within-subset and unpaired; it shows the model
  adds nothing *within* a stratum, not that it adds nothing overall (A-002 P1 shows it beats the global
  mean baseline on paired folds, CI excluding zero). (ii) n = 23 for the noisy stratum throughout.
- **Claims:** **withdraws C12** (A-006 T5). Supersedes the 0.072 figure wherever it appears.
  New **C17**: within every label-quality stratum the model ties a constant predictor.

### A-009 — Three further claims checked against the deposited data (beat-2 open findings)
- **Date:** 2026-09-04. **Trigger:** remaining OPEN/CRITICAL items from L3 beat 2
  (`review-9-adversarial-beat2.md`, `review-10-critical-thinking-beat2.md`, `review-8-verification-beat2.md`).
- **Sources read directly:** `data/ref3_p1p2p3.json` (P2c), `MCA_submission/tables/experimental_delta_est_extracted.csv`.

**(1) The 0.080 eV multi-report MAE has a matched baseline that was not reported.**
  `P2c_report_subsets.multi_report`: $n=102$, Morgan MAE **0.08013**, mean-baseline MAE **0.08908**,
  $\Delta$MAE **0.00896** with a CI spanning zero. The subset's low absolute error reflects its
  narrower label spread, not better prediction — the same artefact that invalidated A-006 T5 (A-008).
  Single-report subset for contrast: $n=129$, MAE 0.1006 vs baseline 0.1229, $\Delta$ 0.0223, CI excludes zero.

**(2) "The DOI split holds out a measurement protocol" has no metadata behind it.**
  In the source table, `phase` is non-null for **0/1520** rows and `atmosphere` for **0/1520**.
  `record_method` is non-null for 1520/1520 but its values are text-parser names
  (`AutoTableParser`, `QuantityModelTemplateParser`, …), not experimental protocols.
  The DOI split holds out a source paper, which is a *proxy* for shared laboratory practice.
  This confirms A-002 caveat (i) and the manuscript now says so wherever the split is described.

**(3) The 79 % vs 30 % SHAP contrast is a dimensionality artefact.**
  Ten descriptors are \qty{29}{\percent} of the 35-dimensional NTO set; ten bits are
  \qty{0.5}{\percent} of a 2048-bit fingerprint. Concentration of attribution follows from feature
  count and is not evidence of superior interpretability. What survives is that NTO variables are
  named physical quantities while Morgan bits are hashed substructures — a property of the
  representation, not of the attribution totals.

- **Interpretation.** All three were stated in the manuscript in a form stronger than the data
  supports, and all three are now corrected in place rather than deleted: (1) the matched baseline
  and the zero-spanning CI are reported alongside the 0.080 figure; (2) "measurement protocol" is
  replaced by "a laboratory's shared practice … a proxy, since the corpus records no measurement
  conditions per value"; (3) the attribution totals are given with the denominators that make them
  comparable, and the interpretability case rests on named variables instead.
- **Caveats.** (i) P2c subsets are not independent of A-002 P2a — the same 102 molecules define both.
  (ii) The parser-name finding means no protocol-controlled analysis is possible in this corpus at all;
  a solvent-controlled study would need a different data source.
- **Claims:** qualifies C5 (no-transfer) — the split tests source-paper transfer, not protocol transfer.
  Withdraws the interpretability-superiority reading of the SHAP contrast.

### A-008b — CORRECTION: A-008 claim (2) was wrong; 0.072 eV does reproduce
- **Date:** 2026-09-04. **Supersedes A-008 (2)**, which asserted that 0.072 eV "is not among"
  the computed estimators and had it replaced by 0.0905 in 14 places and in `qa_score.py`.
  That was my error, and it was already answered by A-003 before I made it.
- **Source:** direct recomputation over the 102 multi-report molecules, and the reconciliation
  table in the archived A-003 (`outputs/analysis/archive/ledger-A001-A003-2026-09.md`).
- **Result.** Both figures are correct estimators of the same quantity under different weightings:

| estimator | value |
|---|---|
| RMS of a single report about its molecule's median, **each molecule weighted equally** | **0.0724** |
| the same quantity, **report-pooled** (each report weighted equally) | 0.0905 |

  A-003 selected the molecule-weighted figure and recorded "the published 0.072 stands".
- **Interpretation.** The published value is right, and the defect was never the number: it was
  that the manuscript printed "0.072 (RMS)" without naming the weighting, so a recomputation that
  reasonably chose report-pooling appeared to contradict it. Two independent reviewers and I all
  landed on 0.0905 for exactly that reason. The fix is to state the estimator, which the manuscript
  now does at both primary sites; report-pooling over-weights molecules carrying many reports, while
  the regression target is one median per molecule, so molecule-weighting is the apt choice.
  All 14 substitutions have been reverted and the `qa_score.py` A4 assertion restored to 0.072.
- **Lesson recorded for this project:** a number that "fails to reproduce" should be checked against
  the ledger's own reconciliation entries before the manuscript is changed. A-003 existed and I did
  not consult it.
- **Claims:** restores C3 as originally stated, now with the estimator named. A-008 claim (1) — the
  constant-predictor control that withdrew the clean-label result — is unaffected and still stands.

### A-010 — CORRECTION: the TD-DFT cost figures were a projection, not a measurement
- **Date:** 2026-09-05. **Trigger:** L3 beat-4 numerical audit (`review-15-numerical-audit.md`, M1–M3).
- **Corrects archived A-005**, which recorded "604.5 core-hours / median 18.2 min / slowest 1.13 h"
  as the measured cost of the TD-DFT run. Those three values are the contents of
  `data/tddft_reference_timing.json → summary.projection_231`, a block whose own name says it is a
  power-law extrapolation (`wall_seconds = 0.2634 · atoms^1.89`). The manuscript then stated them as
  observed run statistics in **12 places across 6 files**.
- **Measured values**, from the 231 per-molecule `wall_seconds` in `data/tddft_reference_231.csv`
  and `summary.elapsed_hours`:

| quantity | published (projection) | measured |
|---|---|---|
| total cost | 604.5 core-hours | **609** core-hours (8 processes × 76.15 h elapsed); Σ per-molecule × 8 = 631.9 |
| median per molecule | 18.2 min | **16.05 min** |
| slowest molecule | 1.13 h | **2.087 h** (4F-m-ν-DABNA) |

- **Interpretation.** The projection was close on the total (0.8 % low) but wrong by 1.85× on the
  slowest molecule, and the manuscript presented all three as measurements. The corrected text now
  quotes 609 core-hours with its basis stated (elapsed wall time × process count) and says explicitly
  that these are measured times rather than a scaling extrapolation. The scientific conclusion is
  unchanged — the point was ever that a several-hundred-core-hour reference buys 0.004 eV — but the
  cost of a computation is exactly the kind of number a referee may try to reproduce.
- **Caveats.** (i) 609 (elapsed × procs) and 632 (Σ per-molecule × 8) differ because the two measure
  occupancy and summed job time; 609 is the resource footprint and is what is published.
  (ii) The power-law fit itself remains useful for extrapolating to larger molecules and is retained
  in the deposited timing file, now clearly distinguished from the measured values.
- **Claims:** amends C9's cost clause. No accuracy result depends on these numbers.

### A-011 — The nearest available condition-controlled test (answers Referee 3, point 2)
- **Date:** 2026-09-05. **Trigger:** the one Digital Discovery objection still answered only by
  explanation. Referee 3 asked for performance "on a strictly solvent- and condition-controlled
  subset rather than attributing model failure entirely to external data noise".
- **Why a proxy:** `phase` and `atmosphere` are non-null for **0/1520** source rows (A-009), so a
  strictly controlled subset cannot be constructed from this corpus at all. The closest available
  proxy is the source paper: molecules whose reported values all come from one DOI were measured
  under a single laboratory's conditions; molecules drawing on two or more DOIs certainly mix them.
- **Source:** `code/a011_condition_controlled_proxy.py` → `data/a011_condition_controlled.json`.
  Same scaffold GroupKFold(5), seed 0, 2000 bootstrap resamples.

| group | n | Morgan MAE | NTO MAE | constant predictor | target SD | advantage over constant |
|---|---|---|---|---|---|---|
| single-source (1 DOI) | 218 | 0.0933 | 0.0965 | 0.1004 | 0.1703 | **+0.0071** |
| multi-source (≥2 DOIs) | 13 | 0.0584 | 0.0802 | 0.0608 | 0.0814 | **+0.0024** |

  Difference in advantage-over-constant (single − multi) = **+0.0047 eV**,
  95% CI **[−0.0138, +0.0345]** — **does not exclude zero**.

- **Interpretation.** The test returns a null. Restricting to molecules measured under one
  laboratory's conditions does not measurably improve what the model adds over a feature-free
  predictor. Two details matter for reading it honestly. First, the multi-source group has the
  *lower* raw MAE (0.058 vs 0.093) — the opposite of what a condition-noise account predicts — but
  its target dispersion is also half as large (SD 0.081 vs 0.170) and its own constant predictor
  scores 0.061, so the raw difference is dispersion, not label quality. This is precisely the
  artefact that invalidated A-006 T5, and the comparison was therefore run against a per-group
  constant predictor from the outset (A-008). Second, only **13** molecules draw on more than one
  source paper, so the test has little power; it constrains the effect rather than excluding it.
  The defensible statement is that we ran the closest test the data permits and found no evidence
  that mixed measurement conditions drive the error — not that conditions are irrelevant.
- **Caveats.** (i) n = 13 in the multi-source group. (ii) One DOI means one laboratory, not one
  solvent: a single paper may itself report several media. (iii) Single-source includes the 129
  single-report molecules, whose condition consistency is untestable rather than verified.
- **Claims:** new **C18** (nearest-available condition-controlled test is null, CI includes zero).
  Converts Referee 3 point 2 from "cannot be tested" to "tested by proxy, null, with the reason the
  strict test is impossible stated".
