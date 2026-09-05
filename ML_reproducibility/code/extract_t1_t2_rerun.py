#!/usr/bin/env python3
"""
extract_t1_t2_rerun.py — F6-B (PLAN260630)
==========================================
Genuine sTDA T1/T2 re-run for the full 231-molecule honest-benchmark corpus.

Motivation
----------
F6-A could only reproduce the 14 ORCA molecules: the combined-features CSV
(CALCULATIONS-MADE/.../combined_features_747mol_full_ct.csv) stores S1 and T1
but NOT T2, and the original 747-mol raw sTDA outputs are no longer on disk.
So F6-A silently fell back to triplet_manifold.csv and produced a meaningless
self-comparison (MAE=0.0, rho=1.0). This script fixes that by re-running the
excited-state part of the ORIGINAL pipeline for every corpus molecule and
capturing T2 (2nd triplet root).

Faithfulness
------------
Reuses the EXACT binaries, flags and parser of the validated F1-B pipeline
(TASK-F1-B/code/featurize_nto_local.py):
    RDKit ETKDGv3+MMFF  ->  ORCA GFN2-xTB TightOpt (gas)  ->  xtb4stda  ->
    stda -xtb -e 10 -nto 5        (singlet: S1 = root 1)
    stda -xtb -e 10 -nto 5 -t     (triplet: T1 = root 1, T2 = root 2)
Energies are parsed with the SAME `_state_energy` regex the F1-B run used to
populate T2 in expanded_energies.csv. We SKIP the Multiwfn/NTO overlap step —
this analysis needs only excited-state energies, so no molden/CT work is done.

Provenance note (documented, not hidden)
----------------------------------------
Geometries are freshly GFN2-xTB-optimised from SMILES (originals were server-side
and are gone), so S1/T1 here may differ by a few tens of meV from the manuscript's
stored training values. The T1-T2 manifold is a SEPARATE auxiliary descriptor;
we validate the sTDA T1-T2 gap against the 14-molecule ORCA reference
(data/triplet_manifold.csv) to bound this.

Integrity: molecules that fail are logged and DROPPED. No energy is ever fabricated.

Outputs
-------
data/t1_t2_full.csv        — one row per successful molecule:
                             molecule, phase(gas), S1_eV, T1_eV, T2_eV,
                             delta_EST_eV, T1_T2_gap_eV, TADF_feasible
data/t1_t2_validation.json — REAL sTDA-vs-ORCA gap validation (n, MAE, rho, p)
data/t1_t2_rerun_report.json — per-molecule status / failures
data/t1_t2_rerun_ckpt/<name>.json — resume-safe checkpoints

Usage
-----
    python code/extract_t1_t2_rerun.py --only 4CzIPN ACRSA 2CzPN --keep-work  # smoke
    python code/extract_t1_t2_rerun.py --workers 4                             # full 231
    python code/extract_t1_t2_rerun.py --assemble-only                         # rebuild CSVs
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CODE = ROOT / "code"
CKPT = DATA / "t1_t2_rerun_ckpt"
WORK = DATA / "t1_t2_rerun_work"

# Reuse the proven F1-B pipeline functions verbatim (no reimplementation).
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "TASK-F1-B" / "code"))
import featurize_nto_local as fz  # noqa: E402  (rdkit_xyz, orca_xtb_opt, _tool_env, _write_xyz, _state_energy, binaries)
from _dataset import load_dataset  # noqa: E402

# TADF-feasibility thresholds (same as F6-A)
T1T2_FEASIBILITY_THRESHOLD = 0.15   # eV
DELTA_EST_THRESHOLD = 0.30          # eV


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))


def run_stda_energies(coords: list[str], workdir: Path, trans: str) -> dict | None:
    """xtb4stda + stda for one transition (gas). Returns root-1 (and root-2 if triplet)
    excitation energies in eV. Skips Multiwfn/NTO. None on failure."""
    workdir.mkdir(parents=True, exist_ok=True)
    fz._write_xyz(coords, workdir / "geom.xyz")
    tenv = fz._tool_env()

    # 1) xtb4stda -> wfn.xtb (gas: no gbsa)
    with open(workdir / "xtb4stda.log", "w") as fh:
        subprocess.run([fz.XTB4STDA, "geom.xyz"], cwd=str(workdir),
                       stdout=fh, stderr=subprocess.STDOUT, env=tenv)
    if not (workdir / "wfn.xtb").exists():
        return None

    # 2) stda -xtb -e 10 -nto 5 [-t]  (same flags as F1-B; energies from stdout log)
    opts = ["-xtb", "-e", "10", "-nto", "5"]
    if trans == "S0T1":
        opts.append("-t")
    log_path = workdir / f"{trans}_stda.log"
    with open(log_path, "w") as fh:
        subprocess.run([fz.STDA] + opts, cwd=str(workdir),
                       stdout=fh, stderr=subprocess.STDOUT, env=tenv)
    text = log_path.read_text(errors="ignore")

    e1 = fz._state_energy(text, 1)
    if e1 is None:
        return None
    out = {"root1_eV": e1}
    if trans == "S0T1":
        out["root2_eV"] = fz._state_energy(text, 2)
    return out


def process_molecule(name: str, smiles: str, keep_work: bool = False) -> dict:
    ck = CKPT / f"{_safe_name(name)}.json"
    if ck.exists():
        return json.loads(ck.read_text())

    mdir = WORK / _safe_name(name)
    res = {"molecule": name, "SMILES": smiles, "status": "ok"}
    try:
        base = fz.rdkit_xyz(smiles)
        if base is None:
            res.update(status="fail", reason="rdkit_embed_failed")
            _save(ck, res); return res
        coords = fz.orca_xtb_opt(base, mdir / "opt_gas", "gas")
        if coords is None:
            res.update(status="fail", reason="orca_xtb_opt_failed")
            _save(ck, res); return res

        s = run_stda_energies(coords, mdir / "gas_S0S1", "S0S1")
        t = run_stda_energies(coords, mdir / "gas_S0T1", "S0T1")
        if not s or not t or s.get("root1_eV") is None or t.get("root1_eV") is None:
            res.update(status="fail", reason="stda_failed")
            _save(ck, res); return res

        s1, t1, t2 = s["root1_eV"], t["root1_eV"], t.get("root2_eV")
        res.update(
            S1_eV=s1, T1_eV=t1, T2_eV=t2,
            delta_EST_eV=round(s1 - t1, 4),
            T1_T2_gap_eV=round(t2 - t1, 4) if t2 is not None else None,
        )
        if t2 is None:
            res.update(status="fail", reason="T2_root2_not_found")
    except Exception as exc:  # noqa: BLE001 — record, never fabricate
        res.update(status="fail", reason=f"exception:{type(exc).__name__}:{exc}")
    finally:
        if not keep_work:
            shutil.rmtree(mdir, ignore_errors=True)

    _save(ck, res)
    return res


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def assemble() -> None:
    rows = [json.loads(p.read_text()) for p in sorted(CKPT.glob("*.json"))]
    ok = [r for r in rows if r.get("status") == "ok" and r.get("T2_eV") is not None]
    n_fail = len(rows) - len(ok)
    if not ok:
        print("No successful molecules to assemble.")
        return

    out_rows = []
    for r in ok:
        gap = r["T1_T2_gap_eV"]
        de = r["delta_EST_eV"]
        feasible = (gap is not None and gap > T1T2_FEASIBILITY_THRESHOLD
                    and de is not None and de <= DELTA_EST_THRESHOLD)
        out_rows.append({
            "molecule": r["molecule"], "phase": "gas",
            "S1_eV": round(r["S1_eV"], 4), "T1_eV": round(r["T1_eV"], 4),
            "T2_eV": round(r["T2_eV"], 4), "delta_EST_eV": de,
            "T1_T2_gap_eV": gap, "TADF_feasible": bool(feasible),
        })
    full = pd.DataFrame(out_rows)
    full.to_csv(DATA / "t1_t2_full.csv", index=False)
    print(f"Wrote data/t1_t2_full.csv: {len(full)} molecules (dropped {n_fail} failures)")

    # ---- REAL sTDA-vs-ORCA validation on the 14-molecule reference ----
    validation = {"n": 0, "note": "triplet_manifold.csv not found"}
    ref_path = DATA / "triplet_manifold.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        ref_gas = ref[ref["phase"] == "gas"][["molecule", "T1_T2_gap_eV"]].rename(
            columns={"T1_T2_gap_eV": "gap_orca"})
        merged = full.merge(ref_gas, on="molecule", how="inner")
        merged = merged.dropna(subset=["T1_T2_gap_eV", "gap_orca"])
        if len(merged) >= 3:
            a = merged["T1_T2_gap_eV"].values
            b = merged["gap_orca"].values
            mae = float(np.mean(np.abs(a - b)))
            rho, p = stats.spearmanr(a, b)
            validation = {
                "n": int(len(merged)),
                "mae_eV": round(mae, 4),
                "spearman_rho": round(float(rho), 4),
                "spearman_p": round(float(p), 4),
                "molecules": merged["molecule"].tolist(),
                "note": ("sTDA-xTB (fresh GFN2 geometry) T1-T2 gap vs ORCA CAM-B3LYP "
                         "reference; independent methods, genuine comparison."),
            }
            print(f"sTDA-vs-ORCA T1-T2 validation (n={len(merged)}): "
                  f"MAE={mae:.4f} eV, rho={rho:.3f}, p={p:.3f}")
        else:
            validation = {"n": int(len(merged)),
                          "note": f"only {len(merged)} overlap molecules — insufficient for stats"}
    (DATA / "t1_t2_validation.json").write_text(json.dumps(validation, indent=2))
    print("Wrote data/t1_t2_validation.json")

    # feasibility stats
    gaps = full["T1_T2_gap_eV"].dropna()
    print(f"\nT1-T2 gap: mean={gaps.mean():.3f} eV, "
          f"frac<0.30eV={(gaps < 0.30).mean():.1%}, "
          f"TADF_feasible={int(full['TADF_feasible'].sum())}/{len(full)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="specific molecule names (smoke test)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    if args.assemble_only:
        assemble()
        return

    ds = load_dataset()
    tasks = list(dict.fromkeys(zip(ds.df.molecule.astype(str), ds.smiles)))  # unique, order-preserving
    if args.only:
        want = set(args.only)
        tasks = [t for t in tasks if t[0] in want]
    if args.limit:
        tasks = tasks[:args.limit]
    todo = [t for t in tasks if not (CKPT / f"{_safe_name(t[0])}.json").exists()]
    print(f"corpus tasks={len(tasks)} remaining(no ckpt)={len(todo)} workers={args.workers}")

    n_ok = n_fail = 0
    if args.workers <= 1 or args.only:
        for name, smi in todo:
            r = process_molecule(name, smi, keep_work=args.keep_work)
            n_ok += r["status"] == "ok"; n_fail += r["status"] != "ok"
            print(f"  {name}: {r['status']}"
                  + (f" ({r.get('reason')})" if r["status"] != "ok"
                     else f"  S1={r['S1_eV']:.3f} T1={r['T1_eV']:.3f} T2={r['T2_eV']:.3f} "
                          f"dEST={r['delta_EST_eV']:.3f} T1T2={r['T1_T2_gap_eV']}"))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_molecule, n, s, args.keep_work): n for n, s in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                n_ok += r["status"] == "ok"; n_fail += r["status"] != "ok"
                if i % 20 == 0 or r["status"] != "ok":
                    print(f"  [{i}/{len(todo)}] {r['molecule']}: {r['status']}"
                          + (f" ({r.get('reason')})" if r["status"] != "ok" else ""))

    report = {
        "n_checkpoints": len(list(CKPT.glob("*.json"))),
        "n_ok_with_T2": sum(1 for p in CKPT.glob("*.json")
                            if (d := json.loads(p.read_text())).get("status") == "ok"
                            and d.get("T2_eV") is not None),
        "this_run_ok": n_ok, "this_run_fail": n_fail,
        "failures": {json.loads(p.read_text())["molecule"]: json.loads(p.read_text()).get("reason")
                     for p in CKPT.glob("*.json")
                     if json.loads(p.read_text()).get("status") != "ok"},
    }
    (DATA / "t1_t2_rerun_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nThis run: ok={n_ok} fail={n_fail}. "
          f"Cumulative ok-with-T2={report['n_ok_with_T2']}. Report -> data/t1_t2_rerun_report.json")
    if not args.only:
        assemble()


if __name__ == "__main__":
    main()
