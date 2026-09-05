#!/usr/bin/env python3
"""
featurize_nto_local.py — F1-B completion (local, faithful reproduction).

Reproduces the ORIGINAL 35-descriptor NTO featurisation for the expanded TADF
corpus using the EXACT ground-truth toolchain (from generate_dmacdps_nto.sh):

    geometry (xTB-opt S0)  ->  xtb4stda  ->  stda -xtb -e 10 -nto 5 [-t]  ->
    nto001.molden  ->  { Multiwfn NTO overlap  +  compute_ct_descriptors }  ->
    Stage A/C/D assembly  ->  35 NTO descriptors (gas, method=stda)

Why this file exists
--------------------
The server run (TASK-F1-B/code/xtb_stda.py) used the standalone Grimme `stda`
WITHOUT NTO export, so it produced excited-state ENERGIES only — the 35 NTO
descriptors the model needs were never computed, and 64/834 molecules failed
outright. This script computes the missing values locally.

Faithfulness
------------
- Uses the same binaries/flags as the original 747-mol pipeline.
- Imports the ORIGINAL extractor + assembly functions verbatim (no re-derivation):
    compute_ct_descriptors_747mol : molden parser + 7 base CT descriptors
    extract_stda_features_747mol  : stda .log parser (energies / osc / HOMO-LUMO)
    add_proxy_respa_features      : Stage C composites
    merge_ct_features_747mol      : Stage D pivot + CT deltas
- Reuses the server's xTB-optimised geometries (mol_*_stage1.xyz = gas,
  mol_*_stage3.xyz = toluene) so no re-optimisation is needed for the 770.
- For the 64 with no geometry: robust RDKit embed -> ORCA GFN2-xTB opt.

Outputs (all under TASK-F1-B/data/)
-----------------------------------
  expanded_features_nto.csv       — 35 NTO descriptors (gas/stda) + mol_id + SMILES
  expanded_energies.csv           — refreshed S1/T1/T2 (gas + toluene)
  expanded_featurization_report.json — per-molecule status / method / failures
  nto_work/<mol_id>/...           — per-molecule working files (molden, logs)
  nto_checkpoints/<mol_id>.json   — per-molecule checkpoint (resume-safe)

Integrity: molecules that still fail are logged and DROPPED. No descriptor is
ever fabricated or zero-filled.

Usage
-----
  python code/featurize_nto_local.py --phase all
  python code/featurize_nto_local.py --only mol_00000 --keep-work   # smoke test
  python code/featurize_nto_local.py --phase reuse --workers 8
  python code/featurize_nto_local.py --assemble-only                 # rebuild CSVs from checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
TASK_ROOT = Path(__file__).resolve().parent.parent          # TASK-F1-B/
PROJECT_ROOT = TASK_ROOT.parent                             # Article3_ML/
DATA = TASK_ROOT / "data"
ORCA_OUT = DATA / "orca_outputs"                            # server geometries live here
WORK = DATA / "nto_work"
CKPT = DATA / "nto_checkpoints"

INPUT_CSV = DATA / "expanded_smiles_resolved.csv"
FEATURES_CSV = DATA / "expanded_features_nto.csv"
ENERGIES_CSV = DATA / "expanded_energies.csv"
REPORT_JSON = DATA / "expanded_featurization_report.json"

# Reuse the ORIGINAL pipeline code (import, do not reimplement)
_DP = PROJECT_ROOT / "CALCULATIONS-MADE" / "data_processing"
sys.path.insert(0, str(_DP))
from compute_ct_descriptors_747mol import (          # noqa: E402
    parse_molden_atoms,
    compute_ct_descriptors_from_nto,
)
from extract_stda_features_747mol import parse_log_file      # noqa: E402
from add_proxy_respa_features import add_proxy_respa_features  # noqa: E402
from merge_ct_features_747mol import pivot_ct_descriptors     # noqa: E402

# Canonical 35-NTO order (single source of truth = code/_dataset.py)
NTO = ['S1_S_he', 'T1_S_he', 'Delta_S_he', 'Abs_Delta_S_he', 'S1_CT_number', 'T1_CT_number',
       'Delta_CT_number', 'Abs_Delta_CT_number', 'S1_Lambda_D', 'S1_Lambda_A', 'T1_Lambda_D',
       'T1_Lambda_A', 'Delta_Lambda_D', 'Abs_Delta_Lambda_D', 'Delta_Lambda_A', 'Abs_Delta_Lambda_A',
       'S1_Delta_r', 'T1_Delta_r', 'Delta_Delta_r', 'Abs_Delta_Delta_r', 'S1_hole_on_A',
       'S1_particle_on_D', 'T1_hole_on_A', 'T1_particle_on_D', 'Char_diff_squared', 'S_NTO_sum',
       'S_NTO_product', 'S_NTO_ratio', 'S1_osc_strength', 'Delta_S_NTO', 'Abs_Delta_S_NTO',
       'Log_Abs_S1', 'Log_Abs_T1', 'S1_overlap', 'T1_overlap']

# ── Tools ────────────────────────────────────────────────────────────────────
ORCA_EXE = os.environ.get("ORCA_EXE", "/home/tchapet/orca-6-1-1/orca")
XTB4STDA = os.environ.get("XTB4STDA_EXE", "/home/tchapet/xtb4stda/xtb4stda")
STDA = os.environ.get("STDA_EXE", "/home/tchapet/stda_v1-6/stda")
MULTIWFN = os.environ.get(
    "MULTIWFN_EXE", "/home/tchapet/Multiwfn/Multiwfn_3.8_dev_bin_Linux/Multiwfn"
)
NUM_NTOS = 5
ORCA_NPROCS = int(os.environ.get("NPROCS", "8"))
ORCA_MAXCORE = int(os.environ.get("MAXCORE_MB", "3500"))

ENVS = ("gas", "toluene")
TRANS = ("S0S1", "S0T1")   # S1 (singlet) / T1,T2 (triplet)


def _tool_env() -> dict:
    """Environment for xtb4stda/stda so parameter files are found."""
    env = dict(os.environ)
    env.setdefault("XTB4STDAHOME", str(Path(XTB4STDA).parent))
    env.setdefault("STDAHOME", str(Path(STDA).parent))
    env["PATH"] = f"{Path(XTB4STDA).parent}:{Path(STDA).parent}:" + env.get("PATH", "")
    env.setdefault("OMP_NUM_THREADS", "2")
    return env


# ── Geometry ─────────────────────────────────────────────────────────────────
def _read_xyz_coord_lines(xyz_path: Path) -> list[str] | None:
    """Return coordinate lines (element x y z) from an ORCA/standard .xyz file."""
    try:
        lines = xyz_path.read_text().splitlines()
    except OSError:
        return None
    if len(lines) < 3:
        return None
    try:
        n = int(lines[0].split()[0])
    except (ValueError, IndexError):
        return None
    coord = [ln for ln in lines[2:2 + n] if ln.strip()]
    return coord if len(coord) == n else None


def _write_xyz(coord_lines: list[str], path: Path) -> None:
    path.write_text(f"{len(coord_lines)}\ngenerated\n" + "\n".join(coord_lines) + "\n")


def reused_geometry(mol_id: str, env: str) -> list[str] | None:
    """Gas -> stage1.xyz (gas-opt), toluene -> stage3.xyz (toluene-opt)."""
    stage = 1 if env == "gas" else 3
    return _read_xyz_coord_lines(ORCA_OUT / f"{mol_id}_stage{stage}.xyz")


def rdkit_xyz(smiles: str) -> list[str] | None:
    """Robust RDKit 3D embed -> coordinate lines, or None."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    ok = False
    for seed in (0xF00D, 42, 7, 2024):
        p = AllChem.ETKDGv3()
        p.randomSeed = seed
        p.useRandomCoords = False
        if AllChem.EmbedMolecule(mol, p) == 0:
            ok = True
            break
    if not ok:                       # last resort: random coordinates
        p = AllChem.ETKDGv3()
        p.useRandomCoords = True
        p.maxIterations = 2000
        if AllChem.EmbedMolecule(mol, p) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass
    conf = mol.GetConformer()
    out = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        out.append(f"{atom.GetSymbol():<2} {pos.x:14.8f} {pos.y:14.8f} {pos.z:14.8f}")
    return out


def orca_xtb_opt(coord_lines: list[str], workdir: Path, env: str) -> list[str] | None:
    """GFN2-xTB geometry optimisation via ORCA; returns optimised coord lines."""
    workdir.mkdir(parents=True, exist_ok=True)
    tag = "ALPB(Toluene)" if env == "toluene" else ""
    inp = workdir / "opt.inp"
    inp.write_text(
        f"! GFN2-xTB Opt TightOpt {tag}\n"
        f"%pal nprocs {ORCA_NPROCS} end\n%maxcore {ORCA_MAXCORE}\n"
        f"%geom MaxIter 500 end\n"
        f"* xyz 0 1\n" + "\n".join(coord_lines) + "\n*\n"
    )
    with open(workdir / "opt.out", "w") as fh:
        r = subprocess.run([ORCA_EXE, str(inp)], stdout=fh, stderr=subprocess.STDOUT,
                           cwd=str(workdir))
    if r.returncode != 0:
        return None
    return _read_xyz_coord_lines(workdir / "opt.xyz")


# ── sTDA + NTO + Multiwfn (one transition) ──────────────────────────────────
def _state_energy(log_text: str, n: int) -> float | None:
    """Excitation energy (eV) of state n from a stda .log."""
    m = re.search(rf'^\s*{n}\s+(\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+', log_text, re.MULTILINE)
    return float(m.group(1)) if m else None


def run_transition(coord_lines: list[str], workdir: Path, env: str, trans: str) -> dict | None:
    """
    Run xtb4stda + stda -nto for one (env, transition); returns:
      { overlap, ct(7 base), log_data(parse_log_file), t2 }  or None on failure.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    xyz = workdir / "geom.xyz"
    _write_xyz(coord_lines, xyz)
    tenv = _tool_env()

    # 1) xtb4stda -> wfn.xtb
    cmd = [XTB4STDA, "geom.xyz"] + (["gbsa", "toluene"] if env == "toluene" else [])
    with open(workdir / "xtb4stda.log", "w") as fh:
        subprocess.run(cmd, cwd=str(workdir), stdout=fh, stderr=subprocess.STDOUT, env=tenv)
    if not (workdir / "wfn.xtb").exists():
        return None

    # 2) stda -xtb -e 10 -nto 5 [-t]  -> nto001.molden + excitation log
    opts = ["-xtb", "-e", "10", "-nto", str(NUM_NTOS)]
    if trans == "S0T1":
        opts.append("-t")
    log_path = workdir / f"{trans}_stda.log"
    with open(log_path, "w") as fh:
        subprocess.run([STDA] + opts, cwd=str(workdir), stdout=fh,
                       stderr=subprocess.STDOUT, env=tenv)
    molden = workdir / "nto001.molden"
    if not molden.exists():
        return None

    log_text = log_path.read_text(errors="ignore")
    log_data = parse_log_file(str(log_path))

    # 3) Multiwfn NTO overlap (fn 200 -> 10 -> 5 -> 6 -> 1,2)
    overlap = _multiwfn_overlap(workdir)

    # 4) 7 base CT descriptors from nto001.molden (mid-split D/A, as original)
    ct = _ct_from_molden(molden)
    if ct is None:
        return None

    return {
        "overlap": overlap,
        "ct": ct,
        "s1_or_t1_eV": log_data.get("energy_ev"),
        "osc": log_data.get("osc_strength"),
        "homo_lumo_gap": log_data.get("homo_lumo_gap"),
        "t2_eV": _state_energy(log_text, 2) if trans == "S0T1" else None,
    }


def _multiwfn_overlap(workdir: Path) -> float | None:
    """Replicate generate_dmacdps_nto.sh Multiwfn overlap EXACTLY.

    The proven recipe feeds the molden filename as the FIRST STDIN LINE (Multiwfn
    launched with no file argument), then a blank line, then fn 200,10,5,6,1,2.
    """
    inp = workdir / "mwfn_overlap_input.txt"
    inp.write_text("nto001.molden\n\n200\n10\n5\n6\n1,2\n0,0\n0\nq\n")
    out = workdir / "mwfn_overlap_output.txt"
    try:
        with open(inp) as fi, open(out, "w") as fo:
            subprocess.run([MULTIWFN], cwd=str(workdir), stdin=fi,
                           stdout=fo, stderr=subprocess.STDOUT, timeout=180,
                           env=_tool_env())
    except subprocess.TimeoutExpired:
        return None
    for line in out.read_text(errors="ignore").splitlines():
        if "Result:" in line:
            parts = line.split()
            try:
                return float(parts[1])
            except (IndexError, ValueError):
                return None
    return None


def _ct_from_molden(molden: Path) -> dict | None:
    """7 base CT descriptors via the ORIGINAL extractor (naive mid-split D/A)."""
    atoms = parse_molden_atoms(str(molden))
    if not atoms:
        return None
    n = len(atoms)
    mid = n // 2
    frag_D = list(range(1, mid + 1))
    frag_A = list(range(mid + 1, n + 1))
    d = compute_ct_descriptors_from_nto(str(molden), atoms, frag_D, frag_A)
    return None if "error" in d else d


# ── One molecule ─────────────────────────────────────────────────────────────
def process_molecule(mol_id: str, smiles: str, keep_work: bool = False) -> dict:
    """Full featurisation for one molecule. Writes a checkpoint; returns status dict."""
    ck = CKPT / f"{mol_id}.json"
    if ck.exists():
        return json.loads(ck.read_text())

    mdir = WORK / mol_id
    result: dict = {"mol_id": mol_id, "SMILES": smiles, "status": "ok",
                    "geom_source": {}, "per": {}}
    try:
        for env in ENVS:
            coords = reused_geometry(mol_id, env)
            if coords is None:
                base = rdkit_xyz(smiles)
                if base is None:
                    result.update(status="fail", reason="rdkit_embed_failed")
                    _save_ckpt(ck, result)
                    return result
                coords = orca_xtb_opt(base, mdir / f"opt_{env}", env)
                if coords is None:
                    result.update(status="fail", reason=f"orca_opt_failed_{env}")
                    _save_ckpt(ck, result)
                    return result
                result["geom_source"][env] = "orca_xtb_opt"
            else:
                result["geom_source"][env] = "reused_server"

            for trans in TRANS:
                r = run_transition(coords, mdir / f"{env}_{trans}", env, trans)
                if r is None:
                    result.update(status="fail", reason=f"stda_failed_{env}_{trans}")
                    _save_ckpt(ck, result)
                    return result
                result["per"][f"{env}_{trans}"] = r
    except Exception as exc:  # noqa: BLE001 — record, never fabricate
        result.update(status="fail", reason=f"exception:{type(exc).__name__}:{exc}")
    finally:
        if not keep_work:
            shutil.rmtree(mdir, ignore_errors=True)

    _save_ckpt(ck, result)
    return result


def _save_ckpt(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


# ── Assembly (Stage A/B/C/D on collected checkpoints) ───────────────────────
def assemble() -> None:
    """Build expanded_features_nto.csv (gas/stda) + expanded_energies.csv from checkpoints."""
    ok = [json.loads(p.read_text()) for p in sorted(CKPT.glob("*.json"))]
    ok = [r for r in ok if r.get("status") == "ok"]

    # ---- 35 NTO descriptors: GAS phase only (matches _dataset.py) ----
    base_rows, energy_rows = [], []
    n_dropped_incomplete = 0
    for r in ok:
        per = r["per"]
        g_s1, g_t1 = per.get("gas_S0S1"), per.get("gas_S0T1")
        # Honest-drop (matches original Stage B): require BOTH gas overlaps and
        # BOTH gas energies, else the row cannot yield real 35 descriptors.
        if (not g_s1 or not g_t1
                or g_s1.get("overlap") is None or g_t1.get("overlap") is None
                or g_s1.get("s1_or_t1_eV") is None or g_t1.get("s1_or_t1_eV") is None
                or g_s1.get("osc") is None):
            n_dropped_incomplete += 1
            continue
        # Stage A/B long-format row (one env, method=stda)
        row = {
            "molecule": r["mol_id"], "environment": "gas", "method": "stda",
            "S1_overlap": g_s1["overlap"], "T1_overlap": g_t1["overlap"],
            "S1_energy_eV": g_s1["s1_or_t1_eV"], "T1_energy_eV": g_t1["s1_or_t1_eV"],
            "S1_osc_strength": g_s1["osc"],
            "HOMO_LUMO_gap_eV": g_s1["homo_lumo_gap"],
            "SMILES": r["SMILES"],
        }
        row["Delta_E_ST_eV"] = row["S1_energy_eV"] - row["T1_energy_eV"]
        # attach 7 base CT for S1 and T1 (long, transition rows for pivot)
        row["_ct_S1"] = g_s1["ct"]
        row["_ct_T1"] = g_t1["ct"]
        base_rows.append(row)

        # energies (gas + toluene, S1/T1/T2)
        er = {"mol_id": r["mol_id"], "SMILES": r["SMILES"]}
        for env in ENVS:
            s1 = per.get(f"{env}_S0S1"); t1 = per.get(f"{env}_S0T1")
            tag = "gas" if env == "gas" else "tol"
            er[f"S1_{tag}_eV"] = s1["s1_or_t1_eV"] if s1 else None
            er[f"T1_{tag}_eV"] = t1["s1_or_t1_eV"] if t1 else None
            er[f"T2_{tag}_eV"] = t1["t2_eV"] if t1 else None
        energy_rows.append(er)

    if n_dropped_incomplete:
        print(f"Dropped {n_dropped_incomplete} 'ok' molecules with incomplete gas "
              f"overlaps/energies (honest-drop, no fabrication).")
    if not base_rows:
        print("No successful molecules to assemble.")
        return

    df = pd.DataFrame(base_rows)

    # Stage C composites (import; identical formulas)
    df = add_proxy_respa_features(df)

    # Stage D: build CT long table -> pivot -> merge -> deltas
    ct_long = []
    for _, row in df.iterrows():
        for trans, key in (("S0S1", "_ct_S1"), ("S0T1", "_ct_T1")):
            c = row[key]
            ct_long.append({"molecule": row["molecule"], "environment": "gas",
                            "method": "stda", "transition": trans, **c})
    ct_pivot = pivot_ct_descriptors(pd.DataFrame(ct_long))
    df = df.drop(columns=["_ct_S1", "_ct_T1"])
    df = df.merge(ct_pivot, on=["molecule", "environment", "method"], how="left")
    for b in ("CT_number", "Lambda_D", "Lambda_A", "Delta_r", "S_he"):
        df[f"Delta_{b}"] = df[f"T1_{b}"] - df[f"S1_{b}"]
        df[f"Abs_Delta_{b}"] = np.abs(df[f"Delta_{b}"])

    # Final 35-descriptor table
    missing = [c for c in NTO if c not in df.columns]
    if missing:
        raise RuntimeError(f"Assembly missing columns: {missing}")
    out = df[["molecule", "SMILES"] + NTO].rename(columns={"molecule": "mol_id"})
    out.to_csv(FEATURES_CSV, index=False)
    pd.DataFrame(energy_rows).to_csv(ENERGIES_CSV, index=False)
    print(f"Wrote {FEATURES_CSV.name}: {len(out)} molecules × {len(NTO)} NTO descriptors")
    print(f"Wrote {ENERGIES_CSV.name}: {len(energy_rows)} molecules")


# ── Driver ───────────────────────────────────────────────────────────────────
def load_todo() -> list[tuple[str, str]]:
    df = pd.read_csv(INPUT_CSV)
    todo = df[df["already_featurised"] != True] if "already_featurised" in df.columns else df  # noqa: E712
    tasks = []
    for idx, (_, row) in enumerate(todo.reset_index(drop=True).iterrows()):
        smi = row.get("smiles", row.get("SMILES", ""))
        if isinstance(smi, str) and smi:
            tasks.append((f"mol_{idx:05d}", smi))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["reuse", "missing", "all"], default="all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", help="specific mol_ids (smoke test)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    if args.assemble_only:
        assemble()
        return

    tasks = load_todo()
    if args.only:
        tasks = [t for t in tasks if t[0] in set(args.only)]
    else:
        has_geom = lambda mid: reused_geometry(mid, "gas") is not None
        if args.phase == "reuse":
            tasks = [t for t in tasks if has_geom(t[0])]
        elif args.phase == "missing":
            tasks = [t for t in tasks if not has_geom(t[0])]
    if args.limit:
        tasks = tasks[:args.limit]

    todo = [t for t in tasks if not (CKPT / f"{t[0]}.json").exists()]
    print(f"Phase={args.phase}  tasks={len(tasks)}  remaining(no ckpt)={len(todo)}  workers={args.workers}")

    n_ok = n_fail = 0
    if args.workers <= 1 or args.only:
        for mid, smi in todo:
            r = process_molecule(mid, smi, keep_work=args.keep_work)
            n_ok += r["status"] == "ok"; n_fail += r["status"] != "ok"
            print(f"  {mid}: {r['status']}" + (f" ({r.get('reason')})" if r["status"] != "ok" else ""))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_molecule, mid, smi, args.keep_work): mid for mid, smi in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                n_ok += r["status"] == "ok"; n_fail += r["status"] != "ok"
                if i % 25 == 0 or r["status"] != "ok":
                    print(f"  [{i}/{len(todo)}] {r['mol_id']}: {r['status']}"
                          + (f" ({r.get('reason')})" if r["status"] != "ok" else ""))

    # Report
    all_ck = [json.loads(p.read_text()) for p in CKPT.glob("*.json")]
    report = {
        "n_checkpoints": len(all_ck),
        "n_ok": sum(r["status"] == "ok" for r in all_ck),
        "n_fail": sum(r["status"] != "ok" for r in all_ck),
        "failures": {r["mol_id"]: r.get("reason") for r in all_ck if r["status"] != "ok"},
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\nThis run: ok={n_ok} fail={n_fail}. "
          f"Cumulative ok={report['n_ok']} fail={report['n_fail']}. Report -> {REPORT_JSON.name}")


if __name__ == "__main__":
    main()
