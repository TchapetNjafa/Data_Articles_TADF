#!/usr/bin/env python3
"""
expand_dataset.py — F1-A: SMILES resolution for master_database STSplit records.

Loads master_database.csv, filters to experimental STSplit ΔE_ST records,
resolves SMILES via PubChem (primary) + OPSIN (fallback), deduplicates by
InChIKey, and reports whether the ≥500 unique structure target is met.

Does NOT featurize — just resolve SMILES and report statistics.

Usage:
  python code/expand_dataset.py                  # offline tiers only
  python code/expand_dataset.py --online          # with PubChem (network, rate-limited)
"""
from __future__ import annotations
import argparse, ast, json, re, shutil, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
THEOEXP = ROOT / "TheoExp" / "scientific literature with ChemDataExtractor"

EST_LO, EST_HI = -0.3, 1.5


def _names(x: str) -> list[str]:
    try:
        return [str(n).strip() for n in ast.literal_eval(x)] if str(x).startswith("[") else [str(x)]
    except Exception:
        return [str(x).strip()]


def _value(v: str) -> float:
    try:
        p = ast.literal_eval(v)
        return float(p[0]) if isinstance(p, list) else float(p)
    except Exception:
        try:
            return float(re.findall(r"[-\d.]+", str(v))[0])
        except Exception:
            return np.nan


def load_stsplit_experimental() -> pd.DataFrame:
    """Long table: one row per (name, value) from master_database STSplit.
    
    Includes both flagged (1.0) and unflagged (NaN) experimental records.
    Excludes is_experimental=0 (computational)."""
    m = pd.read_csv(THEOEXP / "master_database.csv")
    m = m[(m["model_name"] == "STSplit") & (m["is_experimental"].isna() | (m["is_experimental"] == 1.0))]
    rows = []
    for _, r in m.iterrows():
        v = _value(r["standard_value"])
        if not np.isfinite(v) or not (EST_LO <= v <= EST_HI):
            continue
        for n in _names(r["compound.names"]):
            if n and n.lower() != "nan":
                rows.append((n, v))
    return pd.DataFrame(rows, columns=["molecule", "v"])


def dedup_by_name(long: pd.DataFrame) -> pd.DataFrame:
    """Median ΔE_ST per molecule name, with report count and inter-report SD."""
    g = long.groupby("molecule")["v"]
    out = pd.DataFrame({"exp_est": g.median(), "n_reports": g.size(), "sd": g.std().fillna(0.0)})
    return out.reset_index()


# ---- SMILES resolution tiers (in order, first hit wins) --------------------

def tier_dataref() -> dict[str, str]:
    """Data_Ref.csv curated name→SMILES mapping."""
    ref = pd.read_csv(ROOT / "Data_Ref.csv")
    col = "compound.SMILES" if "compound.SMILES" in ref.columns else ref.columns[-1]
    return {str(m): str(s) for m, s in zip(ref["Molecule_id"], ref[col]) if pd.notna(s)}


_IUPAC_RE = re.compile(
    r"(carbazol|phenyl|phenoxazin|phenothiazin|triazin|pyrimidin|pyridin|"
    r"benzene|methanone|benzonitrile|acridin|thioxanthen|dibenzothiophen|"
    r"diphenylphosphine|diphenylamino|dimethyl|tert-butyl|dihydro)",
    re.IGNORECASE)


def _is_iupac_like(name: str) -> bool:
    """Heuristic: long systematic IUPAC names should go to OPSIN, not PubChem."""
    return len(name) > 25 and bool(_IUPAC_RE.search(name))


def tier_pubchem(names: list[str]) -> dict[str, str]:
    """Name → canonical SMILES via PubChem PUG-REST (only for short/trivial names)."""
    import urllib.parse, urllib.request, socket
    out: dict[str, str] = {}
    # Only query names that are NOT IUPAC-like (OPSIN handles those better)
    short_names = [n for n in names if not _is_iupac_like(n)]
    n_total = len(short_names)
    n_skipped = len(names) - n_total
    if n_skipped:
        print(f"  (Skipped {n_skipped} IUPAC-like names → OPSIN)")
    t_start = time.time()
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(4)

    for i, nm in enumerate(short_names, 1):
        try:
            url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/TXT"
            with urllib.request.urlopen(url.format(urllib.parse.quote(nm)), timeout=4) as r:
                smi = r.read().decode().strip().splitlines()[0].strip()
            if smi and valid_smiles(smi):
                out[nm] = smi
        except Exception:
            pass
        if i % 200 == 0 or i == n_total:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_total - i) / rate if rate > 0 else 0
            print(f"  PubChem: {i}/{n_total} ({len(out)} hits, {rate:.1f}/s, "
                  f"ETA {eta/60:.1f}min)")

    socket.setdefaulttimeout(old_timeout)
    return out


_OPSIN_JARS = [
    Path("/tmp/opsin_extract/usr/share/java/opsin-cli-2.8.0.jar"),
    Path("/tmp/opsin_extract/usr/share/java/opsin-core-2.8.0.jar"),
    Path("/tmp/woodstox.jar"),
]
_OPSIN_CP = ":".join(str(j) for j in _OPSIN_JARS if j.exists())


def tier_opsin(names: list[str], batch_size: int = 100) -> dict[str, str]:
    """IUPAC name → SMILES via OPSIN (Java, offline batch)."""
    if not shutil.which("java") or not all(j.exists() for j in _OPSIN_JARS):
        return {}
    out: dict[str, str] = {}
    banner = "Run the jar using the -h flag for help. Enter a chemical name to begin:"
    for start in range(0, len(names), batch_size):
        batch = names[start:start + batch_size]
        inp = "\n".join(batch)
        try:
            p = subprocess.run(
                ["java", "-cp", _OPSIN_CP, "uk.ac.cam.ch.wwmm.opsin.Cli", "-o", "smi"],
                input=inp, capture_output=True, text=True, timeout=120)
            outputs = [l for l in p.stdout.splitlines()
                       if l.strip() and l.strip() != banner]
            for nm, smi in zip(batch, outputs):
                smi = smi.strip()
                if smi and valid_smiles(smi):
                    out[nm] = smi
        except Exception:
            pass
    return out


def valid_smiles(s) -> bool:
    return bool(s) and pd.notna(s) and Chem.MolFromSmiles(str(s)) is not None


# Elements found in organic TADF emitters (incl. B for boron-MR-TADF, Si).
_TADF_ALLOWED_ELEMENTS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}


def is_tadf_candidate(s, hac_min: int = 22) -> bool:
    """Reject non-emitter junk admitted by name→SMILES mis-resolution.

    Added 2026-07-06 after F1-B QC found metals ([Co],[Pt],[U]), salts
    ([Li+].[Cl-]), solvents (water, ethanol) and simple reagents (bromobenzene,
    amino acids) resolved from TADF compound names. A valid donor–acceptor
    emitter candidate is a SINGLE connected organic fragment, built only from
    _TADF_ALLOWED_ELEMENTS, with >= hac_min heavy atoms. The 231-molecule
    training set has min HAC 20 / median 50, so 22 is a safe floor.
    """
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        return False
    if len(Chem.GetMolFrags(mol)) != 1:          # salts / counterions / complexes
        return False
    els = {a.GetSymbol() for a in mol.GetAtoms()}
    if "*" in els or not els <= _TADF_ALLOWED_ELEMENTS:  # attachment points / metals
        return False
    return mol.GetNumHeavyAtoms() >= hac_min


def try_resolve(todo_names: list[str], resolver_fn, label: str) -> tuple[dict[str, str], int]:
    """Try resolver; return (new_smiles_map, newly_resolved_count)."""
    print(f"  Tier '{label}': trying {len(todo_names)} unresolved names...")
    new = resolver_fn(todo_names)
    valid = {nm: smi for nm, smi in new.items() if valid_smiles(smi)}
    print(f"    → resolved {len(valid)} valid SMILES")
    return valid, len(valid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true",
                    help="enable PubChem resolution (network, rate-limited)")
    ap.add_argument("--no-dataref", action="store_true",
                    help="skip Data_Ref.csv tier (debug)")
    args = ap.parse_args()

    print("=" * 60)
    print("F1-A: SMILES resolution for master_database STSplit records")
    print("=" * 60)

    # Step 1: load experimental STSplit records
    long = load_stsplit_experimental()
    mol = dedup_by_name(long)
    print(f"\nLoaded {len(long)} records → {len(mol)} unique name-entries")
    print(f"  ΔE_ST range: {mol['exp_est'].min():.3f} – {mol['exp_est'].max():.3f} eV")
    print(f"  ΔE_ST ≤ 1.0 eV: {(mol['exp_est'] <= 1.0).sum()} entries")

    # Filter to the target ΔE_ST ≤ 1.0 eV
    mol_target = mol[mol["exp_est"] <= 1.0].copy()
    print(f"  Target subset (≤1.0 eV): {len(mol_target)} entries\n")

    # Step 2: resolve SMILES
    tier_counts = {}
    smiles_map: dict[str, str] = {}

    # Tier 0: Data_Ref (offline cache)
    if not args.no_dataref:
        dr = tier_dataref()
        for m in mol_target.molecule:
            if m in dr and valid_smiles(dr[m]):
                smiles_map[m] = dr[m]
        tier_counts["dataref"] = len(smiles_map)
        print(f"  After Data_Ref: {len(smiles_map)} resolved")

    # Tier 1: SMILES_molecules.csv (already-featurised molecules)
    sm_path = ROOT / "SMILES_molecules.csv"
    if sm_path.exists():
        sm = pd.read_csv(sm_path)
        id2smi = dict(zip(sm.molecule_id1.astype(str), sm["compound.SMILES"].astype(str)))
        todo = [m for m in mol_target.molecule if m not in smiles_map]
        n_new = 0
        for m in todo:
            if m in id2smi and valid_smiles(id2smi[m]):
                smiles_map[m] = id2smi[m]
                n_new += 1
        tier_counts["smiles_csv"] = n_new
        print(f"  After SMILES_molecules.csv: {len(smiles_map)} resolved (+{n_new})")
    else:
        tier_counts["smiles_csv"] = 0

    # Tier 2: Name normalization match against all known SMILES sources
    todo = [m for m in mol_target.molecule if m not in smiles_map]
    if todo:
        # Build normalized name→SMILES map from all sources
        all_src = {}
        dr2 = pd.read_csv(ROOT / "Data_Ref.csv")
        for _, r in dr2.iterrows():
            all_src[re.sub(r"[\s\-()\[\]{}]", "", str(r["Molecule_id"]).strip().lower())] = str(r["compound.SMILES"])
        sm2 = pd.read_csv(ROOT / "SMILES_molecules.csv")
        for _, r in sm2.iterrows():
            all_src[re.sub(r"[\s\-()\[\]{}]", "", str(r["molecule_id1"]).strip().lower())] = str(r["compound.SMILES"])
        n_norm = 0
        for m in todo:
            key = re.sub(r"[\s\-()\[\]{}]", "", m.strip().lower())
            if key in all_src and valid_smiles(all_src[key]):
                smiles_map[m] = all_src[key]
                n_norm += 1
        tier_counts["norm_match"] = n_norm
        print(f"  After name normalization: {len(smiles_map)} resolved (+{n_norm})")
    else:
        tier_counts["norm_match"] = 0

    # Tier 3: PubChem (primary online)
    if args.online:
        todo = [m for m in mol_target.molecule if m not in smiles_map]
        print(f"  Starting PubChem for {len(todo)} names...")
        new_map, n = try_resolve(todo, tier_pubchem, "PubChem")
        smiles_map.update(new_map)
        tier_counts["pubchem"] = n
        print(f"  After PubChem: {len(smiles_map)} resolved")
    else:
        tier_counts["pubchem"] = 0
        print("  PubChem skipped (re-run with --online for network resolution)")

    # Tier 3: OPSIN (IUPAC fallback)
    todo = [m for m in mol_target.molecule if m not in smiles_map]
    if todo:
        new_map, n = try_resolve(todo, tier_opsin, "OPSIN")
        smiles_map.update(new_map)
        tier_counts["opsin"] = n
        print(f"  After OPSIN: {len(smiles_map)} resolved")
    else:
        tier_counts["opsin"] = 0

    # Step 3: build resolved DataFrame
    if len(smiles_map) == 0:
        print("\nNo SMILES resolved in offline tiers. Need --online for PubChem.")
        sys.exit(0)
    mol_target["smiles"] = mol_target.molecule.map(smiles_map)
    resolved = mol_target[mol_target.smiles.notna()].copy()

    # TADF-validity filter: drop non-emitter junk from name mis-resolution
    # (metals/salts/solvents/reagents/fragments). See is_tadf_candidate().
    n_before_tadf = len(resolved)
    resolved = resolved[resolved["smiles"].map(is_tadf_candidate)].copy()
    n_tadf_dropped = n_before_tadf - len(resolved)
    if n_tadf_dropped:
        print(f"  TADF-validity filter: dropped {n_tadf_dropped} non-emitter "
              f"resolutions (metals/salts/solvents/fragments, HAC<22)")

    # Step 4: deduplicate by InChIKey
    def inchikey(smi: str) -> str | None:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            return Chem.MolToInchiKey(mol)
        except Exception:
            return None

    resolved["inchikey"] = resolved["smiles"].apply(inchikey)
    n_no_ik = resolved["inchikey"].isna().sum()
    if n_no_ik > 0:
        print(f"\n  Warning: {n_no_ik} resolved molecules could not generate InChIKey (dropped)")
        resolved = resolved.dropna(subset=["inchikey"])

    # Group by InChIKey: keep the entry with the most reports; break ties by lowest exp_est
    resolved["rank"] = resolved.groupby("inchikey")["n_reports"].rank(method="first", ascending=False)
    resolved_unique = resolved[resolved["rank"] == 1].copy()
    n_duplicates = len(resolved) - len(resolved_unique)

    print(f"\n  SMILES resolved: {len(resolved)} name-entries")
    print(f"  After InChIKey dedup: {len(resolved_unique)} unique molecules ({n_duplicates} merged)")
    print(f"  Of which ΔE_ST ≤ 1.0 eV: {(resolved_unique['exp_est'] <= 1.0).sum()}")

    # Step 5: check which unique molecules are already featurised
    feat_path = ROOT / "CALCULATIONS-MADE/data_processing/combined_features_747mol_full_ct.csv"
    if feat_path.exists():
        feat = pd.read_csv(feat_path)
        featurised_names = set(feat.molecule.astype(str))
        resolved_unique["already_featurised"] = resolved_unique.molecule.astype(str).isin(featurised_names)
        n_featurised = resolved_unique["already_featurised"].sum()
        n_new = len(resolved_unique) - n_featurised
    else:
        resolved_unique["already_featurised"] = False
        n_featurised = 0
        n_new = len(resolved_unique)
        print(f"\n  Warning: featurised-molecule file not found at {feat_path}")

    # Step 6: target check
    n_target = int((resolved_unique["exp_est"] <= 1.0).sum())
    target_met = bool(n_target >= 500)

    print(f"\n{'=' * 60}")
    print(f"TARGET CHECK: ≥500 unique molecules with ΔE_ST ≤ 1.0 eV")
    print(f"  Achieved: {n_target}")
    print(f"  Target met: {'✅ YES' if target_met else '❌ NO'}")

    # Summary
    summary = {
        "task": "F1-A",
        "master_database_records": int(len(long)),
        "unique_name_entries": int(len(mol)),
        "name_entries_delta_le_1eV": int(len(mol_target)),
        "resolved_tiers": tier_counts,
        "tadf_filter_dropped": int(n_tadf_dropped),
        "total_resolved_name_entries": int(len(resolved)),
        "after_inchikey_dedup": int(len(resolved_unique)),
        "delta_est_le_1eV": int(n_target),
        "target_met": target_met,
        "already_featurised": int(n_featurised),
        "needs_new_featurisation": int(n_new),
        "duplicate_merged_by_inchikey": int(n_duplicates),
    }

    # Write outputs
    DATA.mkdir(exist_ok=True)
    out_cols = ["molecule", "smiles", "inchikey", "exp_est", "n_reports", "sd"]
    if "already_featurised" in resolved_unique.columns:
        out_cols.append("already_featurised")
    resolved_unique[out_cols].to_csv(DATA / "expanded_smiles_resolved.csv", index=False)

    (DATA / "expand_dataset_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nOutput files:")
    print(f"  {DATA / 'expanded_smiles_resolved.csv'} ({len(resolved_unique)} rows)")
    print(f"  {DATA / 'expand_dataset_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
