#!/usr/bin/env python3
"""Generate two SI figures for PLAN260630 Group 2 (F6-E, F10-D).

Reads only committed data files; writes only to digital_discovery_manuscript/figures/.
  - t1_t2_distribution.pdf         (F6-E): full 231-mol sTDA T1-T2 gap histogram
  - ist_delta_est_distribution.pdf (F10-D): experimental gaps of the inverted-gap candidates

No fabricated values: every number is read from disk. The IST CSV has no resolved
SMILES, so no scaffold figure is produced (would require fabricating structures).
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "digital_discovery_manuscript" / "figures"

plt.rcParams.update({"font.size": 10})


def load_col(path, col, cast=float):
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            v = r.get(col)
            if v not in (None, "", "nan"):
                try:
                    out.append(cast(v))
                except ValueError:
                    pass
    return out


def fig_t1t2():
    gaps = load_col(DATA / "t1_t2_full.csv", "T1_T2_gap_eV")
    assert len(gaps) == 231, f"expected 231 gaps, got {len(gaps)}"
    below = sum(g < 0.3 for g in gaps) / len(gaps)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(gaps, bins=30, range=(0, 2.0), color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(0.3, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"0.3 eV threshold ({below*100:.0f}% below)")
    ax.set_xlabel(r"sTDA-xTB $T_1$--$T_2$ gap (eV)")
    ax.set_ylabel("Count (of 231 emitters)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGS / "t1_t2_distribution.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}  (n=231, {below*100:.1f}% < 0.3 eV)")


def fig_ist():
    gaps = load_col(DATA / "ist_candidates.csv", "delta_est_eV")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(gaps, bins=20, color="#55A868", edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color="#4C72B0", linestyle="--", linewidth=1.2, label="Hund's-rule boundary")
    ax.set_xlabel(r"Reported experimental $\Delta E_\mathrm{ST}$ (eV)")
    ax.set_ylabel(f"Count (of {len(gaps)} records)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGS / "ist_delta_est_distribution.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}  (n={len(gaps)}, min {min(gaps):.3f} max {max(gaps):.3f})")


if __name__ == "__main__":
    fig_t1t2()
    fig_ist()
