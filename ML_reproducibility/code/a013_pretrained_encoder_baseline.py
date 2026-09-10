#!/usr/bin/env python3
"""
a013_pretrained_encoder_baseline.py — DA-5 (L3 pass 2, 2026-09-10).

Question: does a pretrained molecular-transformer representation beat the hand-built
descriptor vectors on the exact 231-molecule experimental dEST task, under the SAME
scaffold GroupKFold(5), seed 0, used everywhere else in the study?

Protocol: frozen ChemBERTa encoder -> mean-pooled sentence embedding per SMILES ->
canonical RandomForest (identical to _dataset.rf(); the estimator used for Morgan/NTO)
AND a leakage-safe RidgeCV head fitted inside each fold. Out-of-fold predictions,
same folds as a006_equivalence_ceiling.py.

Frozen embeddings + strong downstream is the standard low-data protocol. Fine-tuning
a 77M-parameter encoder on 231 labels is guaranteed to overfit and is not a
meaningful test at this scale; that is stated in the ledger entry, not worked around
here.

Writes data/a013_pretrained_encoder.json. Compares to NTO-RF 0.096 / Morgan-RF 0.091.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from _dataset import load_dataset, make_features, rf  # noqa: E402

from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

SEED, N_SPLITS, N_BOOT = 0, 5, 2000
rng = np.random.default_rng(SEED)
ROOT = Path(__file__).parent.parent
CANDIDATE_MODELS = ["DeepChem/ChemBERTa-77M-MLM", "seyonec/ChemBERTa-zinc-base-v1"]


def embed(smiles, model_name):
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), 32):
            batch = [str(s) for s in smiles[i:i + 32]]
            enc = tok(batch, padding=True, truncation=True, max_length=256,
                      return_tensors="pt")
            h = mdl(**enc).last_hidden_state          # (B, T, H)
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)  # mean-pool
            out.append(pooled.cpu().numpy())
    return np.vstack(out).astype(float)


def oof_rf(X, y, cv, groups):
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        p[te] = rf().fit(X[tr], y[tr]).predict(X[te])
    return p


def oof_ridge(X, y, cv, groups):
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict(sc.transform(X[te]))
    return p


def paired_delta(y, p_a, p_b, n=N_BOOT):
    """Bootstrap MAE(b) - MAE(a). Positive => a better than b. Paired on molecules."""
    ea, eb = np.abs(y - p_a), np.abs(y - p_b)
    obs = eb.mean() - ea.mean()
    idx = rng.integers(0, len(y), size=(n, len(y)))
    d = eb[idx].mean(1) - ea[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(delta_MAE=float(obs), CI=[float(lo), float(hi)],
                frac_boot_a_better=float((d > 0).mean()))


def scores(y, p):
    return dict(MAE=float(mean_absolute_error(y, p)), R2=float(r2_score(y, p)),
                rho=float(spearmanr(y, p).statistic))


def main():
    ds = load_dataset()
    y, g = ds.y, ds.scaf
    n = len(y)
    cv = GroupKFold(n_splits=N_SPLITS)

    Xm, _ = make_features(ds, "Morgan")
    Xn, _ = make_features(ds, "NTO")
    pm = oof_rf(Xm, y, cv, g)
    pn = oof_rf(Xn, y, cv, g)

    model_used, emb, err = None, None, None
    for name in CANDIDATE_MODELS:
        try:
            emb = embed(ds.smiles, name)
            model_used = name
            break
        except Exception as e:  # pragma: no cover - network/model fallback
            err = f"{name}: {type(e).__name__}: {e}"
    if emb is None:
        raise RuntimeError(f"no encoder available. last error: {err}")

    p_enc_rf = oof_rf(emb, y, cv, g)
    p_enc_ridge = oof_ridge(emb, y, cv, g)

    res = dict(
        meta=dict(script="code/a013_pretrained_encoder_baseline.py", seed=SEED,
                  n_splits=N_SPLITS, n_molecules=int(n), n_scaffolds=int(len(set(g))),
                  encoder=model_used, embedding_dim=int(emb.shape[1]),
                  pooling="mean over attention mask, frozen encoder",
                  cv="GroupKFold on Bemis-Murcko scaffolds (identical folds to A-006)",
                  downstream="canonical rf() 400 trees seed 0; RidgeCV fitted in-fold",
                  note=("frozen-embedding protocol; fine-tuning a pretrained encoder "
                        "on 231 labels is not a meaningful capacity test at this scale")),
        reference=dict(morgan_rf=scores(y, pm), nto_rf=scores(y, pn)),
        pretrained_encoder=dict(
            rf_head=scores(y, p_enc_rf),
            ridge_head=scores(y, p_enc_ridge),
        ),
        paired_vs_morgan=dict(
            encoder_rf_minus_morgan=paired_delta(y, p_enc_rf, pm),
            wilcoxon_p=float(wilcoxon(np.abs(y - p_enc_rf), np.abs(y - pm)).pvalue),
        ),
        paired_vs_nto=dict(
            encoder_rf_minus_nto=paired_delta(y, p_enc_rf, pn),
        ),
        verdict=None,
    )
    best_enc = min(res["pretrained_encoder"]["rf_head"]["MAE"],
                   res["pretrained_encoder"]["ridge_head"]["MAE"])
    ci = res["paired_vs_morgan"]["encoder_rf_minus_morgan"]["CI"]
    res["verdict"] = (
        "pretrained encoder does NOT beat the hand-built descriptors: "
        f"best encoder MAE {best_enc:.4f} eV vs Morgan-RF {res['reference']['morgan_rf']['MAE']:.4f} eV, "
        f"NTO-RF {res['reference']['nto_rf']['MAE']:.4f} eV; "
        f"paired encoder-RF minus Morgan CI [{ci[0]:+.4f}, {ci[1]:+.4f}] eV"
        if best_enc >= res["reference"]["morgan_rf"]["MAE"] - 0.002 else
        f"pretrained encoder MAY beat descriptors: best encoder MAE {best_enc:.4f} eV "
        f"vs Morgan-RF {res['reference']['morgan_rf']['MAE']:.4f} eV -- inspect CI before claiming"
    )

    outp = ROOT / "data" / "a013_pretrained_encoder.json"
    outp.write_text(json.dumps(res, indent=2))
    print(f"encoder: {model_used}  dim {emb.shape[1]}")
    print(f"  Morgan-RF   MAE {res['reference']['morgan_rf']['MAE']:.4f}  "
          f"R2 {res['reference']['morgan_rf']['R2']:+.3f}  rho {res['reference']['morgan_rf']['rho']:.3f}")
    print(f"  NTO-RF      MAE {res['reference']['nto_rf']['MAE']:.4f}  "
          f"R2 {res['reference']['nto_rf']['R2']:+.3f}  rho {res['reference']['nto_rf']['rho']:.3f}")
    print(f"  encoder+RF  MAE {res['pretrained_encoder']['rf_head']['MAE']:.4f}  "
          f"R2 {res['pretrained_encoder']['rf_head']['R2']:+.3f}  rho {res['pretrained_encoder']['rf_head']['rho']:.3f}")
    print(f"  encoder+Ridge MAE {res['pretrained_encoder']['ridge_head']['MAE']:.4f}  "
          f"R2 {res['pretrained_encoder']['ridge_head']['R2']:+.3f}  rho {res['pretrained_encoder']['ridge_head']['rho']:.3f}")
    print(f"  paired enc-RF - Morgan: {res['paired_vs_morgan']['encoder_rf_minus_morgan']['delta_MAE']:+.4f} eV "
          f"CI {res['paired_vs_morgan']['encoder_rf_minus_morgan']['CI']}")
    print(res["verdict"])
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
