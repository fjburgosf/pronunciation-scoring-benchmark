"""11_enhancements.py — Mejoras A, B, C sobre RQ1 (scoring).

A. IC bootstrap 95% de PCC (baseline, deep, fusion) + test de significancia
   (bootstrap pareado) de deep>baseline y fusion>baseline por dimension.
B. Importancia de features del baseline GBDT (top features por dimension).
C. Modelo de fusion: MFCC(58) + embeddings Wav2Vec2(768) -> Ridge.

Requiere: data/processed/features.csv, embeddings_wav2vec2.npy(+utts.json),
          data/raw/speechocean762/resource/scores.json.
Salida:   results/primary/rq1_enhancements.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

BASE = Path("data/raw/speechocean762")
PROC = Path("data/processed")
OUT = Path("results/primary")
OUT.mkdir(parents=True, exist_ok=True)
DIMS = ["accuracy", "fluency", "completeness", "prosodic", "total"]
RNG = np.random.default_rng(42)
N_BOOT = 2000


def load_targets():
    with open(BASE / "resource" / "scores.json", encoding="utf-8") as f:
        scores = json.load(f)
    rows = [{"utt": u, **{d: v.get(d) for d in DIMS}} for u, v in scores.items()]
    return pd.DataFrame(rows)


def load_embeddings():
    E = np.load(PROC / "embeddings_wav2vec2.npy")
    with open(PROC / "embeddings_wav2vec2_utts.json", encoding="utf-8") as f:
        utts = json.load(f)
    return pd.DataFrame({"utt": utts}), E


def boot_ci_pcc(true, pred, n=N_BOOT):
    idx = np.arange(len(true))
    stats = []
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        if np.std(pred[s]) < 1e-9:
            continue
        stats.append(pearsonr(true[s], pred[s])[0])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def boot_diff_p(true, pred_a, pred_b, n=N_BOOT):
    """p-value (dos colas) de que PCC(pred_b) > PCC(pred_a) via bootstrap pareado."""
    idx = np.arange(len(true))
    diffs = []
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        if np.std(pred_a[s]) < 1e-9 or np.std(pred_b[s]) < 1e-9:
            continue
        diffs.append(pearsonr(true[s], pred_b[s])[0] - pearsonr(true[s], pred_a[s])[0])
    diffs = np.array(diffs)
    frac_le0 = np.mean(diffs <= 0)
    p = 2 * min(frac_le0, 1 - frac_le0)
    return float(min(p, 1.0)), float(np.mean(diffs))


def main():
    def norm(s):
        return s.astype(str).str.lstrip("0")

    feats = pd.read_csv(PROC / "features.csv")
    feats["utt"] = norm(feats["utt"])
    feat_cols = [c for c in feats.columns if c not in ("utt", "split")]
    tgt = load_targets()
    tgt["utt"] = norm(tgt["utt"])
    emb_df, E = load_embeddings()
    emb_df["utt"] = norm(emb_df["utt"])
    emb_df["emb_idx"] = np.arange(len(emb_df))

    df = feats.merge(tgt, on="utt").merge(emb_df, on="utt")
    tr = (df["split"] == "train").values
    te = (df["split"] == "test").values
    Xf = df[feat_cols].values
    Xe = E[df["emb_idx"].values]

    results = {"bootstrap_ci": {}, "significance": {}, "fusion_metrics": {},
               "feature_importance": {}, "config": {"n_boot": N_BOOT, "seed": 42}}

    for d in DIMS:
        y = df[d].values
        ytr, yte = y[tr], y[te]

        # --- Baseline GBDT (B: con importancias) ---
        gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=0)
        gb.fit(Xf[tr], ytr)
        pred_base = gb.predict(Xf[te])

        # --- Deep Ridge ---
        sc_e = StandardScaler().fit(Xe[tr])
        rid_e = Ridge(alpha=1.0).fit(sc_e.transform(Xe[tr]), ytr)
        pred_deep = rid_e.predict(sc_e.transform(Xe[te]))

        # --- C: Fusion (MFCC + embeddings) -> Ridge ---
        Xcat = np.hstack([Xf, Xe])
        sc_c = StandardScaler().fit(Xcat[tr])
        rid_c = Ridge(alpha=1.0).fit(sc_c.transform(Xcat[tr]), ytr)
        pred_fus = rid_c.predict(sc_c.transform(Xcat[te]))
        results["fusion_metrics"][d] = {
            "PCC": round(float(pearsonr(yte, pred_fus)[0]), 4),
            "MSE": round(float(mean_squared_error(yte, pred_fus)), 4),
            "MAE": round(float(mean_absolute_error(yte, pred_fus)), 4),
        }

        # --- A: IC bootstrap ---
        results["bootstrap_ci"][d] = {
            "baseline_PCC_CI": [round(x, 4) for x in boot_ci_pcc(yte, pred_base)],
            "deep_PCC_CI": [round(x, 4) for x in boot_ci_pcc(yte, pred_deep)],
            "fusion_PCC_CI": [round(x, 4) for x in boot_ci_pcc(yte, pred_fus)],
        }
        p_dvb, dm_dvb = boot_diff_p(yte, pred_base, pred_deep)
        p_fvb, dm_fvb = boot_diff_p(yte, pred_base, pred_fus)
        results["significance"][d] = {
            "deep_vs_baseline": {"mean_delta_PCC": round(dm_dvb, 4), "p": round(p_dvb, 4)},
            "fusion_vs_baseline": {"mean_delta_PCC": round(dm_fvb, 4), "p": round(p_fvb, 4)},
        }

        # --- B: importancia de features (top 8) ---
        imp = sorted(zip(feat_cols, gb.feature_importances_), key=lambda t: -t[1])[:8]
        results["feature_importance"][d] = [{"feature": f, "importance": round(float(w), 4)} for f, w in imp]

        print(f"[{d}] fusion PCC={results['fusion_metrics'][d]['PCC']} | "
              f"deep vs base p={p_dvb} | fusion vs base p={p_fvb}", flush=True)

    with open(OUT / "rq1_enhancements.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Guardado: results/primary/rq1_enhancements.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
