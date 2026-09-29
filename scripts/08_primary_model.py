"""08_primary_model.py — Modelo baseline RQ1: predecir puntajes desde features acústicas.

Entrena GradientBoostingRegressor por dimensión sobre el split train oficial,
evalúa PCC/MSE/MAE en el split test oficial.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

BASE = Path("data/raw/speechocean762")
PROC = Path("data/processed")
OUT = Path("results/primary")
OUT.mkdir(parents=True, exist_ok=True)
DIMS = ["accuracy", "fluency", "completeness", "prosodic", "total"]


def load_targets():
    import json
    with open(BASE / "resource" / "scores.json", encoding="utf-8") as f:
        scores = json.load(f)
    rows = []
    for utt, v in scores.items():
        r = {"utt": utt}
        for d in DIMS:
            r[d] = v.get(d)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    feats = pd.read_csv(PROC / "features.csv", dtype={"utt": str})
    tgt = load_targets()
    df = feats.merge(tgt, on="utt")
    feat_cols = [c for c in feats.columns if c not in ("utt", "split")]
    X = df[feat_cols]
    y = df[DIMS]
    tr = (df["split"] == "train").values
    te = (df["split"] == "test").values
    print(f"train={tr.sum()}, test={te.sum()}, n_features={len(feat_cols)}")

    results = {}
    for d in DIMS:
        model = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=42)
        model.fit(X[tr], y.loc[tr, d])
        pred = model.predict(X[te])
        true = y.loc[te, d].values
        pcc, _ = pearsonr(true, pred)
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        results[d] = {"PCC": round(float(pcc), 4), "MSE": round(float(mse), 4), "MAE": round(float(mae), 4)}
        print(f"{d:12s}: PCC={pcc:.3f}  MSE={mse:.3f}  MAE={mae:.3f}")

    with open(OUT / "rq1_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Guardado en results/primary/rq1_baseline_metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
