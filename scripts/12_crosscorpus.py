"""12_crosscorpus.py — Mejora D: generalizacion cross-corpus.

Entrena el scorer en TODO speechocean762 (mandarin, sentence-level 0-10) y lo
aplica al audio de L2-ARCTIC (6 L1s, incl. espanol). Como L2-ARCTIC no tiene
scores 0-10 sino conteos de error por fonema, se valida por CONVERGENCIA: los
puntajes predichos (accuracy/total) deben correlacionar NEGATIVAMENTE con la
tasa de error real por enunciado. Se reporta por L1 y en especial para espanol.

Requiere: features.csv + embeddings_wav2vec2.npy (speechocean762, para entrenar),
          audio de L2-ARCTIC (parquet HF, se decodifica aqui).
Salida:   results/primary/rq1_crosscorpus.json
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

BASE = Path("data/raw/speechocean762")
PROC = Path("data/processed")
OUT = Path("results/primary")
OUT.mkdir(parents=True, exist_ok=True)
DIMS = ["accuracy", "fluency", "completeness", "prosodic", "total"]
SR = 16000
MODEL_NAME = "facebook/wav2vec2-base"
REPO = "chikingsley/l2-arctic-manual-v5.0-16k"
L2_FILES = ["data/train-00000-of-00001.parquet",
            "data/validation-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
            "data/suitcase-00000-of-00001.parquet"]
FEATS_TO_USE = ["accuracy", "total"]  # dimensiones con senal de pronunciacion


def load_targets():
    with open(BASE / "resource" / "scores.json", encoding="utf-8") as f:
        scores = json.load(f)
    return pd.DataFrame([{"utt": u, **{d: v.get(d) for d in DIMS}} for u, v in scores.items()])


def mfcc_feats_from_audio(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    d = librosa.feature.delta(mfcc); dd = librosa.feature.delta(mfcc, order=2)
    f = {}
    for i in range(13):
        f[f"mfcc{i}_mean"] = mfcc[i].mean(); f[f"mfcc{i}_std"] = mfcc[i].std()
        f[f"dmfcc{i}_mean"] = d[i].mean(); f[f"ddmfcc{i}_mean"] = dd[i].mean()
    f["duration"] = librosa.get_duration(y=y, sr=SR)
    rms = librosa.feature.rms(y=y)[0]
    f["rms_mean"] = rms.mean(); f["rms_std"] = rms.std()
    f["zcr_mean"] = librosa.feature.zero_crossing_rate(y)[0].mean()
    f["cent_mean"] = librosa.feature.spectral_centroid(y=y, sr=SR)[0].mean()
    f["bw_mean"] = librosa.feature.spectral_bandwidth(y=y, sr=SR)[0].mean()
    return f


def main():
    device = torch.device("cpu")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device).eval()

    # --- 1. Entrenar scorer en TODO speechocean762 ---
    norm = lambda s: s.astype(str).str.lstrip("0")
    feats = pd.read_csv(PROC / "features.csv"); feats["utt"] = norm(feats["utt"])
    feat_cols = [c for c in feats.columns if c not in ("utt", "split")]
    tgt = load_targets(); tgt["utt"] = norm(tgt["utt"])
    E = np.load(PROC / "embeddings_wav2vec2.npy")
    utts = json.load(open(PROC / "embeddings_wav2vec2_utts.json", encoding="utf-8"))
    emb_df = pd.DataFrame({"utt": [str(u) for u in utts], "emb_idx": np.arange(len(utts))})
    emb_df["utt"] = norm(emb_df["utt"])
    tr = feats.merge(tgt, on="utt").merge(emb_df, on="utt")
    Xf_tr = tr[feat_cols].values
    Xe_tr = E[tr["emb_idx"].values]

    # Tres modelos por dimension: baseline (GBDT sobre MFCC), deep (Ridge sobre
    # embeddings) y fusion (Ridge sobre MFCC+embeddings concatenados).
    MODELS = ["baseline", "deep", "fusion"]
    Xc_tr = np.hstack([Xf_tr, Xe_tr])
    models = {}
    for d in FEATS_TO_USE:
        gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=0).fit(Xf_tr, tr[d].values)
        sc_e = StandardScaler().fit(Xe_tr); rid = Ridge(alpha=1.0).fit(sc_e.transform(Xe_tr), tr[d].values)
        sc_c = StandardScaler().fit(Xc_tr); fus = Ridge(alpha=1.0).fit(sc_c.transform(Xc_tr), tr[d].values)
        models[d] = {"gb": gb, "sc_e": sc_e, "rid": rid, "sc_c": sc_c, "fus": fus}
    print(f"[train] 3 scorers (baseline/deep/fusion) entrenados en {len(tr)} enunciados speechocean762", flush=True)

    # --- 2. Cargar L2-ARCTIC con audio ---
    frames = []
    for fname in L2_FILES:
        p = hf_hub_download(REPO, fname, repo_type="dataset")
        frames.append(pd.read_parquet(p))
    l2 = pd.concat(frames, ignore_index=True)
    l2["L1"] = l2["native_language"].str.lower()
    l2["err_rate"] = (l2["num_substitutions"] + l2["num_deletions"] + l2["num_additions"]) / l2["num_phonemes"]
    print(f"[l2arctic] {len(l2)} enunciados, L1s: {sorted(l2['L1'].unique())}", flush=True)

    # --- 3. Predecir scores (3 modelos) para cada enunciado L2-ARCTIC ---
    preds = {mdl: {d: [] for d in FEATS_TO_USE} for mdl in MODELS}
    keep_idx = []
    with torch.no_grad():
        for i, row in enumerate(l2.itertuples(index=False)):
            try:
                y, sr = sf.read(io.BytesIO(row.audio["bytes"]))
                if y.ndim > 1: y = y.mean(axis=1)
                if sr != SR: y = librosa.resample(y.astype(float), orig_sr=sr, target_sr=SR)
                y = y.astype("float32")
                if len(y) < SR * 0.2: continue
                xf = np.array([[mfcc_feats_from_audio(y)[c] for c in feat_cols]])
                inp = extractor(y, sampling_rate=SR, return_tensors="pt", padding=False)
                h = model(inp.input_values.to(device)).last_hidden_state
                xe = h.mean(dim=1).cpu().numpy()
                xc = np.hstack([xf, xe])
                for d in FEATS_TO_USE:
                    m = models[d]
                    preds["baseline"][d].append(float(m["gb"].predict(xf)[0]))
                    preds["deep"][d].append(float(m["rid"].predict(m["sc_e"].transform(xe))[0]))
                    preds["fusion"][d].append(float(m["fus"].predict(m["sc_c"].transform(xc))[0]))
                keep_idx.append(l2.index[i])
            except Exception as e:
                continue
            if (len(keep_idx)) % 300 == 0:
                print(f"  procesados {len(keep_idx)}/{len(l2)}", flush=True)

    sub = l2.loc[keep_idx].reset_index(drop=True)
    for mdl in MODELS:
        for d in FEATS_TO_USE:
            sub[f"pred_{mdl}_{d}"] = preds[mdl][d]

    # --- 4. Convergencia: pred_score vs err_rate, por modelo, dimension y L1 ---
    results = {"n_scored": int(len(sub)), "train_n": int(len(tr)), "by_model": {}}
    for mdl in MODELS:
        results["by_model"][mdl] = {"by_dimension": {}}
        for d in FEATS_TO_USE:
            col = f"pred_{mdl}_{d}"
            r_all, p_all = pearsonr(sub[col], sub["err_rate"])
            rho_all, _ = spearmanr(sub[col], sub["err_rate"])
            by_l1 = {}
            for lg in sorted(sub["L1"].unique()):
                s = sub[sub["L1"] == lg]
                if len(s) > 5 and s[col].std() > 1e-9:
                    r, pv = pearsonr(s[col], s["err_rate"])
                    by_l1[lg] = {"n": int(len(s)), "pearson_r": round(float(r), 4), "p": round(float(pv), 4)}
            results["by_model"][mdl]["by_dimension"][d] = {
                "pearson_r_all": round(float(r_all), 4), "p_all": round(float(p_all), 6),
                "spearman_rho_all": round(float(rho_all), 4), "by_L1": by_l1,
            }
            print(f"[{mdl}/{d}] r={r_all:.3f} (p={p_all:.2e})", flush=True)

    with open(OUT / "rq1_crosscorpus.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Guardado: results/primary/rq1_crosscorpus.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
