"""13_epadb.py — Mejora D (tercer corpus): generalizacion a EpaDB (L1 espanol).

EpaDB = ingles leido por 50 hablantes nativos de espanol (Argentina), con
transcripcion IPA real por expertos (ipa) y pronunciacion canonica (g2p).
Se entrena el scorer en TODO speechocean762 (mandarin, 0-10) y se aplica al
audio de EpaDB. Validacion por convergencia: el puntaje predicho (accuracy/
total) debe correlacionar NEGATIVAMENTE con la tasa de error fonetica real
(distancia de edicion IPA vs g2p, normalizada por longitud canonica).

Requiere: HF_TOKEN con acceso a KoelLabs/EpaDB.
Salida:   results/primary/rq1_epadb.json
"""
import io
import json
import os
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
REPO = "KoelLabs/EpaDB"
EPA_FILES = ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]
FEATS_TO_USE = ["accuracy", "total"]


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_targets():
    with open(BASE / "resource" / "scores.json", encoding="utf-8") as f:
        scores = json.load(f)
    return pd.DataFrame([{"utt": u, **{d: v.get(d) for d in DIMS}} for u, v in scores.items()])


def mfcc_feats(y, feat_cols):
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
    return [f[c] for c in feat_cols]


def main():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("ERROR: falta HF_TOKEN con acceso a KoelLabs/EpaDB"); return 1

    device = torch.device("cpu")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device).eval()

    # --- 1. Entrenar scorer en speechocean762 ---
    norm = lambda s: s.astype(str).str.lstrip("0")
    feats = pd.read_csv(PROC / "features.csv"); feats["utt"] = norm(feats["utt"])
    feat_cols = [c for c in feats.columns if c not in ("utt", "split")]
    tgt = load_targets(); tgt["utt"] = norm(tgt["utt"])
    E = np.load(PROC / "embeddings_wav2vec2.npy")
    utts = json.load(open(PROC / "embeddings_wav2vec2_utts.json", encoding="utf-8"))
    emb_df = pd.DataFrame({"utt": norm(pd.Series([str(u) for u in utts])), "emb_idx": np.arange(len(utts))})
    tr = feats.merge(tgt, on="utt").merge(emb_df, on="utt")
    Xf_tr = tr[feat_cols].values; Xe_tr = E[tr["emb_idx"].values]
    Xc_tr = np.hstack([Xf_tr, Xe_tr])
    MODELS = ["baseline", "deep", "fusion"]
    models = {}
    for d in FEATS_TO_USE:
        gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=0).fit(Xf_tr, tr[d].values)
        sc_e = StandardScaler().fit(Xe_tr); rid = Ridge(alpha=1.0).fit(sc_e.transform(Xe_tr), tr[d].values)
        sc_c = StandardScaler().fit(Xc_tr); fus = Ridge(alpha=1.0).fit(sc_c.transform(Xc_tr), tr[d].values)
        models[d] = {"gb": gb, "sc_e": sc_e, "rid": rid, "sc_c": sc_c, "fus": fus}
    print(f"[train] 3 scorers (baseline/deep/fusion) entrenados en {len(tr)} enunciados speechocean762", flush=True)

    # --- 2. Cargar EpaDB con audio ---
    frames = []
    for fname in EPA_FILES:
        frames.append(pd.read_parquet(hf_hub_download(REPO, fname, repo_type="dataset", token=tok)))
    epa = pd.concat(frames, ignore_index=True)
    epa["err_rate"] = [levenshtein(str(a), str(g)) / max(len(str(g)), 1)
                       for a, g in zip(epa["ipa"], epa["g2p"])]
    print(f"[epadb] {len(epa)} enunciados, {epa['speaker_code'].nunique()} hablantes espanol", flush=True)

    # --- 3. Predecir scores (3 modelos) por enunciado ---
    preds = {mdl: {d: [] for d in FEATS_TO_USE} for mdl in MODELS}; keep = []
    with torch.no_grad():
        for i in range(len(epa)):
            try:
                y, sr = sf.read(io.BytesIO(epa["audio"].iloc[i]["bytes"]))
                if getattr(y, "ndim", 1) > 1: y = y.mean(axis=1)
                if sr != SR: y = librosa.resample(y.astype(float), orig_sr=sr, target_sr=SR)
                y = y.astype("float32")
                if len(y) < SR * 0.2: continue
                xf = np.array([mfcc_feats(y, feat_cols)])
                inp = extractor(y, sampling_rate=SR, return_tensors="pt", padding=False)
                xe = model(inp.input_values.to(device)).last_hidden_state.mean(dim=1).cpu().numpy()
                xc = np.hstack([xf, xe])
                for d in FEATS_TO_USE:
                    m = models[d]
                    preds["baseline"][d].append(float(m["gb"].predict(xf)[0]))
                    preds["deep"][d].append(float(m["rid"].predict(m["sc_e"].transform(xe))[0]))
                    preds["fusion"][d].append(float(m["fus"].predict(m["sc_c"].transform(xc))[0]))
                keep.append(i)
            except Exception:
                continue
            if len(keep) % 300 == 0:
                print(f"  procesados {len(keep)}/{len(epa)}", flush=True)

    sub = epa.iloc[keep].reset_index(drop=True)
    for mdl in MODELS:
        for d in FEATS_TO_USE:
            sub[f"pred_{mdl}_{d}"] = preds[mdl][d]

    # --- 4. Convergencia: pred_score vs err_rate, por modelo (utt y hablante) ---
    results = {"corpus": "EpaDB", "n_scored": int(len(sub)),
               "n_speakers": int(sub["speaker_code"].nunique()),
               "train_n": int(len(tr)), "by_model": {}}
    agg = {f"pred_{mdl}_{d}": (f"pred_{mdl}_{d}", "mean") for mdl in MODELS for d in FEATS_TO_USE}
    spk = sub.groupby("speaker_code").agg(err_rate=("err_rate", "mean"), **agg).reset_index()
    for mdl in MODELS:
        results["by_model"][mdl] = {"by_dimension": {}}
        for d in FEATS_TO_USE:
            col = f"pred_{mdl}_{d}"
            r_u, p_u = pearsonr(sub[col], sub["err_rate"])
            rho_u, _ = spearmanr(sub[col], sub["err_rate"])
            r_s, p_s = pearsonr(spk[col], spk["err_rate"])
            results["by_model"][mdl]["by_dimension"][d] = {
                "utterance_level": {"pearson_r": round(float(r_u), 4), "p": round(float(p_u), 6),
                                     "spearman_rho": round(float(rho_u), 4)},
                "speaker_level": {"pearson_r": round(float(r_s), 4), "p": round(float(p_s), 6),
                                   "n_speakers": int(len(spk))},
            }
            print(f"[{mdl}/{d}] utt: r={r_u:.3f} (p={p_u:.2e}) | speaker: r={r_s:.3f} (p={p_s:.3f})", flush=True)

    with open(OUT / "rq1_epadb.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Guardado: results/primary/rq1_epadb.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
