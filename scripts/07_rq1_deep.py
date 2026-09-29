"""07_rq1_deep.py — RQ1 con embeddings deep (Wav2Vec2) + regresión Ridge.

Extrae embeddings mean-pooled del último hidden state de Wav2Vec2-base (768-dim)
y entrena Ridge por dimensión. Compara con el baseline MFCC+GBDT (PCC 0.58-0.69).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import librosa
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

BASE = Path("data/raw/speechocean762")
PROC = Path("data/processed")
OUT = Path("results/primary")
OUT.mkdir(parents=True, exist_ok=True)
LOG = Path("logs")
LOG.mkdir(exist_ok=True)
LOG_FILE = LOG / "07_rq1_deep.log"
DIMS = ["accuracy", "fluency", "completeness", "prosodic", "total"]
SR = 16000
MODEL_NAME = "facebook/wav2vec2-base"


def log_progress(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as lf:
        lf.write(msg + "\n")


def read_scp():
    rows = []
    for split in ("train", "test"):
        with open(BASE / split / "wav.scp", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rows.append((parts[0], split, parts[1]))
    return rows


def load_targets():
    with open(BASE / "resource" / "scores.json", encoding="utf-8") as f:
        scores = json.load(f)
    rows = []
    for utt, v in scores.items():
        r = {"utt": utt}
        for d in DIMS:
            r[d] = v.get(d)
        rows.append(r)
    return pd.DataFrame(rows)


def load_checkpoint():
    npy = PROC / "embeddings_wav2vec2.npy"
    utts = PROC / "embeddings_wav2vec2_utts.json"
    if npy.exists() and utts.exists():
        E = np.load(npy)
        with open(utts, encoding="utf-8") as f:
            done = json.load(f)
        return list(E), done
    return [], []


def save_checkpoint(embeds, done):
    E = np.stack(embeds)
    tmp = PROC / "embeddings_wav2vec2_tmp.npy"
    np.save(tmp, E)
    tmp.replace(PROC / "embeddings_wav2vec2.npy")
    with open(PROC / "embeddings_wav2vec2_utts.json", "w", encoding="utf-8") as f:
        json.dump(done, f)


def main():
    device = torch.device("cpu")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device).eval()
    rows = read_scp()
    print(f"utterances: {len(rows)}")
    # limpiar log previo
    LOG_FILE.write_text("", encoding="utf-8")
    log_progress(f"[start] modelo={MODEL_NAME}, utterances={len(rows)}")

    embeds, done = load_checkpoint()
    done_set = set(done)
    if done:
        log_progress(f"[resume] ya procesados {len(done)}/{len(rows)}")

    with torch.no_grad():
        for idx, (utt, split, relpath) in enumerate(rows):
            if utt in done_set:
                continue
            y, sr = librosa.load(BASE / relpath, sr=SR, mono=True)
            inputs = extractor(y, sampling_rate=SR, return_tensors="pt", padding=False)
            out = model(inputs.input_values.to(device))
            h = out.last_hidden_state  # [1, T, 768]
            emb = h.mean(dim=1).squeeze(0).cpu().numpy()
            embeds.append(emb)
            done.append(utt)
            if len(done) % 100 == 0:
                save_checkpoint(embeds, done)
                log_progress(f"  {len(done)}/{len(rows)} ({100.0*len(done)/len(rows):.1f}%)")

    save_checkpoint(embeds, done)
    log_progress(f"[done-extraction] {len(done)}/{len(rows)}")
    E = np.stack(embeds)
    log_progress(f"embeddings shape: {E.shape}")

    split_map = {u: s for u, s, _ in rows}
    feats = pd.DataFrame({"utt": done})
    feats["split"] = feats["utt"].map(split_map)
    df = feats.merge(load_targets(), on="utt")
    tr = (df["split"] == "train").values
    te = (df["split"] == "test").values

    results = {}
    for d in DIMS:
        scaler = StandardScaler().fit(E[tr])
        Xtr = scaler.transform(E[tr])
        Xte = scaler.transform(E[te])
        m = Ridge(alpha=1.0).fit(Xtr, df.loc[tr, d])
        pred = m.predict(Xte)
        true = df.loc[te, d].values
        pcc, _ = pearsonr(true, pred)
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        results[d] = {"PCC": round(float(pcc), 4), "MSE": round(float(mse), 4), "MAE": round(float(mae), 4)}
        print(f"{d:12s}: PCC={pcc:.3f}  MSE={mse:.3f}  MAE={mae:.3f}")

    with open(OUT / "rq1_deep_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Guardado en results/primary/rq1_deep_metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
