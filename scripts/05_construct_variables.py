"""05_construct_variables.py — Extrae features acústicas de cada utterance (baseline handcrafted).

Features: MFCC (13) + delta + delta-delta; duración; RMS; ZCR; centroide/ancho espectral.
Salida: data/processed/features.csv (utt, split, feature_cols...).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

BASE = Path("data/raw/speechocean762")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
SR = 16000


def read_scp():
    rows = []
    for split in ("train", "test"):
        with open(BASE / split / "wav.scp", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rows.append((parts[0], split, parts[1]))
    return rows


def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d = librosa.feature.delta(mfcc)
    dd = librosa.feature.delta(mfcc, order=2)
    feats = {}
    for i in range(13):
        feats[f"mfcc{i}_mean"] = mfcc[i].mean()
        feats[f"mfcc{i}_std"] = mfcc[i].std()
        feats[f"dmfcc{i}_mean"] = d[i].mean()
        feats[f"ddmfcc{i}_mean"] = dd[i].mean()
    feats["duration"] = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = rms.mean()
    feats["rms_std"] = rms.std()
    feats["zcr_mean"] = librosa.feature.zero_crossing_rate(y)[0].mean()
    feats["cent_mean"] = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    feats["bw_mean"] = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    return feats


def main():
    rows = read_scp()
    print(f"utterances a procesar: {len(rows)}")
    records = []
    for idx, (utt, split, relpath) in enumerate(rows):
        feats = extract_features(BASE / relpath)
        rec = {"utt": utt, "split": split}
        rec.update(feats)
        records.append(rec)
        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(rows)}", flush=True)
    df = pd.DataFrame(records)
    df.to_csv(OUT / "features.csv", index=False)
    print(f"features.shape = {df.shape}")
    print("Guardado en data/processed/features.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
