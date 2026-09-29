"""15_transfer_gap.py — Contribucion 1 (brecha de transferencia) + 2 (curva few-shot).

Tarea objetivo en EpaDB (espanol): predecir la tasa de error real por enunciado
(distancia de edicion IPA vs canonica), usando embeddings Wav2Vec2.

Se comparan, sobre un conjunto de TEST fijo de hablantes espanoles:
  - zero-shot (k=0): modelo de scoring entrenado SOLO en mandarin (speechocean762),
    sin ningun dato espanol. Se mide |PCC| entre su score predicho y la tasa de
    error real (el score alto debe implicar menos error, de ahi el valor absoluto).
  - oraculo (k=todos): Ridge entrenado en espanol (los hablantes POOL) para predecir
    la tasa de error, evaluado en TEST. Es el techo in-language.
  - few-shot (k hablantes): Ridge entrenado con k hablantes espanoles muestreados
    del POOL, evaluado en TEST, promediado sobre repeticiones. Curva k -> |PCC|.

Metricas clave: recuperacion zero-shot = |r_zs| / |r_oraculo|, y numero minimo de
hablantes espanoles para igualar al zero-shot.

Salida: results/primary/rq1_transfer_gap.json  y  figures/submission/Figure7.png
Requiere: HF_TOKEN (EpaDB gated), embeddings_wav2vec2.npy (speechocean762).
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

BASE = Path("data/raw/speechocean762")
PROC = Path("data/processed")
RES = Path("results/primary")
OUTFIG = Path("figures/submission")
RES.mkdir(parents=True, exist_ok=True); OUTFIG.mkdir(parents=True, exist_ok=True)
SR = 16000
MODEL_NAME = "facebook/wav2vec2-base"
REPO = "KoelLabs/EpaDB"
EPA_FILES = ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]
CACHE = PROC / "epadb_deep.npz"
DPI = 600
RNG = np.random.default_rng(42)
K_GRID = [1, 2, 3, 5, 10, 15, 20, 30]
N_TEST_SPK = 15   # hablantes de test por particion
R_REPEAT = 20     # repeticiones del muestreo few-shot por particion
R_SPLITS = 10     # particiones aleatorias test/pool (robustez)


def levenshtein(a, b):
    if len(a) < len(b): a, b = b, a
    if len(b) == 0: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def extract_epadb(tok):
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        print(f"[cache] EpaDB embeddings cargados: {d['X'].shape}", flush=True)
        return d["X"], d["err"], d["spk"]
    device = torch.device("cpu")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device).eval()
    frames = [pd.read_parquet(hf_hub_download(REPO, f, repo_type="dataset", token=tok)) for f in EPA_FILES]
    epa = pd.concat(frames, ignore_index=True)
    epa["err"] = [levenshtein(str(a), str(g)) / max(len(str(g)), 1) for a, g in zip(epa["ipa"], epa["g2p"])]
    X, err, spk = [], [], []
    with torch.no_grad():
        for i in range(len(epa)):
            try:
                y, sr = sf.read(io.BytesIO(epa["audio"].iloc[i]["bytes"]))
                if getattr(y, "ndim", 1) > 1: y = y.mean(axis=1)
                if sr != SR: y = librosa.resample(y.astype(float), orig_sr=sr, target_sr=SR)
                y = y.astype("float32")
                if len(y) < SR * 0.2: continue
                inp = extractor(y, sampling_rate=SR, return_tensors="pt", padding=False)
                xe = model(inp.input_values.to(device)).last_hidden_state.mean(dim=1).cpu().numpy()[0]
                X.append(xe); err.append(epa["err"].iloc[i]); spk.append(str(epa["speaker_code"].iloc[i]))
            except Exception:
                continue
            if len(X) % 300 == 0: print(f"  epadb {len(X)}/{len(epa)}", flush=True)
    X = np.array(X); err = np.array(err); spk = np.array(spk)
    np.savez(CACHE, X=X, err=err, spk=spk)
    print(f"[extract] guardado {X.shape} en {CACHE}", flush=True)
    return X, err, spk


def fit_eval(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    r = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    pred = r.predict(sc.transform(Xte))
    if np.std(pred) < 1e-9: return 0.0
    return abs(pearsonr(pred, yte)[0])


def main():
    tok = os.environ.get("HF_TOKEN")
    if not tok and not CACHE.exists():
        print("ERROR: falta HF_TOKEN"); return 1

    X, err, spk = extract_epadb(tok)
    speakers = np.array(sorted(set(spk)))
    print(f"[epadb] {len(X)} enunciados, {len(speakers)} hablantes", flush=True)

    # --- Modelo mandarin (deep) para zero-shot: entrenar una sola vez ---
    Eso = np.load(PROC / "embeddings_wav2vec2.npy")
    utts = json.load(open(PROC / "embeddings_wav2vec2_utts.json", encoding="utf-8"))
    norm = lambda s: s.astype(str).str.lstrip("0")
    with open(BASE / "resource" / "scores.json", encoding="utf-8") as f:
        scores = json.load(f)
    tgt = pd.DataFrame([{"utt": u, "total": v.get("total")} for u, v in scores.items()])
    tgt["utt"] = norm(tgt["utt"])
    emb = pd.DataFrame({"utt": norm(pd.Series([str(u) for u in utts])), "idx": np.arange(len(utts))})
    m = tgt.merge(emb, on="utt")
    Xso = Eso[m["idx"].values]; yso = m["total"].values
    sc_so = StandardScaler().fit(Xso)
    ridge_so = Ridge(alpha=1.0).fit(sc_so.transform(Xso), yso)

    # --- Promediar sobre R_SPLITS particiones aleatorias test(15)/pool(35) ---
    # Esto evita que el resultado dependa de una sola eleccion de hablantes de test.
    zs_list, or_list = [], []
    fs_lists = {k: [] for k in K_GRID}
    for s in range(R_SPLITS):
        rng = np.random.default_rng(100 + s)
        perm = rng.permutation(speakers)
        test_spk = set(perm[:N_TEST_SPK]); pool_spk = list(perm[N_TEST_SPK:])
        te = np.array([sp in test_spk for sp in spk])
        Xte, yte = X[te], err[te]
        pred_zs = ridge_so.predict(sc_so.transform(Xte))
        zs_list.append(abs(pearsonr(pred_zs, yte)[0]))
        tr_pool = np.array([sp in set(pool_spk) for sp in spk])
        or_list.append(fit_eval(X[tr_pool], err[tr_pool], Xte, yte))
        for k in K_GRID:
            if k > len(pool_spk): continue
            vals = []
            for _ in range(R_REPEAT):
                sel = set(rng.choice(pool_spk, size=k, replace=False))
                tr = np.array([sp in sel for sp in spk])
                vals.append(fit_eval(X[tr], err[tr], Xte, yte))
            fs_lists[k].append(float(np.mean(vals)))
        print(f"  split {s+1}/{R_SPLITS}: zs={zs_list[-1]:.3f} oracle={or_list[-1]:.3f}", flush=True)

    r_zs, r_zs_sd = float(np.mean(zs_list)), float(np.std(zs_list))
    r_oracle, r_or_sd = float(np.mean(or_list)), float(np.std(or_list))
    curve = {k: {"mean_abs_r": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
             for k, v in fs_lists.items() if v}
    recovery = r_zs / r_oracle if r_oracle > 0 else float("nan")
    k_match = next((k for k in K_GRID if k in curve and curve[k]["mean_abs_r"] >= r_zs), None)

    results = {
        "target": "EpaDB error rate (per utterance)", "representation": "Wav2Vec2 embeddings",
        "n_utt": int(len(X)), "n_speakers": int(len(speakers)),
        "n_test_speakers": N_TEST_SPK, "n_pool_speakers": int(len(speakers) - N_TEST_SPK),
        "n_splits": R_SPLITS, "n_repeats_per_k": R_REPEAT,
        "zero_shot_abs_r": round(r_zs, 4), "zero_shot_sd": round(r_zs_sd, 4),
        "oracle_abs_r": round(r_oracle, 4), "oracle_sd": round(r_or_sd, 4),
        "zero_shot_recovery_pct": round(recovery * 100, 1),
        "few_shot_curve": curve, "speakers_to_match_zero_shot": k_match,
    }
    with open(RES / "rq1_transfer_gap.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[resumen] {R_SPLITS} splits | zero-shot |r|={r_zs:.3f}+/-{r_zs_sd:.3f} | "
          f"oraculo |r|={r_oracle:.3f}+/-{r_or_sd:.3f} | recuperacion={recovery*100:.0f}% | "
          f"hablantes p/igualar zs={k_match}", flush=True)

    # --- Figura 7 ---
    ks = [k for k in K_GRID if k in curve]
    ys = [curve[k]["mean_abs_r"] for k in ks]
    es = [curve[k]["std"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(ks, ys, yerr=es, marker="o", color="#3b528b", capsize=3, label="In-language model (few-shot)")
    ax.axhline(r_zs, color="#5ec962", ls="--", lw=2, label=f"Zero-shot Mandarin model (|r|={r_zs:.2f})")
    ax.axhline(r_oracle, color="#d62728", ls=":", lw=2, label=f"In-language oracle (|r|={r_oracle:.2f})")
    ax.set_xlabel("Number of Spanish training speakers (k)")
    ax.set_ylabel("|Pearson r| with true error rate (held-out test)")
    ax.set_title("Transfer gap and few-shot curve on EpaDB (Spanish)")
    ax.set_axisbelow(True); ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTFIG / "Figure7.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure7.png guardada ({DPI} dpi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
