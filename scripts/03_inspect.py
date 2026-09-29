"""03_inspect.py — EDA inicial sobre speechocean762 (RQ3 + RQ2 descriptiva).

Carga scores.json (agregado) y scores-detail.json (5 evaluadores), los archivos
de split (train/test) y produce RQ3 (distribución + concordancia inter-evaluador)
y RQ2 descriptiva (correlaciones y diferencias por edad/sexo).
Salidas en results/exploratory/ y figures/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path("data/raw/speechocean762")
OUT = Path("results/exploratory")
FIG = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

DIMS = ["accuracy", "fluency", "completeness", "prosodic", "total"]


def load_scores():
    with open(BASE / "resource" / "scores.json", encoding="utf-8") as f:
        agg = json.load(f)
    with open(BASE / "resource" / "scores-detail.json", encoding="utf-8") as f:
        det = json.load(f)
    return agg, det


def icc1k(mat: np.ndarray) -> float:
    """ICC(1,k): one-way random effects, average of k raters."""
    n, k = mat.shape
    grand = mat.mean()
    ss_total = ((mat - grand) ** 2).sum()
    ss_between = k * ((mat.mean(axis=1) - grand) ** 2).sum()
    ss_within = ss_total - ss_between
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    if ms_between <= 0:
        return float("nan")
    return (ms_between - ms_within) / ms_between


def main() -> int:
    agg, det = load_scores()
    print(f"utterances en scores.json: {len(agg)}")
    print(f"utterances en scores-detail.json: {len(det)}")

    rows = []
    for utt, v in agg.items():
        r = {"utt": utt, "text": v.get("text", "")}
        for d in DIMS:
            r[d] = v.get(d)
        rows.append(r)
    df = pd.DataFrame(rows)
    print("\n== Distribución agregada (scores.json) ==")
    print(df[DIMS].describe().round(3).to_string())
    df.to_csv(OUT / "scores_aggregated.csv", index=False)

    # RQ3: concordancia inter-evaluador (ICC)
    icc_results = {}
    print("\n== Concordancia inter-evaluador (ICC(1,k), 5 evaluadores) ==")
    for d in DIMS:
        mat = np.array([det[u][d] for u in det if isinstance(det[u].get(d), list)])
        if mat.ndim == 2 and mat.shape[1] >= 2:
            icc = icc1k(mat)
            icc_results[d] = icc
            print(f"  {d:12s}: ICC(1,k) = {icc:.4f}   (n={mat.shape[0]}, raters={mat.shape[1]})")

    agg_comp = df["completeness"]
    det_comp = np.array([det[u]["completeness"] for u in det if isinstance(det[u].get("completeness"), list)])
    print(f"\ncompleteness scores.json: min={agg_comp.min()}, max={agg_comp.max()}")
    print(f"completeness scores-detail.json: min={det_comp.min():.3f}, max={det_comp.max():.3f}")

    # RQ2 descriptiva: correlaciones
    corr = df[DIMS].corr(method="pearson")
    print("\n== Correlación de Pearson entre dimensiones ==")
    print(corr.round(3).to_string())
    corr.to_csv(OUT / "correlations_dimensions.csv")
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Correlación entre dimensiones de pronunciación")
    plt.tight_layout()
    plt.savefig(FIG / "03_corr_dimensions.png", dpi=300)
    plt.close()

    # Edad y sexo
    def load_kv(split, fn):
        d = {}
        with open(BASE / split / fn, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    d[parts[0]] = parts[1]
        return d

    utt2spk = {**load_kv("train", "utt2spk"), **load_kv("test", "utt2spk")}
    spk2age = {**load_kv("train", "spk2age"), **load_kv("test", "spk2age")}
    spk2gender = {**load_kv("train", "spk2gender"), **load_kv("test", "spk2gender")}

    df["speaker"] = df["utt"].map(utt2spk)
    df["age"] = df["speaker"].map(spk2age).astype(float)
    df["gender"] = df["speaker"].map(spk2gender)
    df["age_group"] = np.where(df["age"] < 18, "child", "adult")
    print("\n== Composición ==")
    print(df["age_group"].value_counts().to_string())
    print(df["gender"].value_counts().to_string())
    print(f"speakers únicos: {df['speaker'].nunique()}, edad min={df['age'].min()}, max={df['age'].max()}")
    print("\n== Media por grupo de edad ==")
    print(df.groupby("age_group")[DIMS].mean().round(3).to_string())
    print("\n== Media por sexo ==")
    print(df.groupby("gender")[DIMS].mean().round(3).to_string())

    # Discrepancia WAV vs utterances
    wavs = {p.stem for p in (BASE / "WAVE").rglob("*.wav")}
    scp_ids = set()
    for split in ("train", "test"):
        with open(BASE / split / "wav.scp", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    scp_ids.add(parts[0])
    print(f"\n== Discrepancia WAV: en disco={len(wavs)}, en wav.scp={len(scp_ids)}")
    print(f"  extra (sin ID en scp): {len(wavs - scp_ids)}, faltantes: {len(scp_ids - wavs)}")
    if wavs - scp_ids:
        print("  ejemplo extra:", sorted(wavs - scp_ids)[:5])

    summary = {
        "n_utterances": int(len(df)),
        "n_wav": int(len(wavs)),
        "n_speakers": int(df["speaker"].nunique()),
        "icc": icc_results,
    }
    with open(OUT / "summary_eda.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nGuardado en results/exploratory/ y figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
