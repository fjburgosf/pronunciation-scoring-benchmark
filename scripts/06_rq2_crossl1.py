"""06_rq2_crossl1.py — RQ2 cross-L1 con L2-ARCTIC (anotaciones de errores de pronunciación).

Descarga los parquet de chikingsley/l2-arctic-manual-v5.0-16k (sin decodificar audio),
mapea speaker -> L1 y compara tasas de error (sustituciones/omisiones/adiciones)
por L1 (mandarín vs. español vs. otras).
"""
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "chikingsley/l2-arctic-manual-v5.0-16k"
FILES = [
    ("train", "data/train-00000-of-00001.parquet"),
    ("validation", "data/validation-00000-of-00001.parquet"),
    ("test", "data/test-00000-of-00001.parquet"),
    ("suitcase", "data/suitcase-00000-of-00001.parquet"),
]
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# speaker -> L1 (documentado en la página oficial de L2-ARCTIC, verificado 2026-08-24)
SPK2L1 = {
    "ABA": "arabic", "SKA": "arabic", "YBAA": "arabic", "ZHAA": "arabic",
    "BWC": "mandarin", "LXC": "mandarin", "NCC": "mandarin", "TXHC": "mandarin",
    "ASI": "hindi", "RRBI": "hindi", "SVBI": "hindi", "TNI": "hindi",
    "HJK": "korean", "HKK": "korean", "YDCK": "korean", "YKWK": "korean",
    "EBVS": "spanish", "ERMS": "spanish", "MBMPS": "spanish", "NJS": "spanish",
    "HQTV": "vietnamese", "PNV": "vietnamese", "THV": "vietnamese", "TLV": "vietnamese",
}


def main():
    frames = []
    for split, fname in FILES:
        path = hf_hub_download(REPO, fname, repo_type="dataset")
        d = pd.read_parquet(path)
        d["split"] = split
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    df["L1"] = df["native_language"].str.lower()
    df["subst_rate"] = df["num_substitutions"] / df["num_phonemes"]
    df["del_rate"] = df["num_deletions"] / df["num_phonemes"]
    df["add_rate"] = df["num_additions"] / df["num_phonemes"]
    df["err_rate"] = (df["num_substitutions"] + df["num_deletions"] + df["num_additions"]) / df["num_phonemes"]

    print("== utterances por L1 ==")
    print(df["L1"].value_counts().to_string())
    print("\n== tasas de error medias por L1 (por fonema) ==")
    rates = df.groupby("L1")[["subst_rate", "del_rate", "add_rate", "err_rate"]].mean()
    print(rates.round(4).to_string())
    rates.to_csv(OUT / "l2arctic_error_rates_by_l1.csv")

    print("\n== fonemas totales por L1 ==")
    print(df.groupby("L1")["num_phonemes"].sum().to_string())

    # Prueba estadística (Mann-Whitney U) de subst_rate: cada L1 vs español
    from scipy.stats import mannwhitneyu
    es = df[df["L1"] == "spanish"]["subst_rate"]
    print("\n== Mann-Whitney U de subst_rate (vs spanish) ==")
    for lg in sorted(df["L1"].unique()):
        if lg != "spanish":
            other = df[df["L1"] == lg]["subst_rate"]
            u, p = mannwhitneyu(other, es, alternative="two-sided")
            print(f"  {lg:12s} vs spanish: U={u:.1f}, p={p:.4f}")

    # Figura: tasa de error por L1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    order = df.groupby("L1")["err_rate"].mean().sort_values().index
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="L1", y="err_rate", order=list(order), errorbar="sd", palette="viridis")
    plt.title("Tasa de error de pronunciación por L1 (L2-ARCTIC)")
    plt.ylabel("errores / fonema")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/06_err_rate_by_l1.png", dpi=300)
    plt.close()

    print("\nGuardado: results/processed y figures/06_err_rate_by_l1.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
