"""10_figures.py — Genera las figuras finales del manuscrito en ingles a 600 dpi.

Figure 1: distribucion de las 5 dimensiones de puntaje (speechocean762).
Figure 2: matriz de correlacion de Pearson de las 5 dimensiones (speechocean762).
Figure 3: tasas de error por tipo (sustitucion/omision/adicion) y L1 (L2-ARCTIC).
Figure 4: PCC en test del baseline acustico por dimension (+ deep si esta disponible).

Salida: figures/submission/Figure1..4.png (600 dpi, texto en ingles).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("data/raw/speechocean762")
PROC = Path("data/processed")
RES = Path("results/primary")
OUT = Path("figures/submission")
OUT.mkdir(parents=True, exist_ok=True)
DIMS = ["accuracy", "fluency", "completeness", "prosodic", "total"]
LABELS = [d.capitalize() for d in DIMS]
DPI = 600
BLUE = "#3b528b"


def load_scores():
    with open(BASE / "resource" / "scores.json", encoding="utf-8") as f:
        scores = json.load(f)
    return pd.DataFrame([{d: v.get(d) for d in DIMS} for v in scores.values()])


def figure1_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[d].dropna().values for d in DIMS]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(BLUE)
        pc.set_alpha(0.6)
    for key in ("cmeans", "cmaxes", "cmins", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1)
    ax.set_xticks(range(1, len(DIMS) + 1))
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_ylabel("Score (0-10)")
    ax.set_title("Distribution of sentence-level scores (speechocean762)")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "Figure1.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure1.png guardada ({DPI} dpi)")


def figure2_correlation(df):
    corr = df[DIMS].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(DIMS)))
    ax.set_yticks(range(len(DIMS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    for i in range(len(DIMS)):
        for j in range(len(DIMS)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.6 else "black", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    ax.set_title("Correlation of sentence-level score dimensions")
    fig.tight_layout()
    fig.savefig(OUT / "Figure2.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure2.png guardada ({DPI} dpi)")


def figure3_error_types():
    rates = pd.read_csv(PROC / "l2arctic_error_rates_by_l1.csv")
    rates = rates.sort_values("err_rate")
    l1_display = {"chinese": "Mandarin"}
    labels = [l1_display.get(l1, l1.capitalize()) for l1 in rates["L1"]]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, rates["subst_rate"], w, label="Substitutions", color="#440154")
    ax.bar(x, rates["del_rate"], w, label="Deletions", color="#21918c")
    ax.bar(x + w, rates["add_rate"], w, label="Additions", color="#fde725", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Errors per phoneme")
    ax.set_xlabel("First language (L1)")
    ax.set_title("Pronunciation error rates by type and first language (L2-ARCTIC)")
    ax.legend()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "Figure3.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure3.png guardada ({DPI} dpi)")


def figure4_scoring_performance():
    with open(RES / "rq1_baseline_metrics.json", encoding="utf-8") as f:
        base = json.load(f)
    with open(RES / "rq1_deep_metrics.json", encoding="utf-8") as f:
        deep = json.load(f)
    with open(RES / "rq1_enhancements.json", encoding="utf-8") as f:
        enh = json.load(f)
    ci = enh["bootstrap_ci"]
    fus = enh["fusion_metrics"]

    base_pcc = np.array([base[d]["PCC"] for d in DIMS])
    deep_pcc = np.array([deep[d]["PCC"] for d in DIMS])
    fus_pcc = np.array([fus[d]["PCC"] for d in DIMS])

    def err(model_key, pccs):
        lo = np.array([ci[d][model_key][0] for d in DIMS])
        hi = np.array([ci[d][model_key][1] for d in DIMS])
        return np.vstack([pccs - lo, hi - pccs])

    x = np.arange(len(DIMS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - w, base_pcc, w, yerr=err("baseline_PCC_CI", base_pcc), capsize=3,
           label="Baseline (MFCC + GBDT)", color=BLUE)
    ax.bar(x, deep_pcc, w, yerr=err("deep_PCC_CI", deep_pcc), capsize=3,
           label="Deep (Wav2Vec2 + Ridge)", color="#5ec962")
    ax.bar(x + w, fus_pcc, w, yerr=err("fusion_PCC_CI", fus_pcc), capsize=3,
           label="Fusion (MFCC + Wav2Vec2)", color="#fde725", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_ylabel("Pearson correlation (test split)")
    ax.set_ylim(-0.15, 1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Automatic scoring performance by dimension (95% bootstrap CI)")
    ax.legend()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "Figure4.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure4.png guardada ({DPI} dpi, baseline+deep+fusion con IC)")


def figure5_feature_importance():
    with open(RES / "rq1_enhancements.json", encoding="utf-8") as f:
        enh = json.load(f)
    fi = enh["feature_importance"]

    def pretty(f):
        import re as _re
        m = _re.match(r"(dd|d)?mfcc(\d+)_(mean|std)", f)
        if m:
            pre = {"": "MFCC", "d": "delta MFCC", "dd": "delta-delta MFCC"}[m.group(1) or ""]
            return f"{pre} {m.group(2)} ({m.group(3)})"
        names = {"duration": "Duration", "rms_mean": "RMS energy (mean)",
                 "rms_std": "RMS energy (std)", "zcr_mean": "Zero-crossing rate",
                 "cent_mean": "Spectral centroid", "bw_mean": "Spectral bandwidth"}
        return names.get(f, f)

    # top-6 features de la dimension 'total'
    top = fi["total"][:6]
    names = [pretty(t["feature"]) for t in top][::-1]
    vals = [t["importance"] for t in top][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.barh(names, vals, color=BLUE)
    for b, v in zip(bars, vals):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=9)
    ax.set_xlabel("Relative importance (gradient-boosted trees)")
    ax.set_title("Top acoustic features for the total score (speechocean762)")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "Figure5.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure5.png guardada ({DPI} dpi)")


def figure6_crosscorpus():
    cc_path = RES / "rq1_crosscorpus.json"
    if not cc_path.exists():
        print("Figure6: falta rq1_crosscorpus.json, se omite")
        return
    with open(cc_path, encoding="utf-8") as f:
        cc = json.load(f)
    mdl_disp = {"baseline": "Baseline", "deep": "Deep", "fusion": "Fusion"}
    # modelo primario = el de transferencia global mas fuerte (r total mas negativo)
    primary = min(cc["by_model"],
                  key=lambda m: cc["by_model"][m]["by_dimension"]["total"]["pearson_r_all"])
    l1map = {"chinese": "Mandarin"}
    by = cc["by_model"][primary]["by_dimension"]["total"]["by_L1"]
    items = sorted(by.items(), key=lambda kv: kv[1]["pearson_r"])
    labels = [l1map.get(k, k.capitalize()) for k, _ in items]
    rvals = [v["pearson_r"] for _, v in items]
    colors = ["#d62728" if k == "spanish" else BLUE for k, _ in items]

    epa_path = RES / "rq1_epadb.json"
    if epa_path.exists():
        with open(epa_path, encoding="utf-8") as f:
            epa = json.load(f)
        r_epa = epa["by_model"][primary]["by_dimension"]["total"]["utterance_level"]["pearson_r"]
        labels.append("Spanish\n(EpaDB)")
        rvals.append(r_epa)
        colors.append("#9467bd")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, rvals, color=colors, edgecolor="black", linewidth=0.3)
    for b, v in zip(bars, rvals):
        off = -0.008 if v < 0 else 0.004
        va = "top" if v < 0 else "bottom"
        col = "white" if v < -0.03 else "black"
        ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:.2f}",
                ha="center", va=va, fontsize=9, color=col)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Pearson r (predicted total score vs. error rate)")
    ax.set_xlabel("First language (target corpus)")
    ax.set_title(f"Cross corpus generalization of the Mandarin trained scorer ({mdl_disp[primary]} model)")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "Figure6.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure6.png guardada ({DPI} dpi, modelo primario={primary})")


def figure7_transfer_gap():
    """Redibuja la Figura 7 desde el JSON ya calculado por 15_transfer_gap.py."""
    tg_path = RES / "rq1_transfer_gap.json"
    if not tg_path.exists():
        print("Figure7: falta rq1_transfer_gap.json (correr scripts/15_transfer_gap.py), se omite")
        return
    with open(tg_path, encoding="utf-8") as f:
        tg = json.load(f)
    curve = tg["few_shot_curve"]
    ks = sorted(int(k) for k in curve)
    ys = [curve[str(k)]["mean_abs_r"] for k in ks]
    es = [curve[str(k)]["std"] for k in ks]
    r_zs = tg["zero_shot_abs_r"]
    r_or = tg["oracle_abs_r"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(ks, ys, yerr=es, marker="o", color=BLUE, capsize=3,
                label="In-language model (few-shot)")
    ax.axhline(r_zs, color="#5ec962", ls="--", lw=2,
               label=f"Zero-shot Mandarin model (|r|={r_zs:.2f})")
    ax.axhline(r_or, color="#d62728", ls=":", lw=2,
               label=f"In-language oracle (|r|={r_or:.2f})")
    ax.set_xlabel("Number of Spanish training speakers (k)")
    ax.set_ylabel("|Pearson r| with true error rate (held-out test)")
    ax.set_title("Transfer gap and few-shot curve on EpaDB (Spanish)")
    ax.set_axisbelow(True)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "Figure7.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure7.png guardada ({DPI} dpi)")


def main():
    df = load_scores()
    figure1_distribution(df)
    figure2_correlation(df)
    figure3_error_types()
    figure4_scoring_performance()
    figure5_feature_importance()
    figure6_crosscorpus()
    figure7_transfer_gap()
    print(f"\nTodas las figuras en {OUT.resolve()} ({DPI} dpi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
