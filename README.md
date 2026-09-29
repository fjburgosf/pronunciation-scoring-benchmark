# Cross-Lingual Transfer of Automatic Pronunciation Scoring

This repository contains the Python scripts that reproduce all quantitative results in the study: inter-rater reliability (ICC), automatic pronunciation scoring (acoustic, self-supervised, and fusion models) with bootstrap confidence intervals and significance tests, feature-importance analysis, cross-L1 error patterns, and the central cross-corpus generalization analysis (a Mandarin-trained scorer applied to L2-ARCTIC and EpaDB).

---

## Requirements

- Python 3.9+
- ~5 GB disk space (speechocean762 + L2-ARCTIC + EpaDB)
- CPU is sufficient. Wav2Vec2 embedding extraction is the slow part (~1–2 h for 5,000 utterances; the cross-corpus scripts extract embeddings for the target corpora as well).

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

### speechocean762 (scored corpus)

Download manually from the official source:

1. Go to <https://www.openslr.org/101/> or the Interspeech 2021 corpus page.
2. Download and extract to `data/raw/speechocean762/`.

Expected structure:

```
data/raw/speechocean762/
├── train/{wav.scp, wav/}
├── test/{wav.scp, wav/}
└── resource/{scores.json, annotation/}
```

### L2-ARCTIC (six first languages)

Downloaded automatically from the Hugging Face Hub by the pipeline. No manual action required.

### EpaDB (50 native Spanish speakers)

EpaDB is a **gated** dataset on the Hugging Face Hub (`KoelLabs/EpaDB`). To reproduce the EpaDB analysis (Step 8):

1. Request access at <https://huggingface.co/datasets/KoelLabs/EpaDB> (or contact the corpus authors at `jvidal@dc.uba.ar`).
2. Set your Hugging Face token in the environment before running the script:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

The token is read from the `HF_TOKEN` environment variable only. It is never stored in the code.

---

## Pipeline — step by step

Run each script from the project root, in order.

### Step 1 — Download L2-ARCTIC

```bash
python scripts/02_download_l2arctic.py
```

Downloads the manually-annotated L2-ARCTIC subset and saves a CSV to `data/processed/`.

### Step 2 — Inspect speechocean762

```bash
python scripts/03_inspect.py
```

Validates the corpus structure and audio files.

### Step 3 — Construct features and targets

```bash
python scripts/05_construct_variables.py
```

Extracts 58 MFCC-based acoustic features per utterance and saves `data/processed/features.csv`.

### Step 4 — Acoustic baseline (MFCC + GBDT)

```bash
python scripts/08_primary_model.py
```

Trains one GradientBoostingRegressor per scoring dimension and saves `results/primary/rq1_baseline_metrics.json`.

Expected PCC on the test split: accuracy 0.576, fluency 0.693, completeness 0.100, prosodic 0.669, total 0.595.

### Step 5 — Deep scoring (Wav2Vec2 + Ridge)

```bash
python scripts/07_rq1_deep.py
```

Extracts 768-dim mean-pooled Wav2Vec2 embeddings for all 5,000 utterances (checkpointed every 100) and trains a Ridge regressor per dimension. Saves `results/primary/rq1_deep_metrics.json`.

> If interrupted, re-running resumes automatically from `data/processed/embeddings_wav2vec2.npy`.

### Step 6 — Cross-L1 error patterns (L2-ARCTIC)

```bash
python scripts/06_rq2_crossl1.py
```

Computes per-phoneme substitution, deletion, and addition error rates per L1 and runs Mann-Whitney U tests against Spanish. Saves error rates to `data/processed/l2arctic_error_rates_by_l1.csv`.

### Step 7 — Enhancements: CIs, significance, feature importance, fusion

```bash
python scripts/11_enhancements.py
```

Requires the outputs of Steps 3–5. Computes:
- 95% bootstrap confidence intervals for every PCC (2,000 resamples);
- paired-bootstrap significance tests (deep vs. baseline, fusion vs. baseline);
- gradient-boosted-tree feature importances per dimension;
- the fusion model (MFCC + Wav2Vec2 embeddings → Ridge).

Saves `results/primary/rq1_enhancements.json`.

### Step 8 — Cross-corpus generalization

Applies the Mandarin-trained scorer, without adaptation, to independent corpora and correlates predicted scores with reference error rates.

```bash
# L2-ARCTIC (six first languages)
python scripts/12_crosscorpus.py

# EpaDB (50 native Spanish speakers) — requires HF_TOKEN (see Data section)
python scripts/13_epadb.py
```

Saves `results/primary/rq1_crosscorpus.json` and `results/primary/rq1_epadb.json`.

> These scripts extract Wav2Vec2 embeddings for the target corpora and are the most time-consuming steps (~30–45 min each on CPU).

### Step 9 — Transfer gap and few-shot curve

Quantifies how *useful* the transfer is, by comparing the zero-shot Mandarin model against an in-language model trained on Spanish (the oracle) and against few-shot models trained on only `k` Spanish speakers.

```bash
# requires HF_TOKEN (see Data section)
python scripts/15_transfer_gap.py
```

All comparisons are evaluated on held-out Spanish speakers and averaged over 10 random speaker splits (15 test / 35 pool), with 20 draws per few-shot point. Saves `results/primary/rq1_transfer_gap.json` and `figures/submission/Figure7.png`.

> On the first run this extracts Wav2Vec2 embeddings for EpaDB (~30 min on CPU) and caches them in `data/processed/epadb_deep.npz`. Later runs reuse the cache and finish in seconds.

### Step 10 — Figures

```bash
python scripts/10_figures.py
```

Generates **all seven paper figures at 600 dpi** into `figures/submission/` (`Figure1.png` … `Figure7.png`). The script reads the JSON result files produced by the earlier steps, so it runs in seconds and can be re-run any time to redraw the figures without recomputing anything.

Each figure is skipped with a message if its result file is missing, so the script can also be run part-way through the pipeline.

---

## Output files

| File | Description |
|---|---|
| `results/primary/rq1_baseline_metrics.json` | PCC/MSE/MAE, MFCC+GBDT baseline |
| `results/primary/rq1_deep_metrics.json` | PCC/MSE/MAE, Wav2Vec2+Ridge |
| `results/primary/rq1_enhancements.json` | Bootstrap CIs, significance, feature importance, fusion |
| `results/primary/rq1_crosscorpus.json` | Generalization to L2-ARCTIC by L1 |
| `results/primary/rq1_epadb.json` | Generalization to EpaDB (Spanish) |
| `results/primary/rq1_transfer_gap.json` | Zero-shot vs. oracle vs. few-shot on EpaDB |
| `data/processed/l2arctic_error_rates_by_l1.csv` | Cross-L1 error rates |
| `figures/submission/Figure1..7.png` | Paper figures (600 dpi) |

---

## Reproducibility notes

- All figures and tables in the paper are generated by these scripts. No values are entered manually.
- The speechocean762 official train/test split (speaker-disjoint) is used as-is. No test data are used in fitting or hyperparameter selection.
- Wav2Vec2-base weights are loaded from `facebook/wav2vec2-base` on the Hugging Face Hub.
- Random seeds are fixed where applicable (bootstrap seed 42; see individual scripts).

---

## License

Scripts are released under the MIT License. Data licenses follow the original corpus terms (speechocean762: Apache 2.0; L2-ARCTIC and EpaDB: see the respective corpus documentation).
