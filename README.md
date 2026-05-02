# NLPQuantStrat

**Do earnings call transcripts predict post-announcement returns?**  
An end-to-end NLP-based quantitative strategy on Russell 1000 equities.

---

## Overview

This project builds a systematic machine learning pipeline that transforms raw earnings call transcripts into out-of-sample idiosyncratic stock return forecasts. The pipeline covers:

1. **Target construction** — rolling CAPM betas and cumulative idiosyncratic returns at horizons of 1, 3, 5, and 21 trading days.
2. **Feature engineering** — three text representation families:
   - Sparse TF-IDF bag-of-words (200 features, preprocessed transcripts)
   - Dense sentence-transformer embeddings (`all-MiniLM-L6-v2`, 384 dimensions)
   - 17 Loughran–McDonald dictionary-based sentiment scores (7 legacy + 10 alpha features)
3. **Walk-forward cross-validation** — quarterly expanding-window CV with strict no-look-ahead-bias guarantees.
4. **Model comparison** — Ridge, Lasso, Random Forest, and Gradient Boosting across three feature compositions.
5. **Cross-sectional backtesting** — long-only top-quintile strategies evaluated on annualized return, Sharpe ratio, and max drawdown.

All data (transcripts, market returns, embeddings, features, CV results) are stored in and loaded from **Amazon S3**.

---

## Key Results

| Configuration | Hit Rate | MSR | t-stat |
|---|---|---|---|
| `tfidf_sentiment` + Ridge, h=1d | 51.7% | 3.8 × 10⁻³ | **2.24** |
| `sentiment` + Ridge, h=1d | 51.5% | 3.5 × 10⁻³ | **2.10** |
| `tfidf_sentiment` + GBR, h=1d | 51.7% | 3.3 × 10⁻³ | 1.91 |

Best cross-sectional signal: `sentiment_delta` (Sharpe 0.552) and `unc_ratio` (Sharpe 0.549), both outperforming the legacy `polarity_delta` baseline (0.535). Predictive power decays sharply with horizon — signals are essentially noise by h=21d in the Russell 1000 universe.

---

## Project Structure

```
NLPQuantStrat/
├── main.py                        # Pipeline entry point
├── pyproject.toml                 # Project metadata and dependencies (uv)
├── configs/
│   └── run_pipeline_config.json   # All pipeline parameters
├── src/nlp_quant_strat/
│   ├── data/
│   │   ├── data_manager.py        # S3 data loading, alignment, NaN policy
│   │   └── feature_engineering.py # LM sentiment features (17 signals)
│   ├── backtester/
│   │   ├── strategies.py          # Cross-sectional percentile strategy
│   │   ├── portfolio.py           # Equal-weighting scheme, rebalancing
│   │   ├── backtest.py            # Return computation, transaction costs
│   │   ├── analysis.py            # Performance metrics (Sharpe, Max DD, …)
│   │   └── visualization.py       # Cumulative return and rolling metric plots
│   └── utils/
│       ├── config.py              # Typed Config dataclass from JSON
│       └── utils.py               # S3 upload helpers
├── outputs/
│   ├── figures/                   # Strategy plots (per-feature + comparison)
│   └── reports/                   # LaTeX research report + compiled PDF
└── scripts/
    └── example_backtest_pipeline.py
```

---

## Setup

### Prerequisites

- Python ≥ 3.13
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- AWS credentials with read access to the `nlp-quant-strat` S3 bucket (eu-north-1)

### 1. Clone

```bash
git clone https://github.com/mateomolinaro1/NLPQuantStrat.git
cd NLPQuantStrat
```

### 2. Create `.env`

Create a `.env` file at the repository root with your AWS credentials:

```env
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
```

### 3. Create and activate the virtual environment

```bash
uv venv .venv
```

Linux / macOS:
```bash
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
uv sync
```

---

## Configuration

All pipeline parameters live in `configs/run_pipeline_config.json`. The key sections are:

| Section | Key parameters |
|---|---|
| `AWS.S3` | Bucket name, region, S3 keys for each data file |
| `FEATURE_ENGINEERING` | `LOAD_OR_COMPUTE` (`"load"` / `"compute"`), `SAVE_TO_S3`, rolling window |
| `EMBEDDINGS` | `LOAD_OR_COMPUTE`, sentence-transformer model name, TF-IDF vocabulary size |
| `FEATURE_SET` | `MODE` — list of feature compositions to run (e.g. `["sentiment", "tfidf_sentiment", "st_sentiment"]`) |
| `FORECASTING` | Horizons, refit frequency, train/val windows, model grid |
| `BACKTEST` | Quintile breakpoints, rebalancing period, transaction costs, portfolio type |
| `CV_RESULTS` | `LOAD_OR_COMPUTE`, S3 prefix for caching walk-forward results |

### Feature composition modes

| Mode | Features | Dim. |
|---|---|---|
| `sentiment` | LM sentiment scores only | 17 |
| `tfidf` | TF-IDF only | 200 |
| `st` | Sentence-transformer only | 384 |
| `tfidf_sentiment` | TF-IDF + sentiment | 217 |
| `st_sentiment` | Sentence-transformer + sentiment | 401 |
| `all` | All three families | 601 |

---

## Running the pipeline

```bash
python main.py
```

The pipeline runs two phases sequentially:

**Phase 1 — ML Forecasting**  
Loads data, builds CAPM idiosyncratic targets, encodes transcripts, assembles feature matrices, and runs quarterly walk-forward CV for each configured mode. CV results are optionally saved to S3 (`CV_RESULTS.SAVE_RESULTS`).

**Phase 2 — Cross-sectional Backtests**  
Runs a long-only cross-sectional backtest for each of the 17 sentiment features. Saves per-feature and comparison performance charts to `outputs/figures/`.

### Controlling computation vs. loading from cache

Set the following flags in `run_pipeline_config.json` to skip expensive recomputation:

```json
"FEATURE_ENGINEERING": { "LOAD_OR_COMPUTE": "load", "SAVE_TO_S3": false },
"EMBEDDINGS":          { "LOAD_OR_COMPUTE": "load" },
"CV_RESULTS":          { "LOAD_OR_COMPUTE": "load" }
```

Set to `"compute"` to recompute and optionally re-upload to S3.

---

## Sentiment features

### Legacy (Loughran–McDonald)

| Feature | Description |
|---|---|
| `positive_count` | Count of positive-sentiment words |
| `negative_count` | Count of negative-sentiment words |
| `word_count` | Total word count (transcript length proxy) |
| `sentiment_density` | Emotive word proportion |
| `polarity` | `(pos − neg) / (pos + neg + 1)` |
| `polarity_delta` | Quarter-over-quarter change in polarity |
| `pos_polarity_count_q` | Rolling 4-quarter count of positive-polarity quarters |

### Alpha features

| Feature | Description |
|---|---|
| `pos_ratio` | `positive_count / (word_count + 1)` |
| `neg_ratio` | `negative_count / (word_count + 1)` |
| `net_sentiment` | `(pos − neg) / (word_count + 1)` |
| `unc_ratio` | Uncertainty word proportion (LM uncertainty list) |
| `sentiment_delta` | Quarter-over-quarter change in `net_sentiment` |
| `sentiment_surprise` | Polarity minus rolling 4-quarter mean |
| `sentiment_zscore` | Polarity z-score relative to trailing 4-quarter distribution |
| `sent_var` | Variance of sentence-level polarity scores within a transcript |
| `toxic_density` | Fraction of sentences with strongly negative polarity (< −0.3) |
| `sent_vol_interaction` | `sent_var × neg_ratio` (conviction-weighted negativity) |

---

## Evaluation metrics

| Metric | Null baseline | Better direction |
|---|---|---|
| Hit rate | 50% | > 50% |
| Long–short spread | 0 | > 0 |
| Mean signed return (MSR) | 0 | > 0 |
| t-statistic | 0 | > 1.96 (5% significance) |
| Excess RMSE | 0 | < 0 |

Excess RMSE is defined as `RMSE − σ(y)`, where `σ(y)` is the RMS of realized returns. The zero-prediction null model achieves exactly 0 by construction, making the metric comparable across horizons.

---

## Data

All data are stored in the `nlp-quant-strat` S3 bucket (eu-north-1) and are **not included in this repository**.

| File (S3 key) | Description |
|---|---|
| `data/market/riy_asset_returns.parquet` | Daily total returns, Russell 1000 constituents |
| `data/market/russell_returns.parquet` | Russell 1000 index total returns |
| `data/market/risk_free_returns.parquet` | Daily US 1-month T-bill rate |
| `data/market/riy_index_constituents.parquet` | Index constituent membership table |
| `data/transcripts/formatted_unprocessed_transcripts.parquet` | Raw earnings call transcripts |
| `data/transcripts/formatted_preprocessed_transcripts.parquet` | Lowercased, stop-word-removed, stemmed transcripts |
| `data/others/words_dict.parquet` | Loughran–McDonald sentiment dictionary (time-stamped) |
| `data/features/*.parquet` | Pre-computed sentiment feature panels (one file per feature) |
| `data/embeddings/*.parquet` | Cached TF-IDF and sentence-transformer embeddings |
| `data/cv_results/*.parquet` | Cached walk-forward CV predictions and metrics |

---

## Dependencies

Managed via `uv` and declared in `pyproject.toml`:

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation and panel alignment |
| `scikit-learn` | Ridge, Lasso, RF, GBR, StandardScaler |
| `better-aws` | S3 load/save helpers |
| `matplotlib` | Performance visualization |
| `nltk` | Text preprocessing |
| `spacy` | NLP utilities |
| `python-dotenv` | `.env` loading |

---

## Authors

- **Matéo Molinaro** — ENSAE Paris · [mateo.molinaro@ensae.fr](mailto:mateo.molinaro@ensae.fr)
- **Pierre Chuzeville** — ENSAE Paris · [pierre.chuzeville@ensae.fr](mailto:pierre.chuzeville@ensae.fr)

## License

Academic project — ENSAE Paris, 2025–2026.
