"""
Debug/validation script for the forecasting pipeline.
Run in debug mode and step through each section to verify correctness.

Sections are added incrementally as new modules are implemented:
  [STEP 0] Data loading
  [STEP 1] TargetBuilder (targets.py)
  [STEP 2] EmbeddingBuilder (embeddings.py)
  [STEP 3] FeatureSet (feature_set.py)
  [STEP 4] EarningsWalkForwardCV (earnings_wf_cv)
  [STEP 5] ExperimentRunner (experiment.py)        <- TODO
  [STEP 6] ICAnalyser (ic_analysis.py)             <- TODO
"""
import dataclasses
import logging
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("debug_forecasting")

# =============================================================================
# [STEP 0] Data loading
# =============================================================================
logger.info("=" * 60)
logger.info("STEP 0 — Data loading")
logger.info("=" * 60)

from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.data.data_manager import DataManager

config = Config()
data_manager = DataManager(config=config)
data_manager.load_data()

asset_returns   = data_manager.get_asset_returns()
benchmark       = data_manager.get_benchmark_returns()
rf              = data_manager.get_rf_returns()
mapping_df      = data_manager.mapping_df
transcripts_df  = data_manager._get_formatted_unprocessed_transcripts()

# --- Sanity checks ---
assert asset_returns is not None,  "asset_returns is None"
assert benchmark is not None,      "benchmark_returns is None"
assert rf is not None,             "rf_returns is None"
assert mapping_df is not None and not mapping_df.empty, "mapping_df is None or empty"

logger.info("asset_returns   : %s  |  date range: %s → %s", asset_returns.shape,
            asset_returns.index.min().date(), asset_returns.index.max().date())
logger.info("benchmark       : %s", benchmark.shape)
logger.info("rf_returns      : %s", rf.shape)
logger.info("mapping_df      : %d rows  |  columns: %s", len(mapping_df), list(mapping_df.columns))
logger.info("transcripts_df  : %d rows  |  columns: %s", len(transcripts_df), list(transcripts_df.columns))

# Check transcript text is present
has_transcript_col = "transcript" in transcripts_df.columns
logger.info("transcript column present: %s", has_transcript_col)


# Check date alignment between mapping_df and asset_returns
mapping_dates_in_returns = mapping_df["filing_date"].isin(asset_returns.index)
logger.info("mapping_df dates in asset_returns index: %.1f%%",
            mapping_dates_in_returns.mean() * 100)

# Check asset alignment
mapping_assets_in_returns = mapping_df["asset"].isin(asset_returns.columns)
logger.info("mapping_df assets in asset_returns columns: %.1f%%",
            mapping_assets_in_returns.mean() * 100)

logger.info("STEP 0 — OK\n")

# =============================================================================
# [STEP 1] TargetBuilder
# =============================================================================
logger.info("=" * 60)
logger.info("STEP 1 — TargetBuilder")
logger.info("=" * 60)

from nlp_quant_strat.forecasting.targets import TargetBuilder

target_builder = TargetBuilder(
    data=data_manager,
    beta_window=config.beta_window,
    horizons=config.forecasting_horizons,
)
targets = target_builder.build()

# --- Shape and columns ---
expected_cols = {"filing_date", "asset"} | {f"idio_{h}d" for h in config.forecasting_horizons}
assert set(targets.columns) == expected_cols, \
    f"Unexpected columns: {set(targets.columns)} vs {expected_cols}"
logger.info("targets shape   : %s", targets.shape)
logger.info("targets columns : %s", list(targets.columns))

# --- Value range sanity (returns should be in a plausible range) ---
for h in config.forecasting_horizons:
    col = f"idio_{h}d"
    vals = targets[col].dropna()
    logger.info("idio_%dd  — mean: %+.4f | std: %.4f | min: %+.4f | max: %+.4f",
                h, vals.mean(), vals.std(), vals.min(), vals.max())
    assert vals.abs().max() < 5.0, \
        f"Implausible return value for {col}: {vals.abs().max():.2f}  (>500%)"

# --- No lookahead: verify the forward window starts AFTER filing_date ---
# For idio_1d: the return should come from t+1, not t.
# Proxy check: the 1d target should NOT be the same as the same-day excess return.
# Simpler lookahead check: idio_1d for the LAST available date must be NaN
# because there is no t+1 observation.
last_date = asset_returns.index[-1]
last_day_events = targets[targets["filing_date"] == last_date]
if not last_day_events.empty:
    assert last_day_events["idio_1d"].isna().all(), \
        "Events on the last market date should have NaN idio_1d (no t+1 data)"
    logger.info("Lookahead check (last date is NaN) — OK")
else:
    logger.info("Lookahead check — no events on last date, skipping")

# --- Duplicate check ---
n_dupes = targets.duplicated(subset=["filing_date", "asset"]).sum()
assert n_dupes == 0, f"Found {n_dupes} duplicate (filing_date, asset) pairs in targets"
logger.info("Duplicate (filing_date, asset) pairs: %d — OK", n_dupes)

# --- Quick preview ---
logger.info("\nTargets sample (5 rows):\n%s", targets.dropna().head(5).to_string())

logger.info("STEP 1 — OK\n")

# =============================================================================
# [STEP 2] EmbeddingBuilder
# =============================================================================
logger.info("=" * 60)
logger.info("STEP 2 — EmbeddingBuilder")
logger.info("=" * 60)

from nlp_quant_strat.forecasting.embeddings import SentenceTransformerBuilder, TFIDFBuilder

preprocessed_transcripts_df = data_manager._get_formatted_preprocessed_transcripts()
logger.info("preprocessed_transcripts_df : %d rows | columns: %s",
            len(preprocessed_transcripts_df), list(preprocessed_transcripts_df.columns))

n_events = mapping_df[["filing_date", "asset"]].drop_duplicates().shape[0]

# --- TF-IDF (uses stemmed/cleaned text: transcript_cleaned) ---
tfidf_builder = TFIDFBuilder(config=config)
# tfidf_embeddings = tfidf_builder.build(mapping_df=mapping_df, transcripts_df=preprocessed_transcripts_df, aws=data_manager.aws)
tfidf_embeddings = data_manager.aws.s3.load(key="data/embeddings/tfidf_0ed103a5ed.parquet")
# assert set(["filing_date", "asset"]).issubset(tfidf_embeddings.columns), \
#     "tfidf_embeddings missing filing_date or asset column"
# assert tfidf_embeddings.shape[0] == n_events, \
#     f"tfidf_embeddings row count {tfidf_embeddings.shape[0]} != n_events {n_events}"
n_tfidf_feats = tfidf_embeddings.shape[1] - 2
logger.info("tfidf_embeddings : %s | %d features (tfidf_0 .. tfidf_%d)",
            tfidf_embeddings.shape, n_tfidf_feats, n_tfidf_feats - 1)

# --- Sentence Transformer (uses raw text: transcript) ---
st_builder = SentenceTransformerBuilder(config=config)
# st_embeddings = st_builder.build(mapping_df=mapping_df, transcripts_df=transcripts_df, aws=data_manager.aws)
st_embeddings = data_manager.aws.s3.load(key="data/embeddings/st_788990f70d.parquet")
#
# assert set(["filing_date", "asset"]).issubset(st_embeddings.columns), \
#     "st_embeddings missing filing_date or asset column"
# assert st_embeddings.shape[0] == n_events, \
#     f"st_embeddings row count {st_embeddings.shape[0]} != n_events {n_events}"
# n_emb_feats = st_embeddings.shape[1] - 2
# logger.info("st_embeddings    : %s | %d dims (emb_0 .. emb_%d)",
#             st_embeddings.shape, n_emb_feats, n_emb_feats - 1)
#
# logger.info("STEP 2 — OK\n")
#
# data_manager.release_transcripts()
# logger.info("Transcript DataFrames released from memory.")

# =============================================================================
# [STEP 3 + 4] FeatureSet → EarningsWalkForwardCV  (iterated over modes)
# =============================================================================
from nlp_quant_strat.forecasting.feature_set import FeatureSet
from nlp_quant_strat.data.feature_engineering import FeatureEngineering
from nlp_quant_strat.forecasting.earnings_wf_cv import (
    EarningsWalkForwardCV, EarningsWalkForwardResult, build_models_from_config,
)

def long_short_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean signed return: mean(sign(y_pred) * y_true).
    Directly measures expected P&L per unit long/short bet.
    Returns NaN if fewer than 2 valid observations or y_pred is constant.
    """
    mask = ~np.isnan(y_true)
    if mask.sum() < 2:
        return np.nan
    yt, yp = y_true[mask], y_pred[mask]
    if np.std(yp) == 0:
        return np.nan
    return float(np.mean(np.sign(yp) * yt))

# Build sentiment features once — shared across all modes
feature_eng = FeatureEngineering(config=config, data=data_manager)
feature_eng.build_features()

data_manager.release_mapping_texts()
data_manager.release_transcripts()
logger.info("Transcript text released from memory.")

# Build CV engine once — same models / hyperparams apply to all modes
models = build_models_from_config(config.forecasting_models)

cv = EarningsWalkForwardCV(
    models=models,
    scoring_func=long_short_scorer,
    refit_frequency=config.forecasting_refit_frequency,
    train_window=config.forecasting_train_window,
    val_window=config.forecasting_val_window,
    min_train_periods=config.forecasting_min_train_periods,
    min_train_events=config.forecasting_min_train_events,
    min_val_events=config.forecasting_min_val_events,
    load_or_compute=config.cv_results_load_or_compute,
    save_results=config.cv_results_save,
    s3_prefix=config.cv_results_s3_prefix,
    n_jobs=config.cv_results_n_jobs,
)

def _run_mode(mode: str):
    """
    Run STEP 3 (FeatureSet) + STEP 4 (EarningsWalkForwardCV) for one feature mode.

    Called in parallel by joblib.  All inputs (targets, feature_eng, embeddings,
    cv, config, data_manager) are captured from the enclosing scope and treated as
    read-only — FeatureSet.build() produces new DataFrames, cv clones estimators
    internally, so no shared mutable state exists between workers.

    Inner period-parallelism is disabled (n_jobs=1) to avoid oversubscribing the
    CPU when multiple modes run simultaneously.
    """
    # ------------------------------------------------------------------
    # STEP 3 — FeatureSet
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3 — FeatureSet  (mode=%s)", mode)
    logger.info("=" * 60)

    feature_set = FeatureSet(mode=mode)
    X, y = feature_set.build(
        targets=targets,
        feature_eng=feature_eng,
        tfidf_embeddings=tfidf_embeddings,
        st_embeddings=st_embeddings,
    )

    # --- Shape checks ---
    assert X.shape[0] == y.shape[0], \
        f"[mode={mode}] X and y row counts differ: {X.shape[0]} vs {y.shape[0]}"
    assert "filing_date" in X.columns and "asset" in X.columns, \
        f"[mode={mode}] X missing sentinel columns"
    _target_cols = [c for c in y.columns if c.startswith("idio_")]
    assert len(_target_cols) == len(config.forecasting_horizons), \
        f"[mode={mode}] Expected {len(config.forecasting_horizons)} target cols, got {len(_target_cols)}"

    logger.info("X shape              : %s", X.shape)
    logger.info("y shape              : %s", y.shape)
    logger.info("X columns (first 10) : %s", list(X.columns[:10]))
    logger.info("y columns            : %s", list(y.columns))

    _feat_cols = [c for c in X.columns if c not in {"filing_date", "asset"}]
    n_nan = X[_feat_cols].isna().sum().sum()
    assert n_nan == 0, f"[mode={mode}] X has {n_nan} NaN values after FeatureSet.build()"
    logger.info("NaN in X features: %d — OK", n_nan)

    x_keys = X[["filing_date", "asset"]].apply(tuple, axis=1)
    y_keys = y[["filing_date", "asset"]].apply(tuple, axis=1)
    assert (x_keys == y_keys).all(), f"[mode={mode}] X and y not row-aligned"
    logger.info("X/y row alignment — OK")
    logger.info("STEP 3 — OK  (mode=%s)\n", mode)

    # ------------------------------------------------------------------
    # STEP 4 — EarningsWalkForwardCV
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4 — EarningsWalkForwardCV  (mode=%s)", mode)
    logger.info("=" * 60)

    # Disable inner period parallelism: with n_modes workers already running,
    # letting each also spawn n_jobs=-1 threads would oversubscribe the CPU.
    cv_mode = dataclasses.replace(cv, n_jobs=1)
    mode_result: EarningsWalkForwardResult = cv_mode.run(
        x=X,
        y=y,
        horizons=config.forecasting_horizons,
        mode=mode,
        aws=data_manager.aws,
    )

    mode_result.predictions["mode"]       = mode
    mode_result.selection_history["mode"] = mode
    if not mode_result.oos_metrics.empty:
        mode_result.oos_metrics["mode"]   = mode

    logger.info("predictions       : %s", mode_result.predictions.shape)
    logger.info("selection_history : %s", mode_result.selection_history.shape)
    logger.info("STEP 4 — OK  (mode=%s)\n", mode)

    return mode_result.predictions, mode_result.selection_history, mode_result.oos_metrics


logger.info("Running %d mode(s) in parallel (n_jobs=%d): %s",
            len(config.feature_set_mode), len(config.feature_set_mode), config.feature_set_mode)

_mode_results = Parallel(n_jobs=len(config.feature_set_mode), prefer="threads")(
    delayed(_run_mode)(mode) for mode in config.feature_set_mode
)

all_predictions       = [r[0] for r in _mode_results]
all_selection_history = [r[1] for r in _mode_results]
all_oos_metrics       = [r[2] for r in _mode_results]

# --- Combine results across all modes ---
result = EarningsWalkForwardResult(
    predictions=pd.concat(all_predictions, ignore_index=True) if all_predictions
                else pd.DataFrame(columns=["period", "filing_date", "asset", "model", "horizon", "y_pred", "y_true", "mode"]),
    selection_history=pd.concat(all_selection_history, ignore_index=True) if all_selection_history
                      else pd.DataFrame(),
    oos_metrics=pd.concat(all_oos_metrics, ignore_index=True) if all_oos_metrics
                else pd.DataFrame(),
)

# --- Final cross-mode assertions ---
expected_pred_cols = {"filing_date", "asset", "model", "horizon", "y_pred", "y_true", "mode"}
assert expected_pred_cols.issubset(result.predictions.columns), \
    f"predictions missing columns: {expected_pred_cols - set(result.predictions.columns)}"
assert not result.predictions.empty,      "predictions DataFrame is empty"
assert not result.selection_history.empty, "selection_history DataFrame is empty"

logger.info("=== Combined results across %d mode(s): %s ===",
            len(config.feature_set_mode), config.feature_set_mode)
logger.info("predictions shape     : %s", result.predictions.shape)
logger.info("selection_history shape: %s", result.selection_history.shape)
logger.info("modes in output       : %s", sorted(result.predictions["mode"].unique().tolist()))
logger.info("models in output      : %s", result.predictions["model"].unique().tolist())
logger.info("horizons in output    : %s", sorted(result.predictions["horizon"].unique().tolist()))

# --- OOS metrics summary per mode / model / horizon ---
if not result.oos_metrics.empty:
    summary = (
        result.oos_metrics
        .groupby(["mode", "model", "horizon"])[["hit_rate", "spread", "mean_signed_ret", "t_stat", "excess_rmse"]]
        .mean()
        .reset_index()
    )
    for _, row in summary.iterrows():
        logger.info(
            "OOS | mode=%-16s | horizon=%2dd | model=%-6s | "
            "hit_rate=%.3f | spread=%.4f | mean_signed_ret=%.4f | t_stat=%.2f | excess_rmse=%.4f",
            row["mode"], row["horizon"], row["model"],
            row["hit_rate"], row["spread"], row["mean_signed_ret"], row["t_stat"], row["excess_rmse"],
        )
    logger.info("\nOOS metrics sample (10 rows):\n%s", result.oos_metrics.head(10).to_string())

scored = result.selection_history.dropna(subset=["val_score"])
logger.info("Folds with a valid val_score: %d / %d", len(scored), len(result.selection_history))

logger.info("\nSelection history sample (3 rows):\n%s", result.selection_history.head(3).to_string())
logger.info("\nPredictions sample (5 rows):\n%s", result.predictions.head(5).to_string())

logger.info("STEP 3+4 — OK\n")