"""
NLPQuantStrat — main entry point.

Phase 1 — ML Forecasting
    Loads data, builds CAPM-idio targets, loads embeddings, assembles
    feature matrices for each configured mode, and runs the quarterly
    walk-forward CV.  Results are saved to S3 when CV_RESULTS.SAVE_RESULTS
    is true in the config.

Phase 2 — Cross-Sectional Backtests
    Runs a long/short cross-sectional backtest for each sentiment feature
    produced by FeatureEngineering.  Saves performance charts to
    outputs/figures/<feature_name>/.

Both phases share one DataManager, one Config, and one FeatureEngineering
instance, so data is loaded and features are computed exactly once.
"""
import dataclasses
import logging
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Shared initialisation
# ---------------------------------------------------------------------------
from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.data.data_manager import DataManager
from nlp_quant_strat.data.feature_engineering import FeatureEngineering

_t_start = time.time()

logger.info("=" * 60)
logger.info("INITIALISATION — Config + Data + Features")
logger.info("=" * 60)
_t_init = time.time()

config = Config()
data_manager = DataManager(config=config)
data_manager.load_data()

asset_returns  = data_manager.get_asset_returns()
benchmark      = data_manager.get_benchmark_returns()
rf             = data_manager.get_rf_returns()
mapping_df     = data_manager.mapping_df

logger.info("asset_returns : %s  |  %s → %s", asset_returns.shape,
            asset_returns.index.min().date(), asset_returns.index.max().date())
logger.info("benchmark     : %s", benchmark.shape)
logger.info("mapping_df    : %d events", len(mapping_df))

# FeatureEngineering is shared — Phase 1 uses it for sentiment features,
# Phase 2 uses the resulting panel DataFrames for signal construction.
feature_eng = FeatureEngineering(config=config, data=data_manager)
feature_eng.build_features()

logger.info("INITIALISATION — OK  (%.1fs)\n", time.time() - _t_init)


# ===========================================================================
# PHASE 1 — ML Forecasting
# ===========================================================================

def run_forecasting_phase() -> "EarningsWalkForwardResult":  # noqa: F821
    from nlp_quant_strat.forecasting.targets import TargetBuilder
    from nlp_quant_strat.forecasting.embeddings import TFIDFBuilder, SentenceTransformerBuilder
    from nlp_quant_strat.forecasting.feature_set import FeatureSet
    from nlp_quant_strat.forecasting.earnings_wf_cv import (
        EarningsWalkForwardCV, EarningsWalkForwardResult, build_models_from_config,
    )

    # ------------------------------------------------------------------
    # STEP 1 — Targets
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1 | STEP 1 — TargetBuilder")
    logger.info("=" * 60)
    _t = time.time()

    target_builder = TargetBuilder(
        data=data_manager,
        beta_window=config.beta_window,
        horizons=config.forecasting_horizons,
    )
    targets = target_builder.build()
    logger.info("targets : %s  |  NaN rates: %s",
                targets.shape,
                {f"idio_{h}d": f"{targets[f'idio_{h}d'].isna().mean():.1%}"
                 for h in config.forecasting_horizons})
    logger.info("STEP 1 — OK  (%.1fs)\n", time.time() - _t)

    # ------------------------------------------------------------------
    # STEP 2 — Embeddings
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1 | STEP 2 — Embeddings")
    logger.info("=" * 60)
    _t = time.time()

    preprocessed_df = data_manager.get_formatted_preprocessed_transcripts()
    transcripts_df  = data_manager.get_formatted_unprocessed_transcripts()

    tfidf_builder = TFIDFBuilder(config=config)
    tfidf_embeddings = tfidf_builder.build(
        mapping_df=mapping_df,
        transcripts_df=preprocessed_df,
        aws=data_manager.aws,
    )

    st_builder = SentenceTransformerBuilder(config=config)
    st_embeddings = st_builder.build(
        mapping_df=mapping_df,
        transcripts_df=transcripts_df,
        aws=data_manager.aws,
    )

    logger.info("tfidf_embeddings : %s", tfidf_embeddings.shape)
    logger.info("st_embeddings    : %s", st_embeddings.shape)
    logger.info("STEP 2 — OK  (%.1fs)\n", time.time() - _t)

    # Transcripts no longer needed — free memory before the CV loop
    data_manager.release_mapping_texts()
    data_manager.release_transcripts()
    logger.info("Transcript DataFrames released from memory.")

    # ------------------------------------------------------------------
    # STEP 3 + 4 — FeatureSet → WalkForwardCV (parallel over modes)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1 | STEP 3+4 — FeatureSet → EarningsWalkForwardCV")
    logger.info("=" * 60)

    def long_short_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = ~np.isnan(y_true)
        if mask.sum() < 2:
            return np.nan
        yt, yp = y_true[mask], y_pred[mask]
        return np.nan if np.std(yp) == 0 else float(np.mean(np.sign(yp) * yt))

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
        feature_set = FeatureSet(mode=mode)
        X, y = feature_set.build(
            targets=targets,
            feature_eng=feature_eng,
            tfidf_embeddings=tfidf_embeddings,
            st_embeddings=st_embeddings,
        )
        logger.info("[mode=%s] X=%s  y=%s", mode, X.shape, y.shape)

        cv_mode = dataclasses.replace(cv, n_jobs=1)
        mode_result = cv_mode.run(
            x=X, y=y,
            horizons=config.forecasting_horizons,
            mode=mode,
            aws=data_manager.aws,
        )
        mode_result.predictions["mode"]       = mode
        mode_result.selection_history["mode"] = mode
        if not mode_result.oos_metrics.empty:
            mode_result.oos_metrics["mode"]   = mode
        return mode_result.predictions, mode_result.selection_history, mode_result.oos_metrics

    logger.info("Running %d mode(s): %s", len(config.feature_set_mode), config.feature_set_mode)
    _t = time.time()

    _gen = Parallel(
        n_jobs=len(config.feature_set_mode), prefer="threads", return_as="generator_unordered"
    )(delayed(_run_mode)(mode) for mode in config.feature_set_mode)
    _mode_results = list(
        tqdm(_gen, total=len(config.feature_set_mode), desc="Feature modes", unit="mode", leave=True)
    )

    logger.info("All modes done  (%.1fs)\n", time.time() - _t)

    all_preds  = [r[0] for r in _mode_results]
    all_sel    = [r[1] for r in _mode_results]
    all_oos    = [r[2] for r in _mode_results]

    result = EarningsWalkForwardResult(
        predictions=pd.concat(all_preds, ignore_index=True) if all_preds
                    else pd.DataFrame(columns=["period", "filing_date", "asset",
                                               "model", "horizon", "y_pred", "y_true", "mode"]),
        selection_history=pd.concat(all_sel, ignore_index=True) if all_sel else pd.DataFrame(),
        oos_metrics=pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame(),
    )

    # Summary
    logger.info("=== Phase 1 results — %d mode(s) ===", len(config.feature_set_mode))
    logger.info("predictions       : %s", result.predictions.shape)
    logger.info("selection_history : %s", result.selection_history.shape)
    if not result.oos_metrics.empty:
        summary = (
            result.oos_metrics
            .groupby(["mode", "model", "horizon"])[
                ["hit_rate", "spread", "mean_signed_ret", "t_stat", "excess_rmse"]
            ]
            .mean()
            .reset_index()
        )
        for _, row in summary.iterrows():
            logger.info(
                "OOS | mode=%-16s | h=%2dd | model=%-6s | "
                "hit_rate=%.3f | spread=%.4f | msr=%.4f | t=%.2f | xrmse=%.4f",
                row["mode"], row["horizon"], row["model"],
                row["hit_rate"], row["spread"], row["mean_signed_ret"],
                row["t_stat"], row["excess_rmse"],
            )

    logger.info("PHASE 1 — OK\n")
    return result


# ===========================================================================
# PHASE 2 — Cross-Sectional Backtests
# ===========================================================================

def run_backtest_phase() -> None:
    from nlp_quant_strat.backtester.strategies import CrossSectionalPercentiles
    from nlp_quant_strat.backtester.portfolio import EqualWeightingScheme
    from nlp_quant_strat.backtester.backtest import Backtest
    from nlp_quant_strat.backtester.analysis import PerformanceAnalyser
    from nlp_quant_strat.backtester.visualization import Visualizer

    logger.info("=" * 60)
    logger.info("PHASE 2 — Cross-Sectional Backtests")
    logger.info("=" * 60)
    _t = time.time()

    features_to_backtest = feature_eng.feature_names
    logger.info("Features to backtest: %s", features_to_backtest)

    for feature_name in features_to_backtest:
        signal_values = getattr(feature_eng, feature_name, None)
        if signal_values is None:
            logger.warning("feature_eng.%s is None — skipping.", feature_name)
            continue

        logger.info("--- Backtesting feature: %s ---", feature_name)
        _tf = time.time()

        out_dir = config.ROOT_DIR / "outputs" / "figures" / feature_name
        out_dir.mkdir(parents=True, exist_ok=True)

        strategy = CrossSectionalPercentiles(
            returns=asset_returns,
            signal_function=None,
            signal_function_inputs=None,
            signal_values=signal_values,
            percentiles_winsorization=config.percentiles_winsorization,
        )
        strategy.compute_signals_values()
        signals = strategy.compute_signals(
            percentiles_portfolios=config.percentiles_portfolios,
            industry_segmentation=(
                None if config.industry_segmentation == ""
                else "with_industry_segmentation"
            ),
        )

        ptf = EqualWeightingScheme(
            returns=asset_returns,
            signals=signals,
            rebal_periods=config.rebal_periods,
            portfolio_type=config.portfolio_type,
        )
        ptf.compute_weights()
        ptf.rebalance_portfolio()

        backtester = Backtest(
            returns=asset_returns,
            weights=ptf.rebalanced_weights.shift(1),  # shift to avoid lookahead bias
            turnover=ptf.turnover,
            transaction_costs=config.transaction_costs,
            strategy_name=feature_name,
        )
        backtester.run_backtest()

        perf = PerformanceAnalyser(
            portfolio_returns=backtester.cropped_portfolio_net_returns,
            freq=config.market_data_frequency,
            zscores=None,
            bench_returns=benchmark,
            forward_returns=None,
            percentiles=f"({config.percentiles_portfolios[0]}-{config.percentiles_portfolios[1]})",
            industries=(
                "without ind. seg." if config.industry_segmentation == ""
                else "with industries segmentation"
            ),
            rebal_freq=f"{config.rebal_periods} days",
        )
        perf.compute_metrics()

        vizu = Visualizer(performance=perf)
        vizu.plot_cumulative_performance(
            saving_path=out_dir / f"{feature_name}_cumulative_returns.png"
        )
        for metric in ["sharpe", "return", "vol"]:
            vizu.plot_rolling_metric(
                metric=metric,
                saving_path=out_dir / f"{feature_name}_rolling_{metric}.png",
                window=config.rolling_window_performance,
            )
        for metric in ["sharpe", "annualized_return", "vol"]:
            vizu.plot_yearly_metrics(
                metric=metric,
                saving_path=out_dir / f"{feature_name}_yearly_{metric}.png",
            )

        logger.info("  %s — done  (%.1fs)", feature_name, time.time() - _tf)

    logger.info("PHASE 2 — OK  (%.1fs)\n", time.time() - _t)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    forecasting_result = run_forecasting_phase()
    run_backtest_phase()
    logger.info("Pipeline complete  (total: %.1fs)", time.time() - _t_start)