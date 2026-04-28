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

def run_backtest_phase() -> pd.DataFrame:
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

    all_perfs: list = []  # collect (name, PerformanceAnalyser) for comparison plots

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
        perf.compute_rolling_metrics(window=config.rolling_window_performance)
        all_perfs.append((feature_name, perf))

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

    # ------------------------------------------------------------------
    # Cross-strategy comparison plots + summary table
    # ------------------------------------------------------------------
    summary_df = _backtest_comparison(
        all_perfs=all_perfs,
        out_dir=config.ROOT_DIR / "outputs" / "figures" / "comparison",
        rolling_window=config.rolling_window_performance,
    )

    logger.info("=== Strategy summary ===\n%s", summary_df.to_string())
    logger.info("PHASE 2 — OK  (%.1fs)\n", time.time() - _t)
    return summary_df


def _backtest_comparison(all_perfs: list, out_dir, rolling_window: int) -> pd.DataFrame:
    """
    Generate cross-strategy comparison plots and a summary metrics table.

    Parameters
    ----------
    all_perfs : list of (name: str, perf: PerformanceAnalyser)
        Collected from run_backtest_phase(); rolling_metrics must already be computed.
    out_dir : Path
        Directory where comparison figures are saved.
    rolling_window : int
        Window used to compute rolling metrics (for axis labels).

    Returns
    -------
    pd.DataFrame
        Summary table: strategies as columns, metrics as rows.
        Includes a "Benchmark" column derived from the first PerformanceAnalyser.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    if not all_perfs:
        return pd.DataFrame()

    names   = [n for n, _ in all_perfs]
    perfs   = [p for _, p in all_perfs]
    cmap    = plt.get_cmap("tab10")
    colors  = [cmap(i / max(len(names), 1)) for i in range(len(names))]

    # ------------------------------------------------------------------
    # 1. Combined cumulative returns
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    for (name, perf), color in zip(all_perfs, colors):
        cum = perf.cumulative_performance_base_100.squeeze()
        ax.plot(cum.index, cum.values, label=name, color=color)

    # Benchmark (same series for all strategies — use first)
    bench_cum = perfs[0].bench_cumulative_perf_base_100
    if bench_cum is not None:
        bench_cum = bench_cum.squeeze()
        ax.plot(bench_cum.index, bench_cum.values,
                label="Benchmark", color="black", linestyle="--", linewidth=1.5)

    ax.set_title(f"Cumulative Performance — all strategies (base 100)", fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Performance (base 100)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_cumulative_returns.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Combined rolling metrics (Sharpe, Return, Vol)
    # ------------------------------------------------------------------
    for metric, ylabel in [
        ("sharpe", f"Rolling Sharpe ({rolling_window}d)"),
        ("return", f"Rolling Ann. Return ({rolling_window}d)"),
        ("vol",    f"Rolling Volatility ({rolling_window}d)"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))
        for (name, perf), color in zip(all_perfs, colors):
            series = perf.rolling_metrics[metric].squeeze()
            ax.plot(series.index, series.values, label=name, color=color)

        bench_rolling = perfs[0].rolling_metrics_bench
        if bench_rolling is not None:
            b = bench_rolling[metric].squeeze()
            ax.plot(b.index, b.values,
                    label="Benchmark", color="black", linestyle="--", linewidth=1.5)

        ax.set_title(ylabel, fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / f"comparison_rolling_{metric}.png", bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Summary metrics table
    # ------------------------------------------------------------------
    metric_keys = [
        ("annualized_return",    "Ann. Return"),
        ("annualized_volatility", "Ann. Vol"),
        ("annualized_sharpe_ratio", "Sharpe"),
        ("max_drawdown",         "Max DD"),
    ]
    rows = {}
    for mk, label in metric_keys:
        row = {name: perf.metrics[mk] for name, perf in all_perfs}
        # Benchmark column (take scalar from first perf; bench metrics are Series indexed by col name)
        bench_val = perfs[0].metrics.get(f"{mk}_bench")
        if bench_val is not None:
            try:
                row["Benchmark"] = float(bench_val.iloc[0])
            except (AttributeError, TypeError):
                row["Benchmark"] = float(bench_val)
        rows[label] = row

    summary_df = pd.DataFrame(rows).T
    summary_df.index.name = "Metric"

    # Save as CSV
    summary_df.to_csv(out_dir / "comparison_summary.csv")

    # Save as a simple matplotlib table figure
    fig, ax = plt.subplots(figsize=(max(8, int(len(summary_df.columns) * 1.4) + 1), 2.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary_df.applymap(lambda v: f"{v:.3f}" if pd.notna(v) else "—").values,
        rowLabels=summary_df.index.tolist(),
        colLabels=summary_df.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax.set_title("Strategy comparison — full-period metrics", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_summary_table.png", bbox_inches="tight")
    plt.close(fig)

    return summary_df


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    forecasting_result = run_forecasting_phase()
    run_backtest_phase()
    logger.info("Pipeline complete  (total: %.1fs)", time.time() - _t_start)