import logging
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Imports
from nlp_quant_strat.data.data_manager import DataManager
from nlp_quant_strat.data.feature_engineering import FeatureEngineering
from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.backtester.strategies import CrossSectionalPercentiles
from nlp_quant_strat.backtester.portfolio import EqualWeightingScheme
from nlp_quant_strat.backtester.backtest import Backtest
from nlp_quant_strat.backtester.analysis import PerformanceAnalyser
from nlp_quant_strat.backtester.visualization import Visualizer
from nlp_quant_strat.utils.utils import S3Utils

def main():
    # 1. Config & Logging
    load_dotenv()
    config = Config()
    
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("FinalAnalysis")

    # 2. Load Data
    logger.info("Loading data from S3...")
    data_manager = DataManager(config=config)
    data_manager.load_data()

    # 3. Features
    logger.info("Accessing NLP Features...")
    feature_engineering = FeatureEngineering(data=data_manager, config=config)
    feature_engineering.build_features()

    # 4. Multi-Factor Strategy (Refined Point 5)
    logger.info("Computing Multi-Factor Signal (Polarity + Delta)...")
    f1 = feature_engineering.polarity
    f2 = feature_engineering.polarity_delta

    # Combine Raw Signals (No redundant Z-score - Point 5)
    raw_composite = f1.add(f2, fill_value=0) / 2
    raw_composite = raw_composite.replace(0, np.nan) # Handle Sparsity (Point 4)

    logger.info(f"Active NLP signals identified: {raw_composite.notna().sum().sum()}")

    strategy = CrossSectionalPercentiles(
        returns=data_manager.get_asset_returns(),
        signal_values=raw_composite,
        percentiles_winsorization=config.percentiles_winsorization
    )
    
    strategy.compute_signals_values()
    signals = strategy.compute_signals(
        percentiles_portfolios=config.percentiles_portfolios,
        industry_segmentation=None if config.industry_segmentation == "" else "with_industry_segmentation"
    )

    # --- POINT 7: Signal Propagation ---
    signals = signals.replace(0, np.nan).ffill(limit=config.rebal_periods).fillna(0)
    logger.info(f"Signals after propagation: {(signals != 0).sum().sum()}")

    # 5. Portfolio Construction
    logger.info("Rebalancing Portfolio...")
    ptf = EqualWeightingScheme(
        returns=data_manager.get_asset_returns(),
        signals=signals,
        rebal_periods=config.rebal_periods,
        portfolio_type=config.portfolio_type
    )
    
    ptf.compute_weights()
    ptf.rebalance_portfolio()

    # 6. Backtest
    logger.info("Launching Backtest...")
    
    # --- CRITICAL ALIGNMENT FIX ---
    asset_returns = data_manager.get_asset_returns()
    weights_pd = ptf.rebalanced_weights
    
    # Ensure both use the same index name and type
    weights_pd.index.name = "date"
    asset_returns.index.name = "date"
    
    # Force both to be timezone-naive (common source of empty overlaps)
    if weights_pd.index.tz is not None:
        weights_pd.index = weights_pd.index.tz_localize(None)
    if asset_returns.index.tz is not None:
        asset_returns.index = asset_returns.index.tz_localize(None)

    # Check for date overlap before running
    overlap = weights_pd.index.intersection(asset_returns.index)
    logger.info(f"Date overlap between weights and returns: {len(overlap)} days")
    
    if len(overlap) == 0:
        logger.error("❌ ZERO overlap between weights and returns. Check index values!")
        logger.info(f"Weights index example: {weights_pd.index[0]}")
        logger.info(f"Returns index example: {asset_returns.index[0]}")
        return

    backtester = Backtest(
        returns=asset_returns,
        weights=weights_pd.shift(1), 
        turnover=ptf.turnover,
        transaction_costs=config.transaction_costs,
        strategy_name="NLP_MULTI_FACTOR" # Hardcoded string for safety
    )
    backtester.run_backtest()

    # 7. ANALYSE
    logger.info("Analyzing metrics...")
    
    # Get portfolio returns from backtester
    port_ret = backtester.cropped_portfolio_net_returns
    # Get benchmark returns
    bench_ret = data_manager.get_benchmark_returns()

    # --- THE DATE SYNCHRONIZER ---
    # 1. Force index names to match
    port_ret.index.name = "date"
    bench_ret.index.name = "date"

    # 2. Strip timezones from both (The most common fix)
    if port_ret.index.tz is not None:
        port_ret.index = port_ret.index.tz_localize(None)
    if bench_ret.index.tz is not None:
        bench_ret.index = bench_ret.index.tz_localize(None)

    # 3. Ensure they are both Datetime objects (not strings)
    port_ret.index = pd.to_datetime(port_ret.index)
    bench_ret.index = pd.to_datetime(bench_ret.index)

    # DEBUG PRINT: Let's see why they might miss each other
    common_dates = port_ret.index.intersection(bench_ret.index)
    logger.info(f"Common dates found for analysis: {len(common_dates)}")
    
    if len(common_dates) == 0:
        logger.error("❌ BENCHMARK ALIGNMENT FAILURE")
        logger.info(f"Portfolio start: {port_ret.index[0]} | End: {port_ret.index[-1]}")
        logger.info(f"Benchmark start: {bench_ret.index[0]} | End: {bench_ret.index[-1]}")
        return

    perf_analyzer = PerformanceAnalyser(
        portfolio_returns=port_ret,
        freq=config.market_data_frequency,
        bench_returns=bench_ret,
        percentiles=f"({config.percentiles_portfolios[0]}-{config.percentiles_portfolios[1]})",
        rebal_freq=f"{config.rebal_periods} days"
    )
    metrics = perf_analyzer.compute_metrics()

    # 8. Display Results (The "Indestructible" Version)
    print("\n" + "="*60)
    print(f"RESULTS : {config.strategy_name} (Signal: COMPOSITE)")
    print("="*60)

    # Check if metrics were returned or stored in the object
    final_metrics = metrics if metrics is not None else getattr(perf_analyzer, 'metrics', None)

    if final_metrics is not None:
        # If it's a Dictionary
        if isinstance(final_metrics, dict):
            for metric, value in final_metrics.items():
                val = f"{value:.4f}" if isinstance(value, (float, int)) else str(value)
                print(f"{metric:<30} : {val:>10}")
        # If it's a DataFrame or Series
        else:
            print(final_metrics)
    else:
        logger.error("❌ Could not find metrics in the return value or the analyzer object.")
        # Final fallback: manual print of key metrics if accessible
        try:
            print(f"Total Return      : {perf_analyzer.total_return:.4f}")
            print(f"Sharpe Ratio      : {perf_analyzer.sharpe_ratio:.4f}")
        except:
            print("Direct attribute access failed. Check PerformanceAnalyser class definition.")

    print("="*60)

    # 9. Visualization
    vizu = Visualizer(performance=perf_analyzer)
    vizu.plot_cumulative_performance(
        saving_path=config.ROOT_DIR / "outputs" / "figures" / f"{config.strategy_name}_final.png"
    )

if __name__ == "__main__":
    main()