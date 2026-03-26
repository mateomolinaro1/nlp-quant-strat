from nlp_quant_strat.data.data_manager import DataManager
from nlp_quant_strat.data.feature_engineering import FeatureEngineering
from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.backtester.strategies import CrossSectionalPercentiles
from nlp_quant_strat.backtester.portfolio import EqualWeightingScheme
from nlp_quant_strat.backtester.backtest import Backtest
from nlp_quant_strat.backtester.analysis import PerformanceAnalyser
from dotenv import load_dotenv
import logging
import sys

load_dotenv()
config = Config()
logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# Data
data_manager = DataManager(config=config)
data_manager.load_data()

# Features
feature_engineering = FeatureEngineering(data=data_manager, config=config)
feature_engineering.build_features()

# Backtest
strategy = CrossSectionalPercentiles(
    returns=data_manager.get_asset_returns(),
    signal_function=None,
    signal_function_inputs=None,
    signal_values=feature_engineering.positive_count,
    percentiles_winsorization=config.percentiles_winsorization,
)
strategy.compute_signals_values()
signals = strategy.compute_signals(
    percentiles_portfolios=config.percentiles_portfolios,
    industry_segmentation=None if config.industry_segmentation == "" else "with_industry_segmentation",
)

ptf = EqualWeightingScheme(
    returns=data_manager.get_asset_returns(),
    signals=signals,
    rebal_periods=config.rebal_periods,
    portfolio_type=config.portfolio_type
)
ptf.compute_weights()
ptf.rebalance_portfolio()

backtester = Backtest(
    returns=data_manager.get_asset_returns(),
    weights=ptf.rebalanced_weights,
    turnover=ptf.turnover,
    transaction_costs=config.transaction_costs,
    strategy_name=config.strategy_name
)
backtester.run_backtest()

perf_analyzer = PerformanceAnalyser(
    portfolio_returns=backtester.cropped_portfolio_net_returns,
    freq=config.market_data_frequency,
    zscores=None,
    bench_returns=data_manager.get_benchmark_returns(),
    forward_returns=None,
    percentiles=f"{config.percentiles_portfolios[0]}-{config.percentiles_portfolios[1]}",
    industries="" if config.industry_segmentation == "" else "with_industries_segmentation",
    rebal_freq=f"{config.rebal_periods}D"
)
perf_analyzer.compute_metrics()
# bench needs to compute returns and aligns on strategy returns index
import matplotlib.pyplot as plt
# save a plot of the cumulative returns of the strategy
plt.figure(figsize=(10, 6))
plt.plot(perf_analyzer.cumulative_performance, label=config.strategy_name)
plt.savefig(config.ROOT_DIR / "outputs" / "figures" / f"{config.strategy_name}_cumulative_returns.png")
plt.close()
