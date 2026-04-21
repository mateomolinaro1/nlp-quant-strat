from nlp_quant_strat.data.data_manager import DataManager
from nlp_quant_strat.data.feature_engineering import FeatureEngineering
from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.backtester.strategies import CrossSectionalPercentiles
from nlp_quant_strat.backtester.portfolio import EqualWeightingScheme
from nlp_quant_strat.backtester.backtest import Backtest
from nlp_quant_strat.backtester.analysis import PerformanceAnalyser
from nlp_quant_strat.backtester.visualization import Visualizer
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

# !!! au shift des weights pour ne pas faire de lookahead bias
backtester = Backtest(
    returns=data_manager.get_asset_returns(),
    weights=ptf.rebalanced_weights.shift(1),
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
    percentiles=f"({config.percentiles_portfolios[0]}-{config.percentiles_portfolios[1]})",
    industries="without ind. seg." if config.industry_segmentation == "" else "with industries segmentation",
    rebal_freq=f"{config.rebal_periods} days"
)
perf_analyzer.compute_metrics()

vizu = Visualizer(performance=perf_analyzer)
vizu.plot_cumulative_performance(
    saving_path=config.ROOT_DIR / "outputs" / "figures" / f"{config.strategy_name}_cumulative_returns.png"
)
for metric in ["sharpe", "return", "vol"]:
    vizu.plot_rolling_metric(
        metric=metric,
        saving_path=config.ROOT_DIR / "outputs" / "figures" /
                    f"{config.strategy_name}_rolling_{metric}.png",
        window=config.rolling_window_performance
    )
for metric in ["sharpe", "annualized_return", "vol"]:
    vizu.plot_yearly_metrics(
        metric=metric,
        saving_path=config.ROOT_DIR / "outputs" / "figures" / f"{config.strategy_name}_yearly_{metric}.png"
    )

# End