from nlp_quant_strat.backtester import data, signal_utilities, strategies, portfolio, backtest_pandas, analysis
from nlp_quant_strat.utils.config import Config
from dotenv import load_dotenv
load_dotenv()
config = Config()

ds = data.AmazonS3(
    bucket_name=config.aws_bucket_name,
    s3_object_name="data/wrds_gross_query.parquet",
)
dm = data.DataManager(
    data_source=ds,
    max_consecutive_nan=5,
    rebase_prices=False,
    n_implementation_lags=1,
    already_returns=True
)
dm.get_data(
    format_date="%Y-%m-%d",
    crop_lookback_period=0
)

# Strategy setup and benchmark
strategy = strategies.CrossSectionalPercentiles(
    returns=dm.returns,
    signal_function=signal_utilities.Momentum.rolling_momentum,
    signal_function_inputs={"df": dm.cleaned_data,
                            "nb_period": 252,
                            "nb_period_to_exclude": 22,
                            "exclude_last_period": True
                            },
    percentiles_winsorization=(2,98)
)
strategy.compute_signals_values()
strategy.compute_signals(
    percentiles_portfolios=(10,90),
    industry_segmentation=None
)

bench = strategies.BuyAndHold(
    returns=dm.returns
)
bench.compute_signals_values()
bench.compute_signals()

# Portfolio level
ptf = portfolio.EqualWeightingScheme(
    returns=dm.returns,
    signals=strategy.signals,
    rebal_periods=22,
    portfolio_type="long_only"
)
ptf.compute_weights()
ptf.rebalance_portfolio()

ptf_bench = portfolio.EqualWeightingScheme(
    returns=dm.returns,
    signals=bench.signals,
    rebal_periods=22,
    portfolio_type="long_only"
)
ptf_bench.compute_weights()
ptf_bench.rebalance_portfolio()

# Backtesting
backtester = backtest_pandas.Backtest(
    returns=dm.returns,
    weights=ptf.rebalanced_weights.shift(1),  # Shift weights to account for implementation lag
    turnover=ptf.turnover,
    transaction_costs=10,
    strategy_name="CSMOM"
)
backtester.run_backtest()

backtester_bench = backtest_pandas.Backtest(
    returns=dm.returns,
    weights=ptf_bench.rebalanced_weights.shift(1),  # Shift weights to account for implementation lag
    turnover=ptf_bench.turnover,
    transaction_costs=10,
    strategy_name="BUY_AND_HOLD"
)
backtester_bench.run_backtest()


# Performance Analysis
analyzer = analysis.PerformanceAnalyser(
    portfolio_returns=backtester.cropped_portfolio_net_returns,
    bench_returns=backtester_bench.portfolio_net_returns.loc[backtester.start_date:, :],
    freq="d",
    percentiles=str((10,90)),
    # industries="Cross Industries" if self.industry_segmentation is None else "Intra Industries",
    industries="Cross_Industries",
    # rebal_freq=f"{self.rebal_periods} {self.freq_data}"
    rebal_freq=f"{22} {'d'}"
)
analyzer.compute_metrics()
analyzer.plot_cumulative_performance(saving_path="./outputs/backtest/figures/cumulative_performance.png")
bench_net_returns = backtester_bench.portfolio_net_returns.loc[backtester.start_date:, :]
