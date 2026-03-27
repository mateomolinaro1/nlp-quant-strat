"""
A module for visualizing the results of the backtest, including equity curves, drawdowns, and cumulative performance.
"""
import matplotlib.pyplot as plt
from nlp_quant_strat.backtester.analysis import PerformanceAnalyser
import pandas as pd
from itertools import cycle


class Visualizer:
    """Visualize results of the backtest"""

    def __init__(self, performance:PerformanceAnalyser):
        self.performance = performance

    def plot_equity_curve(self, title="Equity Curve", figsize=(10, 6)):
        """Display the equity curve"""
        plt.figure(figsize=figsize)
        plt.plot(self.performance.equity_curve, label="Equity Curve")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show(block=True)

    def plot_drawdowns(self, title="Drawdowns", figsize=(10, 6)):
        """Display the drawdowns"""
        if self.performance.cumulative_performance is None:
            self.performance.compute_cumulative_performance()

        rolling_max = self.performance.cumulative_performance.cummax()
        drawdown = (self.performance.cumulative_performance / rolling_max) - 1

        plt.figure(figsize=figsize)
        plt.plot(drawdown, label="Drawdowns")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.fill_between(drawdown.index, drawdown.iloc[:, 0], 0, color='red', alpha=0.3)
        plt.show(block=True)

    def plot_cumulative_performance(self,
                                    saving_path:str=None,
                                    show:bool=False,
                                    blocking:bool=True):
        """Plot the cumulative performance of the strategy"""
        if self.performance.cumulative_performance is None:
            self.performance.compute_cumulative_performance()
        if self.performance.metrics is None:
            self.performance.metrics = self.performance.compute_metrics()

        plt.figure(figsize=(12, 6))

        # --- Ensure DataFrame format (even if Series) ---
        strategy_df = self.performance.cumulative_performance_base_100
        if isinstance(strategy_df, pd.Series):
            strategy_df = strategy_df.to_frame()

        # --- Plot Strategy ---
        for col in strategy_df.columns:
            plt.plot(
                strategy_df.index,
                strategy_df[col],
                label=f"Strategy - {col}"
            )

        # --- If NO benchmark ---
        if self.performance.bench_returns is None:

            strategy_names = ", ".join(strategy_df.columns)

            plt.title(
                f"Cumulative Performance\n"
                f"Strategy: {strategy_names} | {self.performance.percentiles} | {self.performance.industries} | {self.performance.rebal_freq}\n"
                f"Metrics: ann.ret={self.performance.metrics['annualized_return']:.2%}, "
                f"ann.vol={self.performance.metrics['annualized_volatility']:.2%}, "
                f"SR={self.performance.metrics['annualized_sharpe_ratio']:.2f}, "
                f"maxDD={self.performance.metrics['max_drawdown']:.2%}",
                fontsize=10
            )

        # --- If benchmark exists ---
        else:

            bench_df = self.performance.bench_cumulative_perf_base_100
            if isinstance(bench_df, pd.Series):
                bench_df = bench_df.to_frame()

            # Plot benchmark
            line_styles = cycle(['--', '-.', ':'])  # dashed variations
            markers = cycle(['o', 's', 'D', '^', 'v'])  # optional markers

            for col in bench_df.columns:
                plt.plot(
                    bench_df.index,
                    bench_df[col],
                    label=f"Bench - {col}",
                    linestyle=next(line_styles),
                    marker=next(markers),  # remove this line if too busy
                    markevery=len(bench_df) // 20  # avoid too many markers
                )

            # Clean names
            strategy_names = ", ".join(strategy_df.columns)
            bench_names = ", ".join(bench_df.columns)

            plt.title(
                f"Cumulative Performance\n"
                f"Strategy: {strategy_names} | {self.performance.percentiles} | {self.performance.industries} | {self.performance.rebal_freq}\n"
                f"Strategy Metrics: ann.ret={self.performance.metrics['annualized_return']:.2%}, "
                f"ann.vol={self.performance.metrics['annualized_volatility']:.2%}, "
                f"SR={self.performance.metrics['annualized_sharpe_ratio']:.2f}, "
                f"maxDD={self.performance.metrics['max_drawdown']:.2%}\n"
                f"Bench: {bench_names}\n"
                f"Bench Metrics: ann.ret={self.performance.metrics['annualized_return_bench']:.2%}, "
                f"ann.vol={self.performance.metrics['annualized_volatility_bench']:.2%}, "
                f"SR={self.performance.metrics['annualized_sharpe_ratio_bench']:.2f}, "
                f"maxDD={self.performance.metrics['max_drawdown_bench']:.2%}",
                fontsize=10
            )

        # --- Common formatting ---
        plt.xlabel("Date")
        plt.ylabel("Performance (Base 100)")
        plt.legend()
        plt.grid()

        # --- Save / Show ---
        if saving_path is not None:
            plt.savefig(saving_path, bbox_inches='tight')

        if show:
            plt.show(block=blocking)

        plt.close()