"""
A module for visualizing the results of the backtest, including equity curves, drawdowns, and cumulative performance.
"""
import matplotlib.pyplot as plt
from nlp_quant_strat.backtester.analysis import PerformanceAnalyser
import pandas as pd
import numpy as np
from itertools import cycle
import logging

logger = logging.getLogger(__name__)


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

        logger.info("Plotting cumulative performance...")

        # --- Ensure computations are done ---
        if self.performance.cumulative_performance is None:
            self.performance.compute_cumulative_performance()
        if self.performance.metrics is None:
            self.performance.compute_metrics()

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
                f"Strategy: {strategy_names} | {self.performance.percentiles} | "
                f"{self.performance.industries} | {self.performance.rebal_freq}\n"
                f"Metrics: "
                f"ann.ret={self.performance.metrics['annualized_return']:.2%}, "
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

            # --- Plot benchmark with different styles ---
            line_styles = cycle(['--', '-.', ':'])
            markers = cycle(['o', 's', 'D', '^', 'v'])

            for col in bench_df.columns:
                plt.plot(
                    bench_df.index,
                    bench_df[col],
                    label=f"Bench - {col}",
                    linestyle=next(line_styles),
                    marker=next(markers),
                    markevery=max(len(bench_df) // 20, 1)
                )

            # --- Names ---
            strategy_names = ", ".join(strategy_df.columns)

            # --- Build benchmark metrics string ---
            bench_metrics_str = ""

            for col in bench_df.columns:
                bench_metrics_str += (
                    f"{col}: "
                    f"ret={self.performance.metrics['annualized_return_bench'][col]:.2%}, "
                    f"vol={self.performance.metrics['annualized_volatility_bench'][col]:.2%}, "
                    f"SR={self.performance.metrics['annualized_sharpe_ratio_bench'][col]:.2f}, "
                    f"maxDD={self.performance.metrics['max_drawdown_bench'][col]:.2%}\n"
                )

            # --- Title ---
            plt.title(
                f"Cumulative Performance\n"
                f"Strategy: {strategy_names} | {self.performance.percentiles} | "
                f"{self.performance.industries} | {self.performance.rebal_freq}\n"
                f"Strategy Metrics: "
                f"ann.ret={self.performance.metrics['annualized_return']:.2%}, "
                f"ann.vol={self.performance.metrics['annualized_volatility']:.2%}, "
                f"SR={self.performance.metrics['annualized_sharpe_ratio']:.2f}, "
                f"maxDD={self.performance.metrics['max_drawdown']:.2%}\n"
                f"Bench Metrics:\n{bench_metrics_str}",
                fontsize=10
            )

            # =========================
            # ===== COMMON STYLE ======
            # =========================
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

            logger.info("Plotted cumulative performance.")

    def plot_rolling_metric(
            self,
            metric: str = "sharpe",
            window: int = 252,
            saving_path=None,
            show=False,
            blocking=True
    ) -> None:
        """
        Plot a rolling metric (e.g. rolling Sharpe ratio) for the strategy and benchmark if available.
        :param metric: metric among "return", "vol", "sharpe"
        :param window: the rolling window (e.g. 252 for 1 year), must be the same passed to compute_rolling_metrics(
         method of PerformanceAnalyser
        :param saving_path: the path to save the plot (if None, the plot won't be saved)
        :param show: the flag to show the plot (if False, the plot won't be shown)
        :param blocking: the flag to block the execution when showing the plot
        :return:
        """
        logger.info(f"Plotting rolling {metric} with window {window}...")
        plt.figure(figsize=(12, 6))

        # --- Ensure rolling metrics are computed ---
        if getattr(self.performance, "rolling_metrics", None) is None:
            self.performance.compute_rolling_metrics(window=window)

        # --- Get strategy metric ---
        if metric not in self.performance.rolling_metrics:
            raise ValueError(f"Metric '{metric}' not available")

        rolling = self.performance.rolling_metrics[metric]

        if isinstance(rolling, pd.Series):
            rolling = rolling.to_frame()

        # --- Plot strategy ---
        for col in rolling.columns:
            plt.plot(
                rolling.index,
                rolling[col],
                label=f"Strategy - {col}"
            )

        # =====================================
        # ===== BENCHMARK ======================
        # =====================================
        if self.performance.bench_returns is not None:

            if getattr(self.performance, "rolling_metrics_bench", None) is None:
                self.performance.compute_rolling_metrics(window=window)

            bench_rolling = self.performance.rolling_metrics_bench[metric]

            if isinstance(bench_rolling, pd.Series):
                bench_rolling = bench_rolling.to_frame()

            line_styles = cycle(['--', '-.', ':'])

            for col in bench_rolling.columns:
                plt.plot(
                    bench_rolling.index,
                    bench_rolling[col],
                    label=f"Bench - {col}",
                    linestyle=next(line_styles)
                )

        # --- Labels ---
        ylabel_map = {
            "return": "Rolling Annualized Return",
            "vol": "Rolling Volatility",
            "sharpe": "Rolling Sharpe"
        }

        ylabel = f"{ylabel_map.get(metric, metric)} ({window})"

        title = (
            f"{ylabel}\n"
            f"{self.performance.percentiles} | "
            f"{self.performance.industries} | {self.performance.rebal_freq}"
        )

        plt.title(title, fontsize=10)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()

        # --- Save / Show ---
        if saving_path is not None:
            plt.savefig(saving_path, bbox_inches='tight')

        if show:
            plt.show(block=blocking)

        plt.close()
        logger.info(f"Plotted rolling {metric} with window {window}.")

    def plot_yearly_metrics(
            self,
            metric: str = "annualized_return",
            saving_path=None,
            show=True,
            blocking=False
    ):
        """Plot yearly bar comparison between strategy and benchmarks"""

        logger.info(f"Plotting yearly {metric} comparison...")
        # --- Ensure metrics computed ---
        if getattr(self.performance, "yearly_metrics", None) is None:
            self.performance.compute_yearly_metrics()

        # --- Stack strategy ---
        strat_df = self.performance.stack_yearly_metrics(
            self.performance.yearly_metrics
        )
        strat_df["type"] = "Strategy"

        # --- Stack benchmark ---
        if self.performance.bench_returns is not None:
            bench_df = self.performance.stack_yearly_metrics(
                self.performance.yearly_metrics_bench
            )
            bench_df["type"] = "Bench"

            full_df = pd.concat([strat_df, bench_df], axis=0)
        else:
            full_df = strat_df

        # --- Validate metric ---
        valid_metrics = ["total_return", "annualized_return", "vol", "sharpe"]
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric. Choose from {valid_metrics}")

        df = full_df[["year", "asset", "type", metric]].copy()

        # --- Create label ---
        df["label"] = df["type"] + " - " + df["asset"]

        # --- Pivot ---
        pivot = df.pivot(index="year", columns="label", values=metric)

        # --- Sort years ---
        pivot = pivot.sort_index()

        # --- Plot ---
        plt.figure(figsize=(12, 6))

        years = pivot.index.to_list()
        n_series = pivot.shape[1]

        # safer bar width
        bar_width = min(0.8 / n_series, 0.2)

        x = np.arange(len(years))

        for i, col in enumerate(pivot.columns):
            plt.bar(
                x + i * bar_width,
                pivot[col].values,
                width=bar_width,
                label=col
            )

        # center ticks
        plt.xticks(
            x + bar_width * (n_series - 1) / 2,
            years
        )

        # --- Labels ---
        ylabel_map = {
            "total_return": "Total Return",
            "annualized_return": "Annualized Return",
            "vol": "Volatility",
            "sharpe": "Sharpe Ratio"
        }

        plt.ylabel(ylabel_map[metric])

        plt.title(
            f"Yearly {ylabel_map[metric]} Comparison\n"
            f"{self.performance.percentiles} | "
            f"{self.performance.industries} | {self.performance.rebal_freq}",
            fontsize=10
        )

        plt.legend()
        plt.grid(axis="y")

        # --- Save / Show ---
        if saving_path is not None:
            plt.savefig(saving_path, bbox_inches='tight')

        if show:
            plt.show(block=blocking)

        plt.close()
        logger.info(f"Plotted yearly {metric} comparison.")