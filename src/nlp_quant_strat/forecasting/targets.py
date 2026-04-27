"""
CAPM-based idiosyncratic return target builder for the earnings forecasting pipeline.

For each earnings event (filing_date, asset) in mapping_df, computes cumulative
idiosyncratic returns at multiple forward horizons using rolling-window CAPM betas
estimated exclusively on data up to (and including) the filing date.

Methodology
-----------
Rolling OLS beta at date t:
    beta(t) = Cov[r_i - rf, r_m - rf](t-w+1:t) / Var[r_m - rf](t-w+1:t)

Daily idiosyncratic return:
    idio(t) = (r_i(t) - rf(t)) - beta(t) * (r_m(t) - rf(t))

Cumulative idio return at horizon h for an event filed at date t:
    target = prod(1 + idio(s), s=t+1..t+h) - 1

The beta used is always estimated at t, so there is no forward-looking bias.
The return window [t+1, t+h] starts the day after the filing date to capture
the post-earnings-announcement market reaction without using same-day prices.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from nlp_quant_strat.data.data_manager import DataManager

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS: List[int] = [1, 5, 21, 63]
DEFAULT_BETA_WINDOW: int = 252


class TargetBuilder:
    """
    Builds multi-horizon cumulative CAPM-idiosyncratic return targets aligned
    to the earnings event panel in mapping_df.

    Parameters
    ----------
    data : DataManager
        Loaded DataManager instance (load_data() must have been called).
    beta_window : int
        Rolling window (in trading days) for estimating CAPM betas. Default 252.
    horizons : list[int] | None
        Forward horizons in trading days. Default [1, 5, 21, 63].
    """

    def __init__(
        self,
        data: DataManager,
        beta_window: int = DEFAULT_BETA_WINDOW,
        horizons: List[int] | None = None,
    ):
        self.data = data
        self.beta_window = beta_window
        self.horizons = sorted(horizons or DEFAULT_HORIZONS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """
        Compute targets for every unique (filing_date, asset) event.

        Returns
        -------
        pd.DataFrame
            Columns: filing_date, asset, idio_{h}d for each h in self.horizons.
            Rows near the end of the return series will have NaN targets where
            insufficient forward data exists.
        """
        mapping_df = self.data.mapping_df
        if mapping_df is None or mapping_df.empty:
            raise ValueError("mapping_df is None or empty. Call data.load_data() first.")

        asset_returns = self.data.get_asset_returns()
        benchmark_returns = self.data.get_benchmark_returns()
        rf_returns = self.data.get_rf_returns()

        # --- 1. Build excess return series ---
        rf = rf_returns.squeeze()                          # Series (date,)
        mkt_exc = benchmark_returns.squeeze() - rf         # Series (date,)
        asset_exc = asset_returns.sub(rf, axis=0)          # DataFrame (date x asset)

        # --- 2. Rolling CAPM betas ---
        logger.info("Computing rolling CAPM betas (window=%d days)...", self.beta_window)
        betas = self._rolling_betas(asset_exc, mkt_exc)

        # --- 3. Daily idiosyncratic returns ---
        logger.info("Computing daily idiosyncratic returns...")
        idio_daily = asset_exc - betas.mul(mkt_exc, axis=0)

        # --- 4. Forward cumulative idio returns per horizon ---
        logger.info("Building forward cumulative targets (horizons=%s)...", self.horizons)
        horizon_frames = self._forward_cumulative(idio_daily)

        # --- 5. Look up target for each event ---
        logger.info("Aligning targets to %d earnings events...", len(mapping_df))
        result = self._align_to_events(mapping_df, horizon_frames)

        nan_rates = {
            f"idio_{h}d": f"{result[f'idio_{h}d'].isna().mean():.1%}"
            for h in self.horizons
        }
        logger.info("Targets built. Events: %d | NaN rates: %s", len(result), nan_rates)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rolling_betas(
        self, asset_exc: pd.DataFrame, mkt_exc: pd.Series
    ) -> pd.DataFrame:
        """
        Rolling OLS beta for each asset.

        beta = Cov(r_i_exc, r_m_exc) / Var(r_m_exc)  over a rolling window.
        Returns a DataFrame of the same shape as asset_exc (NaN for first
        beta_window - 1 rows).
        """
        rolling_cov = asset_exc.rolling(self.beta_window).cov(mkt_exc)
        rolling_var = mkt_exc.rolling(self.beta_window).var()
        return rolling_cov.div(rolling_var, axis=0)

    def _forward_cumulative(
        self, idio_daily: pd.DataFrame
    ) -> Dict[int, pd.DataFrame]:
        """
        Vectorised computation of forward cumulative idio returns.

        For each horizon h, the resulting DataFrame has the same index/columns
        as idio_daily. Entry [t, asset] = cumulative idio over [t+1, t+h].

        Trick: rolling(h).sum() at t covers [t-h+1 .. t].
               Shifting by -h maps index t to [t+1 .. t+h].
        Log-sum is used for numerical stability.
        """
        log_idio = np.log1p(idio_daily.clip(lower=-0.9999))
        result: Dict[int, pd.DataFrame] = {}
        for h in self.horizons:
            log_cum = log_idio.rolling(h).sum().shift(-h)
            result[h] = np.expm1(log_cum)
        return result

    @staticmethod
    def _align_to_events(
        mapping_df: pd.DataFrame,
        horizon_frames: Dict[int, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Look up the cumulative target for each (filing_date, asset) event.

        Uses NumPy positional indexing to avoid materialising full long-format
        DataFrames. Events are deduplicated on (filing_date, asset) before lookup.
        """
        events = (
            mapping_df[["filing_date", "asset"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .copy()
        )

        for h, tgt in horizon_frames.items():
            col = f"idio_{h}d"
            tgt_arr = tgt.values
            date_pos = tgt.index.get_indexer(events["filing_date"])   # -1 if missing
            asset_pos = tgt.columns.get_indexer(events["asset"])      # -1 if missing
            valid = (date_pos >= 0) & (asset_pos >= 0)

            values = np.full(len(events), np.nan)
            values[valid] = tgt_arr[date_pos[valid], asset_pos[valid]]
            events[col] = values

        return events