"""
EarningsWalkForwardCV — event-level walk-forward cross-validation for
earnings-transcript → stock-return forecasting.

Each observation is an earnings event (filing_date, asset).
Walk-forward splits are defined by calendar periods (quarterly by default).

Layout per test period P
------------------------
  [ ... train ... | val_window | buffer_periods | P (test) ]
                  ↑            ↑
              val_start     val_end = P − buffer_periods

The buffer prevents forward-looking bias: the longest-horizon target of the
last validation event extends buffer_periods into the future.  With
buffer_periods=1 quarter and max_horizon=63 trading days (≈1 quarter), no
validation target uses prices from the test period.

Model selection
---------------
For each (test period, horizon, model):
  1. Grid-search hyperparameters on validation events using scoring_func
     (scaler fit on train only during search).
  2. Refit on train + val combined with best params
     (scaler refit on train + val).
  3. Predict on test events.
  4. Record predictions and selection metadata.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_SENTINEL_COLS = {"filing_date", "asset"}

# Approximate trading days per calendar period (used for auto buffer computation)
_TRADING_DAYS_PER_PERIOD: Dict[str, int] = {"D": 1, "W": 5, "M": 21, "Q": 63, "Y": 252}

# Registry: sklearn class name → module path
_SKLEARN_MODULE_MAP: Dict[str, str] = {
    "Ridge":                      "sklearn.linear_model",
    "Lasso":                      "sklearn.linear_model",
    "ElasticNet":                 "sklearn.linear_model",
    "BayesianRidge":              "sklearn.linear_model",
    "HuberRegressor":             "sklearn.linear_model",
    "RandomForestRegressor":      "sklearn.ensemble",
    "GradientBoostingRegressor":  "sklearn.ensemble",
    "ExtraTreesRegressor":        "sklearn.ensemble",
    "SVR":                        "sklearn.svm",
    "KNeighborsRegressor":        "sklearn.neighbors",
}


def build_models_from_config(
    model_configs: List[Dict[str, Any]],
) -> List[Tuple[str, Any, List[Dict[str, Any]]]]:
    """
    Instantiate sklearn models from a list of config dicts (from JSON).

    Each dict must contain:
      name       : str         — label used in predictions / selection_history
      class      : str         — sklearn class name (must be in _SKLEARN_MODULE_MAP)
      init_params: dict        — constructor kwargs, e.g. {"random_state": 42}
      param_grid : list[dict]  — hyperparameter candidates for val-set grid search

    Returns
    -------
    List of (name, estimator_instance, param_grid) triples ready for
    EarningsWalkForwardCV.
    """
    models: List[Tuple[str, Any, List[Dict[str, Any]]]] = []
    for mc in model_configs:
        name       = mc["name"]
        cls_name   = mc["class"]
        init_params: Dict[str, Any] = mc.get("init_params", {})
        param_grid: List[Dict[str, Any]] = mc.get("param_grid", [{}])

        module_path = _SKLEARN_MODULE_MAP.get(cls_name)
        if module_path is None:
            raise ValueError(
                f"Unknown model class '{cls_name}'. "
                f"Supported: {sorted(_SKLEARN_MODULE_MAP)}. "
                "Add it to _SKLEARN_MODULE_MAP in earnings_wf_cv.py if needed."
            )
        cls = getattr(import_module(module_path), cls_name)
        models.append((name, cls(**init_params), param_grid))

    logger.info(
        "build_models_from_config | %d models: %s",
        len(models), [m[0] for m in models],
    )
    return models


@dataclass
class EarningsWalkForwardResult:
    """
    Container for EarningsWalkForwardCV.run() output.

    Attributes
    ----------
    predictions : pd.DataFrame
        One row per (test period, filing_date, asset, model, horizon).
        Columns: period (str), filing_date (datetime), asset (str),
        model (str), horizon (int), y_pred (float), y_true (float | NaN).
        y_true is NaN for events whose full forward window extends beyond
        the last available price date — a prediction is still produced but
        cannot be evaluated in the OOS metrics.  When multiple feature modes
        are compared, the caller adds a ``mode`` column before concatenating.

    selection_history : pd.DataFrame
        One row per (period, model, horizon) fold that was attempted.
        Columns: period, model, horizon, train_start, train_end, val_start,
        val_end, n_train, n_val, n_test, best_params, val_score.
        val_score is NaN when fewer than min_val_events were available in
        the validation window; in that case the first param_grid entry is
        used without tuning.

    oos_metrics : pd.DataFrame
        Per-(period, model, horizon) directional performance metrics.
        Columns: period, model, horizon,
          hit_rate        — fraction of events where sign(y_pred)==sign(y_true)
          long_ret        — mean realized return of predicted-positive events
          short_ret       — mean realized return of predicted-negative events
          spread          — long_ret − short_ret  (positive = strategy profitable)
          mean_signed_ret — mean(sign(y_pred)×y_true), expected P&L per unit bet
          t_stat          — spread / (std(signed_rets)/√n)
          excess_rmse     — rmse − std(y_true, ddof=0); 0 = null model, < 0 = beats null
          n_long, n_short, n_obs
    """
    predictions: pd.DataFrame
    selection_history: pd.DataFrame
    oos_metrics: pd.DataFrame


@dataclass
class EarningsWalkForwardCV:
    """
    Walk-forward cross-validation on the earnings event panel.

    Parameters
    ----------
    models : list of (name, estimator, param_grid) triples
        Each triple = model name, sklearn-compatible regressor (cloned per fold),
        and a sequence of hyperparameter dicts to grid-search on the val set.
        Use ``[{}]`` as param_grid to skip tuning and use the estimator as-is.
    scoring_func : Callable[[array-like, array-like], float]
        Validation scorer (y_true, y_pred) → scalar; higher is better.
        Example: ``lambda y, yh: spearmanr(y, yh).statistic``
    refit_frequency : str
        Pandas period alias for test-period granularity.
        ``"Q"`` = quarterly, ``"M"`` = monthly, ``"Y"`` = yearly.
    train_window : int or None
        Rolling window in periods. ``None`` = expanding (all available history).
    val_window : int
        Number of periods in the validation window. Default 4.
    buffer_periods : int or None
        Periods of gap between val end and test start. Prevents val targets'
        forward windows from overlapping with the test period.
        ``None`` (default) = auto: ``ceil(max(horizons) / trading_days_per_period)``.
    min_train_periods : int
        Minimum number of calendar periods of training history required before
        a test period is attempted. For rolling this is already enforced by
        ``train_window``; for expanding it sets an explicit floor so early
        periods with very little history are skipped without even trying.
        Default 4 (= 1 year with quarterly refit).
    min_train_events : int
        Skip a (period, horizon, model) combination if fewer non-NaN train
        events are available. Default 50.
    min_val_events : int
        Minimum non-NaN val events required to score a hyperparameter candidate.
        If below threshold, fall back to the first param_grid entry. Default 5.
    scaler_cls : sklearn scaler class
        During grid search: fit on train only.
        During final fit: refit on train + val combined. Default: StandardScaler.
    load_or_compute : str
        ``"load"``    — try to load a cached result from S3 before running CV.
        ``"compute"`` — always run the full CV (default).
        The cache key is a hash of (mode, horizons, models, all CV hyper-parameters)
        so different experiment configurations never collide.
    save_results : bool
        When True, save predictions, selection_history, and oos_metrics to S3
        after a successful CV run.  Requires ``s3_prefix`` to be set and ``aws``
        to be passed to ``run()``.  Default False.
    s3_prefix : str
        S3 key prefix under which result parquets are stored.
        Default ``"data/cv_results"``.
    n_jobs : int
        Number of parallel workers for the test-period loop.
        ``-1`` uses all available CPU cores (default).
        ``1`` disables parallelism (useful for debugging or when models
        themselves use internal threading — e.g. RandomForestRegressor with
        n_jobs=-1 — to avoid thread oversubscription).
        Uses ``joblib`` with ``prefer="threads"`` so the event-panel DataFrame
        is shared across workers without copying.
    """

    models: List[Tuple[str, Any, Sequence[Dict[str, Any]]]]
    scoring_func: Callable
    refit_frequency: str = "Q"
    train_window: Optional[int] = None
    val_window: int = 4
    buffer_periods: Optional[int] = None
    min_train_periods: int = 8 # should be greater than the rolling window used to compute CAPM idio ret
    min_train_events: int = 50
    min_val_events: int = 5
    scaler_cls: Any = StandardScaler
    load_or_compute: str = "compute"
    save_results: bool = False
    s3_prefix: str = "data/cv_results"
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("models list must not be empty.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        horizons: List[int],
        mode: Optional[str] = None,
        aws=None,
    ) -> EarningsWalkForwardResult:
        """
        Run the walk-forward CV over all test periods, models, and horizons.

        Parameters
        ----------
        x : pd.DataFrame
            Feature matrix with columns [filing_date, asset, feat_1, ...].
            Feature columns must be NaN-free (FeatureSet guarantees this).
        y : pd.DataFrame
            Target matrix with columns [filing_date, asset, idio_{h}d, ...].
            NaN targets are expected near the data boundary and are excluded
            from training / validation; they still receive predictions.
        horizons : list[int]
            Forward horizons in trading days. Must match columns present in y.
        mode : str, optional
            Feature mode label (e.g. ``"tfidf_sentiment"``).  Included in the
            S3 cache key so results from different modes never collide.
        aws : optional
            AWS client from DataManager (``data_manager.aws``).  Required for
            S3 load/save; if None, caching is skipped even when configured.

        Returns
        -------
        EarningsWalkForwardResult
        """
        # --- S3 cache: try to load if configured ---
        cache_prefix = self._cache_key_prefix(horizons, mode)
        if self.load_or_compute == "load" and aws is not None:
            try:
                preds = aws.s3.load(key=f"{cache_prefix}_predictions.parquet")
                sel   = aws.s3.load(key=f"{cache_prefix}_selection_history.parquet")
                oos   = aws.s3.load(key=f"{cache_prefix}_oos_metrics.parquet")
                logger.info("Loaded CV results from S3 (mode=%s): %s", mode, cache_prefix)
                return EarningsWalkForwardResult(
                    predictions=preds, selection_history=sel, oos_metrics=oos
                )
            except Exception:
                logger.info("No S3 cache at %s — running CV.", cache_prefix)

        X = x.copy()
        feat_cols = [c for c in X.columns if c not in _SENTINEL_COLS]
        target_cols = {h: f"idio_{h}d" for h in horizons}

        missing = [c for c in target_cols.values() if c not in y.columns]
        if missing:
            raise ValueError(f"y is missing target columns: {missing}")

        buffer = self._resolve_buffer(horizons)

        data = X.merge(
            y[["filing_date", "asset"] + list(target_cols.values())],
            on=["filing_date", "asset"],
            how="inner",
        ).reset_index(drop=True)

        filing_dates = pd.to_datetime(data["filing_date"])
        date_periods = filing_dates.dt.to_period(self.refit_frequency)

        # Build a typed List[pd.Period] to avoid enumerate inference issues
        all_periods: List[pd.Period] = sorted(date_periods.unique().tolist())
        min_preceding = self.val_window + buffer + max(self.train_window or 0, self.min_train_periods)
        test_periods: List[pd.Period] = all_periods[min_preceding:]

        scheme = "rolling" if self.train_window is not None else "expanding"
        logger.info(
            "EarningsWalkForwardCV | scheme=%s | freq=%s | window=%s | "
            "val_window=%d | buffer=%d (auto=%s) | test_periods=%d | models=%d | "
            "horizons=%s | events=%d",
            scheme, self.refit_frequency, self.train_window,
            self.val_window, buffer, self.buffer_periods is None,
            len(test_periods), len(self.models), horizons, len(data),
        )

        period_results: List[Tuple[List[pd.DataFrame], List[Dict[str, Any]]]] = Parallel(
            n_jobs=self.n_jobs, prefer="threads",
        )(
            delayed(self._run_period)(
                data, date_periods, filing_dates, feat_cols, target_cols,
                horizons, test_period, i, len(test_periods), buffer,
            )
            for i, test_period in enumerate(test_periods)
        )

        pred_chunks       = [c for chunks, _ in period_results for c in chunks]
        selection_records = [r for _, recs  in period_results for r in recs]

        predictions = (
            pd.concat(pred_chunks, ignore_index=True)
            .sort_values(["period", "filing_date", "asset", "model", "horizon"])
            .reset_index(drop=True)
            if pred_chunks
            else pd.DataFrame(
                columns=["period", "filing_date", "asset", "model", "horizon", "y_pred", "y_true"]
            )
        )
        selection_history = pd.DataFrame(selection_records)
        oos_metrics = self._compute_oos_metrics(predictions)

        logger.info(
            "EarningsWalkForwardCV complete | predictions=%d | folds=%d | oos_metric_rows=%d",
            len(predictions), len(selection_history), len(oos_metrics),
        )
        result = EarningsWalkForwardResult(
            predictions=predictions,
            selection_history=selection_history,
            oos_metrics=oos_metrics,
        )

        # --- S3 cache: save if configured ---
        if self.save_results and aws is not None:
            aws.s3.upload(src=result.predictions,        key=f"{cache_prefix}_predictions.parquet")
            aws.s3.upload(src=result.selection_history,  key=f"{cache_prefix}_selection_history.parquet")
            aws.s3.upload(src=result.oos_metrics,        key=f"{cache_prefix}_oos_metrics.parquet")
            logger.info("Saved CV results to S3 (mode=%s): %s", mode, cache_prefix)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_period(
        self,
        data: pd.DataFrame,
        date_periods: pd.Series,
        filing_dates: pd.Series,
        feat_cols: List[str],
        target_cols: Dict[int, str],
        horizons: List[int],
        test_period: pd.Period,
        period_idx: int,
        n_periods: int,
        buffer: int,
    ) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
        """
        Process one test period: scale, grid-search, refit, predict.

        Called by ``run()`` via ``joblib.Parallel``.  All array operations are
        numpy/sklearn and release the GIL, so ``prefer="threads"`` is safe and
        avoids copying the shared ``data`` DataFrame.

        Returns
        -------
        (pred_chunks, selection_records)
            Lists to be concatenated by the caller across all periods.
        """
        test_mask, val_mask, train_mask = self._split_masks(date_periods, test_period, buffer)

        n_train = int(train_mask.sum())
        n_val   = int(val_mask.sum())
        n_test  = int(test_mask.sum())

        if n_test == 0:
            return [], []

        logger.info(
            "Period %s (%d/%d) | train=%d | val=%d | test=%d",
            test_period, period_idx + 1, n_periods, n_train, n_val, n_test,
        )

        X_train_raw   = data.loc[train_mask,   feat_cols].values.astype(float)
        X_val_raw     = data.loc[val_mask,     feat_cols].values.astype(float)
        X_test_raw    = data.loc[test_mask,    feat_cols].values.astype(float)
        test_rows     = data.loc[test_mask, ["filing_date", "asset"]].reset_index(drop=True)
        train_dates   = filing_dates[train_mask]
        val_dates     = filing_dates[val_mask]

        # Period-level scalers (X is horizon- and model-independent)
        scaler_search  = self.scaler_cls()
        X_train_s      = scaler_search.fit_transform(X_train_raw)
        X_val_s        = scaler_search.transform(X_val_raw)

        trainval_mask  = train_mask | val_mask
        X_trainval_raw = data.loc[trainval_mask, feat_cols].values.astype(float)
        scaler_final   = self.scaler_cls()
        X_trainval_s   = scaler_final.fit_transform(X_trainval_raw)
        X_test_s       = scaler_final.transform(X_test_raw)

        pred_chunks: List[pd.DataFrame] = []
        selection_records: List[Dict[str, Any]] = []

        for horizon in horizons:
            t_col = target_cols[horizon]

            y_train_all    = data.loc[train_mask,    t_col].values
            y_val_all      = data.loc[val_mask,      t_col].values
            y_test_vals    = data.loc[test_mask,     t_col].values
            y_trainval_all = data.loc[trainval_mask, t_col].values

            valid_train    = ~np.isnan(y_train_all)
            valid_val      = ~np.isnan(y_val_all)
            valid_trainval = ~np.isnan(y_trainval_all)

            if valid_train.sum() < self.min_train_events:
                logger.debug(
                    "Period %s | horizon %dd: %d valid train events < min=%d, skipping.",
                    test_period, horizon, valid_train.sum(), self.min_train_events,
                )
                continue

            if valid_trainval.sum() < self.min_train_events:
                continue

            for model_name, estimator, param_grid in self.models:
                logger.debug(
                    "  model=%-6s | horizon=%2dd | grid_size=%d | val_scorable=%s",
                    model_name, horizon, len(param_grid),
                    valid_val.sum() >= self.min_val_events,
                )

                # --- Hyperparameter search on validation set ---
                best_params: Dict[str, Any] = dict(param_grid[0])
                best_score  = -np.inf
                can_score   = valid_val.sum() >= self.min_val_events

                for params in param_grid:
                    m = clone(estimator).set_params(**params)
                    m.fit(X_train_s[valid_train], y_train_all[valid_train])

                    if not can_score:
                        best_params = dict(params)
                        break

                    y_val_pred = m.predict(X_val_s[valid_val])
                    score = self.scoring_func(y_val_all[valid_val], y_val_pred)

                    logger.debug(
                        "    params=%s | val_score=%.4f",
                        params, score if not np.isnan(score) else float("nan"),
                    )

                    if not np.isnan(score) and score > best_score:
                        best_score  = score
                        best_params = dict(params)

                # --- Final fit on train + val with best params ---
                m_final = clone(estimator).set_params(**best_params)
                m_final.fit(X_trainval_s[valid_trainval], y_trainval_all[valid_trainval])
                y_pred = m_final.predict(X_test_s)

                logger.debug(
                    "  → best_params=%s | val_score=%.4f | n_test_preds=%d",
                    best_params,
                    float(best_score) if np.isfinite(best_score) else float("nan"),
                    len(y_pred),
                )

                # --- Store predictions ---
                chunk = test_rows.copy()
                chunk["period"]  = str(test_period)
                chunk["model"]   = model_name
                chunk["horizon"] = horizon
                chunk["y_pred"]  = y_pred
                chunk["y_true"]  = y_test_vals
                pred_chunks.append(chunk)

                # --- Store selection metadata ---
                selection_records.append({
                    "period":       str(test_period),
                    "model":        model_name,
                    "horizon":      horizon,
                    "train_start":  train_dates.min() if n_train > 0 else pd.NaT,
                    "train_end":    train_dates.max() if n_train > 0 else pd.NaT,
                    "val_start":    val_dates.min()   if n_val   > 0 else pd.NaT,
                    "val_end":      val_dates.max()   if n_val   > 0 else pd.NaT,
                    "n_train":      int(valid_train.sum()),
                    "n_val":        int(valid_val.sum()),
                    "n_test":       n_test,
                    "best_params":  best_params,
                    "val_score":    float(best_score) if np.isfinite(best_score) else np.nan,
                })

        return pred_chunks, selection_records

    def _cache_key_prefix(self, horizons: List[int], mode: Optional[str]) -> str:
        """
        Build a deterministic S3 key prefix for this (mode, CV config) combination.

        The hash encodes: mode, sorted horizons, model names / classes / init params /
        param grids, and all CV hyper-parameters.  Two runs with identical settings
        always produce the same key; different settings never collide.

        Three parquet files are derived from this prefix:
          ``<prefix>_predictions.parquet``
          ``<prefix>_selection_history.parquet``
          ``<prefix>_oos_metrics.parquet``
        """
        params: Dict[str, Any] = {
            "mode": mode,
            "horizons": sorted(horizons),
            "models": [
                {
                    "name": name,
                    "class": type(est).__name__,
                    "init_params": est.get_params(),
                    "param_grid": list(pg),
                }
                for name, est, pg in self.models
            ],
            "refit_frequency":    self.refit_frequency,
            "train_window":       self.train_window,
            "val_window":         self.val_window,
            "min_train_periods":  self.min_train_periods,
            "min_train_events":   self.min_train_events,
            "min_val_events":     self.min_val_events,
        }
        hash_str = hashlib.md5(
            json.dumps(params, sort_keys=True, default=str).encode()
        ).hexdigest()[:10]
        return self.s3_prefix.rstrip("/") + f"/cv_{hash_str}"

    @staticmethod
    def _compute_oos_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Compute long/short strategy metrics per (period, model, horizon).

        Metrics
        -------
        hit_rate         : proportion of events where sign(y_pred) == sign(y_true)
        long_ret         : mean realized return of predicted-positive events
        short_ret        : mean realized return of predicted-negative events
        spread           : long_ret − short_ret  (target: > 0)
        mean_signed_ret  : mean(sign(y_pred) × y_true)  — expected P&L per unit bet
        t_stat           : spread / (std of signed returns / √n)
        excess_rmse      : rmse − std(y_true, ddof=0).  Baseline is 0 (null model
                           that always predicts 0 achieves rmse = std(y_true) exactly
                           because E[idio] = 0 by CAPM).  Negative = beats null.
        n_long / n_short : number of long and short positions
        n_obs            : total evaluable events (non-NaN y_true)
        """
        scored = predictions.dropna(subset=["y_true"])
        records: List[Dict[str, Any]] = []

        for (period_str, model_name, horizon), grp in scored.groupby(
            ["period", "model", "horizon"], sort=True
        ):
            yt = grp["y_true"].values
            yp = grp["y_pred"].values
            sign_pred = np.sign(yp)

            long_mask  = sign_pred > 0
            short_mask = sign_pred < 0

            long_ret  = float(yt[long_mask].mean())  if long_mask.sum()  > 0 else np.nan
            short_ret = float(yt[short_mask].mean()) if short_mask.sum() > 0 else np.nan
            spread    = (long_ret - short_ret) if not (np.isnan(long_ret) or np.isnan(short_ret)) else np.nan

            signed_rets = sign_pred * yt
            mean_sr     = float(signed_rets.mean())
            n           = len(signed_rets)
            std_sr      = float(signed_rets.std(ddof=1)) if n > 1 else np.nan
            t_stat      = (mean_sr / (std_sr / np.sqrt(n))) if (std_sr and std_sr > 0) else np.nan

            hit_rate = float((sign_pred == np.sign(yt)).mean())

            rmse        = float(np.sqrt(np.mean((yp - yt) ** 2)))
            std_y       = float(np.std(yt, ddof=0))   # ddof=0 so null model gives excess_rmse = 0 exactly
            excess_rmse = rmse - std_y

            records.append({
                "period":          period_str,
                "model":           model_name,
                "horizon":         horizon,
                "hit_rate":        hit_rate,
                "long_ret":        long_ret,
                "short_ret":       short_ret,
                "spread":          spread,
                "mean_signed_ret": mean_sr,
                "t_stat":          t_stat,
                "excess_rmse":     excess_rmse,
                "n_long":          int(long_mask.sum()),
                "n_short":         int(short_mask.sum()),
                "n_obs":           n,
            })

        return pd.DataFrame(records)

    def _resolve_buffer(self, horizons: List[int]) -> int:
        """
        Return buffer_periods to use for this run.

        If the user supplied an explicit value, use it.
        Otherwise compute ceil(max(horizons) / trading_days_per_period) so that
        the longest-horizon val target cannot overlap with the test period.
        """
        if self.buffer_periods is not None:
            return self.buffer_periods
        days_per_period = _TRADING_DAYS_PER_PERIOD.get(self.refit_frequency.upper(), 63)
        return math.ceil(max(horizons) / days_per_period)

    def _split_masks(
        self,
        date_periods: pd.Series,
        test_period: pd.Period,
        buffer: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (test_mask, val_mask, train_mask) as boolean numpy arrays.

        Period layout
        -------------
        train : [train_start, val_start)
        val   : [val_start, val_end)   where val_end = test_period − buffer
        buffer: [val_end, test_period) — unused, isolates val targets from test
        test  : test_period
        """
        val_end   = test_period - buffer
        val_start = val_end     - self.val_window

        test_mask = (date_periods == test_period).values
        val_mask  = ((date_periods >= val_start) & (date_periods < val_end)).values

        if self.train_window is not None:
            train_start = val_start - self.train_window
            train_mask  = ((date_periods >= train_start) & (date_periods < val_start)).values
        else:
            train_mask  = (date_periods < val_start).values

        return test_mask, val_mask, train_mask