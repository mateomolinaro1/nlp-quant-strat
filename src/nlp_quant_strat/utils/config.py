"""
Config class to hold settings for the application. Loads settings from run_pipeline_config.json file.
"""
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import logging
import json
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Centralised configuration for the NLPQuantStrat pipeline.

    Populated automatically from ``configs/run_pipeline_config.json`` on
    instantiation via :meth:`_load_run_pipeline_config`.  Every JSON key maps
    to a Python attribute listed below, with its type, valid values, and
    default.

    AWS / S3
    --------
    bucket_name : str
        JSON path: AWS.S3.BUCKET_NAME
        Name of the S3 bucket storing all pipeline data (e.g. 'nlp-quant-strat').

    region : str
        JSON path: AWS.S3.AWS_DEFAULT_REGION
        AWS region of the bucket (e.g. 'eu-north-1', 'us-east-1').

    output_format : str
        JSON path: AWS.S3.OUTPUT_FORMAT
        Format tag for S3 response payloads.  Currently unused beyond logging.
        Default: 'json'.

    filenames_to_load : list[str]
        JSON path: AWS.S3.FILENAMES_TO_LOAD
        S3 keys (relative to bucket root) to download during
        ``DataManager.load_data()``.  Must include risk_free, asset_returns,
        and benchmark_returns files at minimum.

    filenames_not_to_align : list[str]
        JSON path: AWS.S3.FILENAMES_NOT_TO_ALIGN
        Subset of ``filenames_to_load`` that must not be date-aligned to the
        common trading calendar (e.g. transcripts, index constituents).

    dates_filename : str
        JSON path: AWS.S3.DATES_FILENAME
        S3 key of the asset returns file used to derive the reference
        trading-date calendar.

    benchmark_returns_filename : str
        JSON path: AWS.S3.BENCHMARK_RETURNS_FILENAME
        S3 key of the benchmark (index) returns file.

    which_bench_col_to_keep : str
        JSON path: AWS.S3.WHICH_BENCH_COL_TO_KEEP
        Column inside the benchmark file to use as the market return series
        (e.g. 'Russell 1000').

    rf_returns_filename : str
        JSON path: AWS.S3.RF_RETURNS_FILENAME
        S3 key of the risk-free rate returns file.

    formatted_unprocessed_transcripts_filename : str
        JSON path: AWS.S3.TRANSCRIPTS_FILENAME
        S3 key for raw (unprocessed) earnings call transcripts.

    formatted_preprocessed_transcripts_filename : str
        JSON path: AWS.S3.PREPROCESSED_TRANSCRIPTS_FILENAME
        S3 key for preprocessed (stemmed/cleaned) transcripts.  Required when
        TF-IDF is configured to use 'transcript_cleaned'.

    DATA
    ----
    market_data_frequency : str
        JSON path: DATA.MARKET_DATA_FREQUENCY
        Granularity of the market return data.  Valid: 'd' (daily).

    nan_policy : dict
        JSON path: DATA.NAN_POLICY
        Per-file NaN handling rules, keyed by filename stem.  Each value is a
        dict with 'method' ('ffill' | 'none') and optional 'limit' (int, max
        consecutive periods to forward-fill).
        Example: {"riy_asset_returns": {"method": "none"}}

    FEATURE_ENGINEERING
    -------------------
    load_or_compute_features : str
        JSON path: FEATURE_ENGINEERING.LOAD_OR_COMPUTE
        'load'    — load pre-computed features from S3 (fast path).
        'compute' — recompute from scratch and upload result to S3.

    rolling_window_quarters : int
        JSON path: FEATURE_ENGINEERING.ROLLING_WINDOW_QUARTERS
        Rolling window in calendar quarters for sentiment z-score and delta
        computations.  Typical: 4 (≈ one year of history).  Must be ≥ 1.

    limit_ffill_qdata : int
        JSON path: FEATURE_ENGINEERING.LIMIT_FFILL_QDATA
        Maximum trading days to forward-fill quarterly fundamental data onto
        daily dates.  Typical: 92 (≈ one quarter).  Must be > 0.

    merge_asof_tolerance_days : int
        JSON path: FEATURE_ENGINEERING.MERGE_ASOF_TOLERANCE_DAYS
        Maximum calendar-day gap when asof-merging quarterly data to daily
        dates.  Increase to tolerate longer reporting lags.  Default: 63.

    EMBEDDINGS
    ----------
    embeddings_load_or_compute : str
        JSON path: EMBEDDINGS.LOAD_OR_COMPUTE
        'load'    — load from S3 using a content-hash cache key.
        'compute' — re-encode and upload.
        Default: 'compute'.

    embeddings_s3_cache_prefix : str | None
        JSON path: EMBEDDINGS.S3_CACHE_PREFIX
        S3 key prefix for embedding parquet cache files
        (e.g. 'data/embeddings').  Required for caching to work.

    embeddings_sentence_transformer : dict | None
        JSON path: EMBEDDINGS.SENTENCE_TRANSFORMER
        Sub-keys:
          MODEL_NAME  (str)  — HuggingFace model ID.  Default 'all-MiniLM-L6-v2'.
                               Other common choices: 'all-mpnet-base-v2' (higher
                               quality, slower).
          CHUNK_SIZE  (int)  — Words per chunk when splitting long transcripts.
                               Default 256.  Higher values capture more context
                               per chunk but may exceed the model token limit.
          BATCH_SIZE  (int)  — Chunks per encoder call (CPU).  On GPU the
                               effective batch size is BATCH_SIZE × 8.  Default 32.
          TEXT_COLUMN (str)  — Source column in transcripts_df.  Default 'transcript'.

    embeddings_tfidf : dict | None
        JSON path: EMBEDDINGS.TFIDF
        Sub-keys:
          MAX_FEATURES  (int)        — Vocabulary cap.  Default 200.
                                       Range: 50–50 000.  Larger = richer but
                                       higher-dimensional features.
          NGRAM_RANGE   ([int, int]) — Minimum and maximum n-gram lengths.
                                       Default [1, 2] (unigrams + bigrams).
          TEXT_COLUMN   (str)        — Source column.  Use 'transcript_cleaned'
                                       for preprocessed (stemmed) text; use
                                       'transcript' for raw text.

    FEATURE_SET
    -----------
    feature_set_mode : list[str]
        JSON path: FEATURE_SET.MODE
        Controls which feature groups FeatureSet assembles.
        Valid values:
          'tfidf'           — TF-IDF bag-of-words only
          'st'              — sentence-transformer dense embeddings only
          'sentiment'       — traditional NLP sentiment signals only
          'tfidf_sentiment' — TF-IDF + sentiment signals
          'st_sentiment'    — sentence-transformer + sentiment signals
          'all'             — TF-IDF + sentence-transformer + sentiment signals
        Default: 'st_sentiment'.

    FORECASTING
    -----------
    beta_window : int
        JSON path: FORECASTING.BETA_WINDOW
        Rolling window in trading days for CAPM beta estimation in
        TargetBuilder.  Must be ≥ 2.  Default 252 (≈ one trading year).
        Should be ≥ min_train_periods × trading_days_per_refit_period so
        that training events have reliable betas.

    forecasting_horizons : list[int]
        JSON path: FORECASTING.HORIZONS
        Forward return horizons in trading days.  A separate model is trained
        per horizon.  All values must be positive integers.
        Default [1, 5, 21, 63].
        The auto buffer = ceil(max(horizons) / trading_days_per_period), so
        a larger max horizon requires more historical burn-in.

    forecasting_refit_frequency : str
        JSON path: FORECASTING.REFIT_FREQUENCY
        Calendar period granularity for walk-forward folds.
        Valid: 'D' | 'W' | 'M' | 'Q' | 'Y'.  Default 'Q' (quarterly).
        Controls how often models are retrained.

    forecasting_train_window : int | None
        JSON path: FORECASTING.TRAIN_WINDOW
        Rolling training window in refit_frequency periods.
        null / None = expanding window (all available history).
        If set, must be ≥ min_train_periods.

    forecasting_val_window : int
        JSON path: FORECASTING.VAL_WINDOW
        Validation window in refit_frequency periods.  Must be ≥ 1.
        Default 4 (= one year with quarterly refit).
        Together with the buffer, determines how many periods of history
        are consumed before the first test period can be used.

    forecasting_min_train_periods : int
        JSON path: FORECASTING.MIN_TRAIN_PERIODS
        Minimum calendar periods of training history required before
        attempting a fold.  Folds with fewer are silently skipped.
        Should exceed beta_window expressed in refit_frequency units
        to avoid targets computed from unreliable early rolling betas.
        Default 8.

    forecasting_min_train_events : int
        JSON path: FORECASTING.MIN_TRAIN_EVENTS
        Minimum non-NaN events needed in the training window.  Folds
        below this threshold are skipped for that (horizon, model) pair.
        Default 50.

    forecasting_min_val_events : int
        JSON path: FORECASTING.MIN_VAL_EVENTS
        Minimum non-NaN events in the validation window required to score
        a hyperparameter candidate.  If below threshold, the first
        param_grid entry is used without searching.  Default 5.

    forecasting_models : list[dict]
        JSON path: FORECASTING.MODELS
        List of model specifications.  Each dict has:
          name        (str)        — Label used in output DataFrames.
          class       (str)        — sklearn class name.  Must be registered in
                                     _SKLEARN_MODULE_MAP in earnings_wf_cv.py.
                                     Supported: Ridge, Lasso, ElasticNet,
                                     BayesianRidge, HuberRegressor,
                                     RandomForestRegressor,
                                     GradientBoostingRegressor,
                                     ExtraTreesRegressor, SVR,
                                     KNeighborsRegressor.
          init_params (dict)       — Constructor kwargs, e.g. {"random_state": 42}.
          param_grid  (list[dict]) — Hyperparameter candidates for val-set grid
                                     search, e.g. [{"alpha": 0.01}, {"alpha": 0.1}].
                                     Use [{}] to skip tuning.

    BACKTEST
    --------
    percentiles_winsorization : tuple[int, int]
        JSON path: BACKTEST.PERCENTILES_WINSORIZATION
        Lower and upper percentile bounds for signal winsorisation before
        portfolio construction.  Default [2, 98].

    percentiles_portfolios : tuple[int, int]
        JSON path: BACKTEST.PERCENTILES_PORTFOLIOS
        Percentile breakpoints for long/short portfolio assignment.
        Default [20, 80] → bottom quintile short, top quintile long.

    rebal_periods : int
        JSON path: BACKTEST.REBAL_PERIODS
        Rebalancing frequency in trading days.  Default 66 (≈ one quarter).

    portfolio_type : str
        JSON path: BACKTEST.PORTFOLIO_TYPE
        'long_only' | 'long_short'.

    transaction_costs : float
        JSON path: BACKTEST.TRANSACTION_COSTS_BPS
        One-way transaction cost in basis points (1 bps = 0.01%).
        Applied on each rebalancing leg.  Default 10.

    strategy_name : list[str] | str
        JSON path: BACKTEST.STRATEGY_NAME
        Signal name(s) to backtest.  Must correspond to columns produced by
        FeatureEngineering (e.g. 'polarity', 'polarity_delta').

    OUTPUTS
    -------
    rolling_window_performance : int
        JSON path: OUTPUTS.ROLLING_WINDOW_PERFORMANCE
        Rolling window in trading days for computing rolling Sharpe ratio and
        related performance statistics.  Default 252.
    """
    def __init__(self):
        # Paths
        try:
            self.ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
        except NameError:
            self.ROOT_DIR = Path.cwd()
        logger.info("Root dir: " + str(self.ROOT_DIR))

        self.RUN_PIPELINE_CONFIG_PATH = self.ROOT_DIR / "configs" / "run_pipeline_config.json"
        logger.info("run_pipeline config path: " + str(self.RUN_PIPELINE_CONFIG_PATH))

        # AWS
        self.bucket_name: str|None = None
        self.region: str|None = None
        self.output_format: str|None = None
        self.filenames_to_load: List[str]|None = None
        self.filenames_not_to_align: List[str]|None = None
        self.dates_filename: str|None = None
        self.benchmark_returns_filename: str|None = None
        self.which_bench_col_to_keep: str|None = None
        self.rf_returns_filename: str|None = None
        self.formatted_unprocessed_transcripts_filename: str|None = None
        self.formatted_preprocessed_transcripts_filename: str|None = None

        # Data
        self.market_data_frequency: str|None = None
        self.nan_policy: dict|None = None

        # Feature engineering
        self.load_or_compute_features: str|None = None
        self.rolling_window_quarters: int|None = None
        self.limit_ffill_qdata: int|None = None
        self.merge_asof_tolerance_days: int = 63

        # Backtest
        self.percentiles_winsorization: Tuple[int, int]|None = None
        self.percentiles_portfolios: Tuple[int, int]|None = None
        self.industry_segmentation: pd.DataFrame|None = None
        self.rebal_periods: int|None = None
        self.portfolio_type: str|None = None
        self.transaction_costs: float|int|None = None
        self.strategy_name: str|None = None

        # Outputs
        self.rolling_window_performance: int | None = None

        # Embeddings
        self.embeddings_load_or_compute: str = "compute"
        self.embeddings_s3_cache_prefix: str|None = None
        self.embeddings_sentence_transformer: dict|None = None
        self.embeddings_tfidf: dict|None = None

        # FeatureSet
        self.feature_set_mode: List[str] = ["st_sentiment"]

        # Forecasting
        self.beta_window: int = 252
        self.forecasting_horizons: List[int] = [1, 5, 21, 63]
        self.forecasting_refit_frequency: str = "Q"
        self.forecasting_train_window: Optional[int] = None
        self.forecasting_val_window: int = 21
        self.forecasting_min_train_periods: int = 8
        self.forecasting_min_train_events: int = 50
        self.forecasting_min_val_events: int = 5
        self.forecasting_models: List[dict] = []

        # CV Results
        self.cv_results_save: bool = False
        self.cv_results_load_or_compute: str = "compute"
        self.cv_results_s3_prefix: str = "data/cv_results"
        self.cv_results_n_jobs: int = -1

        # Load JSON config to attributes of Config class
        self._load_run_pipeline_config()

    def _load_run_pipeline_config(self) -> None:
        """
        Parse ``configs/run_pipeline_config.json`` and populate instance attributes.

        Only keys present in the JSON file overwrite the corresponding attribute;
        missing keys leave the dataclass defaults intact.  This allows partial
        config files and makes forward-compatibility easier.

        Raises
        ------
        FileNotFoundError
            If run_pipeline_config.json does not exist at ROOT_DIR/configs/.
        KeyError
            If a required top-level section (e.g. 'AWS') is absent from the JSON.
        """
        with open(self.ROOT_DIR / "configs" / "run_pipeline_config.json" , "r") as f:
            config: dict = json.load(f)

            # AWS
            if config.get("AWS").get("S3").get("BUCKET_NAME") is not None:
                self.bucket_name = config.get("AWS").get("S3").get("BUCKET_NAME")
            if config.get("AWS").get("S3").get("AWS_DEFAULT_REGION") is not None:
                self.region = config.get("AWS").get("S3").get("AWS_DEFAULT_REGION")
            if config.get("AWS").get("S3").get("OUTPUT_FORMAT") is not None:
                self.output_format = config.get("AWS").get("S3").get("OUTPUT_FORMAT")
            if config.get("AWS").get("S3").get("FILENAMES_TO_LOAD") is not None:
                self.filenames_to_load = config.get("AWS").get("S3").get("FILENAMES_TO_LOAD")
            if config.get("AWS").get("S3").get("FILENAMES_NOT_TO_ALIGN") is not None:
                self.filenames_not_to_align = config.get("AWS").get("S3").get("FILENAMES_NOT_TO_ALIGN")
            if config.get("AWS").get("S3").get("DATES_FILENAME") is not None:
                self.dates_filename = config.get("AWS").get("S3").get("DATES_FILENAME")
            if config.get("AWS").get("S3").get("BENCHMARK_RETURNS_FILENAME") is not None:
                self.benchmark_returns_filename = config.get("AWS").get("S3").get("BENCHMARK_RETURNS_FILENAME")
            if config.get("AWS").get("S3").get("WHICH_BENCH_COL_TO_KEEP") is not None:
                self.which_bench_col_to_keep = config.get("AWS").get("S3").get("WHICH_BENCH_COL_TO_KEEP")
            if config.get("AWS").get("S3").get("RF_RETURNS_FILENAME") is not None:
                self.rf_returns_filename = config.get("AWS").get("S3").get("RF_RETURNS_FILENAME")
            if config.get("AWS").get("S3").get("TRANSCRIPTS_FILENAME") is not None:
                self.formatted_unprocessed_transcripts_filename = config.get("AWS").get("S3").get("TRANSCRIPTS_FILENAME")
            if config.get("AWS").get("S3").get("PREPROCESSED_TRANSCRIPTS_FILENAME") is not None:
                self.formatted_preprocessed_transcripts_filename = config.get("AWS").get("S3").get("PREPROCESSED_TRANSCRIPTS_FILENAME")


            # Data
            if config.get("DATA").get("MARKET_DATA_FREQUENCY") is not None:
                self.market_data_frequency = config.get("DATA").get("MARKET_DATA_FREQUENCY")
            if config.get("DATA").get("NAN_POLICY") is not None:
                self.nan_policy = config.get("DATA").get("NAN_POLICY")

            # Feature engineering
            if config.get("FEATURE_ENGINEERING") is not None:
                if config.get("FEATURE_ENGINEERING").get("LOAD_OR_COMPUTE") is not None:
                    self.load_or_compute_features = config.get("FEATURE_ENGINEERING").get("LOAD_OR_COMPUTE")
                if config.get("FEATURE_ENGINEERING").get("ROLLING_WINDOW_QUARTERS") is not None:
                    self.rolling_window_quarters = config.get("FEATURE_ENGINEERING").get("ROLLING_WINDOW_QUARTERS")
                if config.get("FEATURE_ENGINEERING").get("LIMIT_FFILL_QDATA") is not None:
                    self.limit_ffill_qdata = config.get("FEATURE_ENGINEERING").get("LIMIT_FFILL_QDATA")
                if config.get("FEATURE_ENGINEERING").get("MERGE_ASOF_TOLERANCE_DAYS") is not None:
                    self.merge_asof_tolerance_days = config.get("FEATURE_ENGINEERING").get("MERGE_ASOF_TOLERANCE_DAYS")

            # Backtest
            if config.get("BACKTEST") is not None:
                if config.get("BACKTEST").get("PERCENTILES_WINSORIZATION") is not None:
                    self.percentiles_winsorization = tuple(config.get("BACKTEST").get("PERCENTILES_WINSORIZATION"))
                if config.get("BACKTEST").get("PERCENTILES_PORTFOLIOS") is not None:
                    self.percentiles_portfolios = tuple(config.get("BACKTEST").get("PERCENTILES_PORTFOLIOS"))
                if config.get("BACKTEST").get("INDUSTRY_SEGMENTATION") is not None:
                    self.industry_segmentation = config.get("BACKTEST").get("INDUSTRY_SEGMENTATION")
                if config.get("BACKTEST").get("REBAL_PERIODS") is not None:
                    self.rebal_periods = config.get("BACKTEST").get("REBAL_PERIODS")
                if config.get("BACKTEST").get("PORTFOLIO_TYPE") is not None:
                    self.portfolio_type = config.get("BACKTEST").get("PORTFOLIO_TYPE")
                if config.get("BACKTEST").get("TRANSACTION_COSTS_BPS") is not None:
                    self.transaction_costs = config.get("BACKTEST").get("TRANSACTION_COSTS_BPS")
                if config.get("BACKTEST").get("STRATEGY_NAME") is not None:
                    self.strategy_name = config.get("BACKTEST").get("STRATEGY_NAME")

            # Outputs
            if config.get("OUTPUTS").get("ROLLING_WINDOW_PERFORMANCE") is not None:
                self.rolling_window_performance = config.get("OUTPUTS").get("ROLLING_WINDOW_PERFORMANCE")

            # Embeddings
            emb = config.get("EMBEDDINGS")
            if emb is not None:
                if emb.get("LOAD_OR_COMPUTE") is not None:
                    self.embeddings_load_or_compute = emb.get("LOAD_OR_COMPUTE")
                if emb.get("S3_CACHE_PREFIX") is not None:
                    self.embeddings_s3_cache_prefix = emb.get("S3_CACHE_PREFIX")
                st = emb.get("SENTENCE_TRANSFORMER")
                if st is not None:
                    self.embeddings_sentence_transformer = {
                        "model_name": st.get("MODEL_NAME", "all-MiniLM-L6-v2"),
                        "chunk_size": st.get("CHUNK_SIZE", 256),
                        "batch_size": st.get("BATCH_SIZE", 32),
                        "text_column": st.get("TEXT_COLUMN", "transcript"),
                    }
                tfidf = emb.get("TFIDF")
                if tfidf is not None:
                    self.embeddings_tfidf = {
                        "max_features": tfidf.get("MAX_FEATURES", 200),
                        "ngram_range": tuple(tfidf.get("NGRAM_RANGE", [1, 2])),
                        "text_column": tfidf.get("TEXT_COLUMN", "transcript"),
                    }

            # FeatureSet
            fs = config.get("FEATURE_SET")
            if fs is not None:
                mode_val = fs.get("MODE")
                if mode_val is not None:
                    # Accept both a single string and a list of strings.
                    self.feature_set_mode = [mode_val] if isinstance(mode_val, str) else list(mode_val)


            # Forecasting
            fc = config.get("FORECASTING")
            if fc is not None:
                if fc.get("BETA_WINDOW") is not None:
                    self.beta_window = fc.get("BETA_WINDOW")
                if fc.get("HORIZONS") is not None:
                    self.forecasting_horizons = fc.get("HORIZONS")
                if fc.get("REFIT_FREQUENCY") is not None:
                    self.forecasting_refit_frequency = fc.get("REFIT_FREQUENCY")
                if "TRAIN_WINDOW" in fc:
                    self.forecasting_train_window = fc.get("TRAIN_WINDOW")  # may be null → None
                if fc.get("VAL_WINDOW") is not None:
                    self.forecasting_val_window = int(fc.get("VAL_WINDOW"))
                if fc.get("MIN_TRAIN_PERIODS") is not None:
                    self.forecasting_min_train_periods = int(fc.get("MIN_TRAIN_PERIODS"))
                if fc.get("MIN_TRAIN_EVENTS") is not None:
                    self.forecasting_min_train_events = int(fc.get("MIN_TRAIN_EVENTS"))
                if fc.get("MIN_VAL_EVENTS") is not None:
                    self.forecasting_min_val_events = int(fc.get("MIN_VAL_EVENTS"))
                if fc.get("MODELS") is not None:
                    self.forecasting_models = fc.get("MODELS")

            # CV Results
            cvr = config.get("CV_RESULTS")
            if cvr is not None:
                if cvr.get("SAVE_RESULTS") is not None:
                    self.cv_results_save = bool(cvr.get("SAVE_RESULTS"))
                if cvr.get("LOAD_OR_COMPUTE") is not None:
                    self.cv_results_load_or_compute = cvr.get("LOAD_OR_COMPUTE")
                if cvr.get("S3_PREFIX") is not None:
                    self.cv_results_s3_prefix = cvr.get("S3_PREFIX")
                if cvr.get("N_JOBS") is not None:
                    self.cv_results_n_jobs = int(cvr.get("N_JOBS"))
