"""
Config class to hold settings for the application. Loads settings from run_pipeline_config.json file.
"""
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import logging
import json
from typing import List, Tuple

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Configuration object to hold settings for the application.
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
        self.dates_filename: str|None = None
        self.benchmark_returns_filename: str|None = None
        self.formatted_unprocessed_transcripts_filename: str|None = None

        # Data
        self.market_data_frequency: str|None = None

        # Feature engineering
        self.load_or_compute_features: str|None = None

        # Backtest
        self.percentiles_winsorization: Tuple[int, int]|None = None
        self.percentiles_portfolios: Tuple[int, int]|None = None
        self.industry_segmentation: pd.DataFrame|None = None
        self.rebal_periods: int|None = None
        self.portfolio_type: str|None = None
        self.transaction_costs: float|int|None = None
        self.strategy_name: str|None = None

        # Load JSON config to attributes of Config class
        self._load_run_pipeline_config()

    def _load_run_pipeline_config(self)->None:
        """
        Load run_pipeline_config.json file
        :return:
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
            if config.get("AWS").get("S3").get("DATES_FILENAME") is not None:
                self.dates_filename = config.get("AWS").get("S3").get("DATES_FILENAME")
            if config.get("AWS").get("S3").get("BENCHMARK_RETURNS_FILENAME") is not None:
                self.benchmark_returns_filename = config.get("AWS").get("S3").get("BENCHMARK_RETURNS_FILENAME")
            if config.get("AWS").get("S3").get("TRANSCRIPTS_FILENAME") is not None:
                self.formatted_unprocessed_transcripts_filename = config.get("AWS").get("S3").get("TRANSCRIPTS_FILENAME")

            # Data
            if config.get("DATA").get("MARKET_DATA_FREQUENCY") is not None:
                self.market_data_frequency = config.get("DATA").get("MARKET_DATA_FREQUENCY")

            # Feature engineering
            if config.get("FEATURE_ENGINEERING") is not None:
                if config.get("FEATURE_ENGINEERING").get("LOAD_OR_COMPUTE") is not None:
                    self.load_or_compute_features = config.get("FEATURE_ENGINEERING").get("LOAD_OR_COMPUTE")

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
