"""
DataLoader class for loading and preprocessing financial data for NLP-based quantitative strategies.
"""
import pandas as pd
from nlp_quant_strat.utils.config import Config
import logging
from pathlib import Path
from better_aws import AWS

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config: Config):
        self.config = config
        self._data_attrs = []
        self.aws: AWS | None = None
        self.dates: list | None = None
        self.asset_ids: list | None = None
        self.mapping_df: pd.DataFrame | None = None

        for filename in self.config.filenames_to_load:
            if not isinstance(filename, str):
                logger.error(f"Invalid filename in config: {filename}. Expected a string.")
                raise ValueError(f"Invalid filename in config: {filename}. Expected a string.")

            name = Path(filename).stem

            if hasattr(self, name):
                raise ValueError(f"Duplicate attribute name generated: {name}")

            setattr(self, name, None)
            self._data_attrs.append(name)

    def load_data(self):
        if self.aws is None:
            self._init_s3()

        transcript_attr = Path(self.config.formatted_unprocessed_transcripts_filename).stem

        for filename in self.config.filenames_to_load:
            name = Path(filename).stem
            logger.info(f"Loading {filename}...")
            
            obj = self.aws.s3.load(key=filename)
            
            # # --- MODIFICATION ICI ---
            # if name == transcript_attr:
            #     logger.info(f"Optimisation : On supprime la colonne texte pour sauver la RAM")
            #     # On ne garde que les colonnes de mapping (date, ticker, asset)
            #     columns_to_keep = [c for c in obj.columns if c != 'transcript']
            #     obj = obj[columns_to_keep]
            # # --------------------------

            setattr(self, name, obj)

        self._set_dates()
        self._set_asset_ids()
        self._build_mapping()
        self._format_benchmark_data()
        self._align_to_dates()
        self._handle_nan_policy()

    # ---------- Internal helpers ---------- #
    def _init_s3(self) -> None:
        """
        Initialize the AWS client and configure S3 defaults.
        """
        self.aws = AWS(region=self.config.region, verbose=True)
        # Optional sanity check
        self.aws.identity(print_info=True)
        # 2) Configure S3 defaults
        self.aws.s3.config(
            bucket=self.config.bucket_name,
            output_type="pandas",  # tabular loads -> pandas (or "polars")
            file_type="parquet",  # default tabular format for dataframe uploads without extension
            overwrite=True,
        )

    def get_asset_returns(self) -> pd.DataFrame:
        """
        helper to get the asset returns dataframe from the corresponding attribute, which is named after the filename
        specified in the config. This is to avoid hardcoding attribute names and to ensure consistency with the config.
        """
        filename = self.config.dates_filename
        attr_name = Path(filename).stem

        if not hasattr(self, attr_name):
            raise AttributeError(f"Attribute '{attr_name}' not found. Did you load the data?")

        asset_returns_obj = getattr(self, attr_name)
        return asset_returns_obj

    def get_words_dict(self) -> pd.DataFrame:
        attr_name = "words_dict"
        if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
            raise AttributeError(f"Attribute '{attr_name}' not found or not loaded. Did you call load_data()?")
        return getattr(self, attr_name)

    def get_benchmark_returns(self) -> pd.DataFrame:
        """
        helper to get the benchmark returns dataframe from the corresponding attribute, which is named after the
        filename specified in the config. This is to avoid hardcoding attribute names and to ensure consistency with
        the config.
        """
        filename = self.config.benchmark_returns_filename
        attr_name = Path(filename).stem

        if not hasattr(self, attr_name):
            raise AttributeError(f"Attribute '{attr_name}' not found. Did you load the data?")

        asset_returns_obj = getattr(self, attr_name)
        return asset_returns_obj

    def get_rf_returns(self) -> pd.DataFrame:
        """
        helper to get the risk-free returns dataframe from the corresponding attribute, which is named after the
        filename specified in the config. This is to avoid hardcoding attribute names and to ensure consistency with
        the config.
        """
        filename = self.config.rf_returns_filename
        attr_name = Path(filename).stem

        if not hasattr(self, attr_name):
            raise AttributeError(f"Attribute '{attr_name}' not found. Did you load the data?")

        asset_returns_obj = getattr(self, attr_name)
        return asset_returns_obj

    def _format_benchmark_data(self) -> None:
        """
        Format (compute returns and align dates) the benchmark data to have the same structure as the asset returns
        data.
        """
        benchmark_returns = self.get_benchmark_returns()
        if self.config.which_bench_col_to_keep is not None:
            benchmark_returns = benchmark_returns[[self.config.which_bench_col_to_keep]]
        benchmark_returns.ffill(inplace=True, limit=1)
        benchmark_returns = benchmark_returns.pct_change(fill_method=None)
        asset_returns = self.get_asset_returns()
        bench_cols = benchmark_returns.columns
        # Align
        df_merged = pd.merge(
            left=asset_returns,
            right=benchmark_returns,
            left_index=True,
            right_index=True,
            how="left"
        )
        final_df = df_merged[bench_cols]
        setattr(self, Path(self.config.benchmark_returns_filename).stem, final_df)
        return

    def _align_to_dates(self) -> None:
        """Reindex all date-indexed DataFrames to self.dates, forward-filling gaps."""
        dates_index = pd.Index(self.dates)
        for attr in self._data_attrs:
            not_to_align = [p.split("/")[-1].split(".")[0] for p in self.config.filenames_not_to_align]
            if attr in not_to_align:  # skip non-date-indexed df
                continue
            obj = getattr(self, attr)
            if not isinstance(obj, pd.DataFrame):
                continue
            if not isinstance(obj.index, pd.DatetimeIndex):
                continue
            if obj.index.duplicated().any():
                obj = obj[~obj.index.duplicated(keep='last')]
            setattr(self, attr, obj.reindex(dates_index))

    def _handle_nan_policy(self) -> None:
        """Apply per-file NaN handling policy defined in config.nan_policy."""
        if self.config.nan_policy is None:
            return
        for attr, policy in self.config.nan_policy.items():
            obj = getattr(self, attr, None)
            if obj is None or not isinstance(obj, pd.DataFrame):
                continue
            method = policy.get("method", "none")
            if method == "ffill":
                limit = policy.get("limit", None)
                setattr(self, attr, obj.ffill(limit=limit))

    def _get_formatted_unprocessed_transcripts(self) -> pd.DataFrame:
        filename = self.config.formatted_unprocessed_transcripts_filename
        attr_name = Path(filename).stem

        if not hasattr(self, attr_name):
            raise AttributeError(f"Attribute '{attr_name}' not found. Did you load the data?")

        formatted_unprocessed_transcripts_obj = getattr(self, attr_name)
        return formatted_unprocessed_transcripts_obj

    def _get_formatted_preprocessed_transcripts(self) -> pd.DataFrame:
        filename = self.config.formatted_preprocessed_transcripts_filename
        attr_name = Path(filename).stem

        if not hasattr(self, attr_name):
            raise AttributeError(f"Attribute '{attr_name}' not found. Did you load the data?")

        formatted_preprocessed_transcripts_obj = getattr(self, attr_name)
        return formatted_preprocessed_transcripts_obj

    def _set_dates(self) -> None:
        """
        Set self.dates from the index of the configured asset returns df.
        """

        asset_returns_obj = self.get_asset_returns()

        if not hasattr(asset_returns_obj, "index"):
            raise TypeError("Attribute containing the asset returns has no index attribute.")

        self.dates = asset_returns_obj.index.to_list()

    def _set_asset_ids(self) -> None:
        """
        Set self.asset_ids from the columns of the configured asset returns df.
        """
        asset_returns_obj = self.get_asset_returns()

        if not hasattr(asset_returns_obj, "columns"):
            raise TypeError("Attribute containing the asset returns has no columns attribute.")

        self.asset_ids = asset_returns_obj.columns.to_list()

    def _build_mapping(self) -> None:
        """
        Build a mapping from transcript IDs to asset IDs using the index of the transcripts.
        :return:
        """
        """
        Create mapping: ticker -> asset column name
        """

        # 1️ prepare transcripts
        df_transcripts = self._get_formatted_unprocessed_transcripts()
        df_transcripts = df_transcripts.reset_index()
        df_transcripts = df_transcripts.rename(columns={'ticker_api': 'ticker'})
        df_transcripts['filing_date'] = pd.to_datetime(df_transcripts['filing_date'])
        df_transcripts['filing_date'] = df_transcripts['filing_date'].values.astype('datetime64[ns]')
        df_transcripts['ticker'] = df_transcripts['ticker'].astype(str)
        df_transcripts = df_transcripts.sort_values('filing_date')

        # 2️ prepare returns
        df_returns = self.get_asset_returns()
        df_returns = df_returns.reset_index().melt(
            id_vars='index', var_name='asset', value_name='return'
        )
        df_returns['index'] = pd.to_datetime(df_returns['index'])
        df_returns['index'] = df_returns['index'].values.astype('datetime64[ns]')
        df_returns['ticker'] = df_returns['asset'].str.split().str[0].astype(str)
        df_returns = df_returns.sort_values('index')

        # 3 merge_asof
        df_mapped = pd.merge_asof(
            df_transcripts,
            df_returns,
            left_on='filing_date',
            right_on='index',
            by='ticker',
            direction='backward'
        )
        df_mapped.dropna(subset="asset", inplace=True)
        df_mapped.reset_index(drop=True, inplace=True)
        # delete duplicates
        df_mapped = df_mapped[~df_mapped.duplicated(subset=["filing_date", "asset"], keep="last")]
        self.mapping_df = df_mapped

    def release_mapping_texts(self) -> None:
        """Drop transcript text columns from mapping_df to free RAM. Call after feature engineering."""
        if self.mapping_df is None:
            return
        text_cols = [c for c in self.mapping_df.columns if c in ("transcript", "transcript_cleaned")]
        if text_cols:
            self.mapping_df.drop(columns=text_cols, inplace=True)
            logger.info("Released text columns %s from mapping_df.", text_cols)

    def release_transcripts(self) -> None:
        """Set transcript DataFrame attributes to None to free RAM. Call after embeddings are built."""
        for filename in filter(None, [
            self.config.formatted_unprocessed_transcripts_filename,
            self.config.formatted_preprocessed_transcripts_filename,
        ]):
            attr = Path(filename).stem
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
                logger.info("Released '%s' from memory.", attr)