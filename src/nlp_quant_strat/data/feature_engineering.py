"""
This module implements the feature engineering logic for the NLP quant strategy.
"""
import numpy as np
import pandas as pd
import logging
from nlp_quant_strat.data.data_manager import DataManager
from nlp_quant_strat.utils.config import Config

logger = logging.getLogger(__name__)


class FeatureEngineering:

    def __init__(self, data: DataManager, config:Config):
        self.data = data
        self.config = config
        self.positive_count: pd.DataFrame | None = None
        self.negative_count: pd.DataFrame | None = None

    # ***----------------------***
    # ***-- Helper functions --***
    # ***----------------------***
    def _build_yearly_dicts(self):
        """
        Build yearly dictionaries of positive and negative words based on the "Positive" and "Negative" columns
        in the words dictionary, and the filing dates in the mapping dataframe. As some words are removed or added over
        time, we need to build a separate dictionary for each year to accurately count the sentiment words in the
        transcripts. See more details at https://sraf.nd.edu/loughranmcdonald-master-dictionary/
        :return:
        """
        logger.info("Building yearly sentiment dictionaries...")

        years = self.data.mapping_df['filing_date'].dt.year.unique()
        words_df = self.data.words_dict
        words_df["Word"] = words_df["Word"].str.lower()

        yearly_pos = {}
        yearly_neg = {}

        # Convert once to numpy for speed
        words = words_df['Word'].values
        pos_flags = words_df['Positive'].values
        neg_flags = words_df['Negative'].values

        for year in years:
            pos_set = set()
            neg_set = set()

            for word, p, n in zip(words, pos_flags, neg_flags):

                # Positive logic
                if isinstance(p, (int, float, np.integer, np.floating)):
                    if 0 < p <= year:
                        pos_set.add(word)
                    elif p < 0 and -p <= year:
                        pos_set.discard(word)

                # Negative logic
                if isinstance(n, (int, float, np.integer, np.floating)):
                    if 0 < n <= year:
                        neg_set.add(word)
                    elif n < 0 and -n <= year:
                        neg_set.discard(word)

            yearly_pos[year] = pos_set
            yearly_neg[year] = neg_set

            logger.info(f"Year {year}: {len(pos_set)} pos words, {len(neg_set)} neg words")

        return yearly_pos, yearly_neg

    def _format_df(self, df: pd.DataFrame, values_col: str, limit_ffill: int = 23 * 4) -> pd.DataFrame:
        """
        Format a wide-type dataframe to a long-type dataframe that aligned the columns and index to the asset returns
        dataframe, and forward-fill missing values.
        :param df: the dataframe to format, with columns "filing_date" (point-in-time date of the feature),
            "asset" (asset ids), and the feature column specified in values_col
        :param values_col: the name of the column containing the feature values to format
        :param limit_ffill: the maximum number of periods to forward-fill, default is 23*4
        :return: the formatted dataframe with the same columns and index as the asset returns dataframe, and
            the feature values forward-filled
        """
        # Align feature column names to asset returns column names
        df_formatted = df.pivot_table(columns="asset", index="filing_date", values=values_col)
        asset_returns = self.data.get_asset_returns()
        df_formatted = df_formatted.reindex(columns=asset_returns.columns)

        # Align feature index to asset returns index
        df_formatted_aligned = pd.merge_asof(
            asset_returns.reset_index(),
            df_formatted.reset_index(),
            left_on="index",
            right_on="filing_date",
            direction="backward"
        ).set_index("index")
        cols = [c for c in df_formatted_aligned.columns if c.endswith('_y')]
        df_formatted_aligned = df_formatted_aligned[cols]
        cols_cleaned = [c.replace('_y', '') for c in df_formatted_aligned]  # remove suffix for clarity
        df_formatted_aligned.columns = cols_cleaned
        df_formatted_aligned.ffill(inplace=True, limit=limit_ffill)

        return df_formatted_aligned

    # ***----------------------***
    # ***----- Feature 1 ------***
    # ***----------------------***
    def _compute_sentiment_count_feature(self) -> None:
        """
        Compute and stores the positive and negative word counts for each date-ticker-transcript using the yearly
        dictionaries of positive / negative words, and align the resulted features to the asset returns dataframe.
        """

        df = self.data.mapping_df
        n = len(df)

        logger.info(f"Starting sentiment computation for {n} documents...")

        # Precompute dictionaries
        yearly_pos, yearly_neg = self._build_yearly_dicts()

        # Prepare output arrays
        pos_counts = np.zeros(n, dtype=np.int32)
        neg_counts = np.zeros(n, dtype=np.int32)

        # Precompute tokenized text ONCE (huge speed gain)
        logger.info("Tokenizing all transcripts...")
        tokenized = df['transcript'].fillna("").str.lower().str.findall(r'\b\w+\b')

        # Process by year (critical optimization)
        grouped = df.groupby(df['filing_date'].dt.year).groups

        for year, indices in grouped.items():

            logger.info(f"Processing year {year} ({len(indices)} documents)")

            pos_set = yearly_pos[year]
            neg_set = yearly_neg[year]

            for i in indices:
                words = tokenized.iloc[i]

                # Fast counting
                pos_counts[i] = sum(w in pos_set for w in words)
                neg_counts[i] = sum(w in neg_set for w in words)

        # Assign results
        df["pos_count"] = pos_counts
        df["neg_count"] = neg_counts
        cols = df.columns.tolist()
        cols.remove("transcript")
        cols.remove("return")
        df = df[cols]

        df_pos_aligned = self._format_df(df=df, values_col="pos_count")
        df_neg_aligned = self._format_df(df=df, values_col="neg_count")
        self.positive_count = df_pos_aligned
        self.negative_count = df_neg_aligned

        logger.info("Sentiment computation completed.")
        return


    # ***----------------------***
    # ***----- Feature 2 ------***
    # ***----------------------***

    # =========================
    # Public API
    # =========================
    def build_features(self):
        logger.info("Building features...")

        if self.config.load_or_compute_features=="load":
            logger.info("Loading features from s3")
            self.positive_count = self.data.aws.s3.load(key="data/features/positive_count.parquet")
            self.negative_count = self.data.aws.s3.load(key="data/features/negative_count.parquet")

        elif self.config.load_or_compute_features=="compute":
            self._compute_sentiment_count_feature()

        else:
            logger.warning(f"Unknown option for load_or_compute_features: {self.config.load_or_compute_features}."
                           f"No features will be built.")
            raise ValueError(f"Unknown option for load_or_compute_features: {self.config.load_or_compute_features}")