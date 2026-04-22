"""
This module implements the feature engineering logic for the NLP quant strategy.
It computes sentiment metrics from financial transcripts using the Loughran-McDonald logic.
"""
import numpy as np
import pandas as pd
import logging
from nlp_quant_strat.data.data_manager import DataManager
from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.utils.utils import S3Utils

logger = logging.getLogger(__name__)

class FeatureEngineering:

    def __init__(self, data: DataManager, config: Config):
        self.data = data
        self.config = config
        
        # Features mapping to ease loading/saving
        self.feature_names = [
            "positive_count", "negative_count", "word_count", 
            "polarity", "sentiment_density", "pos_polarity_count_q", "polarity_delta"
        ]
        
        # Initialize attributes to None
        for feat in self.feature_names:
            setattr(self, feat, None)

        # rolling window size for momentum feature (number of past quarters to look at)
        self.q = getattr(self.config, 'rolling_window_quarters', 4)

    # ***----------------------***
    # ***-- Helper functions --***
    # ***----------------------***
    
    def _build_yearly_dicts(self):
        """Build yearly sentiment dictionaries (Loughran-McDonald logic)"""
        logger.info("Building yearly sentiment dictionaries...")
        if self.data.mapping_df is None or self.data.mapping_df.empty:
            raise ValueError("mapping_df is None. Impossible to build yearly dict.")

        years = self.data.mapping_df['filing_date'].dt.year.unique()
        words_df = self.data.get_words_dict()
        words_df["Word"] = words_df["Word"].str.lower()

        yearly_pos, yearly_neg = {}, {}
        words = words_df['Word'].values
        pos_flags = words_df['Positive'].values
        neg_flags = words_df['Negative'].values

        for year in years:
            pos_set, neg_set = set(), set()
            for word, p, n in zip(words, pos_flags, neg_flags):
                if isinstance(p, (int, float, np.integer, np.floating)):
                    if 0 < p <= year: pos_set.add(word)
                    elif p < 0 and -p <= year: pos_set.discard(word)
                if isinstance(n, (int, float, np.integer, np.floating)):
                    if 0 < n <= year: neg_set.add(word)
                    elif n < 0 and -n <= year: neg_set.discard(word)
            yearly_pos[year] = pos_set
            yearly_neg[year] = neg_set
        return yearly_pos, yearly_neg

    def _format_df(self, df: pd.DataFrame, values_col: str, limit_ffill: int = 23 * 4) -> pd.DataFrame:
        """Format wide-type dataframe and align to asset returns"""
        df_formatted = df.pivot_table(columns="asset", index="filing_date", values=values_col)
        asset_returns = self.data.get_asset_returns()
        
        df_formatted = df_formatted.reindex(columns=asset_returns.columns)

        df_formatted_aligned = pd.merge_asof(
            asset_returns.reset_index()[['index']], 
            df_formatted.sort_index().reset_index(),
            left_on="index",
            right_on="filing_date",
            direction="backward"
        ).set_index("index")
        
        if "filing_date" in df_formatted_aligned.columns:
            df_formatted_aligned = df_formatted_aligned.drop(columns=["filing_date"])
            
        lim_ffill = getattr(self.config, 'limit_ffill_qdata', limit_ffill)
        df_formatted_aligned.ffill(inplace=True, limit=lim_ffill)
        return df_formatted_aligned

    # ***----------------------***
    # ***-- Feature Compute ---***
    # ***----------------------***

    def _tokenize_transcripts(self, df):
        """Standard regex tokenization (Industry standard for LM Dict)"""
        logger.info("Tokenizing transcripts...")
        return df['transcript'].fillna("").str.lower().str.findall(r'\b\w+\b')

    def _count_sentiment_words(self, tokenized_series, yearly_pos, yearly_neg, years_series):
        """Count positive/negative words in a single pass"""
        n = len(tokenized_series)
        pos_counts = np.zeros(n, dtype=np.int32)
        neg_counts = np.zeros(n, dtype=np.int32)
        word_counts = np.zeros(n, dtype=np.int32)

        for i, (words, year) in enumerate(zip(tokenized_series, years_series)):
            word_counts[i] = len(words)
            if word_counts[i] > 0:
                pos_set, neg_set = yearly_pos[year], yearly_neg[year]
                pos_counts[i] = sum(1 for w in words if w in pos_set)
                neg_counts[i] = sum(1 for w in words if w in neg_set)
        return pos_counts, neg_counts, word_counts

    def _compute_raw_scores(self, df):
        """Compute base sentiment metrics"""
        df["polarity"] = (df["pos_count"] - df["neg_count"]) / (df["pos_count"] + df["neg_count"] + 1)
        df["sentiment_density"] = df["pos_count"] / (df["word_count"] + 1)
        df["is_pos_polarity"] = (df["polarity"] > 0).astype(int)
        return df

    def _compute_temporal_features(self, df):
        """Compute Delta and Rolling Momentum features"""
        df = df.sort_values(['asset', 'filing_date'])
        
        # Momentum: rolling sum of positive quarters
        df[f"pos_polarity_count_{self.q}q"] = (
            df.groupby("asset")["is_pos_polarity"]
            .rolling(window=self.q, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        
        # Delta: Change in tone vs previous report
        df["polarity_delta"] = df.groupby("asset")["polarity"].diff()
        return df

    def _compute_sentiment_features(self) -> None:
        """Main computation orchestration"""
        if self.data.mapping_df is None or self.data.mapping_df.empty:
            logger.error("No documents found in mapping_df.")
            return

        df = self.data.mapping_df.copy()
        yearly_pos, yearly_neg = self._build_yearly_dicts()
        
        # 1. Tokenization & Word Counting
        tokenized = self._tokenize_transcripts(df)
        pos, neg, total = self._count_sentiment_words(tokenized, yearly_pos, yearly_neg, df['filing_date'].dt.year)
        
        df["pos_count"], df["neg_count"], df["word_count"] = pos, neg, total

        # 2. Score Computation
        df = self._compute_raw_scores(df)
        df = self._compute_temporal_features(df)

        # 3. Alignment & Storage
        logger.info("Aligning features to market grid...")
        self.positive_count = self._format_df(df, "pos_count")
        self.negative_count = self._format_df(df, "neg_count")
        self.word_count = self._format_df(df, "word_count")
        self.polarity = self._format_df(df, "polarity")
        self.sentiment_density = self._format_df(df, "sentiment_density")
        self.pos_polarity_count_q = self._format_df(df, f"pos_polarity_count_{self.q}q")
        self.polarity_delta = self._format_df(df, "polarity_delta")

        self._save_features_to_s3()

    # =========================
    # Public API
    # =========================
    def build_features(self):
        """Main entry point: loads from S3 or triggers computation"""
        if getattr(self.config, 'load_or_compute_features', 'load') == "load":
            try:
                for feat in self.feature_names:
                    self.__setattr__(feat, self.data.aws.s3.load(key=f"data/features/{feat}.parquet"))
                logger.info("Features loaded from S3.")
            except Exception as e:
                logger.warning(f"S3 load failed: {e}. Computing...")
                self._compute_sentiment_features()
        else:
            self._compute_sentiment_features()

    def _save_features_to_s3(self):
        """Helper to persist features"""
        for feat in self.feature_names:
            df = getattr(self, feat)
            if df is not None:
                S3Utils.upload_df_with_index(df, self.config.bucket_name, f"data/features/{feat}.parquet")