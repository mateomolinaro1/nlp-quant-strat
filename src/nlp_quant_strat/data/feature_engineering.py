"""
This module implements the feature engineering logic for the NLP quant strategy.
"""
import numpy as np
import pandas as pd
import logging
from nlp_quant_strat.data.data_manager import DataManager
from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.utils.utils import S3Utils
import re

logger = logging.getLogger(__name__)

class FeatureEngineering:

    def __init__(self, data: DataManager, config: Config):
        self.data = data
        self.config = config

        # Features mapping to ease loading/saving
        self.feature_names = [
            "pos_ratio", "neg_ratio", "net_sentiment", "sentiment_surprise",
            "sentiment_zscore", "sentiment_delta", "sent_var",
            "strong_neg_pct", "sent_eps_interaction", "valuation_sentiment_gap"
        ]

        # Initialisation des attributes à None
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
        # Security if mapping is None
        if self.data.mapping_df is None or self.data.mapping_df.empty:
            raise ValueError("mapping_df is None. Impossible to build yearly dict.")

        years = self.data.mapping_df['filing_date'].dt.year.unique()
        words_df = self.data.get_words_dict()
        words_df["Word"] = words_df["Word"].str.lower()

        yearly_pos, yearly_neg, yearly_unc = {}, {}, {}
        words = words_df['Word'].values
        pos_flags = words_df['Positive'].values
        neg_flags = words_df['Negative'].values
        unc_flags = words_df['Uncertainty'].values

        for year in years:
            pos_set, neg_set, unc_set = set(), set(), set()
            for word, p, n, u in zip(words, pos_flags, neg_flags, unc_flags):
                if isinstance(p, (int, float, np.integer, np.floating)):
                    if 0 < p <= year: pos_set.add(word)
                    elif p < 0 and -p <= year: pos_set.discard(word)
                if isinstance(n, (int, float, np.integer, np.floating)):
                    if 0 < n <= year: neg_set.add(word)
                    elif n < 0 and -n <= year: neg_set.discard(word)
                if isinstance(u, (int, float, np.integer, np.floating)):
                    if 0 < u <= year: unc_set.add(word)
                    elif u < 0 and -u <= year: unc_set.discard(word)
            yearly_pos[year] = pos_set
            yearly_unc[year] = unc_set
        return yearly_pos, yearly_neg, yearly_unc

    def _format_df(self, df: pd.DataFrame, values_col: str, limit_ffill: int = 23 * 4) -> pd.DataFrame:
        """Format wide-type dataframe and align to asset returns"""
        # Pivot pour passer en format (Date x Assets)
        df_formatted = df.pivot_table(columns="asset", index="filing_date", values=values_col)
        asset_returns = self.data.get_asset_returns()

        # Realignment on asset universe
        df_formatted = df_formatted.reindex(columns=asset_returns.columns)

        # Merge_asof pour aligner les dates de publication sur les dates de marché
        tolerance_days = getattr(self.config, 'merge_asof_tolerance_days', 63)
        df_formatted_aligned = pd.merge_asof(
            asset_returns.reset_index()[['index']],
            df_formatted.sort_index().reset_index(),
            left_on="index",
            right_on="filing_date",
            direction="backward",
            tolerance=pd.Timedelta(days=tolerance_days),
        ).set_index("index")

        # Cols cleaning after merge
        if "filing_date" in df_formatted_aligned.columns:
            df_formatted_aligned = df_formatted_aligned.drop(columns=["filing_date"])

        # Propagate last known value (Forward Fill)
        if self.config.limit_ffill_qdata is None:
            lim_ffill = limit_ffill
        else:
            lim_ffill = self.config.limit_ffill_qdata
        df_formatted_aligned.ffill(inplace=True, limit=lim_ffill)
        return df_formatted_aligned

    # ***----------------------***
    # ***-- Feature Compute ---***
    # ***----------------------***
    @staticmethod
    def _tokenize_transcripts(df):
        """Standard regex tokenization (Industry standard for LM Dict)"""
        logger.info("Tokenizing transcripts...")
        return df['transcript'].fillna("").str.lower().str.findall(r'\b\w+\b')

    @staticmethod
    def _count_sentiment_words(tokenized_series, yearly_pos, yearly_neg, years_series):
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

    @staticmethod
    def _compute_raw_scores(df):
        """Compute base sentiment metrics"""
        df["polarity"] = (df["pos_count"] - df["neg_count"]) / (df["pos_count"] + df["neg_count"] + 1)
        df["sentiment_density"] = df["pos_count"] / (df["word_count"] + 1)
        df["is_pos_polarity"] = (df["polarity"] > 0).astype(int)
        return df

    @staticmethod
    def _compute_sentence_metrics(text, pos_set, neg_set):
        """Calculates sentiment variance and entropy across sentences."""
        # Split by common sentence delimiters
        sentences = re.split(r'[.!?]+', text)
        sent_sentiments = []

        for s in sentences:
            tokens = re.findall(r'\b\w+\b', s.lower())
            if len(tokens) == 0: continue

            p = sum(1 for w in tokens if w in pos_set)
            n = sum(1 for w in tokens if w in neg_set)
            # Sentence-level polarity
            sent_sentiments.append((p - n) / (p + n + 1))

        if not sent_sentiments:
            return 0.0, 0.0  # variance, % strongly negative

        variance = np.var(sent_sentiments)
        strong_neg_pct = sum(1 for s in sent_sentiments if s < -0.3) / len(sent_sentiments)

        return variance, strong_neg_pct

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
        """Main compute logic with sentence-level dispersion"""
        if self.data.mapping_df is None or self.data.mapping_df.empty:
            return

        df = self.data.mapping_df.copy()
        n = len(df)
        yearly_pos, yearly_neg, yearly_unc = self._build_yearly_dicts()

        # Pre-allocate for speed
        results = {
            "pos": np.zeros(n), "neg": np.zeros(n), "unc": np.zeros(n),
            "words": np.zeros(n), "var": np.zeros(n), "strong_neg": np.zeros(n)
        }

        for i in range(n):
            text = str(df['transcript'].iloc[i]).lower()
            year = df['filing_date'].iloc[i].year

            p_s, n_s, u_s = yearly_pos[year], yearly_neg[year], yearly_unc[year]
            sentences = re.split(r'[.!?]+', text)
            sentence_scores = []

            for s in sentences:
                tokens = re.findall(r'\b\w+\b', s)
                if not tokens: continue

                count_p = sum(1 for w in tokens if w in p_s)
                count_n = sum(1 for w in tokens if w in n_s)

                results["pos"][i] += count_p
                results["neg"][i] += count_n
                results["unc"][i] += sum(1 for w in tokens if w in u_s)
                results["words"][i] += len(tokens)

                # Intra-doc dispersion: Sentence Polarity
                sentence_scores.append((count_p - count_n) / (len(tokens) + 1))

            if sentence_scores:
                results["var"][i] = np.var(sentence_scores)
                results["strong_neg"][i] = sum(1 for s in sentence_scores if s < -0.2) / len(sentence_scores)

        # Level Features
        df["net_sentiment"] = (results["pos"] - results["neg"]) / (results["words"] + 1)
        df["pos_ratio"] = results["pos"] / (results["words"] + 1)
        df["neg_ratio"] = results["neg"] / (results["words"] + 1)
        df["sent_var"], df["strong_neg_pct"] = results["var"], results["strong_neg"]

        # Temporal/Surprise Features
        df = df.sort_values(['asset', 'filing_date'])
        groups = df.groupby("asset")["net_sentiment"]

        df["sentiment_delta"] = df.groupby("asset")["net_sentiment"].diff()

        roll_m = groups.transform(lambda x: x.rolling(self.q, min_periods=2).mean())
        roll_s = groups.transform(lambda x: x.rolling(self.q, min_periods=2).std())

        df["sentiment_surprise"] = df["net_sentiment"] - roll_m
        df["sentiment_zscore"] = (df["sentiment_surprise"] / roll_s).fillna(0)

        # Interactions
        fund_df = self.data.get_fundamentals()
        if fund_df is not None:
            # Ensure proper alignment with fundamentals
            df = pd.merge_asof(
                df.sort_values("filing_date"),
                fund_df.sort_values("filing_date"),
                on="filing_date", by="asset", direction="backward"
            )
            df["sent_eps_interaction"] = df["net_sentiment"] * df.get("eps_surprise", 0)
            df["valuation_sentiment_gap"] = df["net_sentiment"] / (df.get("pe_ratio", 1) + 1)

        # Alignment and Saving
        for feat in self.feature_names:
            if feat in df.columns:
                setattr(self, feat, self._format_df(df, feat))

        self._save_features_to_s3()
        logger.info("Feature engineering completed and saved to S3.")

    def _save_features_to_s3(self):
        """Helper to persist features"""
        for feat in self.feature_names:
            df = getattr(self, feat)
            if df is not None:
                logger.info(f"Uploading {feat} to S3 bucket: {self.config.bucket_name}")
                S3Utils.upload_df_with_index(df, self.config.bucket_name, f"data/features/{feat}.parquet")

    # =========================
    # Public API
    # =========================
    def build_features(self):
        """Main entry point: loads from S3 or triggers computation"""
        logger.info("Building features...")
        if getattr(self.config, 'load_or_compute_features', 'load') == "load":
            try:
                logger.info("Attempting to load all features from S3...")
                for feat in self.feature_names:
                    self.__setattr__(feat, self.data.aws.s3.load(key=f"data/features/{feat}.parquet"))
                logger.info("Features loaded from S3.")
            except Exception as e:
                logger.warning(f"S3 load failed: {e}. Computing...")
                self._compute_sentiment_features()
        else:
            self._compute_sentiment_features()
