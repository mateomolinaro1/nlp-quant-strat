"""
This module implements the feature engineering logic for the NLP quant strategy.
"""

import numpy as np
import pandas as pd
import re
import logging
import gc
from nlp_quant_strat.data.data_manager import DataManager
from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.utils.utils import S3Utils

logger = logging.getLogger(__name__)

# Static worker function for parallel processing (Must be outside the class)
def _process_single_doc(text, year, yearly_pos, yearly_neg, yearly_unc):
    p_s, n_s, u_s = yearly_pos[year], yearly_neg[year], yearly_unc[year]
    sentences = re.split(r'[.!?]+', str(text).lower())
    
    doc_p, doc_n, doc_u, doc_w = 0, 0, 0, 0
    scores = []
    
    for s in sentences:
        tokens = re.findall(r'\b\w+\b', s)
        if not tokens: continue
        
        cp = sum(1 for w in tokens if w in p_s)
        cn = sum(1 for w in tokens if w in n_s)
        cu = sum(1 for w in tokens if u_s and w in u_s)
        cw = len(tokens)
        
        doc_p += cp; doc_n += cn; doc_u += cu; doc_w += cw
        scores.append((cp - cn) / (cw + 1))

    var = np.var(scores) if scores else 0.0
    tox = sum(1 for s in scores if s < -0.2) / len(scores) if scores else 0.0
    return [doc_p, doc_n, doc_u, doc_w, var, tox]

class FeatureEngineering:
    def __init__(self, data: DataManager, config: Config):
        self.data = data
        self.config = config
        
        # Liste complète pour satisfaire FeatureSet.build(mode="sentiment")
        self.feature_names = [
            # Features "Legacy" présentes sur S3 et attendues par le pipeline
            "positive_count", "negative_count", "word_count", 
            "sentiment_density", "polarity", "polarity_delta", "pos_polarity_count_q",
            
            # Features "Alpha" que nous avons développées
            "pos_ratio", "neg_ratio", "net_sentiment", "sentiment_surprise", 
            "sentiment_zscore", "sentiment_delta", "sent_var", 
            "toxic_density", "sent_vol_interaction", "unc_ratio"
        ]
        
        for feat in self.feature_names:
            setattr(self, feat, None)

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
            yearly_neg[year] = neg_set
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
        """Count positive/negative/uncertainty words in a single pass"""
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
    def _count_uncertainty_words(tokenized_series, yearly_unc, years_series):
        """Count uncertainty words in a single pass"""
        n = len(tokenized_series)
        unc_counts = np.zeros(n, dtype=np.int32)
        for i, (words, year) in enumerate(zip(tokenized_series, years_series)):
            if len(words) > 0:
                unc_counts[i] = sum(1 for w in words if w in yearly_unc[year])
        return unc_counts

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
        """Compute Delta, Rolling Momentum, z-score, and surprise features"""
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

        # Net-sentiment delta
        df["sentiment_delta"] = df.groupby("asset")["net_sentiment"].diff()

        # Rolling mean/std of polarity for surprise and z-score
        rolling_mean = (
            df.groupby("asset")["polarity"]
            .rolling(window=self.q, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        rolling_std = (
            df.groupby("asset")["polarity"]
            .rolling(window=self.q, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
        df["sentiment_surprise"] = df["polarity"] - rolling_mean
        df["sentiment_zscore"] = df["sentiment_surprise"] / (rolling_std + 1e-8)
        return df

    def _compute_sentence_level_features(self, df: pd.DataFrame, yearly_pos: dict, yearly_neg: dict) -> pd.DataFrame:
        """Add sent_var, toxic_density, and sent_vol_interaction columns (row-wise sentence parsing)."""
        logger.info("Computing sentence-level features (%d documents)...", len(df))
        years = df['filing_date'].dt.year.values
        texts = df['transcript'].fillna("").values

        sent_var_arr   = np.zeros(len(df), dtype=np.float64)
        toxic_arr      = np.zeros(len(df), dtype=np.float64)

        for i, (text, year) in enumerate(zip(texts, years)):
            var, tox = self._compute_sentence_metrics(text, yearly_pos[year], yearly_neg[year])
            sent_var_arr[i] = var
            toxic_arr[i]    = tox

        df['sent_var']      = sent_var_arr
        df['toxic_density'] = toxic_arr
        # Interaction: intra-doc sentiment dispersion × negative-word density
        df['sent_vol_interaction'] = df['sent_var'] * df['neg_ratio']
        return df

    def _compute_sentiment_features(self) -> None:
        import gc
        if self.data.mapping_df is None:
            return

        # 1. Load transcripts — reset_index so parquet index cols become columns
        logger.info("Loading full transcripts for processing...")
        t_key = getattr(
            self.config, 'TRANSCRIPTS_FILENAME',
            'data/transcripts/formatted_unprocessed_transcripts.parquet',
        )
        df_full = self.data.aws.s3.load(key=t_key)
        if not isinstance(df_full.index, pd.RangeIndex):
            df_full = df_full.reset_index()

        # Normalise column names (ticker_api → ticker)
        if 'ticker_api' in df_full.columns:
            df_full = df_full.rename(columns={'ticker_api': 'ticker'})
        df_full['filing_date'] = pd.to_datetime(df_full['filing_date'])

        available = [c for c in ['ticker', 'filing_date', 'transcript'] if c in df_full.columns]
        df_full = df_full[available]

        # 2. Merge with mapping to get asset identifier
        df = pd.merge(
            self.data.mapping_df[['ticker', 'filing_date', 'asset']],
            df_full,
            on=['ticker', 'filing_date'],
            how='inner',
        )
        del df_full; gc.collect()

        logger.info("Merged transcript events: %d", len(df))

        # 3. Build Loughran-McDonald yearly dictionaries
        yearly_pos, yearly_neg, yearly_unc = self._build_yearly_dicts()

        # 4. Tokenize and count sentiment + uncertainty words
        tokenized = self._tokenize_transcripts(df)
        years = df['filing_date'].dt.year
        pos_counts, neg_counts, word_counts = self._count_sentiment_words(
            tokenized, yearly_pos, yearly_neg, years
        )
        unc_counts = self._count_uncertainty_words(tokenized, yearly_unc, years)

        df['positive_count'] = pos_counts
        df['negative_count'] = neg_counts
        df['word_count']     = word_counts
        # aliases expected by _compute_raw_scores
        df['pos_count'] = pos_counts
        df['neg_count'] = neg_counts

        # 4b. Simple ratio / density features
        df['pos_ratio']     = pos_counts / (word_counts + 1)
        df['neg_ratio']     = neg_counts / (word_counts + 1)
        df['net_sentiment'] = (pos_counts - neg_counts) / (word_counts + 1)
        df['unc_ratio']     = unc_counts / (word_counts + 1)

        # 5. Raw sentiment scores (polarity, sentiment_density, is_pos_polarity)
        df = self._compute_raw_scores(df)

        # 6. Sentence-level features (sent_var, toxic_density, sent_vol_interaction)
        df = self._compute_sentence_level_features(df, yearly_pos, yearly_neg)

        # 7. Temporal features (polarity_delta, pos_polarity_count_Nq, sentiment_delta,
        #    sentiment_surprise, sentiment_zscore)
        df = self._compute_temporal_features(df)
        # Rename so the attribute name matches feature_names and FeatureSet._SENTIMENT_ATTRS
        df = df.rename(columns={f'pos_polarity_count_{self.q}q': 'pos_polarity_count_q'})

        df.drop(columns=['transcript', 'pos_count', 'neg_count', 'is_pos_polarity'],
                errors='ignore', inplace=True)
        gc.collect()

        # 8. Pivot each feature to (dates × assets) panel and forward-fill
        missing = []
        for feat in self.feature_names:
            if feat in df.columns:
                setattr(self, feat, self._format_df(df, feat))
            else:
                missing.append(feat)
        if missing:
            logger.warning("Features not found in df and will be None: %s", missing)

        if getattr(self.config, 'save_features_to_s3', True):
            self._save_features_to_s3()
        
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
            
    def _save_features_to_s3(self):
        """Helper to persist features"""
        for feat in self.feature_names:
            df = getattr(self, feat)
            if df is not None:
                logger.info(f"Uploading {feat} to S3 bucket: {self.config.bucket_name}")
                S3Utils.upload_df_with_index(df, self.config.bucket_name, f"data/features/{feat}.parquet")