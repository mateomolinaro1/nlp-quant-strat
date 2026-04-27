"""
This module implements the feature engineering logic for the NLP quant strategy.
It computes sentiment metrics from financial transcripts using the Loughran-McDonald logic.
"""

import numpy as np
import pandas as pd
import re
import logging
from scipy.stats import entropy
from nlp_quant_strat.data.data_manager import DataManager
from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.utils.utils import S3Utils
import re

logger = logging.getLogger(__name__)

class FeatureEngineering:

    def __init__(self, data: DataManager, config: Config):
        self.data = data
        self.config = config
        
        # Updated to match the "Alpha" features actually computed
        self.feature_names = [
            "pos_ratio", "neg_ratio", "net_sentiment", "sentiment_surprise", 
            "sentiment_zscore", "sentiment_delta", "sent_var", 
            "strong_neg_pct", "sent_eps_interaction", "valuation_sentiment_gap"
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
        
        if "filing_date" in df_formatted_aligned.columns:
            df_formatted_aligned = df_formatted_aligned.drop(columns=["filing_date"])
            
        lim_ffill = getattr(self.config, 'limit_ffill_qdata', limit_ffill)
        df_formatted_aligned.ffill(inplace=True, limit=lim_ffill)
        return df_formatted_aligned

    # ***----------------------***
    # ***-- Feature Compute ---***
    # ***----------------------***
    @staticmethod
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

    def _compute_sentence_metrics(self, text, pos_set, neg_set):
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
            return 0.0, 0.0 # variance, % strongly negative
            
        variance = np.var(sent_sentiments)
        strong_neg_pct = sum(1 for s in sent_sentiments if s < -0.3) / len(sent_sentiments)
        
        return variance, strong_neg_pct

    def _compute_sentiment_features(self) -> None:
        """
        Compute advanced NLP features using Sentiment Surprises, 
        Intra-document Dispersion, and Volatility Interactions.
        """
        if self.data.mapping_df is None or self.data.mapping_df.empty:
            logger.error("Mapping data not found. Ensure DataManager.load_data() was called.")
            return

        # 1. DEFINE KEY AND RELOAD (Order is crucial here!)
        logger.info("Reloading full transcripts for NLP processing...")
        
        # We define 'transcripts_key' FIRST
        transcripts_key = getattr(self.config, 'TRANSCRIPTS_FILENAME', None)

        # Fallback if the first name fails
        if transcripts_key is None:
            transcripts_key = getattr(self.config, 'formatted_unprocessed_transcripts_filename', None)

        if transcripts_key is None:
            logger.error("Could not find transcript path in Config. Check config.json keys.")
            raise AttributeError("Config missing 'TRANSCRIPTS_FILENAME'.")

        # NOW we use 'transcripts_key' to load the data
        df_full = self.data.aws.s3.load(key=transcripts_key)
        
        # 2. SCHEMA ALIGNMENT (Fixing the 'filing_date' and 'ticker' issues)
        if 'filing_date' not in df_full.columns:
            df_full = df_full.reset_index()
            rename_map = {col: 'filing_date' for col in ['index', 'date', 'Date', 'timestamp'] if col in df_full.columns}
            df_full = df_full.rename(columns=rename_map)

        if 'ticker_api' in df_full.columns:
            df_full = df_full.rename(columns={'ticker_api': 'ticker'})

        df_full['filing_date'] = pd.to_datetime(df_full['filing_date'])
        
        # 3. MERGE WITH MAPPING
        mapping_subset = self.data.mapping_df[['ticker', 'filing_date', 'asset']].copy()
        mapping_subset['filing_date'] = pd.to_datetime(mapping_subset['filing_date'])

        df = pd.merge(
            mapping_subset, 
            df_full[['ticker', 'filing_date', 'transcript']], 
            on=['ticker', 'filing_date'], 
            how='inner'
        )
        
        # SAFETY CHECK: If this fails, we stop before the loop
        if 'transcript' not in df.columns or df.empty:
            logger.error(f"Merge failed. Columns available: {df.columns.tolist()}")
            raise KeyError("The 'transcript' column was lost during the merge. Check your ticker/date alignment.")

        n = len(df)
        yearly_pos, yearly_neg, yearly_unc = self._build_yearly_dicts()
        
        # Pre-allocate results
        pos_counts = np.zeros(n)
        neg_counts = np.zeros(n)
        unc_counts = np.zeros(n)
        word_counts = np.zeros(n)
        sent_variance = np.zeros(n)
        toxic_density = np.zeros(n)

        # 2. INTRA-TRANSCRIPT LOOP (Sentence Level)
        logger.info(f"Processing {n} documents at sentence level...")
        for i in range(n):
            text = str(df['transcript'].iloc[i]).lower()
            year = df['filing_date'].iloc[i].year
            
            p_set, n_set, u_set = yearly_pos[year], yearly_neg[year], yearly_unc[year]
            sentences = re.split(r'[.!?]+', text)
            sentence_scores = []
            
            for s in sentences:
                tokens = re.findall(r'\b\w+\b', s)
                if not tokens: continue
                
                s_p = sum(1 for w in tokens if w in p_set)
                s_n = sum(1 for w in tokens if w in n_set)
                s_u = sum(1 for w in tokens if w in u_set)
                
                pos_counts[i] += s_p
                neg_counts[i] += s_n
                unc_counts[i] += s_u
                word_counts[i] += len(tokens)
                
                # Sentence-level polarity
                sentence_scores.append((s_p - s_n) / (len(tokens) + 1))

            if sentence_scores:
                sent_variance[i] = np.var(sentence_scores)
                # Toxic Density: % of sentences that are strongly negative
                toxic_density[i] = sum(1 for s in sentence_scores if s < -0.2) / len(sentence_scores)

        # 3. COMPUTE RATIOS & LEVELS
        df["net_sentiment"] = (pos_counts - neg_counts) / (word_counts + 1)
        df["pos_ratio"] = pos_counts / (word_counts + 1)
        df["neg_ratio"] = neg_counts / (word_counts + 1)
        df["unc_ratio"] = unc_counts / (word_counts + 1)
        df["sent_var"] = sent_variance
        df["toxic_density"] = toxic_density

        # 4. TEMPORAL FEATURES (Surprise & Z-Score)
        df = df.sort_values(['asset', 'filing_date'])
        groups = df.groupby("asset")["net_sentiment"]
        
        df["sentiment_delta"] = df.groupby("asset")["net_sentiment"].diff()
        
        # Rolling stats for Z-Score (Markets care about relative surprise)
        roll_mean = groups.transform(lambda x: x.rolling(window=self.q, min_periods=2).mean())
        roll_std = groups.transform(lambda x: x.rolling(window=self.q, min_periods=2).std())
        
        df["sentiment_surprise"] = df["net_sentiment"] - roll_mean
        df["sentiment_zscore"] = (df["sentiment_surprise"] / roll_std).fillna(0)

        # 5. MARKET INTERACTION (Volatility Proxy)
        logger.info("Computing Sentiment-Volatility interaction...")
        asset_returns = self.data.get_asset_returns()
        # Compute 22-day (1 month) rolling realized volatility
        realized_vol = asset_returns.rolling(window=22).std() * np.sqrt(252)
        
        # Unpivot Vol to join with our transcript DF
        vol_melted = realized_vol.reset_index().melt(id_vars='index', var_name='asset', value_name='market_vol')
        vol_melted.rename(columns={'index': 'filing_date'}, inplace=True)
        
        df = pd.merge_asof(
            df.sort_values("filing_date"),
            vol_melted.sort_values("filing_date"),
            on="filing_date",
            by="asset",
            direction="backward"
        )
        
        # Interaction: High Sentiment + High Volatility = Strong Conviction Signal
        df["sent_vol_interaction"] = df["net_sentiment"] * df["market_vol"]

        # 6. ALIGNMENT & S3 UPLOAD
        self.feature_names = [
            "pos_ratio", "neg_ratio", "net_sentiment", "sentiment_surprise", 
            "sentiment_zscore", "sentiment_delta", "sent_var", 
            "toxic_density", "sent_vol_interaction", "unc_ratio"
        ]

        for feat in self.feature_names:
            if feat in df.columns:
                setattr(self, feat, self._format_df(df, feat))

        self._save_features_to_s3()
        logger.info("Advanced NLP features with Volatility proxies completed.")
        

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
