"""
This module implements the feature engineering logic for the NLP quant strategy.
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
        # Pivot pour passer en format (Date x Assets)
        df_formatted = df.pivot_table(columns="asset", index="filing_date", values=values_col)
        asset_returns = self.data.get_asset_returns()
        
        # Realignment on asset universe
        df_formatted = df_formatted.reindex(columns=asset_returns.columns)

        # Merge_asof pour aligner les dates de publication sur les dates de marché
        df_formatted_aligned = pd.merge_asof(
            asset_returns.reset_index()[['index']], 
            df_formatted.sort_index().reset_index(),
            left_on="index",
            right_on="filing_date",
            direction="backward"
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

    def _compute_sentiment_features(self) -> None:
        """Compute all NLP features in a single pass"""
        if self.data.mapping_df is None or self.data.mapping_df.empty:
            logger.error("No doc found in mapping_df. Check initial loading.")
            return

        df = self.data.mapping_df.copy()
        n = len(df)
        logger.info(f"Starting NLP computation for {n} documents...")

        yearly_pos, yearly_neg = self._build_yearly_dicts()
        
        pos_counts = np.zeros(n, dtype=np.int32)
        neg_counts = np.zeros(n, dtype=np.int32)
        total_word_counts = np.zeros(n, dtype=np.int32)

        logger.info("Tokenizing transcripts...")
        tokenized = df['transcript'].fillna("").str.lower().str.findall(r'\b\w+\b')

        grouped = df.groupby(df['filing_date'].dt.year).groups
        for year, indices in grouped.items():
            logger.info(f"Processing year {year}...")
            pos_set, neg_set = yearly_pos[year], yearly_neg[year]
            
            for i in indices:
                words = tokenized.iloc[i]
                total_word_counts[i] = len(words)
                if total_word_counts[i] > 0:
                    pos_counts[i] = sum(1 for w in words if w in pos_set)
                    neg_counts[i] = sum(1 for w in words if w in neg_set)

        # Calcul des cols brutes
        df["pos_count"] = pos_counts
        df["neg_count"] = neg_counts
        df["word_count"] = total_word_counts
        # Polarity : (Pos - Neg) / (Pos + Neg + 1)
        df["polarity"] = (df["pos_count"] - df["neg_count"]) / (df["pos_count"] + df["neg_count"] + 1)
        df["sentiment_density"] = df["pos_count"] / (df["word_count"] + 1)
        df["is_pos_polarity"] = (df["polarity"] > 0).astype(int)

        # Temporal computations per Asset
        df = df.sort_values(['asset', 'filing_date'])
        
        # Momentum : Count de polarity positive sur Q quarters
        # On utilise une rolling window sur les published rapports
        df[f"pos_polarity_count_{self.q}q"] = (
            df.groupby("asset")["is_pos_polarity"]
            .rolling(window=self.q, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        
        # Delta : change de ton vs rapport précédent
        df["polarity_delta"] = df.groupby("asset")["polarity"].diff()

        # Alignment and storage in attributes
        logger.info("Aligning features to market grid...")
        self.positive_count = self._format_df(df, "pos_count")
        self.negative_count = self._format_df(df, "neg_count")
        self.word_count = self._format_df(df, "word_count")
        self.polarity = self._format_df(df, "polarity")
        self.sentiment_density = self._format_df(df, "sentiment_density")
        self.pos_polarity_count_q = self._format_df(df, f"pos_polarity_count_{self.q}q")
        self.polarity_delta = self._format_df(df, "polarity_delta")

        # Automatic saving après calcul vers S3
        self._save_features_to_s3()
        logger.info("Feature engineering completed and saved to S3.")

    def _save_features_to_s3(self):
        """Helper to persist features using S3Utils to keep index integrity"""
        for feat in self.feature_names:
            df = getattr(self, feat)
            if df is not None:
                key = f"data/features/{feat}.parquet"
                logger.info(f"Uploading {feat} to S3 bucket: {self.config.bucket_name}")
                # CORRECTION : Utilisation de S3Utils.upload_df_with_index au lieu de .save
                S3Utils.upload_df_with_index(
                    df=df, 
                    bucket=self.config.bucket_name, 
                    path=key
                )

    # =========================
    # Public API
    # =========================
    def build_features(self):
        """Main entry point: loads from S3 or triggers computation"""
        logger.info("Building features...")
        
        if self.config.load_or_compute_features == "load":
            try:
                logger.info("Attempting to load all features from S3...")
                for feat in self.feature_names:
                    key = f"data/features/{feat}.parquet"
                    val = self.data.aws.s3.load(key=key)
                    setattr(self, feat, val)
                logger.info("All features loaded successfully from S3.")
            except Exception as e:
                logger.warning(f"Impossible de charger les features ({e}). Launching du calcul...")
                self._compute_sentiment_features()
        else:
            self._compute_sentiment_features()