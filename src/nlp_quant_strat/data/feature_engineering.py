import numpy as np
import pandas as pd
import re
import logging
import gc
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
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

    def _compute_sentiment_features(self) -> None:
        if self.data.mapping_df is None: return

        # 1. Load Transcripts (Remove the 'columns' argument here)
        logger.info("Loading full transcripts for processing...")
        t_key = getattr(self.config, 'TRANSCRIPTS_FILENAME', 'data/transcripts/formatted_unprocessed_transcripts.parquet')
        
        # FIX: No 'columns' argument allowed by better_aws
        df_full = self.data.aws.s3.load(key=t_key)
        
        # IMMEDIATELY select only the columns we need to save RAM
        # This keeps 'transcript', 'ticker_api', and 'filing_date'
        needed_cols = ['ticker_api', 'filing_date', 'transcript']
        df_full = df_full[needed_cols]
        
        # Standardize and Merge
        df_full.rename(columns={'ticker_api': 'ticker'}, inplace=True)
        df_full['filing_date'] = pd.to_datetime(df_full['filing_date'])
        
        df = pd.merge(self.data.mapping_df[['ticker', 'filing_date', 'asset']], df_full, on=['ticker', 'filing_date'], how='inner')
        
        # PURGE df_full immediately to free up the 2x RAM spike
        del df_full 
        import gc
        gc.collect()

        # 2. Setup Dictionaries
        yearly_pos, yearly_neg, yearly_unc = self._build_yearly_dicts()

        # 3. Parallel Processing (The "Turbo" Part)
        logger.info(f"Processing {len(df)} documents on multiple cores...")
        n_cores = multiprocessing.cpu_count() - 1
        
        results = Parallel(n_jobs=n_cores)(
            delayed(_process_single_doc)(
                df['transcript'].iloc[i], 
                df['filing_date'].iloc[i].year,
                yearly_pos, yearly_neg, yearly_unc
            ) for i in tqdm(range(len(df)), desc="NLP Mining")
        )

        # 4. Map Results and Purge Text
        res_arr = np.array(results, dtype='float32')
        df["net_sentiment"] = (res_arr[:, 0] - res_arr[:, 1]) / (res_arr[:, 3] + 1)
        df["pos_ratio"] = res_arr[:, 0] / (res_arr[:, 3] + 1)
        df["neg_ratio"] = res_arr[:, 1] / (res_arr[:, 3] + 1)
        df["unc_ratio"] = res_arr[:, 2] / (res_arr[:, 3] + 1)
        df["sent_var"], df["toxic_density"] = res_arr[:, 4], res_arr[:, 5]
        
        df.drop(columns=['transcript'], inplace=True) # Final Text Purge
        del results, res_arr; gc.collect()

        # 5. Rolling Stats & Interaction (Standard Logic)
        df = df.sort_values(['asset', 'filing_date'])
        groups = df.groupby("asset")["net_sentiment"]
        df["sentiment_delta"] = groups.diff()
        
        # Rolling Z-Score
        roll = groups.rolling(window=self.q, min_periods=2)
        df["sentiment_surprise"] = df["net_sentiment"] - roll.mean().reset_index(0, drop=True)
        df["sentiment_zscore"] = (df["sentiment_surprise"] / roll.std().reset_index(0, drop=True).replace(0, np.nan)).fillna(0)

        # Volatility Join
        asset_returns = self.data.get_asset_returns()
        vol = asset_returns.rolling(22).std() * np.sqrt(252)
        vol_melt = vol.reset_index().melt(id_vars='index', var_name='asset', value_name='v').rename(columns={'index':'filing_date'})
        
        df = pd.merge_asof(df.sort_values("filing_date"), vol_melt.sort_values("filing_date"), on="filing_date", by="asset", direction="backward")
        df["sent_vol_interaction"] = df["net_sentiment"] * df["v"]

        # 6. Pivot and Save
        for feat in self.feature_names:
            if feat in df.columns:
                setattr(self, feat, self._format_df(df, feat))
        
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