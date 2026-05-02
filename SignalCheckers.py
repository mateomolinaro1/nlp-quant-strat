import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.data.data_manager import DataManager

# 1. Setup
config = Config()
dm = DataManager(config)
dm.load_data() # Loads returns and mapping (no text, so it's fast)

# 2. Get Forward Returns (The "Target")
# We shift returns by -1 so that for any date 't', we have the return of 't+1'
returns = dm.get_asset_returns()
forward_returns = returns.shift(-1) 

# 3. Load Features from S3
feature_list = ["sentiment_zscore", "sent_vol_interaction", "toxic_density", "net_sentiment"]
ic_results = {}

print("--- Calculating Feature Correlations (IC) ---")

for feat_name in feature_list:
    # Load the wide-format feature matrix
    feat_df = dm.aws.s3.load(key=f"data/features/{feat_name}.parquet")
    
    # Align shapes (ensure they have the same assets/dates)
    feat_df = feat_df.reindex_like(forward_returns)
    
    # Calculate Spearman Rank Correlation (Information Coefficient)
    # We do this day-by-day and then average it
    daily_ic = feat_df.corrwith(forward_returns, axis=1, method='spearman')
    
    ic_results[feat_name] = daily_ic.mean()
    print(f"Mean IC for {feat_name:25}: {ic_results[feat_name]:.4f}")

# 4. Visualizing the Signal Strength
ic_series = pd.Series(ic_results).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
ic_series.plot(kind='barh', color='skyblue')
plt.title("Signal Strength (Mean Information Coefficient)")
plt.xlabel("Spearman Correlation with Forward Returns")
plt.axvline(0, color='black', lw=1)
plt.show()