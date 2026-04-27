from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.data.data_manager import DataManager
from nlp_quant_strat.data.feature_engineering import FeatureEngineering
import pandas as pd

# 1. Setup
config = Config()
dm = DataManager(config)

# 2. Initialize the AWS connection
# This sets up dm.aws so you can use dm.aws.s3
dm._init_s3() 


# 2. DO NOT LOAD EVERYTHING in DataManager
# Manually load only what is needed for the check

dm.load_data()
# (We only load the dates and tickers of 500 rows to create a 'fake' small mapping)

dm.mapping_df.rename(columns={'ticker_api': 'ticker'}, inplace=True)
dm.mapping_df['asset'] = dm.mapping_df['ticker'] # Dummy asset names for testing

fe = FeatureEngineering(dm, config)

# 3. Use a VERY small subset for the compute
fe._compute_sentiment_features()

print("If you see this, the RAM did not explode!")

print("\n" + "="*30)
print("     SANITY CHECK REPORT")
print("="*30)

# --- CHECK 1: Attribute Alignment ---
print(f"\n[1] Checking attributes...")
for feat in fe.feature_names:
    val = getattr(fe, feat)
    if val is not None:
        print(f"✅ {feat:25} | Shape: {val.shape}")
    else:
        print(f"❌ {feat:25} | IS MISSING!")

# --- CHECK 2: Numerical Stability ---
print(f"\n[2] Checking for NaNs/Infs in Z-Score...")
zscore_df = fe.sentiment_zscore
if zscore_df is not None:
    nans = zscore_df.isna().sum().sum()
    infs = (zscore_df == float('inf')).sum().sum()
    zeros = (zscore_df == 0).sum().sum()
    total = zscore_df.size
    print(f"   - NaNs: {nans}")
    print(f"   - Infs: {infs}")
    print(f"   - Zeros: {zeros} ({zeros/total:.1%} of data)")

# --- CHECK 3: The Interaction Term ---
print(f"\n[3] Interaction Check (Sentiment x Vol)...")
if hasattr(fe, 'sent_vol_interaction'):
    sample_val = fe.sent_vol_interaction.iloc[-5:, :5] # Check a corner of the DF
    print("   Sample values (last 5 days, first 5 assets):")
    print(sample_val)

# --- CHECK 4: Look-ahead Bias Check ---
print(f"\n[4] Date Alignment Check...")
returns = dm.get_asset_returns()
if fe.net_sentiment.index.equals(returns.index):
    print("✅ Index alignment perfect (Feature dates = Return dates)")
else:
    print("❌ Index mismatch! Features and Returns have different timelines.")

print("\n[5] Deep Dive: Verifying Active Asset Data...")

# Common liquid tickers to check
target_tickers = ['AAPL', 'MSFT', 'GS', 'JPM', 'TSLA'] 
found_any = False

for target in target_tickers:
    # Match ticker name in the columns (handles suffixes like ' UN Equity')
    matches = [c for c in fe.sentiment_zscore.columns if target in str(c)]
    
    if matches:
        col = matches[0]
        # Get non-null data for this specific stock
        non_nulls = fe.sentiment_zscore[col].dropna()
        
        print(f"   🔍 {col}: {len(non_nulls)} active sentiment observations.")
        
        if not non_nulls.empty:
            print("      Latest valid Z-Scores:")
            print(non_nulls.tail(5))
            found_any = True
        else:
            print(f"      ⚠️ Found ticker but all values are NaN.")
    else:
        # Ticker might not be in your index constituents
        pass

if not found_any:
    print("   🚩 No target tickers found. Searching for ANY column with data...")
    count = 0
    for col in fe.sentiment_zscore.columns:
        non_nulls = fe.sentiment_zscore[col].dropna()
        if len(non_nulls) > 10: # Look for assets with at least some history
            print(f"   📊 {col}: {len(non_nulls)} observations found.")
            print(non_nulls.tail(3))
            count += 1
        if count >= 3: break

print("\n" + "="*30)
print("     VALIDATION COMPLETE")
print("="*30)
