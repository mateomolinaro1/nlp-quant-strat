from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.data.data_manager import DataManager

# 1. Setup
config = Config() # Ensure your config.json is in the expected path
dm = DataManager(config)
dm.load_data()

# 2. Your Snippet
returns = dm.get_asset_returns()

# 1. Check the median gap between rows
median_gap = returns.index.to_series().diff().median()
print(f"\n--- DATA CHECK ---")
print(f"Median gap: {median_gap}")

# 2. Check total row count
print(f"Total rows: {len(returns)}")
print(f"------------------\n")