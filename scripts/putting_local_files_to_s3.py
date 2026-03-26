from nlp_quant_strat.data.data_loader_old import DataLoader, TranscriptTypes
from nlp_quant_strat.nlp.utils import preprocess_text
from nlp_quant_strat.utils.config import Config
from nlp_quant_strat.utils import utils
import pandas as pd
from better_aws import AWS
from dotenv import load_dotenv
load_dotenv()
config = Config()

pd.set_option("display.max_colwidth", 100000)

# Load data
data = DataLoader()
data.get_data(key=TranscriptTypes.UNPROCESSED.value)

# New preprocessing step
preprocessed_data = preprocess_text(df=data.data)

# Load to AWS S3
aws = AWS(region=config.region, verbose=True)
# Optional sanity check
aws.identity(print_info=True)
# 2) Configure S3 defaults
aws.s3.config(
    bucket=config.bucket_name,
    output_type="pandas",      # tabular loads -> pandas (or "polars")
    file_type="parquet",       # default tabular format for dataframe uploads without extension
    overwrite=True,
)

# 3) Upload the parquet file to S3
# Normally below is the syntax to upload a df to S3
# But as the package is in construction, it misses a function to keep the index of the df
# When uploaded, so I will use mine
# aws.s3.upload(src=df, key="data/FRED-MD-2026-02.parquet")
utils.S3Utils.upload_df_with_index(df=data.data, bucket=config.bucket_name, path="data/transcripts/formatted_unprocessed_transcripts.parquet")

utils.S3Utils.upload_df_with_index(df=data.data, bucket=config.bucket_name, path="data/transcripts/preprocessed_transcripts.parquet")

df = pd.read_feather(config.ROOT_DIR / "data" / "market" / "RIY Index constituents.feather")
df_asset_ret = pd.read_feather(config.ROOT_DIR / "data" / "market" / "RIY Index returns.feather")
df_russell_ret = pd.read_feather(config.ROOT_DIR / "data" / "market" / "total_return_russell.feather")
df_rf = pd.read_csv(config.ROOT_DIR / "data" / "market" / "rf_returns.csv")
df_dict = pd.read_csv(config.ROOT_DIR / "data" / "others" / "Loughran-McDonald_MasterDictionary_1993-2024.csv")

# format df_rf
df_rf["date"] = pd.to_datetime(df_rf["date"], format="%d/%m/%Y")
df_rf.index = df_rf["date"]
df_rf.drop(columns=["date"], inplace=True)

# upload all to s3
utils.S3Utils.upload_df_with_index(df=df, bucket=config.bucket_name, path="data/market/riy_index_constituents.parquet")
utils.S3Utils.upload_df_with_index(df=df_asset_ret, bucket=config.bucket_name, path="data/market/riy_asset_returns.parquet")
utils.S3Utils.upload_df_with_index(df=df_russell_ret, bucket=config.bucket_name, path="data/market/russell_returns.parquet")
utils.S3Utils.upload_df_with_index(df=df_rf, bucket=config.bucket_name, path="data/market/risk_free_returns.parquet")
utils.S3Utils.upload_df_with_index(df=df_dict, bucket=config.bucket_name, path="data/others/words_dict.parquet")

