import pandas as pd
import boto3
from typing import cast, IO
from io import BytesIO


class S3Utils:
    def __init__(self):
        pass

    @staticmethod
    def upload_df_with_index(df: pd.DataFrame, bucket:str, path:str)->None:
        """
        Upload a DataFrame to S3 in parquet format, including the index.
        :param df: DataFrame to upload.
        :param bucket: S3 bucket name.
        :param path: S3 path (including filename) where the DataFrame will be stored.
        :return: None
        """
        # Create an in-memory buffer
        buffer = BytesIO()

        # Write the DataFrame to the buffer in parquet format, including the index
        df.to_parquet(cast(IO[bytes], buffer), engine="pyarrow", index=True)

        # Reset the buffer's position to the beginning
        buffer.seek(0)

        # Initialize S3 client and upload the buffer content to S3
        s3_client = boto3.client('s3')
        s3_client.upload_fileobj(buffer, bucket, path)

        return