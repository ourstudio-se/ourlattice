import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq
import json

from utils.storages.storage import Storage, BoxNotFoundException

class LocalStorage(Storage):

    folder = "data"

    def download_data(self, _id: str, compression="gzip"):
        file_path = f"{LocalStorage.folder}/{_id}.parquet"
        custom_meta_key = f"{_id}.iot"
        try:
            restored_table = pq.read_table(file_path)
            restored_df = restored_table.to_pandas()
            restored_meta_json = restored_table.schema.metadata[custom_meta_key.encode()]
            restored_meta = json.loads(restored_meta_json)
            for k,v in restored_meta.items():
                setattr(restored_df, k, v)
        except:
            raise BoxNotFoundException(f"Could not find box {file_path}")

        return restored_df

    def upload_data(self, _id: str, df: pd.DataFrame, meta: dict = {}, compression="gzip"):
        file_path = f"{LocalStorage.folder}/{_id}.parquet"
        custom_meta_content = meta
        custom_meta_key = f"{_id}.iot"
        table = pa.Table.from_pandas(df)
        custom_meta_json = json.dumps(custom_meta_content)
        existing_meta = table.schema.metadata
        combined_meta = {
            custom_meta_key.encode() : custom_meta_json.encode(),
            **existing_meta
        }
        table = table.replace_schema_metadata(combined_meta)
        pq.write_table(table, file_path, compression=compression)