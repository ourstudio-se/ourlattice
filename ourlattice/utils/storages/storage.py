import pandas as pd

class BoxNotFoundException(Exception):
    pass

class Storage:

    async def download_data(self, _id: str, decompress: str = None):
        pass    

    async def upload_data(self, _id: str, obj: pd.DataFrame, meta: dict = {}, compress: str = None):
        pass