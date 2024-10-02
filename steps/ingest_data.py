import logging
import pandas as pd
from zenml import step

class Ingest_data:
    def __init__(self, data_path : str) :
        self.data_path = data_path
    
    def get_data(self):
        logging.info(f'Ingesting data from{self.data_path}')
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path : str) -> None:
    try:
        ingest_data = Ingest_data(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'Error ingesting data: {e}')
        raise e