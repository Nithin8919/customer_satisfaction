import pandas as pd
import logging
from zenml import step

@step
def clean_df(df: pd.DataFrame) -> None:
    pass