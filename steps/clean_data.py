import pandas as pd
import logging
from zenml import step
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreprocessStrategy
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_df(df:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series,"y_test"]
]:
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df,preprocess_strategy)
        process_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(process_data,divide_strategy)
        X_train,X_test,y_train,y_test = data_cleaning.handle_data()
        logging.info("Data cleaning is completed!")
        return X_train,X_test, y_train, y_test
    except Exception as e:
        logging.info(f"There is an exception in {e}")
        raise e
