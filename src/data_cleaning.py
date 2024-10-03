import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from typing import Union
import numpy as np

class DataStrategy(ABC):
    """
    Abstract class for handling the Data.

    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreprocessStrategy(DataStrategy):
    """
    Strategy for preprocessing the data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame :
        try:
            self.df = self.df.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            self.df["product_weight_g"].fillna(
                self.df["product_weight_g"].median(), inplace=True
            )
            self.df["product_length_cm"].fillna(
                self.df["product_length_cm"].median(), inplace=True
            )
            self.df["product_height_cm"].fillna(
                self.df["product_height_cm"].median(), inplace=True
            )
            self.df["product_width_cm"].fillna(
                self.df["product_width_cm"].median(), inplace=True
            )
            # write "No review" in review_comment_message column
            self.df["review_comment_message"].fillna("No review", inplace=True)

            self.df = self.df.select_dtypes(include=[np.number])
            cols_to_drop = [
                "customer_zip_code_prefix",
                "order_item_id",
            ]
            self.df = self.df.drop(cols_to_drop, axis=1)

            # Catchall fillna in case any where missed
            self.df.fillna(self.df.mean(), inplace=True)

            return self.df
        except Exception as e:
            logging.error(e)
            raise e
        
class DataDivideStrategy(DataStrategy):
    """
    strategy for dividing the data
    """
    def divide_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        It divides the data into train and test data.
        """
        try:
            X = df.drop("review_score", axis=1)
            y = df["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e
        
        
class DataCleaning:
    """
    Class that cleans the data and divides the data
    """
    def __inin__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
        try:
            self.strategy.handle_data(self.data)
            
        except Exception as e:
            logging.info(f"There is an exception {e}")
            raise e