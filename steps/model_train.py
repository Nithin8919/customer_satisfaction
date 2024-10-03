import logging

import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import  RegressorMixin


@step
def train_model(
    X_train : pd.DataFrame,
    X_test : pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    ) -> RegressorMixin:
    
    model = None
    


