import logging
import pandas as pd
from zenml import step
from zenml.client import Client
import mlflow
from src.model_dev import LinearRegressionModel
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from steps.config import ModelNameConfig
import numpy as np

# Assuming experiment_tracker is a valid object from your Client() call
from zenml.client import Client

# Get the active stack
active_stack = Client().active_stack

# Access the experiment tracker
experiment_tracker = active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)  # Corrected assignment here
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,  # Changed to Series for clarity
    config: ModelNameConfig,
) -> RegressorMixin:
    
    model = None
    try:
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            logging.info("Model training completed")
            return trained_model
    
        else:
            raise ValueError(f"Invalid model name: {config.model_name}")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e