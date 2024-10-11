import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client
import numpy as np

from zenml.client import Client

# Get the active stack
active_stack = Client().active_stack

# Access the experiment tracker
experiment_tracker = active_stack.experiment_tracker


@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series  # Assuming y_test is a single column (series)
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"]  # Corrected annotation to rmse if it's RMSE
]:
    try:
        # Model prediction
        prediction = model.predict(X_test)
        
        # Calculate MSE
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)
        # If you want RMSE instead of MSE, take the square root of MSE
        rmse = np.sqrt(mse)
        mlflow.log_metric("rmse", rmse)
        # Calculate R² score
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2", r2)
        # Return r2 first, then rmse
        return r2, rmse

    except Exception as e:
        logging.error("An exception occurred during model evaluation", exc_info=True)
        raise e
