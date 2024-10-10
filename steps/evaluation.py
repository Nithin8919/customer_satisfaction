import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import numpy as np

@step
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
        
        # If you want RMSE instead of MSE, take the square root of MSE
        rmse = np.sqrt(mse)
        
        # Calculate R² score
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        
        # Return r2 first, then rmse
        return r2, rmse

    except Exception as e:
        logging.error("An exception occurred during model evaluation", exc_info=True)
        raise e
