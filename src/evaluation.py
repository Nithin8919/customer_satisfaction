import logging 
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error,root_mean_squared_error

class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray,y_pred : np.ndarray ):
        pass
    
class MSE(Evaluation):
    
        """
        Means square error

        Args:
            Evaluation (_type_): _description_
        """
        
        def calculate_scores(self, y_true: np.ndarray,y_pred : np.ndarray):
            try:
                logging.info("Calculating MSE")
                mse = mean_squared_error(y_true, y_pred)
                logging.info(f"Mean squared error {mse}")
                return mse
            except Exception as e:
                logging.error(f"Error calculating MSE {e}")
                raise e
class R2(Evaluation):
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculate RMSE")
            rmse = root_mean_squared_error(y_true,y_pred)
            logging.info(f'Root Mean Squared Error{rmse}')
            return rmse
        except Exception as e:
            logging.error(f"Error calculating RMSE {e}")
            raise e
        
             
        
        