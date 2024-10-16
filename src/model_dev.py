import logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    abstract class for all the models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class LinearRegressionModel(Model):
    
    def train(self, X_train, y_train, **kwargs):
        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info("MOdel training completed!")
            return reg
        except Exception as e:
            logging.error("Error in training model: ", str(e))
            raise e
        