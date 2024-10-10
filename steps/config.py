from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """
    model name

    Args:
        BaseParameters (_type_): _description_
    """
    model_name:str = "LinearRegression"
     