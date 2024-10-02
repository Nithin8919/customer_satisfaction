from pipelines.training_pipeline import train_pipeline
from pydantic import BaseModel, PydanticUserError

if __name__ == "__main__":
    
    train_pipeline(data_path="/Users/nitin/Documents/customer_satisfaction/data/olist_customers_dataset.csv")



try:

    class Model(BaseModel):
        x: 43 = 123

except PydanticUserError as exc_info:
    assert exc_info.code == 'schema-for-unknown-type'