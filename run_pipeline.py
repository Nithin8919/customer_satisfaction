from pipelines.training_pipeline import train_pipeline
import sys
sys.path.insert(0, '/Users/nitin/Documents/customer_satisfaction/run_pipeline.py')
from zenml.client import Client
active_stack = Client().active_stack
active_stack2 = active_stack.experiment_tracker

if __name__ == "__main__":
    print(active_stack2.get_tracking_uri())
    train_pipeline(data_path="/Users/nitin/Documents/customer_satisfaction/data/olist_customers_dataset.csv")



