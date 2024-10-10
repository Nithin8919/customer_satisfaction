from pipelines.training_pipeline import train_pipeline
import sys
sys.path.insert(0, '/Users/nitin/Documents/customer_satisfaction/run_pipeline.py')

if __name__ == "__main__":
    
    train_pipeline(data_path="/Users/nitin/Documents/customer_satisfaction/data/olist_customers_dataset.csv")



