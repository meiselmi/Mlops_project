from pipelines import training_pipeline
from zenml.client import Client 

if __name__=="__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    training_pipeline(data_path='/home/mei/Desktop/Mlops/data/olist_customers_dataset.csv')


