import logging
import pandas as pd 
from zenml import step 
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker= experiment_tracker.name) 
def train_model( X_train : pd.DataFrame , 
                
                y_train : pd.DataFrame , 
               
                ) -> RegressorMixin :

    try:
        mlflow.sklearn.autolog()
        model = LinearRegressionModel()
        trained_model = model.train(X_train,y_train)
        return trained_model
        
           
    except Exception as e : 
        logging.error("training failed:{} ".format(e))    