import json

import os

import numpy as np
import pandas as pd
#from materializer.custom_materializer import cs_materializer
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
) 
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
#from zenml.steps import BaseParameters
from pipelines.utils import get_data_for_test


import pandas as pd

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step(enable_cache=False)
def dynamic_importer()-> str: 
    data = get_data_for_test()
    return data

@step
def deployement_trigger(
    accuracy : float ,
    min_accuracy : float =0,
):
    """it looks at the model accuracy and decides if it should deploy it or not """
    return accuracy>=min_accuracy

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name :str,
    pipeline_step_name : str,
    running : bool =True, 
    model_name :str= "model",

)-> MLFlowDeploymentService :
    """getting the mlflow deployer stack component"""
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        model_name= model_name,
        running= running,

    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]

@step 
def predictor(
    service : MLFlowDeploymentService,
    data : str,
)-> np.ndarray :
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


    
@pipeline(enable_cache=False , settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path : str,
    min_accuracy : float =0,
    workers : int= 1 ,
    timeout : int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path)
    X_train , X_test, y_train ,y_test = clean_data(df)
    model =train_model(X_train  , y_train )
    mse , r2 = evaluate_model(model , X_test , y_test)

    deployement_decision = deployement_trigger(r2)
    mlflow_model_deployer_step( 
        model =model ,
        deploy_decision = deployement_decision,
        workers = workers,
        timeout = timeout,
    )
@pipeline(enable_cache=False , settings={"docker" : docker_settings})
def inference_pipeline(
    pipeline_name : str,
    pipeline_step_name : str
    
):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        running = False,
    )
    prediction = predictor(service=service , data = data)
    return prediction