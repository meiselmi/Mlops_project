import logging

import pandas as pd 
from zenml import step 
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreProcessStrategy
from typing import Tuple
from typing_extensions import Annotated
@step 
def clean_data (df : pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame , "X_train"],
    Annotated[pd.DataFrame , "X_test"],
    Annotated[pd.Series , "y_train"],
    Annotated[pd.Series , "y_test"],
    ] : 
    
    try : 
        print(df.columns)
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data,divide_strategy)
        X_train , X_test, y_train , y_test = data_cleaning.handle_data()
        logging.info("Data cleaning complete")
        return X_train, X_test, y_train , y_test

    except Exception as e :
        logging.error("error while cleaning: {}".format(e))
        raise e 
