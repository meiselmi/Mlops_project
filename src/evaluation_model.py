import logging 
from abc import ABC ,abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error , r2_score
class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self,y_true : np.ndarray , y_pred : np.ndarray):
        pass
        

class MSE(Evaluation):

    def calculate_scores(self,y_true : np.ndarray , y_pred : np.ndarray):
        
        try : 
            logging.info("calculating MSE")
            mse = mean_squared_error(y_true , y_pred)
            logging.info("The mean squared error value is: " + str(mse))
            return mse 
        except Exception as e :
            logging.error("error in calculating mse : {}".format(e))
            raise e 
        

class R2(Evaluation):
    def calculate_scores(self, y_true, y_pred):
        try : 
            logging.info("calculating R2")
            r2 = r2_score(y_true , y_pred)
            logging.info("The r2 score value is: " + str(r2))
            return r2 
        except Exception as e :
            logging.error("error in calculating r2 :{}".format(e))
            raise e 

        
