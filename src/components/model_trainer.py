import os 
import sys 
sys.path.append('.')
import pandas as pd 
import numpy as np 
from src.logger import logging 
from src.exception import CustomException
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import save_object, model_evaluate
from typing import Any

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts' , 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array , test_array):
        try: 
            logging.info('Split training and testing data')
            x_train , y_train , x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            model_report : dict[str , dict[str, Any]] = model_evaluate(X_train = x_train , X_test = x_test , y_train = y_train , y_test = y_test , models = models)
            
            #get best model 
            best_model_name = max(
                model_report , 
                key= lambda x : model_report[x]['r2_score']
            )
            
            
            print(f"Best model : {best_model_name}")
            
            
            #best model and its score 
            best_model = model_report[best_model_name]["model"]
            best_score = model_report[best_model_name]["r2_score"]
            print(best_score)
            print(type(best_score))
            
            if best_score < 0.6 :
                raise CustomException("No best model found")
            logging.info("Best Model Succesfully found")
            
            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj= best_model    
            )
            
            predicted = best_model.predict(x_test)
            
            r2score = r2_score(y_test , predicted)
            return r2score
            

        except Exception as e:
            raise CustomException(e, sys)
