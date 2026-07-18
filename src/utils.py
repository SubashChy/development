import os 
import sys 
sys.path.append('.')
from src.logger import logging 
from src.exception import CustomException
import dill
from typing import List
from sklearn.metrics import r2_score
from typing import Any

def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok= True)
        
        with open (file_path , "wb") as file_obj:
            dill.dump(obj , file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)


def model_evaluate(X_train, X_test, y_train  , y_test , models) -> dict[str , dict[str , Any]]:
    try:
        report = {}
        for name , model in models.items():
            model.fit(X_train , y_train)
            y_pred = model.predict(X_test)
            
            score = r2_score(y_test , y_pred)
            
            report[name] ={
                "model" : model , 
                "r2_score" : score
            }         
            return report
        
    except Exception as e:
        raise CustomException(e, sys)  