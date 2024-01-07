import os 
import sys 
from dataclasses import dataclass
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exceptions import CustomException
from sklearn.linear_model import SGDRegressor,LinearRegression
from sklearn.ensemble import (AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting train and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Linear Regression":LinearRegression(),
                "Random Forest Regressor":RandomForestRegressor(),
                "SGD Regressor":SGDRegressor(),
                "KNeighbors Regressor":KNeighborsRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "Decision Tree Regressor":DecisionTreeRegressor()
            }


            model_report:dict= evaluate_models(X_train,y_train,X_test,y_test,models)

            ## Get the best model score from dict

            best_model_score=max(sorted(model_report.values()))

            ## Get the best model name fron dict

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ] 

            best_model=models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found: {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_score_value=r2_score(y_test,predicted)
    
            return r2_score_value

        except Exception as e:
            raise CustomException(e,sys)    
