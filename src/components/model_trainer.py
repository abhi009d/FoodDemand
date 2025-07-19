import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainingConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("🔀 Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            model = LogisticRegression(max_iter=1000)

            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'saga'],
                'penalty': ['l2']
            }

            logging.info("🔍 Starting RandomizedSearchCV for Logistic Regression")

            search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=5,
                                        scoring='r2', cv=3, verbose=1, random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)

            best_model = search.best_estimator_

            logging.info(f"✅ Best model parameters: {search.best_params_}")

            # Save the model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("💾 Model saved successfully")

            # Evaluation
            y_pred = best_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"📈 Model Evaluation - MSE: {mse:.2f}, MAE: {mae:.2f}, R2 Score: {r2:.2f}")

            return r2

        except Exception as e:
            raise CustomException(e, sys)
