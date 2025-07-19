import os
import sys
import dill
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves any Python object (e.g., model, transformer) using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads any Python object saved with dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    """
    Trains and evaluates multiple regression models with hyperparameter tuning using R² score.
    """
    try:
        report = {}

        for model_name, model in models.items():
            try:
                print(f"\n🔧 Training model: {model_name}")

                hyperparams = param.get(model_name, {})

                gs = GridSearchCV(
                    model,
                    hyperparams,
                    cv=3,
                    scoring='r2',
                    n_jobs=1,
                    error_score='raise',
                    refit=True
                )

                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_

                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
                test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)

                print(f"\n📊 {model_name} Performance:")
                print(f"Train → R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
                print(f"Test  → R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

                report[model_name] = test_r2

            except Exception as model_err:
                print(f"❌ Skipping model '{model_name}' due to error:\n{model_err}")
                continue

        return report

    except Exception as e:
        raise CustomException(e, sys)
