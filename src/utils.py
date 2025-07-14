import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, classification_report

from src.exception import CustomException


def save_object(file_path: str, obj) -> None:
    """
    Saves any Python object using dill serialization.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Loads a Python object from a file using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(y_true, y_pred) -> dict:
    """
    Returns evaluation metrics for classification models.
    """
    try:
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        return {
            "accuracy": accuracy,
            "report": report
        }
    except Exception as e:
        raise CustomException(e, sys)
