import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("🔄 Loading model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("✅ Model and preprocessor loaded successfully.")

            print("🧪 Transforming input features...")
            data_scaled = preprocessor.transform(features)

            print("📈 Making prediction...")
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 date: str,
                 day_of_week: str,
                 time_of_day: str,
                 food_item: str,
                 temperature: float,
                 rainfall_mm: float,
                 holiday: str,
                 local_event: str,
                 promotion_active: str,
                 restaurant_location: str,
                 customer_footfall: int):
        
        self.date = date
        self.day_of_week = day_of_week
        self.time_of_day = time_of_day
        self.food_item = food_item
        self.temperature = temperature
        self.rainfall_mm = rainfall_mm
        self.holiday = holiday
        self.local_event = local_event
        self.promotion_active = promotion_active
        self.restaurant_location = restaurant_location
        self.customer_footfall = customer_footfall

    def get_data_as_data_frame(self):
        try:
            input_dict = {
                "date": [self.date],
                "day_of_week": [self.day_of_week],
                "time_of_day": [self.time_of_day],
                "food_item": [self.food_item],
                "temperature": [self.temperature],
                "rainfall_mm": [self.rainfall_mm],
                "holiday": [self.holiday],
                "local_event": [self.local_event],
                "promotion_active": [self.promotion_active],
                "restaurant_location": [self.restaurant_location],
                "customer_footfall": [self.customer_footfall]
            }

            return pd.DataFrame(input_dict)

        except Exception as e:
            raise CustomException(e, sys)
