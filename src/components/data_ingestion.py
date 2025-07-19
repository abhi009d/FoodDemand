import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("📥 Starting data ingestion...")

        try:
            # Load the dataset
            df = pd.read_csv("notebook/data/restaurant_food_demand.csv")
            logging.info("✅ Dataset loaded successfully.")

            # Optional: Print column names for debug
            logging.info(f"Columns found in dataset: {df.columns.tolist()}")

            # Check if 'orders' column exists
            if "orders" not in df.columns:
                raise CustomException("❌ Target column 'orders' not found in dataset.", sys)

            # Create necessary directories
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("💾 Raw data saved.")

            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("🔀 Train-test split completed.")

            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("✅ Train and test data saved.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("❌ Error during data ingestion.")
            raise CustomException(e, sys)
