import os
import sys
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


def main():
    try:
        logging.info("🚀 Starting the training pipeline...")

        # 1️⃣ Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        # 2️⃣ Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        # 3️⃣ Model Training
        trainer = ModelTrainer()
        final_score = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"🎉 Training complete. Final test score: {final_score:.4f}")
        print(f"\n✅ Final Model Test Score: {final_score:.4f}")

    except Exception as e:
        logging.error("❌ Training pipeline failed.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
