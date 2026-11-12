"""Run the full training pipeline: ingest data, preprocess it, train and save the best model.

Usage (from the project root):
    python -m src.pipeline.train_pipeline
"""
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging


def run_training_pipeline():
    logging.info("Training pipeline started")

    train_path, test_path = DataIngestion().initiate_data_ingestion()
    train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_path, test_path)
    r2 = ModelTrainer().initiate_model_trainer(train_arr, test_arr)

    logging.info(f"Training pipeline finished. Best model test R2: {r2:.4f}")
    return r2


if __name__ == "__main__":
    score = run_training_pipeline()
    print(f"Training complete. Best model test R2 score: {score:.4f}")
    print("Saved model and preprocessor to the artifacts/ folder.")
