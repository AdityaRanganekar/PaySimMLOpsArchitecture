import os
import sys
import numpy as np
import pandas as pd


"""
common constant variable for training pipeline
"""
TARGET_COLUMN = "isFraud"
PIPELINE_NAME: str = "FraudDetection"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "data.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR =os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"




"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "PaySim_Records"
DATA_INGESTION_DATABASE_NAME: str = "TRANSACTION_DATA"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2