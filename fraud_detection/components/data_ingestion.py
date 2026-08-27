import os
import sys
import pymongo
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd

from fraud_detection.entity.config_entity import DataIngestionConfig
from fraud_detection.logging.logger import logging
from fraud_detection.exception.exception import FraudDetectionException
from fraud_detection.entity.artifact_entity import DataIngestionArtifact

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        try:
            self.config= config
        except Exception as e:
            raise FraudDetectionException(e, sys)

        
    def export_collection_as_dataframe(self):
        """
        Read data from mongodb using Smart Extraction for Imbalanced Data
        """
        try:
            database_name = self.config.database_name
            collection_name = self.config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            logging.info("Starting Smart Data Extraction from local MongoDB...")
            
            # Gets ALL Fraudulent transactions (~8,200 rows)
            logging.info("Extracting all fraud records...")
            fraud_cursor = collection.find({"isFraud": 1})
            df_fraud = pd.DataFrame(list(fraud_cursor))
            
            # Gets a random sample of Normal transactions (e.g., 200,000 rows)
            # This prevents your PC from crashing while providing enough data to train.
            logging.info("Extracting a sample of 200,000 normal records...")
            normal_cursor = collection.aggregate([
                { "$match": { "isFraud": 0 } },
                { "$sample": { "size": 200000 } }
            ])
            df_normal = pd.DataFrame(list(normal_cursor))
            
            # Combines them and shuffle
            logging.info("Combining and shuffling the dataset...")
            df = pd.concat([df_fraud, df_normal], ignore_index=True)
            
            # Shuffles the dataframe so frauds aren't all at the top
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            logging.info(f"Final Extracted Dataframe Shape: {df.shape}")

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])
            
            df.replace({"na": np.nan}, inplace=True)

            return df
            
        except Exception as e:
            raise FraudDetectionException(e, sys)


    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path= self.config.feature_store_file_path
            dir_path= os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False, header=True)

            return dataframe
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.config.train_test_split_ratio, stratify=dataframe["isFraud"])

            logging.info("Performed train-test split on Dataframe")

            logging.info("Exited split_data_as_train_test method of Data_ingestion class")

            dir_path= os.path.dirname(self.config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file path")

            train_set.to_csv(self.config.training_file_path, index=False, header=True)
            test_set.to_csv(self.config.testing_file_path, index=False, header=True)

            logging.info("Exported train and test file path")
        except Exception as e:
            raise FraudDetectionException(e,sys)
    
    def initiate_data_ingestion(self):
        try:
            dataframe= self.export_collection_as_dataframe()
            dataframe= self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            dataingestionartifact= DataIngestionArtifact(trained_file_path= self.config.training_file_path,
                                                         test_file_path= self.config.testing_file_path)

            return dataingestionartifact
        except Exception as e:
            raise FraudDetectionException(e, sys)