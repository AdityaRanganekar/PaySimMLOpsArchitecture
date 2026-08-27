from fraud_detection.entity.config_entity import DataIngestionConfig, DataValidationConfig
from fraud_detection.entity.config_entity import TrainingPipelineConfig
from fraud_detection.components.data_ingestion import DataIngestion
from fraud_detection.components.data_validation import DataValidation
from fraud_detection.logging.logger import logging
from fraud_detection.exception.exception import FraudDetectionException

import sys

if __name__=='__main__':
    try:
        training_pipeline_config= TrainingPipelineConfig()
        data_ingestion_config= DataIngestionConfig(training_pipeline_config)
        data_ingestion= DataIngestion(data_ingestion_config)

        logging.info("Initiate Data Ingestion")
        data_ingestion_artifact= data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed")
        print(data_ingestion_artifact)

        data_validation_config= DataValidationConfig(training_pipeline_config)
        data_validation= DataValidation(data_ingestion_artifact, data_validation_config)

        logging.info("Initiate Data validation")
        data_validation_artifact= data_validation.initiate_data_validation()
        logging.info("Data Validation completed")
        print(data_validation_artifact)
        
    except Exception as e:
           raise FraudDetectionException(e,sys)

