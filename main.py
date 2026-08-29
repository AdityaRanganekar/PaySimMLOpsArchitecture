from fraud_detection.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from fraud_detection.entity.config_entity import TrainingPipelineConfig
from fraud_detection.components.data_ingestion import DataIngestion
from fraud_detection.components.data_validation import DataValidation
from fraud_detection.components.data_transformation import DataTransformation
from fraud_detection.logging.logger import logging
from fraud_detection.exception.exception import FraudDetectionException

import sys

if __name__=='__main__':
    try:
        training_pipeline_config= TrainingPipelineConfig()

        # Data Ingestion
        data_ingestion_config= DataIngestionConfig(training_pipeline_config)
        data_ingestion= DataIngestion(data_ingestion_config)

        logging.info("Initiate Data Ingestion")
        data_ingestion_artifact= data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed")
        print(data_ingestion_artifact)

        # Data Validation
        data_validation_config= DataValidationConfig(training_pipeline_config)
        data_validation= DataValidation(data_ingestion_artifact, data_validation_config)

        logging.info("Initiate Data validation")
        data_validation_artifact= data_validation.initiate_data_validation()
        logging.info("Data Validation completed")
        print(data_validation_artifact)

        # Data Transfromation
        data_transformation_config= DataTransformationConfig(training_pipeline_config)
        data_transformation= DataTransformation(data_validation_artifact, data_transformation_config)

        logging.info("Initiate Data Transformation")
        data_transformation_artifact= data_transformation.initiate_data_transformation()
        logging.info("Data Transformation completed")
        print(data_transformation_artifact)
        
    except Exception as e:
           raise FraudDetectionException(e,sys)

