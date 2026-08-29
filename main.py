import sys
import time
from fraud_detection.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from fraud_detection.entity.config_entity import TrainingPipelineConfig
from fraud_detection.components.data_ingestion import DataIngestion
from fraud_detection.components.data_validation import DataValidation
from fraud_detection.components.data_transformation import DataTransformation
from fraud_detection.components.model_trainer import ModelTrainer
from fraud_detection.logging.logger import logging
from fraud_detection.exception.exception import FraudDetectionException

if __name__=='__main__':
    try:
        training_pipeline_config= TrainingPipelineConfig()
        pipeline_start_time = time.time()

        # Data Ingestion
        data_ingestion_config= DataIngestionConfig(training_pipeline_config)
        data_ingestion= DataIngestion(data_ingestion_config)

        logging.info("Initiate Data Ingestion")
        ingestion_start = time.time()
        data_ingestion_artifact= data_ingestion.initiate_data_ingestion()
        ingestion_end = time.time()
        logging.info(f"Data Ingestion completed in {ingestion_end - ingestion_start:.2f} seconds")
        print(data_ingestion_artifact)

        # Data Validation
        data_validation_config= DataValidationConfig(training_pipeline_config)
        data_validation= DataValidation(data_ingestion_artifact, data_validation_config)

        logging.info("Initiate Data validation")
        validation_start = time.time()
        data_validation_artifact= data_validation.initiate_data_validation()
        validation_end = time.time()
        logging.info(f"Data Validation completed in {validation_end - validation_start:.2f} seconds")
        print(data_validation_artifact)

        # Data Transformation
        data_transformation_config= DataTransformationConfig(training_pipeline_config)
        data_transformation= DataTransformation(data_validation_artifact, data_transformation_config)

        logging.info("Initiate Data Transformation")
        transformation_start = time.time()
        data_transformation_artifact= data_transformation.initiate_data_transformation()
        transformation_end = time.time()
        logging.info(f"Data Transformation completed in {transformation_end - transformation_start:.2f} seconds")
        print(data_transformation_artifact)

        # Model Training
        model_trainer_config= ModelTrainerConfig(training_pipeline_config)
        model_trainer= ModelTrainer(data_transformation_artifact, model_trainer_config)
        
        logging.info("Initiate Model Training")
        training_start = time.time()
        model_trainer_artifact= model_trainer.initiate_model_trainer()
        training_end = time.time()
        logging.info(f"Model Training completed in {training_end - training_start:.2f} seconds")
        print(model_trainer_artifact)

        pipeline_end_time = time.time()
        total_time = pipeline_end_time - pipeline_start_time
        logging.info(f"========== TRAINING PIPELINE SUCCESS ==========")
        logging.info(f"Total Pipeline Execution Time: {total_time / 60:.2f} minutes")

    except Exception as e:
           raise FraudDetectionException(e,sys)