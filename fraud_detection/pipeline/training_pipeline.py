import os
import sys
import time

from fraud_detection.exception.exception import FraudDetectionException
from fraud_detection.logging.logger import logging

from fraud_detection.components.data_ingestion import DataIngestion
from fraud_detection.components.data_validation import DataValidation
from fraud_detection.components.data_transformation import DataTransformation
from fraud_detection.components.model_trainer import ModelTrainer

from fraud_detection.constant.training_pipeline import TRAINING_BUCKET_NAME, SAVED_MODEL_DIR
from fraud_detection.cloud.s3_syncer import S3Sync

from fraud_detection.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from fraud_detection.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact


class TrainingPipeline:

    def __init__(self):
        self.training_pipeline_config= TrainingPipelineConfig()
        self.s3_sync = S3Sync()
    
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config= DataIngestionConfig(self.training_pipeline_config)
            logging.info("Initiate Data Ingestion")
            ingestion_start = time.time()

            data_ingestion= DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact= data_ingestion.initiate_data_ingestion()
            ingestion_end = time.time()
            logging.info(f"Data Ingestion completed in {ingestion_end - ingestion_start:.2f} seconds")
            logging.info(data_ingestion_artifact)

            return data_ingestion_artifact

        except Exception as e:
            raise FraudDetectionException(e, sys)

    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config= DataValidationConfig(self.training_pipeline_config)
            logging.info("Initiate Data validation")
            validation_start = time.time()

            data_validation= DataValidation(data_ingestion_artifact, data_validation_config)
            data_validation_artifact= data_validation.initiate_data_validation()
            validation_end = time.time()
            logging.info(f"Data Validation completed in {validation_end - validation_start:.2f} seconds")
            logging.info(data_validation_artifact)

            return data_validation_artifact
        
        except Exception as e:
            raise FraudDetectionException(e, sys)
            
    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config= DataTransformationConfig(self.training_pipeline_config)
            logging.info("Initiate Data Transformation")
            transformation_start = time.time()

            data_transformation= DataTransformation(data_validation_artifact, data_transformation_config)
            data_transformation_artifact= data_transformation.initiate_data_transformation()
            transformation_end = time.time()
            logging.info(f"Data Transformation completed in {transformation_end - transformation_start:.2f} seconds")
            logging.info(data_transformation_artifact)

            return data_transformation_artifact
        
        except Exception as e:
            raise FraudDetectionException(e, sys)


    def start_model_training(self, data_transformation_artifact:DataTransformationArtifact):
        try:
            # Model Training
            model_trainer_config= ModelTrainerConfig(self.training_pipeline_config)
            logging.info("Initiate Model Training")
            training_start = time.time()

            model_trainer= ModelTrainer(data_transformation_artifact, model_trainer_config)
            model_trainer_artifact= model_trainer.initiate_model_trainer()
            training_end = time.time()
            logging.info(f"Model Training completed in {(training_end - training_start)/60:.2f} minutes")
            logging.info(model_trainer_artifact)
    
            

            return model_trainer_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys)


    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir, aws_bucket_url = aws_bucket_url)
        except Exception as e:
            raise FraudDetectionException(e,sys)
        
        
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.model_dir, aws_bucket_url = aws_bucket_url)
        except Exception as e:
            raise FraudDetectionException(e,sys)
        

    def run_pipeline(self):
        try:
            pipeline_start_time = time.time()
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_training(data_transformation_artifact=data_transformation_artifact)

            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
            
            pipeline_end_time = time.time()
            total_time = pipeline_end_time - pipeline_start_time
            logging.info(f"========== TRAINING PIPELINE SUCCESS ==========")
            logging.info(f"Total Pipeline Execution Time: {total_time / 60:.2f} minutes")
            
            return model_trainer_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys)