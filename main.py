from fraud_detection.entity.config_entity import DataIngestionConfig
from fraud_detection.entity.config_entity import TrainingPipelineConfig
from fraud_detection.components.data_ingestion import DataIngestion
from fraud_detection.logging.logger import logging
from fraud_detection.exception.exception import FraudDetectionException

import sys

if __name__=='__main__':
    try:
        trainingpipelineconfig= TrainingPipelineConfig()
        dataingestionconfig= DataIngestionConfig(trainingpipelineconfig)
        data_ingestion= DataIngestion(dataingestionconfig)

        logging.info("Initiate Data Ingestion")

        dataingestionartifact= data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
    except Exception as e:
           raise FraudDetectionException(e,sys)

