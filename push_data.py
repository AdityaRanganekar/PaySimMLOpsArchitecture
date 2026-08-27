import os
import sys
import pandas as pd
import pymongo
from dotenv import load_dotenv

from fraud_detection.exception.exception import FraudDetectionException
from fraud_detection.logging.logger import logging

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(f"Connecting to MongoDB at: {MONGO_DB_URL}")

class FraudDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise FraudDetectionException(e, sys)
        
    def push_data_in_chunks(self, file_path, database_name, collection_name, chunk_size=50000):
        """
        Reads the massive CSV in chunks and inserts them sequentially into MongoDB.
        """
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[database_name]
            self.collection = self.database[collection_name]
            
            total_inserted = 0

            for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                chunk.reset_index(drop=True, inplace=True)

                records = chunk.to_dict(orient='records')

                self.collection.insert_many(records)
                total_inserted += len(records)
                
                print(f"Batch {chunk_idx + 1} pushed. Total records so far: {total_inserted}")
                
            return total_inserted

        except Exception as e:
            raise FraudDetectionException(e, sys)
        
if __name__ == '__main__':

    FILE_PATH = "transactions_data/data.csv"
    DATABASE = "TRANSACTION_DATA"
    COLLECTION = "PaySim_Records"
    
    networkobj = FraudDataExtract()
    
    print("Initiating batch ingestion for PaySim dataset...")
    
    no_of_records = networkobj.push_data_in_chunks(
        file_path=FILE_PATH,
        database_name=DATABASE,
        collection_name=COLLECTION,
        chunk_size=50000
    )
    
    print(f"\nSuccess! Total number of records inserted: {no_of_records}")