import sys
import os
import pymongo
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from uvicorn import run as app_run
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse

from fraud_detection.exception.exception import FraudDetectionException
from fraud_detection.logging.logger import logging
from fraud_detection.pipeline.training_pipeline import TrainingPipeline
from fraud_detection.utils.ml_utils.model.estimator import FraudDetectionModel

from fraud_detection.utils.main_utils.utils import load_object
from fraud_detection.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from fraud_detection.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

from fastapi.templating import Jinja2Templates

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

mongo_db_url = os.getenv("MONGODB_URL_KEY")

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

templates = Jinja2Templates(directory="./templates")
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_training_in_background():
    """Runs the heavy ML pipeline safely in the background"""
    train_pipeline = TrainingPipeline()
    train_pipeline.run_pipeline()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route(background_tasks: BackgroundTasks):
    try:

        background_tasks.add_task(run_training_in_background)
        return JSONResponse(content={"message": "Training started in background. Check terminal for logs."})
    except Exception as e:
        raise FraudDetectionException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        
        detection_model = FraudDetectionModel(preprocessor=preprocessor, model=final_model)
        
        # Generate predictions
        y_pred = detection_model.predict(df)
        df['predicted_column'] = y_pred
        
        os.makedirs('prediction_output', exist_ok=True)
        df.to_csv('prediction_output/output.csv', index=False)

        table_html = df.head(50).to_html(classes='table table-striped', index=False)
        
        return templates.TemplateResponse(request=request, name="table.html", context={"table": table_html})
        
    except Exception as e:
        raise FraudDetectionException(e, sys)
    
if __name__=="__main__":
    app_run(app, host="0.0.0.0", port=8000)
