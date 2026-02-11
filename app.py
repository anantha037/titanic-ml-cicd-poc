from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import logging


# LOGGING SETUP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# MODEL VERSION (for rollback)

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_PATH = f"models/model_{MODEL_VERSION}.joblib"

logger.info(f"Loading model version: {MODEL_VERSION}")

model = joblib.load(MODEL_PATH)


# FASTAPI APP

app = FastAPI()


# INPUT SCHEMA
class TitanicInput(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    Sibsp: int
    Parch: int
    Fare: float
    Embarked: int


# ROUTES

@app.get("/")
def home():
    return {"message": "Titanic ML API running", "model_version":MODEL_VERSION}


@app.post("/predict")
def predict(data: TitanicInput):
    
    logger.info(f"Prediction request received")

    features = np.array([[
        data.Pclass,
        data.Sex,
        data.Age,
        data.Sibsp,
        data.Parch,
        data.Fare,
        data.Embarked
    ]])

    prediction = model.predict(features)[0]

    logger.info(f"Prediction result: {prediction}")

    return {"prediction": int(prediction)}