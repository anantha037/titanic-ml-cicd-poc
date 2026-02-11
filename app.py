from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import logging
import time
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest


# LOGGING SETUP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# MODEL VERSION (for rollback)

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_PATH = f"models/model_{MODEL_VERSION}.joblib"

logger.info(f"Loading model version: {MODEL_VERSION}")

model = joblib.load(MODEL_PATH)


# PROMETHEUS METRICS
REQUEST_COUNT = Counter(
    "titanic_requests_total",
    "Total prediction requests"
)

REQUEST_LATENCY = Histogram(
    "titanic_prediction_latency_seconds",
    "Prediction latency in seconds"
)



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

    start_time = time.time()
    REQUEST_COUNT.inc()

    logger.info("Prediction request received")

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

    REQUEST_LATENCY.observe(time.time() - start_time)

    return {"prediction": int(prediction)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")