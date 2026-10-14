"""API FastAPI qui sert le modèle Iris."""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv(
    "MODEL_DIR", Path(__file__).resolve().parents[1] / "model"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("iris-api")

PREDICTIONS = Counter(
    "iris_predictions_total", "Nombre de prédictions par classe", ["species"])

state = {}


@asynccontextmanager
async def lifespan(app):
    state["model"] = joblib.load(MODEL_DIR / "model.joblib")
    with open(MODEL_DIR / "metadata.json") as f:
        state["metadata"] = json.load(f)
    logger.info("Modèle chargé : %s", state["metadata"]["model_name"])
    yield
    state.clear()


app = FastAPI(title="Iris ML API", version="1.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)  # endpoint /metrics


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path,
                response.status_code, duration_ms)
    return response


class IrisRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


@app.get("/")
def home():
    return {"message": "Iris ML API is running"}


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ok", "model": state["metadata"]["model_name"]}


@app.post("/predict")
def predict(data: IrisRequest):
    meta = state["metadata"]
    X = pd.DataFrame([[data.sepal_length, data.sepal_width,
                       data.petal_length, data.petal_width]],
                     columns=meta["features"])
    proba = state["model"].predict_proba(X)[0]
    pred = int(proba.argmax())
    species = meta["classes"][pred]
    PREDICTIONS.labels(species=species).inc()
    logger.info("input=%s prediction=%s", data.model_dump(), species)
    return {
        "prediction": pred,
        "species": species,
        "confidence": float(proba[pred]),
        "probabilities": dict(zip(meta["classes"], proba.round(4).tolist())),
    }
