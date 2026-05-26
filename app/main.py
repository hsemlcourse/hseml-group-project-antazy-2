"""FastAPI service for adoption speed prediction."""

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from petfinder.constants import DEFAULT_MODEL_PATH
from petfinder.inference import Predictor, PredictionResult

from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PetFeatures,
    PredictionResponse,
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor(model_path=MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_predictor()
    yield


app = FastAPI(
    title="PetFinder Adoption Speed API",
    description="Predict AdoptionSpeed for shelter animals (tabular model).",
    version="1.0.0",
    lifespan=lifespan,
)


def _to_response(result: PredictionResult) -> PredictionResponse:
    return PredictionResponse(
        adoption_speed=result.adoption_speed,
        class_label_ru=result.class_label_ru,
        probabilities=result.probabilities,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = MODEL_PATH.exists()
    return HealthResponse(
        status="ok" if loaded else "model_missing",
        model_path=str(MODEL_PATH.resolve()),
        model_loaded=loaded,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PetFeatures) -> PredictionResponse:
    try:
        result = get_predictor().predict_one(features.to_record())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _to_response(result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest) -> BatchPredictResponse:
    records = [r.to_record() for r in body.records]
    try:
        results = get_predictor().predict_batch(records)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BatchPredictResponse(predictions=[_to_response(r) for r in results])
