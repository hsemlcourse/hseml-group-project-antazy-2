"""API tests for FastAPI prediction service."""

from pathlib import Path

import pytest
from petfinder.constants import DEFAULT_MODEL_PATH

MODEL_EXISTS = DEFAULT_MODEL_PATH.exists()

SAMPLE_RECORD = {
    "Type": 2,
    "Name": "Nibble",
    "Age": 3,
    "Breed1": 299,
    "Breed2": 0,
    "Gender": 1,
    "Color1": 1,
    "Color2": 7,
    "Color3": 0,
    "MaturitySize": 1,
    "FurLength": 1,
    "Vaccinated": 2,
    "Dewormed": 2,
    "Sterilized": 2,
    "Health": 1,
    "Quantity": 1,
    "Fee": 100,
    "State": 41326,
    "RescuerID": "8480853f516546f6cf33aa88cd76c379",
    "VideoAmt": 0,
    "Description": "Friendly cat for adoption.",
    "PetID": "86e1089a3",
    "PhotoAmt": 1.0,
}


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "model_missing")
    assert "model_path" in data


@pytest.mark.skipif(not MODEL_EXISTS, reason="best_model.joblib not present")
def test_predict(client):
    response = client.post("/predict", json=SAMPLE_RECORD)
    assert response.status_code == 200
    data = response.json()
    assert "adoption_speed" in data
    assert 0 <= data["adoption_speed"] <= 4
    assert "class_label_ru" in data


@pytest.mark.skipif(not MODEL_EXISTS, reason="best_model.joblib not present")
def test_predict_batch(client):
    response = client.post(
        "/predict/batch",
        json={"records": [SAMPLE_RECORD, SAMPLE_RECORD]},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
