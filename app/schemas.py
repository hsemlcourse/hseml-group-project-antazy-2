"""Pydantic schemas for the prediction API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PetFeatures(BaseModel):
    """Input features matching train.csv (without AdoptionSpeed)."""

    Type: int = Field(..., ge=1, le=4, description="1=Dog, 2=Cat, 3=Other")
    Name: Optional[str] = None
    Age: int = Field(..., ge=0, description="Age in months")
    Breed1: int
    Breed2: int = 0
    Gender: int = Field(..., ge=1, le=3)
    Color1: int
    Color2: int = 0
    Color3: int = 0
    MaturitySize: int = Field(..., ge=1, le=4)
    FurLength: int = Field(..., ge=1, le=3)
    Vaccinated: int = Field(..., ge=1, le=3)
    Dewormed: int = Field(..., ge=1, le=3)
    Sterilized: int = Field(..., ge=1, le=3)
    Health: int = Field(..., ge=1, le=3)
    Quantity: int = 1
    Fee: int = Field(..., ge=0)
    State: int
    RescuerID: str
    VideoAmt: int = Field(..., ge=0)
    Description: Optional[str] = None
    PetID: Optional[str] = None
    PhotoAmt: float = Field(..., ge=0)

    def to_record(self) -> dict:
        return self.model_dump(exclude_none=False)


class PredictionResponse(BaseModel):
    adoption_speed: int
    class_label_ru: str
    probabilities: Optional[Dict[str, float]] = None


class BatchPredictRequest(BaseModel):
    records: List[PetFeatures] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    model_loaded: bool
