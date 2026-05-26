"""Model loading and prediction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from petfinder.constants import CLASS_LABELS_RU, DEFAULT_MODEL_PATH
from petfinder.preprocess import clean_dataframe, record_to_dataframe, records_to_dataframe


@dataclass
class PredictionResult:
    adoption_speed: int
    class_label_ru: str
    probabilities: Optional[Dict[str, float]] = None


class Predictor:
    """Wrapper around the saved sklearn pipeline."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path.resolve()}")
        self.pipeline = joblib.load(self.model_path)

    def predict_one(self, record: Dict[str, Any]) -> PredictionResult:
        df = clean_dataframe(record_to_dataframe(record))
        pred = int(self.pipeline.predict(df)[0])
        proba = self._predict_proba(df)
        return PredictionResult(
            adoption_speed=pred,
            class_label_ru=CLASS_LABELS_RU.get(pred, f"Класс {pred}"),
            probabilities=proba,
        )

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[PredictionResult]:
        df = clean_dataframe(records_to_dataframe(records))
        preds = self.pipeline.predict(df)
        probas = self._predict_proba_batch(df)
        results = []
        for i, pred in enumerate(preds):
            pred_int = int(pred)
            results.append(
                PredictionResult(
                    adoption_speed=pred_int,
                    class_label_ru=CLASS_LABELS_RU.get(pred_int, f"Класс {pred_int}"),
                    probabilities=probas[i] if probas else None,
                )
            )
        return results

    def _predict_proba(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        if not hasattr(self.pipeline, "predict_proba"):
            return None
        try:
            proba = self.pipeline.predict_proba(df)[0]
            classes = self.pipeline.classes_
            return {str(int(c)): float(p) for c, p in zip(classes, proba)}
        except Exception:
            return None

    def _predict_proba_batch(
        self, df: pd.DataFrame
    ) -> List[Optional[Dict[str, float]]]:
        if not hasattr(self.pipeline, "predict_proba"):
            return [None] * len(df)
        try:
            probas = self.pipeline.predict_proba(df)
            classes = self.pipeline.classes_
            return [
                {str(int(c)): float(p) for c, p in zip(classes, row)}
                for row in probas
            ]
        except Exception:
            return [None] * len(df)
