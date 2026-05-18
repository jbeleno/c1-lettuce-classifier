"""Pydantic 2 schemas. The API surface is shaped after the ``Prediction``
dataclass in ``src.inference`` so the contract between the model code and
the HTTP layer stays explicit."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model: str


class PredictionRow(PredictionResponse):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    image_sha256: str
    image_filename: str | None
    image_width: int
    image_height: int


class ModelInfo(BaseModel):
    name: str
    framework: str
    test_accuracy: float | None = None
    macro_f1: float | None = None
    classes: list[str] | None = None


class MetricsRow(BaseModel):
    model: str
    test_accuracy: float | None
    macro_f1: float | None
    macro_precision: float | None
    macro_recall: float | None
    best_val_accuracy: float | None


class MetricsResponse(BaseModel):
    rows: list[MetricsRow]
