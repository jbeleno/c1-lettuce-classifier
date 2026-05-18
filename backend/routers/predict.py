"""POST /predict — accept an image upload, run inference, persist the result."""
from __future__ import annotations

import hashlib
import io
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import PredictionRecord
from backend.schemas import PredictionResponse
from src.inference import REGISTRY, available_models, predict

log = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])
DEFAULT_MODEL_PREFERENCE = (
    "ensemble_avg",
    "swin_t",
    "vit_b_16",
    "efficientnet_b0",
    "resnet50",
    "mobilenet_v3_small",
)


def _pick_default_model() -> str:
    avail = set(available_models())
    for name in DEFAULT_MODEL_PREFERENCE:
        if name in avail:
            return name
    raise HTTPException(
        status_code=503,
        detail="no trained models available — run `make train-all` first",
    )


@router.post("", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    if model and model not in REGISTRY:
        raise HTTPException(status_code=400, detail=f"unknown model: {model}")
    chosen = model or _pick_default_model()
    if chosen not in available_models():
        raise HTTPException(
            status_code=503,
            detail=f"model {chosen} is registered but its artifact is not on disk",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file upload")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc}")

    result = predict(image, chosen)
    record = PredictionRecord(
        model=result.model,
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        image_sha256=hashlib.sha256(raw).hexdigest(),
        image_filename=file.filename,
        image_width=image.width,
        image_height=image.height,
    )
    db.add(record)
    db.commit()
    return PredictionResponse(**result.to_dict())
