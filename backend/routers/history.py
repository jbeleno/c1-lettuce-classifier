"""GET /history — recent predictions for the audit / 'what did the model
predict on these last N images' view."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import PredictionRecord
from backend.schemas import PredictionRow

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=list[PredictionRow])
def list_predictions(
    limit: int = Query(default=50, ge=1, le=500),
    label: str | None = Query(default=None),
    model: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    stmt = select(PredictionRecord).order_by(PredictionRecord.created_at.desc())
    if label:
        stmt = stmt.where(PredictionRecord.label == label)
    if model:
        stmt = stmt.where(PredictionRecord.model == model)
    stmt = stmt.limit(limit)
    rows = db.execute(stmt).scalars().all()
    return rows
