"""GET /metrics — model comparison surface used by the dashboard. Backed by
the metadata.json files produced during training."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas import MetricsResponse, MetricsRow
from src.evaluate_all import collect

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", response_model=MetricsResponse)
def metrics():
    df = collect()
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="no trained models found")
    rows = [
        MetricsRow(
            model=r["model"],
            test_accuracy=r["test_accuracy"],
            macro_f1=r["macro_f1"],
            macro_precision=r["macro_precision"],
            macro_recall=r["macro_recall"],
            best_val_accuracy=r["best_val_accuracy"],
        )
        for r in df.to_dict(orient="records")
    ]
    return MetricsResponse(rows=rows)
