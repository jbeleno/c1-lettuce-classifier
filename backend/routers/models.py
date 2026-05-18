"""GET /models — what's loaded / loadable right now."""
from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import ModelInfo
from src.inference import REGISTRY, available_models, model_metadata

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
def list_models():
    out: list[ModelInfo] = []
    avail = set(available_models())
    for name, spec in REGISTRY.items():
        if name not in avail:
            continue
        meta = model_metadata(name) or {}
        cr = meta.get("classification_report", {}) or {}
        macro = cr.get("macro avg", {}) if isinstance(cr, dict) else {}
        out.append(
            ModelInfo(
                name=name,
                framework=spec.framework,
                test_accuracy=meta.get("test_accuracy"),
                macro_f1=macro.get("f1-score") if isinstance(macro, dict) else None,
                classes=meta.get("classes"),
            )
        )
    return out
