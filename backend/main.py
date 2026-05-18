"""FastAPI application factory. Endpoints:

  POST /predict   — upload an image, get a Prediction, persist it
  GET  /history   — recent predictions, optional filter by label/model
  GET  /metrics   — model comparison table
  GET  /models    — list of trained, loadable models
  GET  /healthz   — liveness probe

Run with:  uvicorn backend.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db import init_db
from backend.routers import history, metrics, models, predict
from src.utils import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logging.getLogger(__name__).info("starting C1 backend")
    init_db()
    yield


app = FastAPI(
    title="C1 — Hydroponic lettuce growth-stage classifier",
    version="0.1.0",
    description="Image-classification backend for the USCO AI course project (C1).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(history.router)
app.include_router(metrics.router)
app.include_router(models.router)


@app.get("/healthz", tags=["meta"])
def healthz():
    return {"status": "ok"}
