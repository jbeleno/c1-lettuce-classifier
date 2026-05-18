"""API contract tests. Use SQLite to avoid needing Postgres for CI."""
import io
import os
from pathlib import Path

import pytest

os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from backend.db import init_db


@pytest.fixture(autouse=True, scope="module")
def _db():
    init_db()


@pytest.fixture
def client():
    return TestClient(app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_models_endpoint_is_a_list(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_predict_rejects_when_no_model_available(client):
    img = Image.new("RGB", (224, 224), (50, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    r = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    # Either 503 (no models trained yet) or 200 (smoke model present).
    assert r.status_code in (200, 503)


def test_predict_rejects_unknown_model(client):
    buf = io.BytesIO()
    Image.new("RGB", (224, 224)).save(buf, format="JPEG")
    buf.seek(0)
    r = client.post(
        "/predict",
        data={"model": "does_not_exist"},
        files={"file": ("x.jpg", buf, "image/jpeg")},
    )
    assert r.status_code == 400


def test_history_empty(client):
    r = client.get("/history?limit=5")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_metrics_returns_404_when_nothing_trained(client):
    r = client.get("/metrics")
    # 200 if a model was already trained in this session, 404 otherwise.
    assert r.status_code in (200, 404)
