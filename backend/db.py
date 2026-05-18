"""SQLAlchemy 2.x async engine + session + declarative base. Postgres in
production, SQLite when ``DATABASE_URL`` overrides it (used in tests)."""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

from src.config import POSTGRES_DSN

DATABASE_URL = os.environ.get("DATABASE_URL", POSTGRES_DSN)

_engine_kwargs: dict = {"future": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    # In-memory SQLite is per-connection by default; pin one connection so the
    # test client and the request handler share a single schema instance.
    if ":memory:" in DATABASE_URL:
        _engine_kwargs["poolclass"] = StaticPool

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from backend import models  # noqa: F401 — register ORM classes

    Base.metadata.create_all(bind=engine)
