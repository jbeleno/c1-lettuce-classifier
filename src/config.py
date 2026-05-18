"""Project-wide configuration. Everything that varies between machines or
runs lives here so the rest of the codebase can stay declarative."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models_saved"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, SPLITS_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_DATASET_PATH = Path(
    os.environ.get(
        "RAW_DATASET_PATH",
        str(DATA_DIR / "lettuce pallets.v2i.multiclass"),
    )
)

SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "42"))
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

CLASSES = ["empty_pod", "germination", "young", "pod", "Ready"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Class colors mirror the IDE-syntax palette from C2 (slate-900 design system).
CLASS_COLORS = {
    "empty_pod": "#64748B",   # slate / muted — nothing growing
    "germination": "#A78BFA", # violet — keyword
    "young": "#FBBF24",       # amber — warning / nascent
    "pod": "#38BDF8",         # sky — developing
    "Ready": "#22C55E",       # green — harvest-ready (added diff line)
}

IMG_SIZE = 224
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))

# Two-phase TF training (CNNs)
TF_HEAD_EPOCHS = int(os.environ.get("TF_HEAD_EPOCHS", "12"))
TF_FINE_TUNE_EPOCHS = int(os.environ.get("TF_FINE_TUNE_EPOCHS", "4"))

# Single-phase PyTorch training (transformers)
TORCH_EPOCHS = int(os.environ.get("TORCH_EPOCHS", "8"))

# Mixed precision: turned on automatically when a GPU is detected, unless
# explicitly disabled with ``MIXED_PRECISION=0``. On the Ampere / Turing
# generation that the lab workstation runs this typically gives a 1.7-2x
# wall-time speedup with no accuracy loss.
MIXED_PRECISION = os.environ.get("MIXED_PRECISION", "auto").lower()

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN",
    "postgresql+psycopg://postgres:postgres@localhost:5432/lettuce",
)
