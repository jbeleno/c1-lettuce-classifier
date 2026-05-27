# C1 — Hydroponic Lettuce Growth-Stage Classifier

University course project (USCO BEINSOF52, Artificial Intelligence — 2026).
Image-classification pipeline that labels individual hydroponic lettuce pods
into one of five growth stages: `empty_pod`, `germination`, `young`, `pod`,
`Ready`. Five backbones are compared (MobileNetV3, EfficientNet-B0, ResNet50,
ViT-B/16, Swin-Tiny) plus a heterogeneous soft-voting ensemble.

**Authors:** Jesús Beleño · Juan Forero

---

## Deliverables (for the grading committee)

The full set of artifacts is committed under [docs/exports/](docs/exports/):

| File | Format | Pages / size | Purpose |
|---|---|---|---|
| [docs/exports/documentation.pdf](docs/exports/documentation.pdf) | PDF | 24 pages · 1.1 MB | **Main written report.** 15 sections in English covering the rubric (Introduction, Problem, Objectives, State of the Art, Requirements, Use Cases, ER + Data Dictionary, Class Diagrams, GUI Mockups, API Catalogue, Testing, Model Architecture, Results, Future Work, References). |
| [docs/exports/slides.pdf](docs/exports/slides.pdf) | PDF | 13 slides · 432 KB | **Presentation deck** for the oral defense (English, 16:9, slate-900 visual identity). |
| [docs/exports/slides.pptx](docs/exports/slides.pptx) | PPTX | 13 slides · 200 KB | **Same deck, editable** in Keynote / PowerPoint / LibreOffice Impress. |

To open all three at once on macOS:

```bash
open docs/exports/documentation.pdf docs/exports/slides.pdf docs/exports/slides.pptx
```

To rebuild any of them from sources:

```bash
make docs-pdf     # documentation.md  → docs/exports/documentation.pdf
make slides       # scripts/build_slides.py → docs/exports/slides.{pptx,pdf}
make diagrams     # PlantUML → docs/diagrams/png/*.png (used by both)
```

The supporting Markdown source for the document is [docs/documentation.md](docs/documentation.md);
the design-system specification is [docs/design-system.md](docs/design-system.md).

---

## Quick start (laptop)

```bash
make install
cp .env.example .env            # tweak knobs if needed
make data                       # ingest + preprocess + 70/15/15 split + balance + report
make train-all                  # all 5 models + ensemble (long; better on a GPU box)
make eval-all                   # comparison tables
make cv MODEL=resnet50          # K-fold CV (Topic 3, robustness check)
make backend                    # FastAPI on http://127.0.0.1:8000/docs
make frontend-install && make frontend  # React dashboard on http://localhost:5173
make test                       # 29/29 cases
```

## Quick start (Ubuntu workstation with a CUDA GPU)

The lab workstation has an RTX 2090. The training stack auto-detects CUDA
and turns on fp16 mixed precision — no flag needed.

```bash
# Prereqs (one-time): system NVIDIA driver + uv
curl -LsSf https://astral.sh/uv/install.sh | sh
nvidia-smi   # confirm the driver sees the card

# Project
git clone <repo-url> && cd c1-lettuce-classifier
make install                       # uv resolves CUDA wheels for torch + TF

# Put the Roboflow folder somewhere and point .env at it
cp .env.example .env
# edit RAW_DATASET_PATH= ...

make data
make train-all
make eval-all
```

Knobs you'll want to tune in `.env` on a workstation vs a laptop:

| Var | Laptop | Workstation |
|---|---|---|
| `BATCH_SIZE` | 16-32 | 64-128 |
| `NUM_WORKERS` | 2-4 | 8-16 |
| `TF_HEAD_EPOCHS` | 12 | 20 |
| `TF_FINE_TUNE_EPOCHS` | 4 | 8 |
| `TORCH_EPOCHS` | 8 | 12 |
| `MIXED_PRECISION` | auto | auto |

Set `MIXED_PRECISION=0` to force float32 everywhere — useful on CPU-only
boxes or when debugging numerical issues.

## Hold-out split vs K-fold CV

Two independent evaluations, on purpose:

- **Hold-out (70/15/15)** — feeds `make train-all`, `make eval-all`, and the
  FastAPI backend. Produces the *final* models that go to `models_saved/`
  and serve predictions through the API.
- **K-fold CV** — reads `data/processed/all.csv` (the union of all splits)
  and runs 5 stratified, group-aware folds. Reports mean ± std per metric
  to `reports/cv_<model>/`. Doesn't touch `models_saved/`.

Both use `SPLIT_SEED=42` and both group by `source_frame` to prevent crops
from the same original image leaking across partitions.

## Two-layer class balancing

```
data/splits/train.csv         →  balance.py  →  data/splits/train_balanced.csv
14,088 rows · 5 imbalanced               23,565 rows · 4,713 per class
                                              ▼
                                       trainer reads the
                                       balanced CSV by default
                                              +
                                       class_weight="balanced"
                                       still applied as a
                                       safety net in the loss
```

`balance.py` writes a physically balanced training set;
`class_weight` reweights the loss as a second layer.
Val and test are never balanced — they stay representative of production.

## Layout

```
src/
  data/        ingest, preprocess, group-aware split, balance, class report
  models/      one file per backbone + base libs + ensemble
  cv.py        K-fold cross-validation
  inference.py unified predict(image, model_name) -> Prediction
  evaluate_all.py
backend/       FastAPI + SQLAlchemy + Postgres (predict / history / metrics)
frontend/     React + Vite + TS + Tailwind (predict / history / metrics / models)
tests/        pytest (29 cases — data leakage, splits, balance, API, CV)
docs/         documentation source + PlantUML diagrams + mockups + exports
scripts/      build_slides.py (python-pptx)
```

## What's implemented

- 70/15/15 split with group-aware stratification (no leakage).
- Reproducibility CSVs for every split.
- **Two-layer class balancing** — sample-level (oversampling → `train_balanced.csv`)
  + loss-level (`class_weight="balanced"`).
- Augmentation pipeline (flip, rotation, brightness, contrast, zoom).
- ≥ 3 models — actually 5 + ensemble.
- Mixed precision when running on CUDA.
- Two-phase fine-tune for the CNN backbones; end-to-end for transformers.
- K-fold cross-validation (5 folds, group-aware).
- Per-class precision / recall / F1 + confusion matrix.
- Model serialization (`model.keras` for TF, `model.pt` for Torch).
- Strict separation between training and inference code.
- FastAPI backend with PostgreSQL persistence (4 endpoints + `/healthz`).
- React front-end with webcam capture and pre-prediction crop tool.
- Full test suite — 29 cases all green.
