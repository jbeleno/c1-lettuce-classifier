# C1 — Hydroponic Lettuce Growth-Stage Classifier

University course project (USCO BEINSOF52, Artificial Intelligence — 2026).
Image-classification pipeline that labels individual hydroponic lettuce pods
into one of five growth stages: `empty_pod`, `germination`, `young`, `pod`,
`Ready`. Five backbones are compared (MobileNetV3, EfficientNet-B0, ResNet50,
ViT-B/16, Swin-Tiny) plus a heterogeneous soft-voting ensemble.

## Quick start (laptop)

```bash
make install
cp .env.example .env       # tweak knobs if needed
make data                  # ingest + preprocess + 70/15/15 split + class report
make train-all             # all 5 models + ensemble (long; better on a GPU box)
make eval-all              # comparison tables
make cv MODEL=resnet50     # K-fold CV (Topic 3, robustness check)
make backend               # FastAPI on http://127.0.0.1:8000/docs
make test
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

## Layout

```
src/
  data/        ingest, preprocess, group-aware split, class report
  models/      one file per backbone + base libs + ensemble
  cv.py        K-fold cross-validation
  inference.py unified predict(image, model_name) -> Prediction
  evaluate_all.py
backend/       FastAPI + SQLAlchemy + Postgres (predict / history / metrics)
tests/         pytest (data leakage, splits schema, API contract, CV math)
```

## Notes on the rubric

Implemented:
- 70/15/15 split with group-aware stratification (no leakage).
- Reproducibility CSVs for every split.
- Class balancing (`class_weight="balanced"`).
- ≥ 3 models — actually 5 + ensemble.
- Mixed precision when running on CUDA.
- Two-phase fine-tune for the CNN backbones.
- K-fold cross-validation (5 folds).
- Per-class precision / recall / F1 + confusion matrix.
- Model serialization (`model.keras` for TF, `model.pt` for Torch).
- Strict separation between training and inference code.
- FastAPI backend with PostgreSQL persistence.

Intentionally skipped:
- IoT (Topic 6) — outside scope per project decision.
- Hyperband hyperparameter search (Topic 2).
- GAN-based augmentation (Topic 1, optional).
