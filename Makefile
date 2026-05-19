.PHONY: install ingest preprocess split report data \
        train-mobilenet train-efficientnet train-resnet train-vit train-swin \
        train-cnns train-transformers train-all \
        eval-all ensemble \
        cv cv-mobilenet cv-efficientnet cv-resnet cv-vit cv-swin cv-smoke \
        diagrams mockups docs-pdf docs-docx docs slides \
        frontend frontend-install frontend-build \
        test backend smoke clean

UV ?= uv

install:
	$(UV) sync --extra dev

# ── Data pipeline ────────────────────────────────────────────────
ingest:
	$(UV) run python -m src.data.ingest

preprocess:
	$(UV) run python -m src.data.preprocess

split:
	$(UV) run python -m src.data.split

report:
	$(UV) run python -m src.data.report

data: ingest preprocess split report

# ── Training ─────────────────────────────────────────────────────
train-mobilenet:
	$(UV) run python -m src.models.mobilenet

train-efficientnet:
	$(UV) run python -m src.models.efficientnet

train-resnet:
	$(UV) run python -m src.models.resnet

train-vit:
	$(UV) run python -m src.models.vit

train-swin:
	$(UV) run python -m src.models.swin

train-cnns: train-mobilenet train-efficientnet train-resnet
train-transformers: train-vit train-swin
train-all: train-cnns train-transformers ensemble

ensemble:
	$(UV) run python -m src.models.ensemble

eval-all:
	$(UV) run python -m src.evaluate_all

# ── Cross-validation (Topic 3: K-Fold ***) ───────────────────────────
# Default model is mobilenet_v3_small — override with MODEL=<name>.
CV_MODEL ?= mobilenet_v3_small
CV_FOLDS ?= 5
CV_EPOCHS ?= 6

cv:
	$(UV) run python -m src.cv --model $(CV_MODEL) --n-folds $(CV_FOLDS) --epochs $(CV_EPOCHS)

cv-mobilenet:
	$(UV) run python -m src.cv --model mobilenet_v3_small --n-folds $(CV_FOLDS) --epochs $(CV_EPOCHS)

cv-efficientnet:
	$(UV) run python -m src.cv --model efficientnet_b0 --n-folds $(CV_FOLDS) --epochs $(CV_EPOCHS)

cv-resnet:
	$(UV) run python -m src.cv --model resnet50 --n-folds $(CV_FOLDS) --epochs $(CV_EPOCHS)

cv-vit:
	$(UV) run python -m src.cv --model vit_b_16 --n-folds $(CV_FOLDS) --epochs $(CV_EPOCHS)

cv-swin:
	$(UV) run python -m src.cv --model swin_t --n-folds $(CV_FOLDS) --epochs $(CV_EPOCHS)

cv-smoke:
	$(UV) run python -m src.cv --model mobilenet_v3_small --smoke

# ── Backend ──────────────────────────────────────────────────────
backend:
	$(UV) run uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# ── Frontend (React + Vite + Tailwind) ───────────────────────────
frontend-install:
	cd frontend && npm install

frontend:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

# ── Documentation ────────────────────────────────────────────────
diagrams:
	cd docs/diagrams && plantuml -tpng -o ../png puml/01_use_cases.puml \
	  puml/02_er_diagram.puml puml/03_class_offline.puml \
	  puml/04_class_online.puml puml/05_architecture_pipeline.puml \
	  puml/06_architecture_models.puml puml/07_architecture_ensemble.puml

mockups:
	cd docs/diagrams && plantuml -tpng -o ../mockups \
	  puml/mockup_01_predict.puml puml/mockup_02_history.puml \
	  puml/mockup_03_metrics.puml puml/mockup_04_models.puml

docs-pdf: diagrams mockups
	mkdir -p docs/exports
	pandoc docs/documentation.md \
	  -o docs/exports/documentation.pdf \
	  --toc \
	  --pdf-engine=typst \
	  --resource-path=docs:docs/diagrams/png:docs/mockups:reports

docs-docx: diagrams mockups
	mkdir -p docs/exports
	pandoc docs/documentation.md \
	  -o docs/exports/documentation.docx \
	  --toc \
	  --resource-path=docs:docs/diagrams/png:docs/mockups:reports

docs: docs-pdf

slides:
	mkdir -p docs/exports
	$(UV) run python scripts/build_slides.py

# ── Tests ────────────────────────────────────────────────────────
test:
	$(UV) run pytest tests/ -q

smoke:
	$(UV) run python scripts/smoke_test.py

clean:
	rm -rf data/raw data/processed data/splits/*.csv models_saved .venv reports
