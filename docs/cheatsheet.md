# Defense cheat sheet — where everything lives

Quick reference for the oral defense. One block per topic, with the file
that owns it, the key function, and a one-line "what to say".

---

## 1. Dataset

| Pieza | Archivo |
|---|---|
| Carpeta cruda | `data/raw/lettuce_pallets/` (symlink al export Roboflow) |
| Ingest | [src/data/ingest.py](../src/data/ingest.py) — symlink + valida estructura |
| Preprocess | [src/data/preprocess.py](../src/data/preprocess.py) — `_load_split()` parsea `_classes.csv` y valida one-hot |
| Output | `data/processed/all.csv` — 19,721 filas (`filepath, class, source_frame`) |

**Decir**: "*Dataset público de Roboflow Universe — `lettuce-pallets v2i.multiclass` — 19,721 crops de pods individuales en 5 etapas de crecimiento, pre-anotado con Isolate-Objects.*"

## 2. Balance

| Capa | Archivo | Función |
|---|---|---|
| **Sample-level** | [src/data/balance.py](../src/data/balance.py) | `main()` — oversampling random seeded → `train_balanced.csv` con **4,713 por clase** |
| **Loss-level (TF)** | [src/models/_tf_lib.py](../src/models/_tf_lib.py) | `class_weights()` |
| **Loss-level (Torch)** | [src/models/_torch_lib.py](../src/models/_torch_lib.py) | `class_weight_tensor(device)` |
| **Tests** | [tests/test_balance.py](../tests/test_balance.py) | 5 casos — conteo igual + reproducible |

**Decir**: "*Doble capa de balanceo. Capa 1: `balance.py` aplica oversampling random con seed=42 hasta que cada clase tiene 4,713 filas. Capa 2: `class_weight="balanced"` se sigue pasando a la loss como red de seguridad. Val y test no se balancean — la distribución natural se preserva para que las métricas reportadas sean creíbles.*"

## 3. Augmentation

| Framework | Archivo | Función | Transformaciones |
|---|---|---|---|
| TF | [src/models/_tf_lib.py](../src/models/_tf_lib.py) | `_make_dataset()` | `RandomFlip`, `RandomRotation(0.15)`, `RandomBrightness(0.15)`, `RandomContrast(0.15)`, `RandomZoom(0.10)` |
| Torch | [src/models/_torch_lib.py](../src/models/_torch_lib.py) | `_train_transforms()` | `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(15)`, `ColorJitter` + ImageNet `Normalize` |

**Decir**: "*Augmentation aleatoria solo en train. Cada época, cada fila pasa por una transformación distinta, lo que rompe la duplicación literal del oversampling: 4 copias de la misma imagen ven 4 variantes visuales distintas.*"

## 4. Data split

| Pieza | Archivo |
|---|---|
| El split | [src/data/split.py](../src/data/split.py) — `StratifiedGroupKFold` peel |
| Anti-leakage helper | [src/utils.py](../src/utils.py) — `source_frame_id()` |
| Seed | [src/config.py](../src/config.py) — `SPLIT_SEED=42` |
| Outputs | `data/splits/{train,val,test}.csv` (14,088 / 2,816 / 2,817) |
| Tests | [tests/test_split.py](../tests/test_split.py) — proporciones + no-leakage + clases presentes |

**Decir**: "*Split 70/15/15 estratificado por clase y agrupado por `source_frame` — el prefijo antes de `_jpg.rf.<hash>` que identifica el frame original Roboflow. Eso evita que recortes del mismo frame caigan en train y test. Seed=42 fija → reproducible.*"

## 5. Fine-tuning / hyperparameters

### Fine-tuning de modelos pre-entrenados

| Familia | Archivo | Función | Estrategia |
|---|---|---|---|
| TF CNNs | [src/models/_tf_lib.py](../src/models/_tf_lib.py) | `fit_transfer()` | **2 fases**: head-only (lr=1e-3, 12 ep) + top-30 % unfrozen (lr=1e-5, 4 ep), BN siempre frozen |
| Torch transformers | [src/models/_torch_lib.py](../src/models/_torch_lib.py) | `train_torch_model()` | **End-to-end** `AdamW(lr=3e-5, weight_decay=1e-4)`, 8 ep, AMP en CUDA |

### Hyperparámetros configurables (sustituye al Hyperband del Topic 2)

[src/config.py](../src/config.py) + `.env`:

```bash
BATCH_SIZE=64                # laptop 16-32, workstation 64-128
NUM_WORKERS=8
TF_HEAD_EPOCHS=20
TF_FINE_TUNE_EPOCHS=8
TORCH_EPOCHS=12
MIXED_PRECISION=auto
SPLIT_SEED=42
```

**Decir**: "*Transfer learning desde ImageNet1K. Las CNN usan fine-tuning de 2 fases para evitar corromper los estadísticos de BatchNorm; los transformers entrenan end-to-end porque usan LayerNorm. Los hiperparámetros son configurables por `.env` para poder cambiar entre laptop y la workstation con CUDA.*"

## 6. Construcción de modelos

| Modelo | Archivo | Backbone |
|---|---|---|
| MobileNetV3-Small | [src/models/mobilenet.py](../src/models/mobilenet.py) | `tf.keras.applications.MobileNetV3Small` |
| EfficientNet-B0 | [src/models/efficientnet.py](../src/models/efficientnet.py) | `tf.keras.applications.EfficientNetB0` |
| ResNet50 | [src/models/resnet.py](../src/models/resnet.py) | `tf.keras.applications.ResNet50` |
| ViT-B/16 | [src/models/vit.py](../src/models/vit.py) | `torchvision.models.vit_b_16` |
| Swin-Tiny | [src/models/swin.py](../src/models/swin.py) | `torchvision.models.swin_t` |
| Ensemble | [src/models/ensemble.py](../src/models/ensemble.py) | soft-voting sobre los 5 `test_probs.parquet` |

**Topología común**: `Input(224×224×3) → preprocess → backbone → pool → Dropout(0.3, solo TF) → Dense(5, softmax, dtype=float32)`.

**Decir**: "*5 backbones pre-entrenados en ImageNet1K, todos con la misma cabeza de clasificación de 5 clases. La capa final tiene `dtype=float32` para estabilidad numérica cuando la global policy es `mixed_float16`.*"

## 7. Métricas

| Pieza | Archivo / Ruta |
|---|---|
| Cálculo TF | [src/models/_tf_lib.py](../src/models/_tf_lib.py) — `evaluate_and_save()` |
| Cálculo Torch | [src/models/_torch_lib.py](../src/models/_torch_lib.py) — `evaluate_and_save_torch()` |
| Persistencia | `models_saved/<name>/metadata.json` |
| Matriz visual | `models_saved/<name>/confusion_matrix.png` |
| Tabla agregada | [src/evaluate_all.py](../src/evaluate_all.py) → `reports/model_comparison.{csv,md}` |
| Endpoint vivo | `GET /metrics` en [backend/routers/metrics.py](../backend/routers/metrics.py) |

**Resultados finales**:

```
                test_acc   macro_f1   macro_recall
swin_t          0.9386     0.9319     0.9339        ← mejor accuracy
ensemble_avg    0.9365     0.9322     0.9406        ← mejor macro_recall  ⭐
vit_b_16        0.9365     0.9281     0.9257
resnet50        0.9098     0.9056     0.9172
efficientnet    0.8939     0.8932     0.9178
mobilenet       0.8832     0.8855     0.9019
```

**Decir**: "*Swin-T gana accuracy puro. El ensemble gana macro-recall (0.9406), la métrica alineada con el costo operacional — cada falso negativo en `Ready` cuesta cosecha tardía, cada falso negativo en `empty_pod` cuesta re-siembra fuera de tiempo.*"

## 8. Implementación web

### Backend (FastAPI + Postgres)

| Pieza | Archivo |
|---|---|
| App | [backend/main.py](../backend/main.py) |
| Predict | [backend/routers/predict.py](../backend/routers/predict.py) |
| History | [backend/routers/history.py](../backend/routers/history.py) |
| Metrics | [backend/routers/metrics.py](../backend/routers/metrics.py) |
| Models | [backend/routers/models.py](../backend/routers/models.py) |
| DB | [backend/db.py](../backend/db.py) + [backend/models.py](../backend/models.py) |
| Inference lazy load | [src/inference.py](../src/inference.py) |

### Frontend (React + Vite + Tailwind)

| Pieza | Archivo |
|---|---|
| App + Router | [frontend/src/App.tsx](../frontend/src/App.tsx) |
| Predict + webcam + crop | [frontend/src/routes/Predict.tsx](../frontend/src/routes/Predict.tsx) |
| History | [frontend/src/routes/History.tsx](../frontend/src/routes/History.tsx) |
| Metrics | [frontend/src/routes/Metrics.tsx](../frontend/src/routes/Metrics.tsx) |
| Models | [frontend/src/routes/Models.tsx](../frontend/src/routes/Models.tsx) |
| API client | [frontend/src/api.ts](../frontend/src/api.ts) |
| Design system tokens | [frontend/src/index.css](../frontend/src/index.css) (`@theme`) |

### Levantar todo

```bash
docker start lettuce-pg          # Postgres con historial intacto
make backend                      # http://127.0.0.1:8000  (terminal 1)
make frontend                     # http://localhost:5173    (terminal 2)
```

**Decir**: "*Backend FastAPI con 4 routers (predict, history, metrics, models) + Postgres para persistencia. Frontend React con 4 vistas — Predict con webcam vía getUserMedia nativo y un crop tool entre captura y predicción para garantizar que el modelo reciba un pod por imagen.*"

---

## Compacta para imprimir

```
1. DATASET        → src/data/ingest.py + preprocess.py · 19,721 crops
2. BALANCE        → src/data/balance.py + _tf_lib.class_weights()
3. AUGMENTATION   → _tf_lib._make_dataset · _torch_lib._train_transforms
4. SPLIT          → src/data/split.py · StratifiedGroupKFold · seed=42
5. FINE-TUNING    → _tf_lib.fit_transfer (2-fase) · _torch_lib.train_torch_model
6. MODELOS        → 5 build_model() en src/models/ + ensemble.py
7. MÉTRICAS       → evaluate_and_save → metadata.json + /metrics endpoint
8. WEB            → backend/ FastAPI + frontend/ React
```
