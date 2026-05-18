"""Light tests for the CV module. Heavy training runs are not exercised here;
those belong to a smoke `make cv-smoke` invocation."""
import numpy as np
import pandas as pd
import pytest

from src.cv import CVConfig, MODELS, _fold_metrics, _summarise, _write_markdown


def test_registry_matches_known_models():
    assert set(MODELS) == {
        "mobilenet_v3_small",
        "efficientnet_b0",
        "resnet50",
        "vit_b_16",
        "swin_t",
    }
    for name, spec in MODELS.items():
        assert spec["framework"] in {"tf", "torch"}
        assert "module" in spec


def test_smoke_config_shrinks_folds_and_epochs():
    cfg = CVConfig(model_name="mobilenet_v3_small", smoke=True)
    assert cfg.n_folds == 2
    assert cfg.epochs == 1


def test_fold_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    m = _fold_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert len(m["confusion_matrix"]) == 5


def test_fold_metrics_mixed():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 0])  # last one wrong
    m = _fold_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(4 / 5)


def test_summarise_writes_markdown(tmp_path):
    per_fold = [
        {
            "fold": 0,
            "n_train": 100,
            "n_val": 25,
            "accuracy": 0.8,
            "macro_f1": 0.75,
            "macro_precision": 0.78,
            "macro_recall": 0.74,
        },
        {
            "fold": 1,
            "n_train": 100,
            "n_val": 25,
            "accuracy": 0.84,
            "macro_f1": 0.79,
            "macro_precision": 0.81,
            "macro_recall": 0.77,
        },
    ]
    cfg = CVConfig(model_name="mobilenet_v3_small", n_folds=2, epochs=1)
    summary = _summarise(per_fold, cfg)
    assert summary["mean"]["accuracy"] == pytest.approx(0.82)
    assert summary["std"]["accuracy"] > 0
    out = tmp_path / "summary.md"
    _write_markdown(summary, out)
    body = out.read_text()
    assert "Cross-validation" in body
    assert "mobilenet_v3_small" in body
