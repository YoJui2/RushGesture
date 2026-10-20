"""Pruebas de entrenamiento, persistencia y carga de modelos."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_rush import config, features, synthetic_data, train
from gesture_rush import dataset as dataset_mod


@pytest.fixture(scope="module")
def small_df():
    return synthetic_data.generate(
        per_class_per_participant=25,
        participants=5,
        sessions_per_participant=1,
        seed=3,
    )


@pytest.fixture(autouse=True)
def _fast_models(monkeypatch):
    """Reduce la rejilla para que las pruebas sean rapidas."""
    from sklearn.ensemble import RandomForestClassifier

    def _build():
        return {
            "random_forest": {
                "estimator": RandomForestClassifier(
                    random_state=config.RANDOM_STATE, n_jobs=-1
                ),
                "param_grid": {"n_estimators": [50], "max_depth": [10]},
            }
        }

    monkeypatch.setattr(train, "build_models", _build)


def test_train_and_persist(small_df, tmp_path):
    result = train.train_and_select(
        small_df,
        model_path=tmp_path / "model.joblib",
        metadata_path=tmp_path / "model.json",
        evaluation_path=tmp_path / "report.md",
        confusion_plot_path=tmp_path / "cm.png",
    )
    assert result["model_path"].exists()
    assert result["metadata_path"].exists()
    assert result["evaluation_path"].exists()
    assert result["confusion_plot"].exists()
    assert result["test_metrics"]["f1_macro"] >= 0.0


def test_load_model_and_predict(small_df, tmp_path):
    train.train_and_select(
        small_df,
        model_path=tmp_path / "model.joblib",
        metadata_path=tmp_path / "model.json",
        evaluation_path=tmp_path / "report.md",
        confusion_plot_path=tmp_path / "cm.png",
    )
    model, metadata = train.load_model(tmp_path / "model.joblib")
    # metadata se busca en la ruta por defecto; verificamos el modelo directamente.
    assert hasattr(model, "predict")
    label = model.predict(np.zeros((1, features.feature_dim())))[0]
    assert label in config.GESTURES


def test_cross_validate_by_participant(small_df, tmp_path):
    result = train.cross_validate_by_participant(
        small_df,
        report_path=tmp_path / "cv_report.md",
        summary_path=tmp_path / "cv_summary.json",
        confusion_plot_path=tmp_path / "cv_cm.png",
    )
    assert len(result["folds"]) == small_df["participant"].nunique()
    assert result["report_path"].exists()
    assert result["summary_path"].exists()
    assert result["confusion_plot"].exists()
    assert 0.0 <= result["aggregate"]["f1_macro_mean"] <= 1.0
