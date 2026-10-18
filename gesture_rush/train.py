"""Entrenamiento, ajuste y comparacion de clasificadores de gestos.

Entrena tres modelos con el mismo esquema de validacion agrupada por
participante y el mismo conjunto de prueba reservado:

- Random Forest (referencia)
- SVM con kernel RBF (candidato principal), escalado en un Pipeline
- MLP (red pequena)

Selecciona el mejor por macro-F1 en validacion, lo reajusta con train+val y lo
evalua *una sola vez* en el conjunto de prueba.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np

from . import config, dataset as dataset_mod, evaluate, features


def build_models() -> dict[str, dict]:
    """Define los modelos y sus rejillas de hiperparametros."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    rs = config.RANDOM_STATE
    return {
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=rs, n_jobs=-1),
            "param_grid": {
                "n_estimators": [100, 300],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2],
            },
        },
        "svm_rbf": {
            "estimator": Pipeline(
                [("scaler", StandardScaler()),
                 ("svc", SVC(kernel="rbf", probability=True, random_state=rs))]
            ),
            "param_grid": {
                "svc__C": [1.0, 10.0, 100.0],
                "svc__gamma": ["scale", 0.01, 0.1],
            },
        },
        "mlp": {
            "estimator": Pipeline(
                [("scaler", StandardScaler()),
                 ("mlp", MLPClassifier(max_iter=1500, random_state=rs))]
            ),
            "param_grid": {
                "mlp__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "mlp__alpha": [1e-4, 1e-3, 1e-2],
            },
        },
    }


def _make_cv(groups: np.ndarray, max_splits: int = 5):
    """Devuelve un separador de CV agrupado, o estratificado si no alcanza."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    n_groups = len(np.unique(groups))
    if n_groups >= 2:
        return GroupKFold(n_splits=min(max_splits, n_groups)), True
    return StratifiedKFold(n_splits=min(max_splits, 3), shuffle=True,
                           random_state=config.RANDOM_STATE), False


def _fit_best(name: str, spec: dict, X, y, groups) -> tuple[object, dict, float]:
    """Ajusta hiperparametros con GridSearchCV sobre train+val (agrupado)."""
    from sklearn.model_selection import GridSearchCV

    cv, use_groups = _make_cv(groups)
    search = GridSearchCV(
        spec["estimator"], spec["param_grid"], cv=cv,
        scoring=config.SELECTION_METRIC, n_jobs=-1, refit=True,
    )
    if use_groups:
        search.fit(X, y, groups=groups)
    else:
        search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def train_and_select(
    df=None,
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    evaluation_path: str | Path | None = None,
    confusion_plot_path: str | Path | None = None,
) -> dict:
    """Pipeline completo de entrenamiento y evaluacion.

    Devuelve un diccionario con el resultado de cada modelo, el nombre del
    mejor y las rutas de los artefactos guardados.
    """
    if df is None:
        df = dataset_mod.load_dataset()
    report = dataset_mod.validate_dataset(df)

    X, y, groups = dataset_mod.featurize(df)
    split = dataset_mod.group_split(X, y, groups)
    tr, va, te = split["train"], split["val"], split["test"]

    models = build_models()
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    for name, spec in models.items():
        t0 = time.perf_counter()
        best_est, best_params, cv_score = _fit_best(name, spec, X[tr], y[tr], groups[tr])
        val_metrics = evaluate.compute_metrics(y[va], best_est.predict(X[va]))
        fitted[name] = best_est
        results[name] = {
            "cv_score": cv_score,
            "best_params": best_params,
            "val": val_metrics,
            "fit_seconds": time.perf_counter() - t0,
        }

    best_name = max(results, key=lambda n: results[n]["val"]["f1_macro"])

    # Reajuste del mejor con train+val y evaluacion unica en test.
    best_est = fitted[best_name]
    X_tv = np.vstack([X[tr], X[va]])
    y_tv = np.concatenate([y[tr], y[va]])
    best_est.fit(X_tv, y_tv)

    test_pred = best_est.predict(X[te])
    test_metrics = evaluate.compute_metrics(y[te], test_pred)
    test_metrics["inference_ms_per_sample"] = evaluate.measure_inference_time(
        best_est, X[te]
    )
    cm, labels = evaluate.confusion_matrix(y[te], test_pred)
    test_metrics["confusion_matrix"] = cm.tolist()
    test_metrics["labels"] = labels
    test_metrics["top_confusions"] = evaluate.top_confusions(cm, labels)

    # Persistencia.
    model_path = Path(model_path or config.MODEL_PATH)
    metadata_path = Path(metadata_path or config.MODEL_METADATA_PATH)
    evaluar_path = Path(evaluation_path or config.REPORTS_DIR / "evaluation_report.md")
    cm_plot = Path(confusion_plot_path or config.REPORTS_DIR / "confusion_matrix.png")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_est, model_path)
    evaluate.save_confusion_matrix_plot(cm, labels, cm_plot)

    comparison = {
        name: {
            "cv_score": r["cv_score"],
            "best_params": r["best_params"],
            "val_accuracy": r["val"]["accuracy"],
            "val_f1_macro": r["val"]["f1_macro"],
            "fit_seconds": r["fit_seconds"],
        }
        for name, r in results.items()
    }
    metadata = {
        "best_model": best_name,
        "selection_metric": config.SELECTION_METRIC,
        "classes": [str(c) for c in best_est.classes_],
        "feature_spec": features.feature_spec(),
        "dataset": report,
        "split_groups": {
            "train": [str(g) for g in split["train_groups"]],
            "val": [str(g) for g in split["val_groups"]],
            "test": [str(g) for g in split["test_groups"]],
        },
        "cv_score": results[best_name]["cv_score"],
        "best_params": results[best_name]["best_params"],
        "test_metrics": {
            k: v for k, v in test_metrics.items()
            if k not in ("report", "per_class", "confusion_matrix", "labels")
        },
        "test_per_class": test_metrics["per_class"],
        "comparison": comparison,
        "versions": _versions(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    evaluate.save_metrics_report(
        {
            name: {**r["val"], "inference_ms_per_sample": 0.0}
            for name, r in results.items()
        },
        evaluar_path,
    )

    return {
        "best_model": best_name,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "evaluation_path": evaluar_path,
        "confusion_plot": cm_plot,
    }


def _versions() -> dict:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def load_model(path: str | Path | None = None):
    """Carga el modelo seleccionado y sus metadatos."""
    path = Path(path or config.MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el modelo en {path}")
    sibling = path.with_suffix(".json")
    meta_path = sibling if sibling.exists() else Path(config.MODEL_METADATA_PATH)
    model = joblib.load(path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return model, metadata
