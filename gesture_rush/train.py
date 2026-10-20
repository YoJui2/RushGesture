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


def _fit_best(name: str, spec: dict, X, y, groups) -> tuple[object, dict, float, bool]:
    """Ajusta hiperparametros con GridSearchCV sobre train+val (agrupado).

    Devuelve tambien un booleano que indica si la CV interna pudo agrupar por
    participante. Si no hay al menos dos grupos, cae a CV estratificada dentro
    del mismo participante y el score deja de ser una estimacion entre personas.
    """
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
    return search.best_estimator_, search.best_params_, float(search.best_score_), use_groups


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
        best_est, best_params, cv_score, cv_grouped = _fit_best(
            name, spec, X[tr], y[tr], groups[tr]
        )
        val_metrics = evaluate.compute_metrics(y[va], best_est.predict(X[va]))
        fitted[name] = best_est
        results[name] = {
            "cv_score": cv_score,
            "cv_grouped": cv_grouped,
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
            "cv_grouped": r["cv_grouped"],
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
            f"{name} - validacion": {
                **r["val"],
                "inference_ms_per_sample": 0.0,
                "subset": "validacion (seleccion de hiperparametros)",
            }
            for name, r in results.items()
        }
        | {
            f"{best_name} - TEST (reservado)": {
                **test_metrics,
                "subset": "test reservado, evaluado una sola vez",
            }
        },
        evaluar_path,
        notes=[
            f"- Mejor modelo: {best_name}",
            f"- Participantes en train: {', '.join(map(str, split['train_groups']))}",
            f"- Participantes en validacion: {', '.join(map(str, split['val_groups']))}",
            f"- Participantes en test: {', '.join(map(str, split['test_groups']))}",
        ],
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


def cross_validate_by_participant(
    df=None,
    validation_fraction: float = 0.25,
    report_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    confusion_plot_path: str | Path | None = None,
) -> dict:
    """Validacion cruzada dejando un participante fuera en cada fold.

    En cada fold:
    1. Un participante queda como test.
    2. De los participantes restantes se reserva al menos uno para validacion.
    3. Los hiperparametros se ajustan solo con los participantes de train,
       usando CV agrupada cuando hay al menos dos grupos.
    4. El mejor modelo se reajusta con train+validacion y se evalua en test.

    Cada participante actua como test exactamente una vez. El resultado
    agregado (media y desviacion estandar) es una estimacion mas estable que
    un unico split cuando hay pocos participantes.
    """
    from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

    if df is None:
        df = dataset_mod.load_dataset()
    report = dataset_mod.validate_dataset(df)

    X, y, groups = dataset_mod.featurize(df)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError(
            "La validacion cruzada por participante necesita al menos 3 "
            f"participantes para separar train/validacion/test; hay "
            f"{len(unique_groups)}."
        )

    labels = list(config.GESTURES)
    models = build_models()
    outer = LeaveOneGroupOut()
    folds: list[dict] = []
    total_cm = np.zeros((len(labels), len(labels)), dtype=int)

    for fold_idx, (trainval_idx, test_idx) in enumerate(
        outer.split(X, y, groups), start=1
    ):
        trainval_groups = np.unique(groups[trainval_idx])
        n_val = max(1, int(round(validation_fraction * len(trainval_groups))))
        n_val = min(n_val, len(trainval_groups) - 1)
        if n_val < 1:
            raise ValueError(
                "No hay suficientes participantes para reservar validacion en "
                f"el fold {fold_idx}."
            )

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=n_val / len(trainval_groups),
            random_state=config.RANDOM_STATE,
        )
        rel_train, rel_val = next(
            gss.split(X[trainval_idx], y[trainval_idx], groups[trainval_idx])
        )
        tr_idx = trainval_idx[rel_train]
        va_idx = trainval_idx[rel_val]

        model_results: dict[str, dict] = {}
        fitted: dict[str, object] = {}
        for name, spec in models.items():
            t0 = time.perf_counter()
            best_est, best_params, cv_score, cv_grouped = _fit_best(
                name, spec, X[tr_idx], y[tr_idx], groups[tr_idx]
            )
            val_metrics = evaluate.compute_metrics(
                y[va_idx], best_est.predict(X[va_idx]), labels=labels
            )
            fitted[name] = best_est
            model_results[name] = {
                "cv_score": cv_score,
                "cv_grouped": cv_grouped,
                "best_params": best_params,
                "val": val_metrics,
                "fit_seconds": time.perf_counter() - t0,
            }

        best_name = max(
            model_results, key=lambda n: model_results[n]["val"]["f1_macro"]
        )
        best_est = fitted[best_name]
        best_est.fit(
            np.vstack([X[tr_idx], X[va_idx]]),
            np.concatenate([y[tr_idx], y[va_idx]]),
        )
        test_pred = best_est.predict(X[test_idx])
        test_metrics = evaluate.compute_metrics(y[test_idx], test_pred, labels=labels)
        test_metrics["inference_ms_per_sample"] = evaluate.measure_inference_time(
            best_est, X[test_idx]
        )
        cm, _ = evaluate.confusion_matrix(y[test_idx], test_pred, labels=labels)
        total_cm += cm

        folds.append(
            {
                "fold": fold_idx,
                "test_groups": [str(g) for g in np.unique(groups[test_idx])],
                "train_groups": [str(g) for g in np.unique(groups[tr_idx])],
                "val_groups": [str(g) for g in np.unique(groups[va_idx])],
                "best_model": best_name,
                "best_params": model_results[best_name]["best_params"],
                "cv_score": model_results[best_name]["cv_score"],
                "cv_grouped": model_results[best_name]["cv_grouped"],
                "val_f1_macro": model_results[best_name]["val"]["f1_macro"],
                "val_accuracy": model_results[best_name]["val"]["accuracy"],
                "test": test_metrics,
                "comparison": {
                    name: {
                        "cv_score": r["cv_score"],
                        "cv_grouped": r["cv_grouped"],
                        "val_accuracy": r["val"]["accuracy"],
                        "val_f1_macro": r["val"]["f1_macro"],
                        "best_params": r["best_params"],
                    }
                    for name, r in model_results.items()
                },
            }
        )

    def _mean_std(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    accuracy_mean, accuracy_std = _mean_std(
        [f["test"]["accuracy"] for f in folds]
    )
    f1_mean, f1_std = _mean_std([f["test"]["f1_macro"] for f in folds])
    f1w_mean, f1w_std = _mean_std([f["test"]["f1_weighted"] for f in folds])

    per_class: dict[str, dict] = {}
    for label in labels:
        f1_vals = [f["test"]["per_class"][label]["f1"] for f in folds]
        precision_vals = [f["test"]["per_class"][label]["precision"] for f in folds]
        recall_vals = [f["test"]["per_class"][label]["recall"] for f in folds]
        f1_m, f1_s = _mean_std(f1_vals)
        precision_m, precision_s = _mean_std(precision_vals)
        recall_m, recall_s = _mean_std(recall_vals)
        per_class[label] = {
            "f1_mean": f1_m,
            "f1_std": f1_s,
            "precision_mean": precision_m,
            "precision_std": precision_s,
            "recall_mean": recall_m,
            "recall_std": recall_s,
            "support": int(sum(f["test"]["per_class"][label]["support"] for f in folds)),
        }

    aggregate = {
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "f1_macro_mean": f1_mean,
        "f1_macro_std": f1_std,
        "f1_weighted_mean": f1w_mean,
        "f1_weighted_std": f1w_std,
        "per_class": per_class,
        "confusion_matrix": total_cm.tolist(),
        "labels": labels,
        "top_confusions": evaluate.top_confusions(total_cm, labels),
    }

    report_path = Path(report_path or config.REPORTS_DIR / "cv_report.md")
    summary_path = Path(summary_path or config.REPORTS_DIR / "cv_summary.json")
    cm_plot = Path(confusion_plot_path or config.REPORTS_DIR / "cv_confusion_matrix.png")

    evaluate.save_confusion_matrix_plot(total_cm, labels, cm_plot)

    lines = [
        "# Validacion cruzada agrupada por participante - Gesture Rush",
        "",
        "- Metodo: Leave-One-Participant-Out (cada participante es test una vez)",
        f"- Participantes: {', '.join(map(str, unique_groups))}",
        f"- Muestras: {report['n_samples']}",
        f"- Validacion interna: al menos {max(1, int(round(validation_fraction * (len(unique_groups) - 1))))} "
        "participante(s) de los restantes",
        "- Los hiperparametros se eligen solo con participantes de train.",
        "",
        "## Resultados por fold",
        "",
        "| Fold | Test | Train | Validacion | Mejor modelo | F1 validacion | Accuracy test | F1 macro test |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in folds:
        lines.append(
            f"| {f['fold']} | {', '.join(f['test_groups'])} | "
            f"{', '.join(f['train_groups'])} | {', '.join(f['val_groups'])} | "
            f"{f['best_model']} | {f['val_f1_macro']:.3f} | "
            f"{f['test']['accuracy']:.3f} | {f['test']['f1_macro']:.3f} |"
        )
    lines += [
        "",
        "## Agregado en test",
        "",
        f"- accuracy: {accuracy_mean:.4f} +/- {accuracy_std:.4f}",
        f"- f1_macro: {f1_mean:.4f} +/- {f1_std:.4f}",
        f"- f1_weighted: {f1w_mean:.4f} +/- {f1w_std:.4f}",
        "",
        "## F1 por clase (media +/- desviacion en test)",
        "",
        "| Clase | F1 | Precision | Recall | Soporte |",
        "|---|---|---|---|---|",
    ]
    for label in labels:
        info = per_class[label]
        lines.append(
            f"| {label} | {info['f1_mean']:.3f} +/- {info['f1_std']:.3f} | "
            f"{info['precision_mean']:.3f} | {info['recall_mean']:.3f} | "
            f"{info['support']} |"
        )
    lines += [
        "",
        "## Confusiones frecuentes (matriz acumulada)",
        "",
    ]
    for c in aggregate["top_confusions"]:
        lines.append(f"- {c['true']} -> {c['pred']}: {c['count']}")
    lines += [
        "",
        f"Matriz de confusion acumulada: `{cm_plot.name}`",
        "",
        "> Para el modelo final de despliegue, reentrena con todos los "
        "participantes usando `scripts/run_train.py`.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "method": "leave_one_group_out",
        "n_participants": int(len(unique_groups)),
        "n_samples": int(report["n_samples"]),
        "validation_fraction": float(validation_fraction),
        "folds": folds,
        "aggregate": aggregate,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )

    return {
        "folds": folds,
        "aggregate": aggregate,
        "report_path": report_path,
        "summary_path": summary_path,
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
