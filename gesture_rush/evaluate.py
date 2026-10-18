"""Metricas, graficos y analisis de errores.

Concentra la evaluacion para no duplicarla entre entrenamiento y scripts.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import config


def compute_metrics(y_true, y_pred, labels=None) -> dict:
    """Accuracy, macro-F1 y metricas por clase."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_recall_fscore_support,
    )

    labels = list(labels or sorted(set(y_true) | set(y_pred)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, digits=3
        ),
    }


def confusion_matrix(y_true, y_pred, labels=None):
    """Matriz de confusion con orden de etiquetas estable."""
    from sklearn.metrics import confusion_matrix as sk_cm

    labels = list(labels or sorted(set(y_true) | set(y_pred)))
    return sk_cm(y_true, y_pred, labels=labels), labels


def measure_inference_time(estimator, X: np.ndarray, repeats: int = 3) -> float:
    """Tiempo medio de inferencia por muestra, en milisegundos."""
    if len(X) == 0:
        return 0.0
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        estimator.predict(X)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return (best / len(X)) * 1000.0


def top_confusions(cm: np.ndarray, labels: list[str], top: int = 5) -> list[dict]:
    """Lista de confusiones (fuera de la diagonal) mas frecuentes."""
    items: list[dict] = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                items.append(
                    {"true": labels[i], "pred": labels[j], "count": int(cm[i, j])}
                )
    items.sort(key=lambda d: d["count"], reverse=True)
    return items[:top]


def save_confusion_matrix_plot(cm, labels, path: str | Path) -> Path:
    """Guarda la matriz de confusion como imagen (sin depender de display)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 1.2 * len(labels) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_metrics_report(metrics_by_model: dict, path: str | Path) -> Path:
    """Guarda un informe de texto con metricas por modelo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Informe de evaluacion - Gesture Rush", ""]
    for name, info in metrics_by_model.items():
        lines.append(f"## Modelo: {name}")
        lines.append(f"- accuracy: {info['accuracy']:.4f}")
        lines.append(f"- f1_macro: {info['f1_macro']:.4f}")
        lines.append(f"- f1_weighted: {info['f1_weighted']:.4f}")
        if "inference_ms_per_sample" in info:
            lines.append(f"- inferencia: {info['inference_ms_per_sample']:.4f} ms/muestra")
        lines.append("")
        lines.append("```")
        lines.append(info["report"].rstrip())
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
