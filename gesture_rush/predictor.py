"""Inferencia en tiempo real con el modelo seleccionado.

Incluye un suavizado temporal para evitar activaciones falsas: una prediccion
solo se acepta si se repite de forma estable durante una ventana de frames y
supera un umbral de confianza. La ausencia de mano es un estado explicito.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from . import config, features

NO_HAND = "__no_hand__"


class TemporalSmoother:
    """Voto mayoritario sobre una ventana deslizante de predicciones."""

    def __init__(
        self,
        window: int = config.SMOOTHING_WINDOW,
        min_votes: int = config.SMOOTHING_MIN_VOTES,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    ) -> None:
        self.window = window
        self.min_votes = min_votes
        self.confidence_threshold = confidence_threshold
        self._votes: deque[str] = deque(maxlen=window)

    def reset(self) -> None:
        self._votes.clear()

    def update(self, label: str | None, confidence: float = 0.0) -> str | None:
        """Agrega una observacion y devuelve la etiqueta estable o ``None``.

        ``label=None`` representa la ausencia de mano.
        """
        if label is None or confidence < self.confidence_threshold:
            self._votes.append(NO_HAND)
            return None
        self._votes.append(label)
        counts = {lab: 0 for lab in set(self._votes) if lab != NO_HAND}
        for lab in self._votes:
            if lab in counts:
                counts[lab] += 1
        if not counts:
            return None
        best = max(counts, key=counts.get)
        if counts[best] >= self.min_votes:
            return best
        return None

    @property
    def filled(self) -> int:
        """Cuantos frames entran en la ventana ahora mismo."""
        return len(self._votes)

    def snapshot(self) -> tuple[str | None, int]:
        """Etiqueta con mas votos y cuantos tiene, aunque aun no confirme.

        Sirve para mostrar la tendencia del voto temporal antes de alcanzar el
        minimo; no altera la etiqueta estable que devuelve ``update``.
        """
        counts: dict[str, int] = {}
        for label in self._votes:
            if label != NO_HAND:
                counts[label] = counts.get(label, 0) + 1
        if not counts:
            return None, 0
        winner = max(counts, key=counts.get)
        return winner, counts[winner]


class Predictor:
    """Envoltorio de inferencia: modelo + especificacion de features."""

    def __init__(self, model, metadata: dict | None = None) -> None:
        self.model = model
        self.metadata = metadata or {}
        self.feature_spec = self.metadata.get("feature_spec") or features.feature_spec()
        spec_version = str(
            self.feature_spec.get("version", config.FEATURE_SPEC_VERSION)
        )
        if spec_version != config.FEATURE_SPEC_VERSION:
            raise ValueError(
                "El modelo fue entrenado con features "
                f"v{spec_version}, pero el codigo usa "
                f"v{config.FEATURE_SPEC_VERSION}. Reentrena con "
                "scripts/run_train.py."
            )
        self.use_z = bool(self.feature_spec.get("use_z", config.USE_Z))
        self.classes = list(getattr(model, "classes_", self.metadata.get("classes", [])))
        expected_dim = int(self.feature_spec.get("dim", features.feature_dim(self.use_z)))
        actual_dim = getattr(model, "n_features_in_", expected_dim)
        if actual_dim != expected_dim:
            raise ValueError(
                "Incompatibilidad entre la especificacion de features y el modelo: "
                f"modelo espera {actual_dim}, la especificacion define {expected_dim}"
            )
        # Un modelo entrenado con otro vocabulario devolveria etiquetas que ya no
        # existen en config.GESTURES, y el juego las interpretaria como validas.
        expected_classes = set(config.GESTURES)
        if self.classes and set(self.classes) != expected_classes:
            missing = sorted(expected_classes - set(self.classes))
            extra = sorted(set(self.classes) - expected_classes)
            raise ValueError(
                "El modelo no coincide con config.GESTURES: "
                f"faltan {missing} y sobran {extra}. Reentrena con "
                "scripts/run_train.py."
            )

    @classmethod
    def load(cls, model_path: str | Path | None = None) -> "Predictor":
        import json

        import joblib

        model_path = Path(model_path or config.MODEL_PATH)
        meta_path = Path(config.MODEL_METADATA_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo en {model_path}. Entrena primero con "
                "scripts/run_train.py."
            )
        model = joblib.load(model_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(model, metadata)

    def predict_proba(self, landmarks: np.ndarray) -> np.ndarray:
        vec = features.normalize_landmarks(landmarks, use_z=self.use_z).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(vec)[0]
        # Fallback: decision_function -> softmax aproximado.
        scores = np.atleast_1d(self.model.decision_function(vec)[0]).astype(np.float64)
        scores = scores - scores.max()
        exp = np.exp(scores)
        return exp / exp.sum()

    def predict(self, landmarks: np.ndarray | None) -> tuple[str, float]:
        """Devuelve ``(etiqueta, confianza)``; ``(NO_HAND, 0.0)`` si no hay mano."""
        if landmarks is None:
            return NO_HAND, 0.0
        probs = self.predict_proba(landmarks)
        idx = int(np.argmax(probs))
        return str(self.classes[idx]), float(probs[idx])

    def predict_top_k(
        self, landmarks: np.ndarray, k: int = 3
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """``(etiqueta, confianza, top-k)``, de mayor a menor probabilidad.

        El top-k permite ver como se reparte la probabilidad entre clases
        parecidas; es la vista que usa el modo Prueba para explicar el modelo.
        """
        if landmarks is None:
            return NO_HAND, 0.0, []
        probs = self.predict_proba(landmarks)
        order = np.argsort(probs)[::-1][: max(1, int(k))]
        top = [(str(self.classes[i]), float(probs[i])) for i in order]
        return top[0][0], top[0][1], top
