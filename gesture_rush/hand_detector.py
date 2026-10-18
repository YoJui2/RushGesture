"""Integracion con MediaPipe Hand Landmarker (Tasks API).

Se usa el modelo preentrenado de MediaPipe *solo* para detectar la mano y sus
21 landmarks. La clasificacion de gestos es trabajo propio (ver ``train.py``).

`mediapipe` se importa de forma diferida para que el resto del paquete
(features, dataset, entrenamiento, pruebas) funcione sin el instalado.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from . import config

# Conexiones entre landmarks para dibujar el esqueleto de la mano.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indice
    (5, 9), (9, 10), (10, 11), (11, 12),     # medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anular
    (13, 17), (17, 18), (18, 19), (19, 20),  # menique
    (0, 17),
)


@dataclass
class HandResult:
    """Resultado de deteccion para una mano."""

    landmarks: np.ndarray          # forma (21, 3), coords normalizadas [0, 1]
    handedness: str                # "Left" o "Right"
    score: float                   # confianza de la clasificacion de lateralidad
    index: int = 0                 # orden de deteccion en el frame
    extra: dict = field(default_factory=dict)

    @property
    def centroid(self) -> np.ndarray:
        """Centro (x, y) de la mano, util para asociarla a un jugador."""
        return self.landmarks[:, :2].mean(axis=0)


def ensure_model(path=None, url: str | None = None) -> str:
    """Garantiza que el modelo ``.task`` exista; si no, lo descarga una vez.

    Tras la primera descarga la aplicacion funciona sin Internet.
    """
    from pathlib import Path

    path = Path(path or config.HAND_LANDMARKER_TASK)
    url = url or config.HAND_LANDMARKER_URL
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[hand_detector] Descargando modelo MediaPipe -> {path}")
        urllib.request.urlretrieve(url, path)
    return str(path)


class HandDetector:
    """Detector de manos basado en MediaPipe Tasks."""

    def __init__(
        self,
        model_path=None,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_HAND_DETECTION_CONFIDENCE,
        min_presence_confidence: float = config.MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
        running_mode: str = "video",
        auto_download: bool = True,
    ) -> None:
        # Import diferido.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._mp_python = mp_python
        self._vision = vision

        if auto_download:
            model_path = ensure_model(model_path)
        else:
            from pathlib import Path

            model_path = str(Path(model_path or config.HAND_LANDMARKER_TASK))
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"No se encontro el modelo en {model_path}. "
                    "Ejecuta scripts/download_assets.py o desactiva auto_download=False."
                )

        self.running_mode_name = running_mode
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        common = dict(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        if running_mode == "video":
            options = vision.HandLandmarkerOptions(
                running_mode=vision.RunningMode.VIDEO, **common
            )
        else:
            options = vision.HandLandmarkerOptions(
                running_mode=vision.RunningMode.IMAGE, **common
            )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:
            pass

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- Deteccion -----------------------------------------------------------
    def _timestamp_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

    def detect(self, frame_bgr: np.ndarray) -> list[HandResult]:
        """Devuelve la lista de manos detectadas en el frame (puede estar vacia)."""
        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB sin copia extra innecesaria
        rgb = np.ascontiguousarray(rgb)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        if self.running_mode_name == "video":
            result = self._detector.detect_for_video(mp_image, self._timestamp_ms())
        else:
            result = self._detector.detect(mp_image)

        hands: list[HandResult] = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float64
            )
            label, score = "Unknown", 0.0
            if i < len(result.handedness) and result.handedness[i]:
                category = result.handedness[i][0]
                label = category.category_name or "Unknown"
                score = float(category.score)
            hands.append(
                HandResult(landmarks=pts, handedness=label, score=score, index=i)
            )
        return hands


def draw_hand(frame: np.ndarray, hand: HandResult) -> None:
    """Dibuja landmarks y conexiones de una mano sobre el frame (in-place)."""
    import cv2

    h, w = frame.shape[:2]
    points = [
        (int(np.clip(lm[0], 0, 1) * w), int(np.clip(lm[1], 0, 1) * h))
        for lm in hand.landmarks
    ]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 0), 2, cv2.LINE_AA)
    for idx, (x, y) in enumerate(points):
        color = (0, 0, 255) if idx in (0, 4, 8, 12, 16, 20) else (255, 120, 0)
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
