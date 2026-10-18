"""Herramienta de recoleccion y etiquetado de landmarks con la webcam.

Guarda unicamente los landmarks derivados (63 valores) y metadatos; no guarda
video ni imagenes completas. Incluye deduplicacion de muestras casi identicas
y control de balance de clases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features
from .camera import Camera, CameraError
from .hand_detector import HandDetector, draw_hand


class DataCollector:
    """Recolector interactivo de muestras etiquetadas."""

    def __init__(
        self,
        participant: str = "participant_01",
        session: str = "s01",
        output_path: str | Path | None = None,
        dedup_distance: float = 0.02,
        dedup_memory: int = 5,
    ) -> None:
        self.participant = participant
        self.session = session
        self.output_path = Path(output_path or config.DATASET_CSV)
        self.dedup_distance = dedup_distance
        self.dedup_memory = dedup_memory

        self.rows: list[dict] = []
        self._recent: dict[str, list[np.ndarray]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if self.output_path.exists():
            existing = pd.read_csv(self.output_path)
            self.rows = existing.to_dict("records")

    def _is_duplicate(self, label: str, landmarks: np.ndarray) -> bool:
        recent = self._recent.setdefault(label, [])
        for prev in recent:
            if np.linalg.norm(prev - landmarks) < self.dedup_distance:
                return True
        recent.append(landmarks.copy())
        if len(recent) > self.dedup_memory:
            recent.pop(0)
        return False

    def add(self, label: str, landmarks: np.ndarray, handedness: str = "Unknown") -> bool:
        """Agrega una muestra. Devuelve False si se descarto por duplicada."""
        if self._is_duplicate(label, landmarks):
            return False
        row = {
            f"lm_{i}_{axis}": float(landmarks[i, a])
            for i in range(features.N_LANDMARKS)
            for a, axis in enumerate("xyz")
        }
        row.update(
            {
                "label": label,
                "participant": self.participant,
                "session": self.session,
                "handedness": handedness,
                "source": "real",
            }
        )
        self.rows.append(row)
        return True

    def counts(self) -> dict[str, int]:
        df = pd.DataFrame(self.rows)
        if df.empty:
            return {g: 0 for g in config.GESTURES}
        return {g: int((df["label"] == g).sum()) for g in config.GESTURES}

    def save(self) -> Path:
        df = pd.DataFrame(self.rows)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        return self.output_path


def run_interactive(
    participant: str = "participant_01",
    session: str = "s01",
    output_path: str | Path | None = None,
) -> int:
    """Bucle interactivo de recoleccion.

    Teclas:
        1..5   seleccionar gesto objetivo
        SPACE  iniciar/detener captura
        q      guardar y salir
    """
    import cv2

    gestures = list(config.GESTURES)
    if not gestures:
        print("[error] No hay gestos configurados.")
        return 1

    collector = DataCollector(participant=participant, session=session,
                              output_path=output_path)
    try:
        cam = Camera()
        cam.open()
    except CameraError as exc:
        print(f"[error] {exc}")
        return 1

    detector = HandDetector()
    current = 0
    capturing = False
    print("Recolector. 1..5 = gesto, SPACE = capturar, q = guardar y salir.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("[warn] Sin imagen de camara.")
                break

            hands = detector.detect(frame)
            label = gestures[current]
            for hand in hands:
                draw_hand(frame, hand)
                if capturing:
                    collector.add(label, hand.landmarks, hand.handedness)

            counts = collector.counts()
            _draw_hud(frame, cam.fps, label, capturing, len(hands), counts)

            cv2.imshow("Gesture Rush - Recolector (Fase 2)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                capturing = not capturing
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(gestures):
                    current = idx
                    capturing = True
    finally:
        detector.close()
        cam.release()
        cv2.destroyAllWindows()
        path = collector.save()
        print(f"Guardadas {len(collector.rows)} muestras en {path}")
        print("Balance:", collector.counts())
    return 0


def _draw_hud(frame, fps, label, capturing, n_hands, counts) -> None:
    import cv2

    status = "CAPTURANDO" if capturing else "EN PAUSA"
    color = (0, 0, 255) if capturing else (0, 255, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesto: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{status}  manos:{n_hands}  FPS:{fps:4.1f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    balance = "  ".join(f"{k[:4]}:{v}" for k, v in counts.items())
    cv2.putText(frame, balance, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
