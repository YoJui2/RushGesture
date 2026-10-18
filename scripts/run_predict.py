"""Fase 4: prediccion de gestos en tiempo real con el modelo entrenado.

Uso:
    python scripts/run_predict.py

Teclas:
    q / ESC  salir
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from gesture_rush import config  # noqa: E402
from gesture_rush.camera import Camera, CameraError  # noqa: E402
from gesture_rush.hand_detector import HandDetector, draw_hand  # noqa: E402
from gesture_rush.predictor import NO_HAND, Predictor, TemporalSmoother  # noqa: E402


def main() -> int:
    try:
        predictor = Predictor.load()
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        return 1

    try:
        cam = Camera()
        cam.open()
    except CameraError as exc:
        print(f"[error] {exc}")
        return 1

    detector = HandDetector()
    smoother = TemporalSmoother()
    print(f"Modelo: {predictor.metadata.get('best_model', '?')}. 'q' para salir.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("[warn] Sin imagen de camara.")
                break

            hands = detector.detect(frame)
            primary = hands[0] if hands else None
            label, conf = predictor.predict(primary.landmarks if primary else None)
            stable = smoother.update(
                None if label == NO_HAND else label, conf
            )

            if primary is not None:
                draw_hand(frame, primary)
                text = f"{label} {conf*100:4.1f}%"
                if stable:
                    text += f"  [ESTABLE: {stable}]"
                    color = (0, 255, 0)
                else:
                    color = (0, 255, 255)
                cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, color, 2, cv2.LINE_AA)
            else:
                smoother.update(None, 0.0)
                cv2.putText(frame, "Sin mano detectada", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, f"FPS: {cam.fps:5.1f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Gesture Rush - Prediccion (Fase 4)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        detector.close()
        cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
