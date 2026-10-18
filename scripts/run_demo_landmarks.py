"""Fase 1: demo de camara + deteccion de manos + landmarks.

Uso:
    python scripts/run_demo_landmarks.py

Teclas:
    q / ESC  salir
    d        alternar dibujo de landmarks
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from gesture_rush.camera import Camera, CameraError  # noqa: E402
from gesture_rush.hand_detector import HandDetector, draw_hand  # noqa: E402


def main() -> int:
    try:
        cam = Camera()
        cam.open()
    except CameraError as exc:
        print(f"[error] {exc}")
        return 1

    draw = True
    detector = HandDetector()
    print("Demo de landmarks. Presiona 'q' para salir, 'd' para alternar dibujo.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("[warn] No se pudo leer de la camara.")
                break

            hands = detector.detect(frame)
            if draw:
                for hand in hands:
                    draw_hand(frame, hand)

            cv2.putText(
                frame,
                f"FPS: {cam.fps:5.1f}   manos: {len(hands)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Gesture Rush - Landmarks (Fase 1)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                draw = not draw
    finally:
        detector.close()
        cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
