"""Lista las camaras disponibles y sus indices.

Sirve para identificar la entrada de video de la camara fisica.

Uso:
    python scripts/list_cameras.py            # solo lista indices que abren
    python scripts/list_cameras.py --preview  # muestra un frame de cada una
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

_BACKENDS = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]


def _probe(index: int, backend: int) -> tuple[bool, int, int]:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return False, 0, 0
    ok, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok or frame is None:
        return False, 0, 0
    return True, width, height


def main() -> int:
    parser = argparse.ArgumentParser(description="Enumera camaras disponibles")
    parser.add_argument("--max-index", type=int, default=8)
    parser.add_argument("--preview", action="store_true",
                        help="Muestra un frame de cada camara encontrada (revisa 'q')")
    args = parser.parse_args()

    found: list[tuple[int, int, int, int]] = []
    for index in range(args.max_index):
        for name, backend in _BACKENDS:
            ok, width, height = _probe(index, backend)
            if ok:
                print(f"[OK] indice={index} backend={name} {width}x{height}")
                found.append((index, backend, width, height))
                break
        else:
            print(f"[--] indice={index} sin senal")

    if not found:
        print("\nNo se detecto ninguna camara. Revisa que este conectada, "
              "encendida y que ninguna otra aplicacion la este usando.")
        return 1

    print("\nElige el indice de tu camara fisica (usa --preview para ver cada una):")
    for index, _, width, height in found:
        print(f"  indice {index} ({width}x{height})  ->  set GESTURE_CAMERA_INDEX={index}")
    print("  (o edita CAMERA_INDEX en gesture_rush/config.py)")

    if args.preview:
        for index, backend, _, _ in found:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                continue
            cv2.putText(frame, f"indice {index}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow(f"camara {index}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
