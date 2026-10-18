"""Descarga los assets necesarios (modelo MediaPipe Hand Landmarker).

Uso:
    python scripts/download_assets.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_rush.hand_detector import ensure_model  # noqa: E402


def main() -> int:
    path = ensure_model()
    print(f"Modelo listo: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
