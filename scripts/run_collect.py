"""Recolector interactivo de datos reales con la webcam.

Uso:
    python scripts/run_collect.py --participant participant_01 --session s01
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_rush import config  # noqa: E402
from gesture_rush.collect_data import run_interactive  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Recolector de landmarks")
    parser.add_argument("--participant", default="participant_01")
    parser.add_argument("--session", default="s01")
    parser.add_argument("--output", default=str(config.DATASET_CSV))
    args = parser.parse_args()
    return run_interactive(
        participant=args.participant, session=args.session, output_path=args.output
    )


if __name__ == "__main__":
    raise SystemExit(main())
