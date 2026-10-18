"""Genera un dataset sintetico de landmarks para arrancar el pipeline.

Uso:
    python scripts/run_generate_synthetic.py --participants 6 --per-class 150
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_rush import config, synthetic_data  # noqa: E402
from gesture_rush.dataset import save_dataset, validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera dataset sintetico")
    parser.add_argument("--participants", type=int, default=6)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--per-class", type=int, default=150,
                        help="muestras por gesto y por participante")
    parser.add_argument("--output", default=str(config.DATASET_CSV))
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args()

    df = synthetic_data.generate(
        per_class_per_participant=args.per_class,
        participants=args.participants,
        sessions_per_participant=args.sessions,
        seed=args.seed,
    )
    report = validate_dataset(df)
    path = save_dataset(df, args.output)

    print(f"Dataset sintetico guardado en {path}")
    print(f"  muestras: {report['n_samples']}")
    print(f"  participantes: {report['n_participants']}")
    print(f"  por clase: {report['class_counts']}")
    print("Recuerda: los resultados con datos sinteticos no demuestran "
          "generalizacion a personas reales.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
