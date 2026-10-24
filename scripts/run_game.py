"""Fase 6: interfaz grafica del minijuego "Duelo de gestos".

Uso:
    python scripts/run_game.py                 # camara real + modelo entrenado
    python scripts/run_game.py --demo          # sin camara, jugadores simulados
    python scripts/run_game.py --rounds 3      # partida corta
    python scripts/run_game.py --players 3     # tres franjas en el encuadre

Teclas:
    ESPACIO / ENTER  iniciar ronda
    R                reiniciar la partida
    ESC              volver al menu (desde el menu, salir)
    1 / 2 / 3        en --demo, forzar el acierto de ese jugador
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_rush import config  # noqa: E402
from gesture_rush.camera import CameraError  # noqa: E402
from gesture_rush.game import Duel  # noqa: E402
from gesture_rush.live import CameraPipeline, DemoPipeline  # noqa: E402
from gesture_rush.ui import GestureRushApp  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture Rush - Duelo de gestos")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="sin camara: dos jugadores simulados (para probar la interfaz)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help=f"indice de camara (por defecto {config.CAMERA_INDEX})",
    )
    parser.add_argument(
        "--players",
        type=int,
        choices=(2, 3),
        default=2,
        help="jugadores en el encuadre (por defecto 2)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="rondas por partida (0 = sin limite)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    duel = Duel(num_players=args.players, total_rounds=args.rounds)

    if args.demo:
        pipeline = DemoPipeline(duel)
    else:
        try:
            from gesture_rush.predictor import Predictor

            predictor = Predictor.load()
        except FileNotFoundError as exc:
            print(f"[error] {exc}")
            print("Proba con --demo para ver la interfaz sin modelo.")
            return 1
        pipeline = CameraPipeline(
            predictor, camera_index=args.camera, num_players=args.players
        )
        try:
            pipeline.start()
        except (CameraError, FileNotFoundError) as exc:
            print(f"[error] {exc}")
            print("Proba con --demo para ver la interfaz sin camara.")
            return 1

    app = GestureRushApp(duel, pipeline, demo=args.demo)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
