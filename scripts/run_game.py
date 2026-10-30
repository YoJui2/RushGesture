"""Fase 6: interfaz grafica de los minijuegos de Gesture Rush.

Uso:
    python scripts/run_game.py                 # camara real + modelo entrenado
    python scripts/run_game.py --demo          # sin camara, jugadores simulados
    python scripts/run_game.py --rounds 3      # partida corta
    python scripts/run_game.py --players 3     # tres franjas en el encuadre
    python scripts/run_game.py --game practice # diagnostico del modelo

Teclas:
    ESPACIO / ENTER  iniciar ronda o carrera (o pulgar arriba de todos)
    R                reiniciar la partida (en Prueba, el voto temporal)
    ESC              volver al menu (desde el menu, salir)
    1 / 2 / 3        en el menu, elegir minijuego
    1 / 2            en --demo: pulgar arriba fuera de ronda, acierto en ronda
    V                abrir/cerrar el control de volumen
    M                silenciar / recuperar la musica
    + / -            subir / bajar el volumen
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_rush import config  # noqa: E402
from gesture_rush.camera import CameraError  # noqa: E402
from gesture_rush.game import ChainRace, Duel, PracticeSession  # noqa: E402
from gesture_rush.live import CameraPipeline, DemoPipeline  # noqa: E402
from gesture_rush.ui import GestureRushApp  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture Rush - minijuegos de gestos")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="sin camara: jugadores simulados (para probar la interfaz)",
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
        choices=(1, 2),
        default=2,
        help="jugadores para Duelo y Cadena: 1 o 2 (por defecto 2)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="rondas por partida del duelo: 3, 5 u 8 (por defecto 5; 0 = sin limite)",
    )
    parser.add_argument(
        "--chain",
        type=int,
        default=5,
        help="gestos por cadena en el modo cadena (por defecto 5)",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="arrancar en ventana (por defecto pantalla completa; F11 alterna)",
    )
    parser.add_argument(
        "--game",
        choices=("menu", "duel", "chain", "practice"),
        default="menu",
        help="pantalla inicial (por defecto el menu)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    duel = Duel(num_players=args.players, total_rounds=args.rounds)
    chain = ChainRace(num_players=args.players, chain_length=args.chain)
    practice = PracticeSession()

    if args.demo:
        pipeline = DemoPipeline(practice if args.game == "practice" else duel)
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

    app = GestureRushApp(
        duel, chain, practice, pipeline,
        demo=args.demo, fullscreen=not args.windowed,
    )
    if args.game != "menu":
        app.start_game(args.game)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
