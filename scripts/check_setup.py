"""Verifica que la maquina tenga todo lo necesario para correr Gesture Rush.

Comprueba la version de Python, las dependencias de ``requirements.txt`` y los
artefactos locales (dataset, modelo y asset de MediaPipe). No abre la camara.

Uso:
    python scripts/check_setup.py
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK = "[ok]  "
WARN = "[warn]"
FAIL = "[fail]"

_failures = 0
_warnings = 0


def _ok(msg: str) -> None:
    print(f"{OK} {msg}")


def _warn(msg: str) -> None:
    global _warnings
    _warnings += 1
    print(f"{WARN} {msg}")


def _fail(msg: str) -> None:
    global _failures
    _failures += 1
    print(f"{FAIL} {msg}")


def _parse_requirements(path: Path) -> list[tuple[str, str]]:
    """Devuelve [(paquete, version)] de las lineas ``paquete==version``."""
    pinned: list[tuple[str, str]] = []
    if not path.exists():
        return pinned
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pinned.append((name.strip(), version.strip()))
    return pinned


def check_python() -> None:
    print("\n== Python ==")
    version = sys.version.split()[0]
    if sys.version_info[:2] == (3, 11):
        _ok(f"Python {version} (version recomendada 3.11.x)")
    else:
        _warn(
            f"Python {version}: el proyecto esta probado con 3.11.x. "
            "mediapipe/opencv pueden no tener ruedas para tu version."
        )
    print(f"      ejecutable: {sys.executable}")


def check_dependencies() -> None:
    print("\n== Dependencias ==")
    pinned = _parse_requirements(ROOT / "requirements.txt")
    if not pinned:
        _warn("No se pudo leer requirements.txt; no se verifican versiones.")
        return

    for name, expected in pinned:
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            _fail(f"{name} no esta instalado (se esperaba {expected}).")
            continue
        if installed == expected:
            _ok(f"{name}=={installed}")
        else:
            _warn(
                f"{name}=={installed} (requirements.txt fija {expected}). "
                "Si algo falla, reinstala con: pip install -r requirements.txt"
            )


def check_artifacts() -> None:
    print("\n== Artefactos locales ==")
    from gesture_rush import config

    # Asset de MediaPipe.
    if config.HAND_LANDMARKER_TASK.exists():
        size_mb = config.HAND_LANDMARKER_TASK.stat().st_size / (1024 * 1024)
        _ok(f"asset MediaPipe: {config.HAND_LANDMARKER_TASK} ({size_mb:.1f} MB)")
    else:
        _warn(
            "falta assets/hand_landmarker.task; ejecuta: "
            "python scripts/download_assets.py"
        )

    # Dataset.
    if not config.DATASET_CSV.exists():
        _warn(
            "no hay dataset en data/landmarks.csv. Opciones: recolectar con "
            "scripts/run_collect.py, generar datos sinteticos con "
            "scripts/run_generate_synthetic.py, o copiar el CSV desde el equipo "
            "original."
        )
    else:
        try:
            import pandas as pd

            df = pd.read_csv(config.DATASET_CSV)
            participants = (
                int(df["participant"].nunique())
                if "participant" in df.columns
                else 0
            )
            classes = int(df["label"].nunique()) if "label" in df.columns else 0
            _ok(
                f"dataset: {len(df)} muestras, {participants} participantes, "
                f"{classes} clases ({config.DATASET_CSV})"
            )
        except Exception as exc:  # pragma: no cover - diagnostico
            _fail(f"no se pudo leer el dataset: {exc}")

    # Modelo + metadatos.
    if not config.MODEL_PATH.exists():
        _warn(
            "no hay modelo en models/gesture_model.joblib; ejecuta: "
            "python scripts/run_train.py (o copia models/ desde el equipo original)"
        )
    else:
        try:
            meta = (
                json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
                if config.MODEL_METADATA_PATH.exists()
                else {}
            )
            best = meta.get("best_model", "?")
            accuracy = meta.get("test_metrics", {}).get("accuracy")
            detail = f"modelo: {best}"
            if accuracy is not None:
                detail += f", accuracy test {accuracy:.3f}"
            _ok(detail)
        except Exception as exc:  # pragma: no cover - diagnostico
            _warn(f"modelo presente pero metadatos ilegibles: {exc}")


def main() -> int:
    print("Gesture Rush - verificacion del entorno")
    print(f"Directorio del proyecto: {ROOT}")
    check_python()
    check_dependencies()
    check_artifacts()

    print("\n== Resumen ==")
    if _failures:
        print(f"{FAIL} {_failures} problema(s) bloqueante(s), {_warnings} aviso(s).")
        print("Corrige lo marcado como [fail] y vuelve a ejecutar este script.")
        return 1
    if _warnings:
        print(f"{WARN} {_warnings} aviso(s). El proyecto puede correr, pero revisa lo anterior.")
    else:
        print(f"{OK} Todo listo.")

    print("\nSiguiente paso sugerido:")
    print("  python scripts/list_cameras.py --preview   # elegir camara")
    print("  python scripts/run_predict.py              # probar en vivo")
    print("  python -m pytest -q                        # correr pruebas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
