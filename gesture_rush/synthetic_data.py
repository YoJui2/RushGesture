"""Generador de landmarks sinteticos para arrancar el pipeline.

Sirve como *bootstrap*: permite entrenar y ejecutar todo el sistema hoy mismo,
sin depender de recolectar miles de muestras reales con la webcam. Los datos
generados se marcan con ``source="synthetic"`` para no confundirlos con datos
reales.

El modelo de mano es una plantilla geometrica: una palma rigida y cinco dedos
que interpolan entre una pose extendida y una flexionada. Sobre esa plantilla
se aplican variaciones de rotacion, escala, traslacion, ruido y lateralidad,
simulando distintos participantes y sesiones.

IMPORTANTE: los resultados con datos sinteticos NO demuestran generalizacion a
personas reales. Se usan solo para validar la tuberia completa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import LANDMARK_NAMES

# -- Plantilla geometrica de la mano (frame local, +y "hacia arriba") --------

_PALM: dict[int, tuple[float, float]] = {
    0: (0.00, 0.00),    # wrist
    1: (-0.35, 0.30),   # thumb_cmc
    2: (-0.62, 0.55),   # thumb_mcp
    5: (-0.30, 1.00),   # index_mcp
    9: (0.00, 1.05),    # middle_mcp
    13: (0.28, 1.00),   # ring_mcp
    17: (0.52, 0.85),   # pinky_mcp
}

# (indice de la cadena de 4 landmark: mcp, pip, dip, tip)
_FINGER_CHAINS: dict[str, tuple[int, int, int, int]] = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

_FINGER_LENGTH: dict[str, float] = {
    "index": 0.95,
    "middle": 1.05,
    "ring": 0.95,
    "pinky": 0.78,
}

# Offsets (dx, dy) respecto al MCP, para las poses extendida y flexionada.
_EXTENDED = {"pip": (0.00, 0.55), "dip": (0.00, 0.85), "tip": (0.00, 1.05)}
_CURLED = {"pip": (0.00, 0.38), "dip": (-0.06, 0.14), "tip": (-0.10, -0.06)}

# Offsets del pulgar respecto al thumb_mcp (2).
_THUMB_EXTENDED = {"ip": (0.05, 0.32), "tip": (0.10, 0.62)}
_THUMB_CURLED = {"ip": (-0.02, 0.20), "tip": (0.14, 0.10)}


def _lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (a[0] * t + b[0] * (1 - t), a[1] * t + b[1] * (1 - t))


def canonical_template(extensions: dict[str, float]) -> np.ndarray:
    """Construye una mano 3D (21x3) a partir de la extension de cada dedo.

    ``extensions`` mapea "thumb", "index", "middle", "ring", "pinky" a valores
    en [0, 1], donde 1 = extendido y 0 = flexionado.
    """
    pts = np.zeros((21, 2), dtype=np.float64)
    for idx, (x, y) in _PALM.items():
        pts[idx] = (x, y)

    # Pulgar.
    thumb_e = float(extensions.get("thumb", 1.0))
    base = pts[2]
    ip = _lerp(_THUMB_EXTENDED["ip"], _THUMB_CURLED["ip"], thumb_e)
    tip = _lerp(_THUMB_EXTENDED["tip"], _THUMB_CURLED["tip"], thumb_e)
    pts[3] = base + np.array(ip)
    pts[4] = base + np.array(tip)

    # Dedos restantes.
    for name, (mcp, pip_i, dip_i, tip_i) in _FINGER_CHAINS.items():
        e = float(extensions.get(name, 0.0))
        f = _FINGER_LENGTH[name]
        base = pts[mcp]
        pip = _lerp(
            (_EXTENDED["pip"][0], _EXTENDED["pip"][1] * f),
            (_CURLED["pip"][0], _CURLED["pip"][1] * f),
            e,
        )
        dip = _lerp(
            (_EXTENDED["dip"][0], _EXTENDED["dip"][1] * f),
            (_CURLED["dip"][0], _CURLED["dip"][1] * f),
            e,
        )
        tip = _lerp(
            (_EXTENDED["tip"][0], _EXTENDED["tip"][1] * f),
            (_CURLED["tip"][0], _CURLED["tip"][1] * f),
            e,
        )
        pts[pip_i] = base + np.array(pip)
        pts[dip_i] = base + np.array(dip)
        pts[tip_i] = base + np.array(tip)

    z = np.zeros((21, 1), dtype=np.float64)
    return np.concatenate([pts, z], axis=1)


# Extension de cada dedo por gesto. Debe cubrir todas las clases de
# ``config.GESTURES``: el generador sintetico y las pruebas lo exigen.
# La plantilla interpola cada dedo entre extendido y flexionado, asi que
# ``ok_sign`` es solo una aproximacion (no puede representar el contacto entre
# pulgar e indice) y sirve para validar la tuberia, no como evidencia.
GESTURE_EXTENSIONS: dict[str, dict[str, float]] = {
    "open_palm": {"thumb": 1.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "fist": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
    "thumbs_up": {"thumb": 1.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
    "victory": {"thumb": 0.1, "index": 1.0, "middle": 1.0, "ring": 0.0, "pinky": 0.0},
    "pointing_index": {"thumb": 0.3, "index": 1.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
    "ok_sign": {"thumb": 0.55, "index": 0.35, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "rock": {"thumb": 0.3, "index": 1.0, "middle": 0.0, "ring": 0.0, "pinky": 1.0},
    "shaka": {"thumb": 1.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 1.0},
    "pinky_up": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 1.0},
    "l_shape": {"thumb": 1.0, "index": 1.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
}


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_sample(
    label: str,
    rng: np.random.Generator,
    rotation_bias: float = 0.0,
    scale_bias: float = 1.0,
    noise: float = 0.02,
) -> tuple[np.ndarray, str]:
    """Genera una muestra (21x3) para ``label`` con variaciones aleatorias."""
    template = canonical_template(GESTURE_EXTENSIONS[label])

    # Rotacion en el plano (orientacion de la mano), escala y traslacion.
    theta = rotation_bias + rng.uniform(-np.pi / 6, np.pi / 6)
    scale = scale_bias * rng.uniform(0.8, 1.25)
    pts = template @ _rotation_matrix(theta).T
    pts *= scale
    pts += rng.uniform(-0.4, 0.4, size=3)

    # Profundidad (z): dedos flexionados tienden a acercarse a la camara.
    curl = 1.0 - np.mean(list(GESTURE_EXTENSIONS[label].values()))
    pts[:, 2] = rng.normal(0.0, 0.03, size=21) + 0.15 * curl

    # Ruido de medicion y lateralidad (espejado horizontal).
    pts[:, :2] += rng.normal(0.0, noise, size=(21, 2))
    handedness = "Right"
    if rng.random() < 0.5:
        pts[:, 0] *= -1.0
        handedness = "Left"
    return pts, handedness


def generate(
    per_class_per_participant: int = 120,
    participants: int = 4,
    sessions_per_participant: int = 2,
    gestures: tuple[str, ...] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera un dataset sintetico completo y devuelve un ``DataFrame``.

    Cada participante y sesion introduce un sesgo propio de rotacion/escala,
    de modo que la separacion por grupos (participante) tenga sentido.
    """
    from . import config

    gestures = gestures or config.GESTURES
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for p in range(1, participants + 1):
        participant = f"participant_{p:02d}"
        p_rot = rng.uniform(-0.25, 0.25)
        p_scale = rng.uniform(0.9, 1.15)
        p_noise = rng.uniform(0.015, 0.035)
        for s in range(1, sessions_per_participant + 1):
            session = f"s{p:02d}_{s}"
            s_rot = p_rot + rng.uniform(-0.12, 0.12)
            s_scale = p_scale * rng.uniform(0.95, 1.05)
            for label in gestures:
                for _ in range(per_class_per_participant):
                    pts, handedness = make_sample(
                        label, rng, s_rot, s_scale, p_noise
                    )
                    row = _sample_to_row(pts)
                    row.update(
                        {
                            "label": label,
                            "participant": participant,
                            "session": session,
                            "handedness": handedness,
                            "source": "synthetic",
                        }
                    )
                    rows.append(row)

    return pd.DataFrame(rows)


def _sample_to_row(pts: np.ndarray) -> dict:
    row: dict[str, float] = {}
    for i in range(len(LANDMARK_NAMES)):
        row[f"lm_{i}_x"] = float(pts[i, 0])
        row[f"lm_{i}_y"] = float(pts[i, 1])
        row[f"lm_{i}_z"] = float(pts[i, 2])
    return row
