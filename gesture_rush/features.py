"""Extraccion y normalizacion de landmarks de la mano.

Pipeline de features reproducible:

1. Recibe los 21 landmarks (x, y, z) tal como los entrega MediaPipe
   (coordenadas normalizadas respecto al frame, z relativo a la muneca).
2. Traslada las coordenadas para que la muneca (indice 0) sea el origen.
3. Normaliza la escala usando el tamano de la palma (distancia muneca -> MCP
   del dedo medio, por ser estable frente a la flexion de los dedos).
4. Aplana en un vector de dimension fija (42 sin z, 63 con z).

El mismo modulo se usa en recoleccion, entrenamiento y tiempo real, evitando
discrepancias entre las tres etapas. La especificacion (orden y version) se
guarda junto al modelo.
"""

from __future__ import annotations

import numpy as np

from . import config

WRIST = config.WRIST_INDEX
SCALE_REF = config.SCALE_REFERENCE_INDEX

N_LANDMARKS = 21

# Nombres de los 21 landmarks, en el orden de MediaPipe Hands.
LANDMARK_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)


def landmark_columns(n: int = N_LANDMARKS) -> list[str]:
    """Nombres de las columnas para almacenar landmarks crudos (3*n columnas)."""
    cols: list[str] = []
    for i in range(n):
        cols.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
    return cols


def landmarks_from_row(row) -> np.ndarray:
    """Reconstruye un arreglo (21, 3) a partir de una fila del dataset."""
    cols = landmark_columns()
    return np.array([row[c] for c in cols], dtype=np.float64).reshape(N_LANDMARKS, 3)


def feature_names(use_z: bool | None = None) -> list[str]:
    """Devuelve los nombres de las features en el orden canonico."""
    if use_z is None:
        use_z = config.USE_Z
    axes = ["x", "y", "z"] if use_z else ["x", "y"]
    return [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in axes]


def feature_dim(use_z: bool | None = None) -> int:
    """Dimension del vector de features."""
    return N_LANDMARKS * (3 if (config.USE_Z if use_z is None else use_z) else 2)


def feature_spec(use_z: bool | None = None) -> dict:
    """Especificacion completa de las features (se guarda con el modelo)."""
    if use_z is None:
        use_z = config.USE_Z
    return {
        "version": config.FEATURE_SPEC_VERSION,
        "use_z": bool(use_z),
        "dim": feature_dim(use_z),
        "wrist_index": WRIST,
        "scale_reference_index": SCALE_REF,
        "landmark_names": list(LANDMARK_NAMES),
        "names": feature_names(use_z),
    }


def validate_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Valida forma y finitud de un arreglo de landmarks. Devuelve float64."""
    arr = np.asarray(landmarks, dtype=np.float64)
    if arr.shape == (N_LANDMARKS, 2):
        arr = np.concatenate([arr, np.zeros((N_LANDMARKS, 1))], axis=1)
    if arr.shape != (N_LANDMARKS, 3):
        raise ValueError(
            f"Se esperaban landmarks de forma (21, 2) o (21, 3); se obtuvo {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Los landmarks contienen valores no finitos (NaN o inf)")
    return arr


def normalize_landmarks(landmarks: np.ndarray, use_z: bool | None = None) -> np.ndarray:
    """Normaliza landmarks y devuelve el vector de features aplanado.

    Traslada a la muneca, escala por el tamano de palma y aplana.
    """
    if use_z is None:
        use_z = config.USE_Z
    arr = validate_landmarks(landmarks)

    wrist = arr[WRIST]
    centered = arr - wrist  # traslacion: muneca en el origen

    # Escala: distancia (estable) entre la muneca y el MCP del dedo medio.
    palm_size = np.linalg.norm(centered[SCALE_REF])
    if palm_size < 1e-6:
        raise ValueError("Tamano de palma invalido (cercano a cero); no se puede normalizar")
    scaled = centered / palm_size

    coords = scaled[:, :3] if use_z else scaled[:, :2]
    return coords.reshape(-1).astype(np.float64)


def extract_features(landmarks: np.ndarray, use_z: bool | None = None) -> np.ndarray:
    """Alias semantico de :func:`normalize_landmarks`."""
    return normalize_landmarks(landmarks, use_z=use_z)
