"""Parametros y rutas configurables de Gesture Rush.

Todo lo ajustable del proyecto vive aqui para evitar valores magicos
repartidos por los modulos.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Dataset de landmarks recolectados (o sinteticos de arranque).
DATASET_CSV = DATA_DIR / "landmarks.csv"

# Modelo seleccionado + metadatos (especificacion de features, clases, metricas).
MODEL_PATH = MODELS_DIR / "gesture_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "gesture_model.json"

# Modelo preentrenado de MediaPipe Hand Landmarker (se descarga una sola vez).
HAND_LANDMARKER_TASK = ASSETS_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# ----------------------------------------------------------------------------
# Gestos
# ----------------------------------------------------------------------------
# Orden canonico de clases. Agregar clases aqui (y ampliar el dataset) es
# suficiente para extender el sistema.
GESTURES: tuple[str, ...] = (
    "open_palm",
    "fist",
    "thumbs_up",
    "victory",
    "pointing_index",
)

GESTURE_LABELS_ES: dict[str, str] = {
    "open_palm": "Mano abierta",
    "fist": "Puno cerrado",
    "thumbs_up": "Pulgar arriba",
    "victory": "Victoria (dos dedos)",
    "pointing_index": "Indice extendido",
}

# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
FEATURE_SPEC_VERSION = "1.0"
# Si es True se incluyen las coordenadas z normalizadas (63 dims); si no, 42.
USE_Z = False
# Landmark de referencia para escalar (indice del MCP del dedo medio).
SCALE_REFERENCE_INDEX = 9  # middle_finger_mcp
WRIST_INDEX = 0

# ----------------------------------------------------------------------------
# Camara
# ----------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# ----------------------------------------------------------------------------
# Deteccion de manos (MediaPipe)
# ----------------------------------------------------------------------------
MAX_NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ----------------------------------------------------------------------------
# Predictor / estabilidad temporal
# ----------------------------------------------------------------------------
# Ventana de suavizado temporal (numero de frames) y votos minimos.
SMOOTHING_WINDOW = 7
SMOOTHING_MIN_VOTES = 5
# Umbral de confianza minima para aceptar una prediccion.
CONFIDENCE_THRESHOLD = 0.6

# ----------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2  # fraccion de (train+val) reservada para validacion
RANDOM_STATE = 42
# Metrica usada para seleccionar el mejor modelo.
SELECTION_METRIC = "f1_macro"

# Recoleccion de datos
SAMPLES_PER_SESSION_TARGET = 200
