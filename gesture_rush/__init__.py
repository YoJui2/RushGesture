"""Gesture Rush - paquete principal.

Demo nucleo reconocedor de gestos (Fases 1-4 del prompt):
camara + MediaPipe + features + dataset + entrenamiento + predictor.
"""

import warnings

# MediaPipe arrastra una version vieja de protobuf que emite este aviso en cada
# arranque. No afecta al funcionamiento y ensucia la consola, asi que se filtra
# aqui: el paquete se importa antes de tocar mediapipe en cualquier entrada.
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated",
    category=UserWarning,
)

__version__ = "0.1.0"
