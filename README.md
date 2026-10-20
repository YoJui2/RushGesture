# Gesture Rush

Videojuego competitivo de reconocimiento de gestos por webcam. Este repositorio
corresponde al **demo del núcleo reconocedor** (Fases 1–4 del plan): captura de
cámara, detección de manos, extracción de features, dataset, entrenamiento y
comparación de tres clasificadores propios e inferencia en tiempo real.

> El detector de manos (MediaPipe) es un modelo preentrenado; **la clasificación
> de gestos, el dataset y su evaluación son trabajo propio**.

## Estado del demo

| Fase | Contenido | Estado |
|------|-----------|--------|
| 1 | Cámara + MediaPipe + landmarks | Implementado |
| 2 | Recolector + dataset etiquetado | Implementado |
| 3 | Random Forest / SVM-RBF / MLP + evaluación | Implementado |
| 4 | Predictor en tiempo real con suavizado temporal | Implementado |

Las fases 5–8 del prompt (juego multijugador, UI completa, pruebas con tres
participantes) quedan **fuera** de este demo.

## Requisitos

- Python 3.11.x
- Webcam o cámara física conectada

## Cámara

El proyecto captura desde la cámara física configurada en
`gesture_rush/config.py` (`CAMERA_INDEX = 1` en este equipo). Si tu cámara
enumera con otro índice, puedes comprobarlo y cambiarlo sin editar código:

```bat
python scripts/list_cameras.py --preview
set GESTURE_CAMERA_INDEX=0
```

El backend de captura se elige automáticamente (Media Foundation → DirectShow →
cualquiera). También puedes forzarlo con `GESTURE_CAMERA_BACKEND=dshow`,
`msmf` o `any`.

## Instalación

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si alguna versión fijada no está disponible en tu índice:

```bat
pip install mediapipe opencv-python scikit-learn pandas matplotlib joblib pytest
pip freeze > requirements.lock.txt
```

## Uso rápido (sin webcam, con datos sintéticos)

Sirve para validar toda la tubería hoy mismo.

```bat
python scripts/run_generate_synthetic.py --participants 6 --per-class 150
python scripts/run_train.py
python scripts/run_predict.py
```

> Los resultados con datos sintéticos **no** demuestran generalización a personas
> reales; solo validan el pipeline.

## Uso con datos reales

1. Descarga el modelo de MediaPipe (una sola vez; luego funciona sin Internet):

   ```bat
   python scripts/download_assets.py
   ```

2. Verifica cámara y landmarks:

   ```bat
   python scripts/run_demo_landmarks.py
   ```

3. Recolecta muestras por gesto y participante. Teclas: `1..5` gesto,
   `SPACE` iniciar/detener captura, `q` guardar y salir:

   ```bat
   python scripts/run_collect.py --participant participant_01 --session s01
   ```

   Repite con varias personas y sesiones (objetivo inicial: 200–500 por gesto y
   participante, con variación de distancia, orientación, velocidad e iluminación).

4. Entrena y compara modelos:

   ```bat
   python scripts/run_train.py
   ```

5. Predicción en vivo:

   ```bat
   python scripts/run_predict.py
   ```

## Pruebas

```bat
python -m pytest
```

Las pruebas no requieren webcam ni `mediapipe`; cubren normalización de features,
dataset, entrenamiento/carga de modelos, predicción de las clases y estabilidad
temporal.

## Gestos reconocidos

`open_palm`, `fist`, `thumbs_up`, `victory`, `pointing_index`. Para añadir clases,
edítalas en `gesture_rush/config.py` y amplía el dataset.

## Estructura

```
gesture_rush/
  config.py          Parámetros y rutas
  camera.py          Captura, FPS y gestión de errores
  hand_detector.py   Integración MediaPipe + landmarks
  features.py        Normalización y especificación de features
  synthetic_data.py  Generador de landmarks sintéticos (bootstrap)
  collect_data.py    Recolector etiquetado
  dataset.py         Carga, validación, featurización y split por grupo
  train.py           Entrenamiento y comparación (RF / SVM / MLP)
  evaluate.py        Métricas, matriz de confusión, informes
  predictor.py       Inferencia + suavizado temporal
scripts/             Entradas ejecutables por fase
tests/               Pruebas pytest
models/ data/ reports/ assets/   Artefactos (no versionados)
```

## Extracción de features

Pipeline reproducible (`features.py`): se traslada la muñeca al origen, se
normaliza la escala por el tamaño de la palma (muñeca→MCP del dedo medio) y se
aplana en un vector de 42 dimensiones (63 si `USE_Z = True`). La **especificación
de features** (orden, dimensión, versión) se guarda junto al modelo para evitar
desajustes entre entrenamiento e inferencia.

## Metodología de evaluación

- División **agrupada por participante** (`GroupShuffleSplit`): ninguna sesión o
  persona aparece a la vez en entrenamiento y prueba.
- Hiperparámetros ajustados con `GridSearchCV` sobre train/validación con
  `GroupKFold`.
- Selección por **macro-F1** en validación; el modelo elegido se reajusta con
  train+validación y se evalúa **una sola vez** en el test reservado.
- Métricas: accuracy, precision/recall/F1 por clase, macro-F1, matriz de
  confusión, tiempo de inferencia. Informe en `reports/`.

## Limitaciones conocidas

- Con pocos participantes, la generalización a personas desconocidas no está
  garantizada y así se declara en los metadatos.
- Los datos sintéticos sirven de bootstrap, no de evidencia experimental.
- MediaPipe puede no detectar de forma fiable tres manos simultáneas en todas
  las condiciones de iluminación y encuadre (relevante para las fases 5–7).
