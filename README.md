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

- **Python 3.11.x** (probado con 3.11.9). Evita 3.12/3.13: `mediapipe==0.10.14`
  y las versiones fijadas de OpenCV pueden no tener ruedas para esas versiones.
- **Git** para clonar el repositorio.
- **Webcam** para los comandos de cámara (`run_demo_landmarks`, `run_collect`,
  `run_predict`). El entrenamiento y las pruebas funcionan sin cámara.
- Windows 10/11 es el entorno probado. En macOS/Linux el código de cámara usa el
  backend genérico de OpenCV; el resto del pipeline es multiplataforma.
- ~500 MB de disco para el entorno virtual y dependencias.
- Internet solo para clonar, instalar dependencias y descargar el modelo de
  MediaPipe la primera vez.

## Clonar e instalar desde otra máquina

Todos los comandos se ejecutan **desde la raíz del proyecto**.

### 1. Clonar

```bat
git clone https://github.com/YoJui2/RushGesture.git
cd RushGesture
```

Si el repositorio es privado necesitas acceso a la cuenta `YoJui2` y
autenticarte con GitHub (HTTPS con token o SSH). Alternativa SSH:

```bat
git clone git@github.com:YoJui2/RushGesture.git
cd RushGesture
```

> No copies la carpeta `.venv` de otra máquina: el entorno virtual contiene rutas
> absolutas y binarios específicos del sistema. Se crea de nuevo en 2 minutos.

### 2. Crear el entorno virtual

**Windows (cmd):**

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea el script de activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Las versiones están fijadas en `requirements.txt` para reproducibilidad. Si
alguna no está disponible en tu índice:

```bat
python -m pip install mediapipe opencv-python scikit-learn pandas matplotlib joblib pytest
python -m pip freeze > requirements.lock.txt
```

### 4. Verificar la instalación

```bat
python scripts/check_setup.py
```

El script comprueba la versión de Python, las dependencias, el dataset, el modelo
y el asset de MediaPipe, y te dice qué falta. Después descarga el modelo de manos
(una sola vez; luego funciona sin Internet):

```bat
python scripts/download_assets.py
```

Prueba rápida del código:

```bat
python -m pytest -q
```

## Datos y artefactos: qué viaja en el clon

**Importante:** el repositorio versiona el código y la documentación, pero no los
artefactos pesados. El dataset, el modelo entrenado, las matrices/imágenes y el
asset de MediaPipe están en `.gitignore`, así que **un clon nuevo no los trae**.
Son artefactos locales que se regeneran o se copian aparte.

| Ruta | Contenido | ¿Viene en el clon? |
|------|-----------|--------------------|
| `data/landmarks.csv` | Dataset de landmarks etiquetados | No |
| `models/gesture_model.joblib` | Modelo clasificador entrenado | No |
| `models/gesture_model.json` | Metadatos (clases, features, métricas) | No |
| `reports/*.md` | Informes de evaluación | Sí (si están commiteados) |
| `reports/*.png`, `reports/*.json` | Matrices y resúmenes | No |
| `assets/hand_landmarker.task` | Modelo preentrenado de MediaPipe | No (se descarga) |
| `gesture_rush/`, `scripts/`, `tests/` | Código y pruebas | Sí |
| `docs/` | Diagramas del pipeline | Sí |

### Opción A: copiar los artefactos del equipo original (recomendado)

Si quieres conservar **exactamente** el dataset y el modelo ya entrenados, copia
`data/`, `models/` y `reports/` desde la máquina original.

En la máquina original, desde la raíz del proyecto:

```powershell
powershell Compress-Archive -Path data,models,reports -DestinationPath gesture_rush_artifacts.zip
```

En la máquina nueva, después de clonar y crear el entorno, desde la raíz:

```powershell
powershell Expand-Archive -Path gesture_rush_artifacts.zip -DestinationPath . -Force
```

En macOS/Linux:

```bash
zip -r gesture_rush_artifacts.zip data models reports
unzip gesture_rush_artifacts.zip -d .
```

Añade `assets` al ZIP solo si la máquina nueva no tiene Internet; si no, basta
con `python scripts/download_assets.py`.

Verifica que todo llegó:

```bat
python scripts/check_setup.py
```

### Opción B: regenerar los datos

- **Con webcam y personas reales:** sigue la sección
  [Uso con webcam](#uso-con-webcam) y recolecta con `scripts/run_collect.py`.
- **Sin webcam (solo para validar el pipeline):** genera datos sintéticos con
  `scripts/run_generate_synthetic.py`. Los resultados sintéticos **no**
  demuestran generalización a personas reales.

### Estado de los artefactos del equipo original

Estos son los números de los artefactos que quedaron entrenados localmente
(pueden cambiar si vuelves a entrenar):

| Dato | Valor |
|------|-------|
| Muestras | 7.423 (todas reales, sin sintéticas) |
| Participantes | 4 (`participant_01`, `participant_02`, `participant_03`, `participant_4`) |
| Sesiones | 3 |
| Clases | 5 (`open_palm`, `fist`, `thumbs_up`, `victory`, `pointing_index`) |
| Modelo elegido | MLP (capas 64, 32), 42 features |
| Test reservado | accuracy 0,685 · macro-F1 0,675 |
| Validación cruzada LOPO | accuracy 0,763 ± 0,058 · macro-F1 0,761 ± 0,061 |

Los informes completos están en `reports/evaluation_report.md` y
`reports/cv_report.md`.

> Si quieres que estos artefactos viajen con Git en lugar de copiarlos a mano,
> quita las líneas correspondientes de `.gitignore` y haz commit. El dataset
> pesa ~9 MB y el modelo ~130 KB, así que es viable; para datasets mucho más
> grandes usa Git LFS.

## Uso rápido sin webcam (datos sintéticos)

Sirve para validar toda la tubería en un clon limpio. **Ojo:** si ya tienes el
dataset o el modelo reales, no ejecutes esto con las rutas por defecto, porque
`run_train.py` sobrescribe `models/gesture_model.*`.

En un clon limpio, sin artefactos:

```bat
python scripts/run_generate_synthetic.py --participants 6 --per-class 150 --output data/landmarks_synthetic.csv
python scripts/run_train.py --dataset data/landmarks_synthetic.csv
python scripts/run_predict.py
```

Si ya copiaste los artefactos reales y quieres experimentar sin pisarlos, usa
rutas alternativas:

```bat
python scripts/run_generate_synthetic.py --participants 6 --per-class 150 --output data/landmarks_synthetic.csv
python scripts/run_train.py --dataset data/landmarks_synthetic.csv --model models/synthetic_model.joblib --metadata models/synthetic_model.json --report reports/synthetic_evaluation.md --confusion-plot reports/synthetic_confusion_matrix.png
```

`run_predict.py` siempre carga `models/gesture_model.joblib`; el modelo
`synthetic_model.joblib` sirve para comparar métricas, no para la demo en vivo.

> Los resultados con datos sintéticos **no** demuestran generalización a personas
> reales; solo validan el pipeline.

## Uso con webcam

1. Descarga el modelo de MediaPipe (una sola vez):

   ```bat
   python scripts/download_assets.py
   ```

2. Identifica el índice de tu cámara:

   ```bat
   python scripts/list_cameras.py --preview
   ```

   El proyecto usa `CAMERA_INDEX = 1` por defecto (configurado en el equipo
   original). Si tu cámara aparece con otro índice, no edites código: define la
   variable de entorno en la terminal actual.

   **Windows (cmd):**

   ```bat
   set GESTURE_CAMERA_INDEX=0
   ```

   **Windows (PowerShell):**

   ```powershell
   $env:GESTURE_CAMERA_INDEX=0
   ```

   **macOS / Linux:**

   ```bash
   export GESTURE_CAMERA_INDEX=0
   ```

   El backend se elige automáticamente (Media Foundation → DirectShow →
   cualquiera). Puedes forzarlo con `GESTURE_CAMERA_BACKEND=dshow`, `msmf` o
   `any`.

3. Verifica cámara y landmarks. Teclas: `q`/`ESC` salir, `d` alternar dibujo:

   ```bat
   python scripts/run_demo_landmarks.py
   ```

4. Recolecta muestras por gesto y participante. Teclas: `1..5` gesto,
   `SPACE` iniciar/detener captura, `q` guardar y salir. El recolector **agrega**
   filas al CSV existente:

   ```bat
   python scripts/run_collect.py --participant participant_01 --session s01
   ```

   Repite con varias personas y sesiones (objetivo inicial: 200–500 muestras por
   gesto y participante, con variación de distancia, orientación, velocidad e
   iluminación). Para que la división train/val/test funcione hacen falta al
   menos **3 participantes**.

5. Entrena y compara modelos (RF, SVM-RBF y MLP):

   ```bat
   python scripts/run_train.py
   ```

6. Evaluación con validación cruzada Leave-One-Participant-Out:

   ```bat
   python scripts/run_cv.py
   ```

7. Predicción en vivo con el modelo entrenado. Teclas: `q`/`ESC` salir:

   ```bat
   python scripts/run_predict.py
   ```

## Comandos de referencia

| Comando | Para qué |
|---------|----------|
| `python scripts/check_setup.py` | Diagnóstico de Python, dependencias y artefactos |
| `python scripts/download_assets.py` | Descarga el modelo de MediaPipe |
| `python scripts/list_cameras.py --preview` | Lista cámaras e índices |
| `python scripts/run_demo_landmarks.py` | Demo de cámara + landmarks |
| `python scripts/run_collect.py --participant X --session Y` | Recolecta datos reales |
| `python scripts/run_generate_synthetic.py --output data/landmarks_synthetic.csv` | Genera datos sintéticos |
| `python scripts/run_train.py` | Entrena, compara y guarda el modelo |
| `python scripts/run_cv.py` | Validación cruzada por participante |
| `python scripts/run_predict.py` | Predicción en tiempo real |
| `python -m pytest -q` | Ejecuta las pruebas |

## Pruebas

```bat
python -m pytest -q
```

Las pruebas no requieren webcam ni `mediapipe`; cubren normalización de features,
dataset, entrenamiento/carga de modelos, predicción de las clases y estabilidad
temporal.

## Problemas frecuentes

| Síntoma | Solución |
|---------|----------|
| `git` no se reconoce | Instala Git y vuelve a abrir la terminal |
| `python` abre la Microsoft Store o usa otra versión | Usa `py -3.11` en Windows y verifica `python --version` |
| `pip` no encuentra `mediapipe==0.10.14` | Estás en Python 3.12/3.13; crea el venv con 3.11 |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| No se pudo abrir la cámara | Prueba otro índice con `list_cameras.py` y `GESTURE_CAMERA_INDEX` |
| La cámara abre pero se ve negro | Fuerza `GESTURE_CAMERA_BACKEND=dshow` o `msmf` |
| `No se encontro el modelo` | Copia `models/` o ejecuta `python scripts/run_train.py` |
| `No existe el dataset` | Copia `data/landmarks.csv`, recolecta o genera datos sintéticos |
| `El modelo fue entrenado con features v...` | El código y el modelo no coinciden; reentrena con `run_train.py` |
| No hay Internet para MediaPipe | Copia `assets/hand_landmarker.task` desde otra máquina |
| Solo hay 1 o 2 participantes | La división agrupada exige al menos 3; recolecta más datos |

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
scripts/             Entradas ejecutables por fase + check_setup.py
tests/               Pruebas pytest
models/ data/ reports/ assets/   Artefactos locales (no versionados)
```

## Extracción de features

Pipeline reproducible (`features.py`): se traslada la muñeca al origen, se
normaliza la escala por el tamaño de la palma (muñeca→MCP del dedo medio) y se
aplana en un vector de 42 dimensiones (63 si `USE_Z = True`). La **especificación
de features** (orden, dimensión, versión) se guarda junto al modelo para evitar
desajustes entre entrenamiento e inferencia.

## Metodología de evaluación

- **Evaluación principal (`scripts/run_train.py`)**: división **agrupada por
  participante** (`GroupShuffleSplit`) en train/validación/test. El test se
  evalúa **una sola vez**.
- **Validación cruzada (`scripts/run_cv.py`)**: *Leave-One-Participant-Out*.
  Cada participante es test exactamente una vez; en cada fold los participantes
  restantes se separan en train y validación. Los hiperparámetros se eligen solo
  con train y el mejor modelo se reajusta con train+validación antes de evaluar
  el test. Se informa media ± desviación estándar.
- Hiperparámetros ajustados con `GridSearchCV` sobre train/validación con
  `GroupKFold` cuando hay al menos dos participantes en train.
- Selección por **macro-F1** en validación; el modelo elegido se reajusta con
  train+validación y se evalúa **una sola vez** en el test reservado.
- Métricas: accuracy, precision/recall/F1 por clase, macro-F1, matriz de
  confusión, tiempo de inferencia. Informes en `reports/`.

> La validación cruzada no reemplaza la necesidad de más participantes:
> mejora la estimación, pero no agrega datos nuevos. Con un solo participante
> en train, la CV interna cae a folds dentro de la misma persona y se marca
> como `(intra-participante)`.

## Limitaciones conocidas

- Con pocos participantes, la generalización a personas desconocidas no está
  garantizada y así se declara en los metadatos.
- Los datos sintéticos sirven de bootstrap, no de evidencia experimental.
- MediaPipe puede no detectar de forma fiable tres manos simultáneas en todas
  las condiciones de iluminación y encuadre (relevante para las fases 5–7).
