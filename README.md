# Gesture Rush

Videojuego competitivo de reconocimiento de gestos por webcam. Este repositorio
corresponde al **demo del núcleo reconocedor y del minijuego clásico** (Fases
1–6 del plan): captura de cámara, detección de manos, extracción de features,
dataset, entrenamiento y comparación de tres clasificadores propios, inferencia
en tiempo real y una interfaz de juego en Pygame.

> El detector de manos (MediaPipe) es un modelo preentrenado; **la clasificación
> de gestos, el dataset y su evaluación son trabajo propio**.

## Estado del demo

| Fase | Contenido | Estado |
|------|-----------|--------|
| 1 | Cámara + MediaPipe + landmarks | Implementado |
| 2 | Recolector + dataset etiquetado | Implementado |
| 3 | Random Forest / SVM-RBF / MLP + evaluación | Implementado |
| 4 | Predictor en tiempo real con suavizado temporal | Implementado |
| 5 | Asociación de manos a jugadores, rondas y puntuación | Implementado |
| 6 | Interfaz visual: menú + «Duelo» + «Cadena» + «Prueba» | En curso |

Quedan pendientes la calibración de regiones, el modo técnico con landmarks
sobre el vídeo y las pruebas con tres personas en la máquina real (fases 6–8).

## Requisitos

- **Python 3.11.x** (probado con 3.11.9). Evita 3.12/3.13: `mediapipe==0.10.14`
  y las versiones fijadas de OpenCV pueden no tener ruedas para esas versiones.
- **Git** para clonar el repositorio.
- **Webcam** para los comandos de cámara (`run_demo_landmarks`, `run_collect`,
  `run_predict`). El entrenamiento y las pruebas funcionan sin cámara.
- Windows 10/11 es el entorno probado. En macOS/Linux el código de cámara usa el
  backend genérico de OpenCV; el resto del pipeline es multiplataforma.
- ~500 MB de disco para el entorno virtual y dependencias.
- Internet solo para clonar e instalar dependencias. El modelo de MediaPipe ya
  viene en `assets/`; si falta, `scripts/download_assets.py` lo descarga.

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
y el asset de MediaPipe, y te dice qué falta. El modelo de manos ya viene en
`assets/`; si falta, descárgalo (una sola vez; luego funciona sin Internet):

```bat
python scripts/download_assets.py
```

Prueba rápida del código:

```bat
python -m pytest -q
```

## Datos y artefactos: qué viaja en el clon

Este repositorio **versiona los artefactos del demo** para que un clon se pueda
usar sin reentrenar: dataset real, modelo, metadatos, informes y el modelo de
MediaPipe. Eso vale para una campaña cerrada. La campaña actual (10 clases)
**todavía no tiene dataset ni modelo**: se recolecta y se entrena de nuevo, y los
artefactos de la campaña anterior quedaron archivados en `data/legacy/`,
`models/legacy/` y `reports/legacy/`.

| Ruta | Contenido | ¿Viene en el clon? |
|------|-----------|--------------------|
| `data/landmarks.csv` | Dataset real de landmarks | No: se genera al recolectar |
| `models/gesture_model.joblib` | Modelo entrenado | No: se genera con `run_train.py` |
| `models/gesture_model.json` | Metadatos (clases, features, métricas) | No: se genera con `run_train.py` |
| `reports/evaluation_report.md`, `reports/cv_report.md` | Informes de evaluación | No: se generan al entrenar |
| `reports/confusion_matrix.png`, `reports/cv_confusion_matrix.png` | Matrices de confusión | No: se generan al entrenar |
| `reports/cv_summary.json` | Resumen de la validación cruzada | No: se genera al validar |
| `assets/hand_landmarker.task` | Modelo preentrenado de MediaPipe | Sí |
| `data/legacy/`, `models/legacy/`, `reports/legacy/` | Artefactos de la campaña de 5 clases | No (locales, ignorados por git) |
| `data/landmarks_synthetic.csv` | Dataset sintético temporal | No |
| `models/synthetic_model.*` | Modelo de experimentos | No |
| `.venv/`, `__pycache__/`, `.pytest_cache/` | Entorno local | No |

> El `.joblib` y el `.json` del modelo son un par: si actualizas uno, actualiza
> el otro. El JSON guarda la especificación de features y las clases, y el
> predictor lo usa para validar que el modelo sea compatible con el código.

### ¿Por qué se versionan los artefactos?

Para este proyecto tiene sentido porque:

- El dataset pesa ~9 MB, el modelo ~130 KB y el asset ~7,8 MB: en total ~18 MB,
  muy por debajo de los límites de GitHub.
- Un clon queda **listo para demostrar** sin webcam para entrenar ni pasos
  manuales de transferencia.
- Se reproducen exactamente las métricas de `reports/` con el mismo modelo.
- El paquete `.joblib` + `.json` + informe queda congelado como evidencia.

Cuándo **no** conviene:

- Si el dataset crece a cientos de MB o GB, o si habrá muchas versiones del
  modelo: usa **Git LFS** o **DVC** en lugar de commitear los binarios.
- Si los datos son sensibles y el repositorio va a ser público. Aquí se guardan
  landmarks (no video ni imágenes), pero siguen siendo datos biométricos
  derivados: asegúrate de tener consentimiento de los participantes.
- Si el modelo cambia muy seguido: cada versión del `.joblib` genera un diff
  binario nuevo. Para el demo actual no es un problema.

El entorno virtual (`.venv/`) y los archivos `.env` **nunca** se versionan: el
primero es específico de cada máquina y el segundo puede contener secretos.

### Cómo actualizar los artefactos después de reentrenar

```bat
python scripts/run_train.py
git add data/landmarks.csv models/gesture_model.joblib models/gesture_model.json reports/
git commit -m "data: actualiza dataset, modelo e informes"
git push
```

### Si necesitas regenerar o copiar los artefactos

- **Con webcam y personas reales:** sigue la sección
  [Uso con webcam](#uso-con-webcam) y recolecta con `scripts/run_collect.py`.
- **Sin webcam (solo para validar el pipeline):** genera datos sintéticos con
  `scripts/run_generate_synthetic.py`. Los resultados sintéticos **no**
  demuestran generalización a personas reales.
- **Desde otra copia del proyecto:** si trabajas con un commit viejo o un ZIP
  sin artefactos, puedes transferirlos. En la máquina original:

  ```powershell
  powershell Compress-Archive -Path data,models,reports,assets -DestinationPath gesture_rush_artifacts.zip
  ```

  En la máquina nueva, desde la raíz del proyecto:

  ```powershell
  powershell Expand-Archive -Path gesture_rush_artifacts.zip -DestinationPath . -Force
  ```

  En macOS/Linux:

  ```bash
  zip -r gesture_rush_artifacts.zip data models reports assets
  unzip gesture_rush_artifacts.zip -d .
  ```

  Verifica que todo llegó con `python scripts/check_setup.py`.

### Campaña anterior (5 clases)

El vocabulario anterior se archivó en `data/legacy/`, `models/legacy/` y
`reports/legacy/`; no está versionado (la evidencia queda en el historial de
git). Sus números fueron:

| Dato | Valor |
|------|-------|
| Muestras | 7.423 (todas reales, sin sintéticas) |
| Participantes | 4 (`participant_01`, `participant_02`, `participant_03`, `participant_4`) |
| Sesiones | 3 |
| Clases | 5 (`open_palm`, `fist`, `thumbs_up`, `victory`, `pointing_index`) |
| Modelo elegido | MLP (capas 64, 32), 42 features |
| Test reservado | accuracy 0,685 · macro-F1 0,675 |
| Validación cruzada LOPO | accuracy 0,763 ± 0,058 · macro-F1 0,761 ± 0,061 |

> Con 10 clases esos números **no** son comparables: cambia el vocabulario, el
> dataset se recolecta de cero y hay que volver a evaluar. Mientras no exista
> `models/gesture_model.joblib`, `run_predict.py` no arranca y `check_setup.py`
> avisa que falta el modelo.

Los informes completos están en `reports/evaluation_report.md` y
`reports/cv_report.md`.

## Uso rápido sin webcam (datos sintéticos)

El clon ya trae el dataset y el modelo reales, así que este flujo es solo para
validar el pipeline o experimentar **sin pisar** los artefactos del demo. Usa
rutas alternativas para el dataset y el modelo:

```bat
python scripts/run_generate_synthetic.py --participants 6 --per-class 150 --output data/landmarks_synthetic.csv
python scripts/run_train.py --dataset data/landmarks_synthetic.csv --model models/synthetic_model.joblib --metadata models/synthetic_model.json --report reports/synthetic_evaluation.md --confusion-plot reports/synthetic_confusion_matrix.png
```

`run_predict.py` siempre carga `models/gesture_model.joblib`; el modelo
`synthetic_model.joblib` sirve para comparar métricas, no para la demo en vivo.

Si borraste los artefactos y quieres reconstruir un modelo por defecto para
probar la demo, **sin datos reales**:

```bat
python scripts/run_generate_synthetic.py --participants 6 --per-class 150 --output data/landmarks_synthetic.csv
python scripts/run_train.py --dataset data/landmarks_synthetic.csv
python scripts/run_predict.py
```

> Los resultados con datos sintéticos **no** demuestran generalización a personas
> reales; solo validan el pipeline.

## Uso con webcam

1. El modelo de MediaPipe ya viene en `assets/`. Si falta, descárgalo (una sola
   vez):

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

4. Recolecta muestras por gesto y participante. Teclas: `1..9` y `0` eligen uno
   de los diez gestos, `n`/`p` pasan al siguiente/anterior, `SPACE` inicia o
   detiene la captura y `q` guarda y sale. El recolector **agrega** filas al CSV
   existente:

   ```bat
   python scripts/run_collect.py --participant participant_01 --session s01
   ```

   Repite con varias personas y sesiones (objetivo inicial: 200–500 muestras por
   gesto y participante, con variación de distancia, orientación, velocidad e
   iluminación). Para que la división train/val/test funcione hacen falta al
   menos **3 participantes**, y **cada participante debe cubrir las diez
   clases**: la validación Leave-One-Participant-Out compara personas completas.

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

## Jugar: minijuegos (Fase 6)

Menú con tres modos sobre **una sola cámara**: Duelo y Cadena compiten (1 o 2
jugadores, cada uno en su mitad del encuadre) y Prueba diagnostica el modelo.

```bat
python scripts/run_game.py                    :: camara + modelo entrenado
python scripts/run_game.py --demo             :: sin camara: jugadores simulados
python scripts/run_game.py --demo --game chain
python scripts/run_game.py --game duel --rounds 3
python scripts/run_game.py --game practice    :: diagnostico del modelo
```

| Opción | Para qué |
|--------|----------|
| `--demo` | No abre cámara ni carga el modelo; jugadores simulados. Ideal para trabajar la interfaz. |
| `--game menu\|duel\|chain\|practice` | Pantalla inicial (por defecto el menú). |
| `--players 1\|2` | Jugadores para Duelo y Cadena: 1 (solo) o 2 (por defecto). |
| `--rounds N` | Rondas del duelo: 3, 5 u 8 (5 por defecto; `0` = sin límite). |
| `--windowed` | Arranca en ventana en vez de pantalla completa. |
| `--chain N` | Gestos por cadena en el modo cadena (5 por defecto). |
| `--camera N` | Índice de cámara; si se omite, usa `GESTURE_CAMERA_INDEX`. |

Teclas: en el menú, `1`/`2`/`3` o clic eligen modo; las píldoras (o `P`)
cambian jugadores y `R` rota las rondas (3, 5 u 8). `F11` alterna pantalla
completa. Para iniciar una ronda o carrera alcanza con que **todos levanten el
pulgar** (👍), o `ESPACIO`/`ENTER`/clic en el botón. En el duelo, la rueda del
mouse o `↑`/`↓` recorren el historial de rondas. En `--demo`, `1` y `2` simulan
el pulgar arriba fuera de ronda y el acierto durante la ronda.

Los iconos de cada gesto son PNG en `assets/hands/` (uno por clase). Si falta
alguno, la interfaz dibuja un icono vectorial de respaldo, así que la app
funciona igual sin esas imágenes.

En Duelo y Cadena cada panel muestra el **encuadre completo** y resalta la
franja del jugador (el resto queda atenuado): la cámara se ve más grande y queda
claro qué mitad le corresponde a cada uno.

### Duelo de gestos

El gesto objetivo es el mismo para todos; el primero en hacerlo se lleva el punto.

- Cuenta atrás de 3 s, luego 8 s para hacer el gesto (barra de progreso).
- El primero en acertar se lleva el punto; la ronda espera hasta 2 s al rival
  para registrar su tiempo, y la pantalla muestra la diferencia en segundos.
- Si nadie acierta, la ronda queda desierta. Tiempos idénticos son empate.
- La ronda arranca cuando **todos levantan el pulgar** (o con el botón/`ENTER`).
- Al terminar las rondas se muestra el ganador y el **historial completo**: la
  rueda del mouse o `↑`/`↓` recorren todas las rondas jugadas.
- Las rondas se eligen en el menú (3, 5 u 8; por defecto 5).
- **Solo (`--players 1`)**: contrarreloj. Sin rival, la ronda cierra en cuanto
  acertás y el panel muestra tu tiempo; la partida resume las rondas jugadas y
  el mejor tiempo.

### Cadena de gestos

**Los dos jugadores reciben la misma cadena** (5 gestos sorteados al iniciar;
pueden repetirse dentro de la cadena, nunca dos iguales seguidos). Cada jugador
avanza a su ritmo completando el gesto que le toca.

- Orden estricto: solo cuenta el gesto de la posición actual; los demás no avanzan.
- Fallar no penaliza, solo cuesta tiempo: el objetivo no cambia hasta acertarlo.
- Gana quien completa el último gesto antes. Cuando el primero termina, el rival
  tiene 2 s para llegar; si llega, gana el tiempo total más bajo.
- Si nadie termina en 45 s, la carrera queda desierta. Los tiempos idénticos son empate.
- El marcador inferior muestra progreso, tiempo total y tiempo medio por gesto.
- **Solo (`--players 1`)**: contrarreloj. La carrera termina al completar la
  cadena y el resumen muestra el tiempo total y la media por gesto.
- La carrera arranca cuando **todos levantan el pulgar** (o con el botón/`ENTER`).

### Prueba del modelo

Modo sin reglas ni puntaje, pensado para **inspeccionar y explicar el
clasificador en vivo**. Una sola vista de cámara (se clasifica la mano más
centrada del encuadre, con sus **21 landmarks** dibujados encima; `L` los
oculta) y una columna de diagnóstico:

| Elemento | Qué muestra |
|----------|-------------|
| Predicción cruda | Clase con mayor probabilidad (`argmax`) y su confianza, en % y en barra con el umbral marcado. |
| Top 3 | Las tres clases más probables; barras parejas = el modelo duda entre gestos parecidos. |
| Voto temporal | Etiqueta con más votos en la ventana y cuántos frames la sostienen (cada frame aporta un voto; los puntos son los 7 lugares). |
| El juego aceptaría | La etiqueta que el juego aceptaría con esta lectura: requiere 5 de los últimos 7 frames y confianza ≥ 60 %. |
| Rendimiento | FPS de la cámara, ms de clasificar un frame y la configuración vigente del suavizado. |

`R` reinicia el voto temporal (útil para mostrar cómo se confirma un gesto desde
cero). La columna resume la configuración vigente y el compromiso del umbral:
**más alto = más estricto y más lento; más bajo = más reactivo y más propenso a
falsos positivos**.

Cómo se llega a un gesto confirmado:

1. MediaPipe entrega 21 landmarks de la mano y se normalizan (muñeca al origen,
   escala por el tamaño de la palma) → vector de 42 features.
2. El clasificador devuelve una probabilidad por cada una de las 10 clases; la
   predicción cruda es la de mayor probabilidad.
3. El **voto temporal** descarta parpadeos: cada frame clasificado aporta un
   voto y la etiqueta se acepta solo si reúne 5 votos en los últimos 7
   (~0,23 s a 30 fps).
4. En Duelo y Cadena cada franja del encuadre se clasifica por separado, con su
   propio suavizado: una mano en la mitad izquierda nunca puntúa para el jugador
   de la derecha. Prueba usa una sola franja (el encuadre completo).

La interfaz dibuja sobre un lienzo de 1600×900 (16:9) y pygame lo escala a la
pantalla, así que en un monitor 1080p ocupa los 1920×1080 exactos, sin franjas.
Arranca en **pantalla completa**; `--windowed` (o `F11`) la pasa a ventana. Si la
cámara o el modelo faltan, usá `--demo`; es también el modo de respaldo para la
feria.

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
| `python scripts/run_game.py` | Interfaz de los minijuegos (Fase 6) |
| `python scripts/run_game.py --demo` | Interfaz sin cámara ni modelo |
| `python scripts/run_game.py --game chain --chain 8` | Cadena de ocho gestos |
| `python scripts/run_game.py --game practice` | Diagnóstico del modelo (top 3, voto temporal, FPS) |
| `python -m pytest -q` | Ejecuta las pruebas |

## Pruebas

```bat
python -m pytest -q
```

Las pruebas no requieren webcam ni `mediapipe`; cubren normalización de features,
dataset, entrenamiento/carga de modelos, predicción de las clases, estabilidad
temporal, reglas del duelo y de la cadena, asignación de manos a jugadores,
top-3 y voto temporal del modo Prueba, y el modo demo de punta a punta.

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

Diez clases, en el orden que mapean las teclas del recolector:

| Tecla | Clase | Gesto |
|---|---|---|
| `1` | `open_palm` | Mano abierta |
| `2` | `fist` | Puño cerrado |
| `3` | `thumbs_up` | Pulgar arriba |
| `4` | `victory` | Victoria (dos dedos) |
| `5` | `pointing_index` | Índice extendido |
| `6` | `ok_sign` | Señal OK |
| `7` | `rock` | Cuernos (rock) |
| `8` | `shaka` | Pulgar y meñique |
| `9` | `pinky_up` | Meñique extendido |
| `0` | `l_shape` | Pistola (mano en L) |

Para añadir una clase hay que tocar tres sitios y ampliar el dataset:

1. `GESTURES` y `GESTURE_LABELS_ES` en `gesture_rush/config.py`.
2. `GESTURE_EXTENSIONS` en `gesture_rush/synthetic_data.py` (el generador
   sintético y las pruebas exigen que cubra todas las clases).
3. Recolectar la clase nueva con **todos** los participantes existentes.

> `l_shape` y `pointing_index` difieren únicamente en el pulgar; es el par más
> propenso a confundirse del vocabulario (medido: 1,20 σ de separación, el más
> bajo de todos los candidatos). Si la evaluación lo confirma, conviene quedarse
> con uno de los dos.

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
  player_tracker.py  Asociación de manos a jugadores (franjas del encuadre)
  game.py            Reglas del duelo y de la cadena: fases, tiempos y puntos
  live.py            Fuentes de datos para la UI (cámara real / demo)
  ui.py              Interfaz Pygame: menú, duelo, cadena y modo prueba
scripts/             Entradas ejecutables por fase + check_setup.py
tests/               Pruebas pytest
assets/hands/        Iconos PNG de los 10 gestos (respaldo vectorial si falta)
models/ data/ reports/ assets/   Artefactos del demo (versionados)
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
