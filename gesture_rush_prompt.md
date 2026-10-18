# PROMPT PARA AGENTE DE CÓDIGO — GESTURE RUSH

## 1. Objetivo del proyecto

Desarrollar **Gesture Rush**, un videojuego competitivo de reconocimiento de gestos mediante webcam, pensado para una feria de ciencias universitaria.

Entre 2 y 3 jugadores se colocan frente a una cámara. En cada ronda aparece un gesto objetivo en pantalla. Los jugadores deben reproducirlo lo más rápido posible y el sistema asigna puntos según quién lo hace primero correctamente.

El proyecto debe demostrar machine learning real: recopilación de un dataset propio, entrenamiento de varios clasificadores, evaluación rigurosa, selección del mejor modelo e inferencia en tiempo real.

No quiero solamente un prototipo que invoque un modelo preentrenado para reconocer gestos. La clasificación final debe utilizar modelos entrenados y evaluados por nosotros.

**Plazo objetivo:** 2–4 semanas.

**Prioridades:** funcionamiento fiable, baja latencia, experiencia visual atractiva, código modular, reproducibilidad experimental y facilidad para explicar el sistema en una defensa académica.

## 2. Antes de programar

Primero inspeccioná el repositorio y el entorno existentes.

1. Identificá el sistema operativo, versión de Python, estructura del proyecto y dependencias instaladas.
2. Detectá si ya existe código reutilizable.
3. Proponé una arquitectura breve y un plan de implementación por etapas.
4. Identificá incompatibilidades potenciales entre OpenCV, MediaPipe, NumPy y scikit-learn.
5. Implementá progresivamente, verificando cada etapa antes de continuar.

No reemplaces archivos ni configuraciones existentes sin entender su propósito. No ejecutes comandos destructivos. Si falta información esencial, preguntá antes de tomar una decisión que pueda comprometer el proyecto.

## 3. Stack tecnológico

Utilizá preferentemente:

- **Python:** lenguaje principal.
- **OpenCV:** captura de webcam y procesamiento de fotogramas.
- **MediaPipe Hand Landmarker:** detección de manos y extracción de 21 puntos de referencia por mano.
- **NumPy:** procesamiento de características.
- **scikit-learn:** entrenamiento, validación y comparación de clasificadores.
- **Pygame:** interfaz del juego, animaciones, marcador y rondas.
- **Matplotlib y scikit-learn metrics:** visualización y evaluación de resultados.
- **pytest:** pruebas automatizadas.

Usá versiones compatibles y fijá las dependencias en un archivo reproducible. No agregues frameworks innecesarios ni servicios externos que puedan introducir latencia o exigir conexión a Internet durante la feria.

MediaPipe es el detector de manos preentrenado. Los clasificadores de gestos, sus datos de entrenamiento y su evaluación deben ser trabajo propio.

## 4. Arquitectura

Separá el proyecto en módulos con responsabilidades claras:

- `camera.py`: captura de imágenes, control de FPS y gestión de errores de cámara.
- `hand_detector.py`: integración con MediaPipe y extracción de landmarks.
- `features.py`: normalización y transformación de landmarks.
- `collect_data.py`: interfaz para recolectar y etiquetar ejemplos.
- `dataset.py`: carga, validación y división de los datos.
- `train.py`: entrenamiento y comparación de modelos.
- `evaluate.py`: métricas, gráficos, matriz de confusión y análisis de errores.
- `predictor.py`: carga del modelo seleccionado e inferencia.
- `player_tracker.py`: asociación de manos con jugadores.
- `game.py`: lógica de rondas, tiempos, puntuaciones y condiciones de victoria.
- `ui.py`: pantalla, instrucciones, feedback y marcador.
- `config.py`: parámetros configurables.
- `tests/`: pruebas automatizadas.

Podés adaptar esta estructura si el repositorio ya tiene una organización mejor. Evitá duplicar lógica entre módulos.

## 5. Gestos a reconocer

Empezá con cinco clases:

1. `open_palm`: mano abierta.
2. `fist`: puño cerrado.
3. `thumbs_up`: pulgar arriba.
4. `victory`: señal de victoria con dos dedos.
5. `pointing_index`: índice extendido.

Permití añadir clases sin modificar todo el sistema.

La ausencia de una mano detectada debe ser un estado del detector, no una predicción artificial del clasificador. Considerá incorporar una postura neutral como clase adicional si las pruebas demuestran que ayuda a prevenir activaciones accidentales.

La aplicación debe mostrar en tiempo real el gesto objetivo, el estado de cada jugador y la predicción actual.

## 6. Extracción de características

No alimentes directamente el clasificador con coordenadas de píxeles sin normalizar.

Implementá un pipeline de características reproducible:

1. Obtener los 21 landmarks de la mano.
2. Trasladar las coordenadas para que la muñeca sea el origen.
3. Normalizar la escala utilizando una medida estable de la palma.
4. Conservar las relaciones geométricas necesarias para distinguir los gestos.
5. Investigar cómo manejar la orientación de la mano y las diferencias entre manos izquierdas y derechas.
6. Comprobar que el mismo procesamiento se aplica durante entrenamiento, validación y ejecución del juego.

No elimines información de orientación indiscriminadamente: podría ser relevante para distinguir ciertos gestos. Compará la normalización básica con alternativas cuando sea útil.

Guardá la definición y el orden de las características junto al modelo para impedir discrepancias entre entrenamiento e inferencia.

## 7. Recolección del dataset propio

Implementá una herramienta sencilla para recopilar ejemplos con la webcam.

Requisitos:

- Mostrar el gesto solicitado.
- Permitir iniciar y detener la captura por clase.
- Guardar los landmarks, etiqueta, participante, sesión y metadatos relevantes.
- Evitar duplicados consecutivos excesivos.
- Permitir revisar el balance de clases.
- Validar que las muestras no contengan valores inválidos.
- Guardar los datos en CSV o un formato equivalente fácil de inspeccionar.

Empezá con un piloto de 3–5 personas y comprobá que las cinco clases pueden distinguirse. Después ampliá la recolección.

Como objetivo inicial, recopilá aproximadamente 200–500 muestras por gesto y por participante, distribuidas entre varias sesiones. Ajustá la cantidad según la diversidad y la calidad de los datos, no solo por el número de filas.

Incluí variaciones realistas de distancia, orientación, velocidad de ejecución e iluminación. No recolectes todas las muestras con una única postura estática y un solo fondo.

Conservá la identidad de cada participante mediante un identificador no personal, por ejemplo `participant_01`. No guardes video ni imágenes completas salvo que sean necesarios y exista autorización. Por defecto, guardá únicamente las características derivadas de los landmarks.

## 8. Modelos que hay que entrenar

Entrená y compará estos tres clasificadores:

### A. Random Forest — modelo de referencia

Usá `sklearn.ensemble.RandomForestClassifier`.

Es el primer modelo que debe funcionar. Tiene una buena relación entre simplicidad, velocidad y capacidad de modelar relaciones no lineales.

Probá algunos valores razonables de `n_estimators`, `max_depth` y `min_samples_leaf`, evitando una búsqueda excesiva.

### B. SVM con kernel RBF — candidato principal

Usá `sklearn.svm.SVC`.

Escalá las características mediante un `Pipeline` de scikit-learn. Ajustá `C` y `gamma` utilizando validación cruzada agrupada por participante cuando el tamaño de los datos lo permita.

Si necesitás probabilidades, evaluá `probability=True` o una calibración apropiada. No confundas las probabilidades estimadas con certezas reales.

### C. MLP — red neuronal pequeña

Usá `sklearn.neural_network.MLPClassifier`.

Probá una arquitectura pequeña y regularización. Evitá redes profundas innecesarias para un dataset de landmarks de tamaño moderado.

Compará los tres modelos con el mismo esquema de validación y el mismo conjunto de prueba.

No asumas que uno será superior antes de medirlo.

## 9. Entrenamiento y evaluación rigurosa

Implementá un pipeline de entrenamiento reproducible.

1. Validar y limpiar el dataset.
2. Verificar que todas las clases tengan suficientes muestras.
3. Dividir los datos por participante o sesión.
4. Ajustar el preprocesamiento únicamente con los datos de entrenamiento.
5. Entrenar los modelos.
6. Ajustar hiperparámetros usando exclusivamente los datos de entrenamiento y validación.
7. Seleccionar el mejor modelo con una métrica definida previamente.
8. Evaluar una sola vez el modelo seleccionado en el conjunto de prueba reservado.
9. Guardar resultados, configuración y versión del dataset.

**Prohibido dividir aleatoriamente fotogramas correlacionados de una misma sesión entre entrenamiento y prueba y presentar ese resultado como generalización a personas nuevas.**

Usá `GroupShuffleSplit`, `GroupKFold` o una estrategia equivalente para mantener separados los participantes. Si la cantidad de participantes es insuficiente, declaralo explícitamente y no afirmes que el modelo generaliza a usuarios desconocidos.

Métricas obligatorias:

- Accuracy.
- Precision, recall y F1 por clase.
- F1 macro.
- Matriz de confusión.
- Tiempo de inferencia.
- FPS de la aplicación.
- Latencia desde la captura hasta la predicción.

Reportá también ejemplos de errores frecuentes, como confundir pulgar arriba con puño cerrado.

No inventes resultados ni fijes métricas de éxito antes de medirlas.

Como objetivo inicial de ingeniería, buscá un F1 macro superior al 90 % en un conjunto de prueba representativo, pero priorizá una evaluación honesta. Si no se alcanza, identificá las causas y proponé mejoras.

## 10. Reglas del juego

Implementá partidas configurables para dos o tres jugadores.

Cada jugador tendrá una región fija de la imagen: izquierda, centro y derecha.

Flujo de una ronda:

1. Mostrar una cuenta regresiva.
2. Mostrar un gesto objetivo aleatorio.
3. Esperar una respuesta correcta y estable de cada jugador.
4. Registrar el primer instante válido.
5. Asignar puntos según el orden de respuesta.
6. Mostrar quién ganó la ronda.
7. Esperar un breve intervalo antes de iniciar la siguiente.

Usá como configuración inicial:

- Primer jugador correcto: 100 puntos.
- Segundo jugador correcto: 60 puntos.
- Tercer jugador correcto: 30 puntos.
- Respuesta incorrecta: 0 puntos.
- Gesto detectado antes de la señal de inicio: penalización configurable.

Aceptá una predicción únicamente si se mantiene estable durante un intervalo configurable, inicialmente alrededor de 100–200 ms. Investigá si un umbral de confianza o un filtro temporal mejora el resultado sin agregar demasiada latencia.

Evaluá a todos los jugadores sobre el mismo fotograma o instante lógico. Evitá que el orden secuencial de procesamiento favorezca a una región de la imagen.

Si hay empate dentro de la resolución temporal del sistema, registralo como empate en lugar de inventar un orden.

## 11. Asociación de manos y jugadores

La aplicación debe intentar soportar tres participantes simultáneos.

Usá la posición de la mano y las regiones de la imagen para asignar cada mano al jugador correspondiente.

Consideraciones:

- No asignar automáticamente todas las manos al primer jugador detectado.
- Evitar que una mano que cruza otra región genere una puntuación para otro participante.
- Mostrar una advertencia si no se detecta la mano de un jugador.
- Permitir ajustar las regiones de cada jugador.
- Probar con personas de distintas alturas y distancias.

MediaPipe puede detectar varias manos, pero no des por garantizada una detección fiable de tres participantes en cualquier condición. Probá el límite de manos, la iluminación y el campo de visión de la cámara. Si la webcam no permite una separación fiable, implementá un modo de calibración y explicá sus limitaciones.

## 12. Interfaz y experiencia de usuario

La interfaz debe ser visual, legible desde cierta distancia y apta para una feria.

Pantallas:

1. Menú inicial y selección de dos o tres jugadores.
2. Calibración de cámara y regiones.
3. Pantalla de partida con gesto objetivo, jugadores y puntuaciones.
4. Resultado de cada ronda.
5. Tabla final de posiciones.
6. Panel experimental para comparar modelos y métricas.

Mostrá una representación visual de los landmarks cuando el modo técnico esté activado.

No bloquees el hilo de la interfaz mientras se captura la cámara o se realiza la inferencia. Medí el rendimiento en la máquina real.

La aplicación debe funcionar sin conexión a Internet una vez descargadas las dependencias y los recursos necesarios.

## 13. Pruebas obligatorias

Incluí pruebas automatizadas para:

- Normalización de landmarks.
- Orden y dimensiones de las características.
- Entrenamiento y carga de modelos.
- Predicción de las cinco clases.
- Ausencia de mano detectada.
- Estabilidad temporal de las predicciones.
- Asignación de manos a jugadores.
- Cálculo de puntuaciones.
- Empates y penalizaciones.
- Inicio y finalización de rondas.
- Recuperación ante desconexión de la webcam.

Probá manualmente con dos y tres personas. Medí los errores de asociación, las predicciones falsas y la latencia completa.

## 14. Entregables

Al finalizar, el repositorio debe contener:

- Código fuente modular.
- Archivo de dependencias con versiones compatibles.
- Instrucciones de instalación y ejecución.
- Herramienta de recolección de datos.
- Dataset propio o instrucciones reproducibles para generarlo.
- Scripts de entrenamiento y evaluación.
- Modelos entrenados y metadatos.
- Resultados reales de los experimentos.
- Matrices de confusión y gráficos.
- Pruebas automatizadas.
- README en español.
- Guion breve para demostrar el juego en la feria.
- Explicación de la arquitectura, los modelos, las métricas y las limitaciones.

No incluyas archivos pesados, videos ni datasets innecesarios en Git. Proporcioná instrucciones claras para generarlos localmente.

## 15. Orden de implementación

Trabajá en este orden y verificá cada fase antes de avanzar:

**Fase 1:** cámara, detección de manos y visualización de landmarks.

**Fase 2:** recolector de datos y dataset etiquetado.

**Fase 3:** entrenamiento, validación y comparación de Random Forest, SVM y MLP.

**Fase 4:** predictor en tiempo real con el modelo seleccionado.

**Fase 5:** asociación de jugadores, reglas, puntuación y rondas.

**Fase 6:** interfaz visual, calibración y feedback.

**Fase 7:** pruebas con tres participantes, optimización de latencia y corrección de errores.

**Fase 8:** documentación, resultados experimentales y preparación de la demostración.

En cada fase, informá qué implementaste, qué pruebas ejecutaste, qué resultados obtuviste y qué falta. Si algo falla, diagnosticá la causa antes de agregar complejidad.

## Criterio de finalización

El proyecto se considera terminado cuando tres personas pueden jugar con una sola webcam, el sistema reconoce los gestos solicitados, las puntuaciones son consistentes, el modelo seleccionado fue evaluado con datos separados de entrenamiento y la aplicación funciona de forma estable sin conexión a Internet.

Priorizá primero una versión funcional y medible. Los efectos visuales avanzados, los modos adicionales y las animaciones deben añadirse después de garantizar el funcionamiento del núcleo del juego.
