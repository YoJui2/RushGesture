# Validacion cruzada agrupada por participante - Gesture Rush

- Metodo: Leave-One-Participant-Out (cada participante es test una vez)
- Participantes: participant_01, participant_02, participant_03, participant_4
- Muestras: 7423
- Validacion interna: al menos 1 participante(s) de los restantes
- Los hiperparametros se eligen solo con participantes de train.

## Resultados por fold

| Fold | Test | Train | Validacion | Mejor modelo | F1 validacion | Accuracy test | F1 macro test |
|---|---|---|---|---|---|---|---|
| 1 | participant_01 | participant_03, participant_4 | participant_02 | random_forest | 0.661 | 0.832 | 0.831 |
| 2 | participant_02 | participant_03, participant_4 | participant_01 | mlp | 0.828 | 0.685 | 0.675 |
| 3 | participant_03 | participant_02, participant_4 | participant_01 | svm_rbf | 0.868 | 0.801 | 0.803 |
| 4 | participant_4 | participant_02, participant_03 | participant_01 | random_forest | 0.774 | 0.733 | 0.737 |

## Agregado en test

- accuracy: 0.7628 +/- 0.0576
- f1_macro: 0.7613 +/- 0.0607
- f1_weighted: 0.7578 +/- 0.0597

## F1 por clase (media +/- desviacion en test)

| Clase | F1 | Precision | Recall | Soporte |
|---|---|---|---|---|
| open_palm | 0.873 +/- 0.070 | 0.899 | 0.861 | 1359 |
| fist | 0.616 +/- 0.134 | 0.635 | 0.614 | 1567 |
| thumbs_up | 0.779 +/- 0.098 | 0.774 | 0.796 | 1575 |
| victory | 0.798 +/- 0.111 | 0.857 | 0.756 | 1338 |
| pointing_index | 0.741 +/- 0.018 | 0.722 | 0.784 | 1584 |

## Confusiones frecuentes (matriz acumulada)

- fist -> pointing_index: 256
- thumbs_up -> fist: 195
- fist -> thumbs_up: 154
- pointing_index -> thumbs_up: 130
- pointing_index -> fist: 124

Matriz de confusion acumulada: `cv_confusion_matrix.png`

> Para el modelo final de despliegue, reentrena con todos los participantes usando `scripts/run_train.py`.
