# Validacion cruzada agrupada por participante - Gesture Rush

- Metodo: Leave-One-Participant-Out (cada participante es test una vez)
- Participantes: participant_01, participant_02, participant_03
- Muestras: 4930
- Validacion interna: al menos 1 participante(s) de los restantes
- Los hiperparametros se eligen solo con participantes de train.

## Resultados por fold

| Fold | Test | Train | Validacion | Mejor modelo | F1 validacion | Accuracy test | F1 macro test |
|---|---|---|---|---|---|---|---|
| 1 | participant_01 | participant_02 | participant_03 | mlp | 0.643 | 0.685 | 0.678 |
| 2 | participant_02 | participant_01 | participant_03 | mlp | 0.810 | 0.599 | 0.598 |
| 3 | participant_03 | participant_01 | participant_02 | svm_rbf | 0.621 | 0.696 | 0.676 |

## Agregado en test

- accuracy: 0.6600 +/- 0.0437
- f1_macro: 0.6508 +/- 0.0370
- f1_weighted: 0.6559 +/- 0.0384

## F1 por clase (media +/- desviacion en test)

| Clase | F1 | Precision | Recall | Soporte |
|---|---|---|---|---|
| open_palm | 0.808 +/- 0.100 | 0.808 | 0.816 | 831 |
| fist | 0.657 +/- 0.121 | 0.708 | 0.630 | 1033 |
| thumbs_up | 0.581 +/- 0.098 | 0.500 | 0.740 | 1141 |
| victory | 0.678 +/- 0.076 | 0.708 | 0.662 | 909 |
| pointing_index | 0.531 +/- 0.143 | 0.738 | 0.456 | 1016 |

## Confusiones frecuentes (matriz acumulada)

- fist -> thumbs_up: 293
- pointing_index -> thumbs_up: 242
- pointing_index -> victory: 138
- thumbs_up -> fist: 114
- victory -> pointing_index: 102

Matriz de confusion acumulada: `cv_confusion_matrix.png`

> Para el modelo final de despliegue, reentrena con todos los participantes usando `scripts/run_train.py`.
