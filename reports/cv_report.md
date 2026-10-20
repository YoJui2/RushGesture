# Validacion cruzada agrupada por participante - Gesture Rush

- Metodo: Leave-One-Participant-Out (cada participante es test una vez)
- Participantes: participant_01, participant_02, participant_03, participant_4
- Muestras: 5927
- Validacion interna: al menos 1 participante(s) de los restantes
- Los hiperparametros se eligen solo con participantes de train.

## Resultados por fold

| Fold | Test | Train | Validacion | Mejor modelo | F1 validacion | Accuracy test | F1 macro test |
|---|---|---|---|---|---|---|---|
| 1 | participant_01 | participant_03, participant_4 | participant_02 | random_forest | 0.661 | 0.801 | 0.786 |
| 2 | participant_02 | participant_03, participant_4 | participant_01 | mlp | 0.793 | 0.662 | 0.651 |
| 3 | participant_03 | participant_02, participant_4 | participant_01 | svm_rbf | 0.822 | 0.800 | 0.802 |
| 4 | participant_4 | participant_02, participant_03 | participant_01 | random_forest | 0.762 | 0.651 | 0.664 |

## Agregado en test

- accuracy: 0.7285 +/- 0.0720
- f1_macro: 0.7259 +/- 0.0684
- f1_weighted: 0.7243 +/- 0.0721

## F1 por clase (media +/- desviacion en test)

| Clase | F1 | Precision | Recall | Soporte |
|---|---|---|---|---|
| open_palm | 0.834 +/- 0.083 | 0.853 | 0.835 | 1040 |
| fist | 0.615 +/- 0.150 | 0.613 | 0.624 | 1252 |
| thumbs_up | 0.727 +/- 0.106 | 0.689 | 0.789 | 1296 |
| victory | 0.774 +/- 0.118 | 0.829 | 0.735 | 1056 |
| pointing_index | 0.680 +/- 0.069 | 0.729 | 0.669 | 1283 |

## Confusiones frecuentes (matriz acumulada)

- pointing_index -> thumbs_up: 172
- thumbs_up -> fist: 168
- fist -> thumbs_up: 164
- fist -> pointing_index: 147
- victory -> pointing_index: 132

Matriz de confusion acumulada: `cv_confusion_matrix.png`

> Para el modelo final de despliegue, reentrena con todos los participantes usando `scripts/run_train.py`.
