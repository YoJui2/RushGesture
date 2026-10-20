# Informe de evaluacion - Gesture Rush

- Mejor modelo: mlp
- Participantes en train: participant_02
- Participantes en validacion: participant_03
- Participantes en test: participant_01

## random_forest - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.6556
- f1_macro: 0.5997
- f1_weighted: 0.5902
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.750     0.017     0.033       180
     open_palm      0.894     0.995     0.942       196
pointing_index      0.453     0.740     0.562       177
     thumbs_up      0.577     0.896     0.702       125
       victory      0.857     0.682     0.759       132

      accuracy                          0.656       810
     macro avg      0.706     0.666     0.600       810
  weighted avg      0.711     0.656     0.590       810
```

## svm_rbf - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.5198
- f1_macro: 0.4949
- f1_weighted: 0.4988
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.850     0.283     0.425       180
     open_palm      0.916     0.949     0.932       196
pointing_index      0.019     0.011     0.014       177
     thumbs_up      0.318     0.896     0.470       125
       victory      0.787     0.530     0.633       132

      accuracy                          0.520       810
     macro avg      0.578     0.534     0.495       810
  weighted avg      0.592     0.520     0.499       810
```

## mlp - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.7136
- f1_macro: 0.6427
- f1_weighted: 0.6449
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.900     0.050     0.095       180
     open_palm      0.947     1.000     0.973       196
pointing_index      0.695     0.966     0.809       177
     thumbs_up      0.610     0.864     0.715       125
       victory      0.553     0.712     0.623       132

      accuracy                          0.714       810
     macro avg      0.741     0.718     0.643       810
  weighted avg      0.765     0.714     0.645       810
```

## mlp - TEST (reservado)
- conjunto: test reservado, evaluado una sola vez
- accuracy: 0.6853
- f1_macro: 0.6783
- f1_weighted: 0.6777
- inferencia: 0.0005 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.765     0.510     0.612       557
     open_palm      0.683     0.825     0.748       332
pointing_index      0.777     0.497     0.606       364
     thumbs_up      0.649     0.803     0.718       695
       victory      0.644     0.785     0.708       381

      accuracy                          0.685      2329
     macro avg      0.704     0.684     0.678      2329
  weighted avg      0.701     0.685     0.678      2329
```
