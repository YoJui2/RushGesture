# Informe de evaluacion - Gesture Rush

- Mejor modelo: mlp
- Participantes en train: participant_03, participant_4
- Participantes en validacion: participant_01
- Participantes en test: participant_02

## random_forest - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.7158
- f1_macro: 0.6505
- f1_weighted: 0.6752
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.616     0.912     0.735       557
     open_palm      0.789     0.994     0.880       332
pointing_index      0.841     0.102     0.181       364
     thumbs_up      0.811     0.783     0.796       695
       victory      0.668     0.651     0.660       381

      accuracy                          0.716      2329
     macro avg      0.745     0.688     0.651      2329
  weighted avg      0.743     0.716     0.675      2329
```

## svm_rbf - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.7304
- f1_macro: 0.6558
- f1_weighted: 0.6866
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.681     0.917     0.782       557
     open_palm      0.630     1.000     0.773       332
pointing_index      0.912     0.085     0.156       364
     thumbs_up      0.835     0.777     0.805       695
       victory      0.774     0.753     0.763       381

      accuracy                          0.730      2329
     macro avg      0.766     0.707     0.656      2329
  weighted avg      0.771     0.730     0.687      2329
```

## mlp - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.8171
- f1_macro: 0.7928
- f1_weighted: 0.8003
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.793     0.941     0.860       557
     open_palm      0.811     0.997     0.895       332
pointing_index      0.850     0.357     0.503       364
     thumbs_up      0.804     0.819     0.811       695
       victory      0.875     0.916     0.895       381

      accuracy                          0.817      2329
     macro avg      0.826     0.806     0.793      2329
  weighted avg      0.821     0.817     0.800      2329
```

## mlp - TEST (reservado)
- conjunto: test reservado, evaluado una sola vez
- accuracy: 0.6622
- f1_macro: 0.6515
- f1_weighted: 0.6578
- inferencia: 0.0009 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.531     0.686     0.599       296
     open_palm      0.875     0.670     0.759       303
pointing_index      0.684     0.876     0.768       475
     thumbs_up      0.613     0.517     0.561       321
       victory      0.664     0.500     0.571       396

      accuracy                          0.662      1791
     macro avg      0.674     0.650     0.651      1791
  weighted avg      0.674     0.662     0.658      1791
```
