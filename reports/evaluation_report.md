# Informe de evaluacion - Gesture Rush

- Mejor modelo: svm_rbf
- Participantes en train: participant_03, participant_04
- Participantes en validacion: participant_01
- Participantes en test: participant_02

## random_forest - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.8282
- f1_macro: 0.8286
- f1_weighted: 0.8233
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.870     0.645     0.741       622
       l_shape      0.701     0.961     0.811       515
       ok_sign      0.916     1.000     0.956       456
     open_palm      0.991     0.923     0.956       454
      pinky_up      0.899     0.891     0.895       512
pointing_index      0.865     0.798     0.830       475
          rock      0.940     0.949     0.944       591
         shaka      0.549     0.838     0.664       525
     thumbs_up      0.789     0.379     0.512       514
       victory      0.993     0.962     0.977       451

      accuracy                          0.828      5115
     macro avg      0.851     0.835     0.829      5115
  weighted avg      0.848     0.828     0.823      5115
```

## svm_rbf - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.9228
- f1_macro: 0.9245
- f1_weighted: 0.9212
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.783     0.900     0.838       622
       l_shape      0.952     0.957     0.955       515
       ok_sign      0.972     0.993     0.983       456
     open_palm      0.932     1.000     0.965       454
      pinky_up      0.940     0.955     0.948       512
pointing_index      0.931     0.943     0.937       475
          rock      0.984     0.959     0.972       591
         shaka      0.868     0.901     0.884       525
     thumbs_up      0.939     0.656     0.772       514
       victory      0.996     0.989     0.992       451

      accuracy                          0.923      5115
     macro avg      0.930     0.925     0.924      5115
  weighted avg      0.926     0.923     0.921      5115
```

## mlp - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.8655
- f1_macro: 0.8663
- f1_weighted: 0.8621
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.890     0.712     0.791       622
       l_shape      0.889     0.981     0.933       515
       ok_sign      0.948     0.996     0.971       456
     open_palm      0.905     0.989     0.945       454
      pinky_up      0.848     0.707     0.771       512
pointing_index      0.852     0.874     0.863       475
          rock      0.853     0.956     0.902       591
         shaka      0.688     0.907     0.782       525
     thumbs_up      0.950     0.630     0.758       514
       victory      0.933     0.962     0.948       451

      accuracy                          0.865      5115
     macro avg      0.876     0.871     0.866      5115
  weighted avg      0.874     0.865     0.862      5115
```

## svm_rbf - TEST (reservado)
- conjunto: test reservado, evaluado una sola vez
- accuracy: 0.8516
- f1_macro: 0.8497
- f1_weighted: 0.8531
- inferencia: 0.3350 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.706     0.779     0.741       385
       l_shape      0.919     0.788     0.848       490
       ok_sign      0.810     0.884     0.845       550
     open_palm      0.923     0.960     0.941       551
      pinky_up      0.780     0.879     0.827       489
pointing_index      0.734     0.869     0.796       525
          rock      0.889     0.826     0.856       437
         shaka      0.911     0.816     0.861       477
     thumbs_up      0.924     0.865     0.894       490
       victory      0.972     0.817     0.888       552

      accuracy                          0.852      4946
     macro avg      0.857     0.848     0.850      4946
  weighted avg      0.860     0.852     0.853      4946
```
