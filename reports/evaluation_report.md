# Informe de evaluacion - Gesture Rush

- Mejor modelo: mlp
- Participantes en train: participant_02
- Participantes en validacion: participant_03
- Participantes en test: participant_01

## random_forest - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.7695
- f1_macro: 0.7659
- f1_weighted: 0.7640
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.871     0.883     0.877       436
       l_shape      0.839     0.696     0.761       434
       ok_sign      0.760     0.960     0.848       474
     open_palm      0.954     0.733     0.829       484
      pinky_up      0.608     0.803     0.692       471
pointing_index      0.691     0.801     0.742       478
          rock      0.713     0.813     0.760       459
         shaka      0.687     0.381     0.490       483
     thumbs_up      0.768     0.919     0.837       433
       victory      0.943     0.731     0.823       472

      accuracy                          0.769      4624
     macro avg      0.783     0.772     0.766      4624
  weighted avg      0.783     0.769     0.764      4624
```

## svm_rbf - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.7695
- f1_macro: 0.7632
- f1_weighted: 0.7631
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.782     0.830     0.805       436
       l_shape      0.662     0.668     0.665       434
       ok_sign      0.792     0.973     0.873       474
     open_palm      0.955     0.870     0.910       484
      pinky_up      0.719     0.809     0.761       471
pointing_index      0.656     0.611     0.633       478
          rock      0.731     0.882     0.800       459
         shaka      0.852     0.429     0.570       483
     thumbs_up      0.707     0.910     0.796       433
       victory      0.930     0.731     0.819       472

      accuracy                          0.769      4624
     macro avg      0.779     0.771     0.763      4624
  weighted avg      0.781     0.769     0.763      4624
```

## mlp - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.7727
- f1_macro: 0.7686
- f1_weighted: 0.7672
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.914     0.828     0.869       436
       l_shape      0.882     0.671     0.762       434
       ok_sign      0.645     0.964     0.773       474
     open_palm      0.843     0.876     0.859       484
      pinky_up      0.650     0.516     0.575       471
pointing_index      0.731     0.726     0.728       478
          rock      0.758     0.861     0.806       459
         shaka      0.822     0.518     0.635       483
     thumbs_up      0.739     0.949     0.831       433
       victory      0.860     0.835     0.847       472

      accuracy                          0.773      4624
     macro avg      0.784     0.774     0.769      4624
  weighted avg      0.783     0.773     0.767      4624
```

## mlp - TEST (reservado)
- conjunto: test reservado, evaluado una sola vez
- accuracy: 0.8070
- f1_macro: 0.8036
- f1_weighted: 0.8047
- inferencia: 0.0010 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.965     0.836     0.896       622
       l_shape      0.620     0.955     0.752       515
       ok_sign      0.719     1.000     0.837       456
     open_palm      0.930     0.733     0.820       454
      pinky_up      0.749     0.578     0.653       512
pointing_index      0.871     0.798     0.833       475
          rock      0.940     0.936     0.938       591
         shaka      0.672     0.524     0.589       525
     thumbs_up      0.787     0.776     0.782       514
       victory      0.932     0.942     0.937       451

      accuracy                          0.807      5115
     macro avg      0.819     0.808     0.804      5115
  weighted avg      0.821     0.807     0.805      5115
```
