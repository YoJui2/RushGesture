# Informe de evaluacion - Gesture Rush

- Mejor modelo: mlp
- Participantes en train: participant_03, participant_4
- Participantes en validacion: participant_01
- Participantes en test: participant_02

## random_forest - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.7689
- f1_macro: 0.7538
- f1_weighted: 0.7578
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.648     0.827     0.726       872
     open_palm      0.856     0.997     0.921       651
pointing_index      0.718     0.386     0.502       665
     thumbs_up      0.851     0.809     0.829       974
       victory      0.785     0.793     0.789       663

      accuracy                          0.769      3825
     macro avg      0.772     0.763     0.754      3825
  weighted avg      0.771     0.769     0.758      3825
```

## svm_rbf - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.7903
- f1_macro: 0.7702
- f1_weighted: 0.7765
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.712     0.888     0.790       872
     open_palm      0.739     0.994     0.848       651
pointing_index      0.867     0.383     0.532       665
     thumbs_up      0.865     0.815     0.839       974
       victory      0.849     0.834     0.842       663

      accuracy                          0.790      3825
     macro avg      0.807     0.783     0.770      3825
  weighted avg      0.806     0.790     0.777      3825
```

## mlp - validacion
- conjunto: validacion (seleccion de hiperparametros)
- accuracy: 0.8340
- f1_macro: 0.8275
- f1_weighted: 0.8289
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.791     0.847     0.818       872
     open_palm      0.852     0.940     0.894       651
pointing_index      0.775     0.564     0.653       665
     thumbs_up      0.847     0.860     0.854       974
       victory      0.894     0.944     0.919       663

      accuracy                          0.834      3825
     macro avg      0.832     0.831     0.828      3825
  weighted avg      0.831     0.834     0.829      3825
```

## mlp - TEST (reservado)
- conjunto: test reservado, evaluado una sola vez
- accuracy: 0.6845
- f1_macro: 0.6746
- f1_weighted: 0.6803
- inferencia: 0.0009 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.532     0.706     0.607       296
     open_palm      0.910     0.670     0.772       303
pointing_index      0.667     0.916     0.772       475
     thumbs_up      0.720     0.530     0.610       321
       victory      0.728     0.528     0.612       396

      accuracy                          0.685      1791
     macro avg      0.712     0.670     0.675      1791
  weighted avg      0.709     0.685     0.680      1791
```
