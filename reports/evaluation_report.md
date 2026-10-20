# Informe de evaluacion - Gesture Rush

## Modelo: random_forest
- accuracy: 0.9927
- f1_macro: 0.9927
- f1_weighted: 0.9927
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      1.000     0.997     0.998       300
     open_palm      1.000     0.967     0.983       300
pointing_index      1.000     1.000     1.000       300
     thumbs_up      0.997     1.000     0.998       300
       victory      0.968     1.000     0.984       300

      accuracy                          0.993      1500
     macro avg      0.993     0.993     0.993      1500
  weighted avg      0.993     0.993     0.993      1500
```

## Modelo: svm_rbf
- accuracy: 0.9987
- f1_macro: 0.9987
- f1_weighted: 0.9987
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      0.993     1.000     0.997       300
     open_palm      1.000     1.000     1.000       300
pointing_index      1.000     0.993     0.997       300
     thumbs_up      1.000     1.000     1.000       300
       victory      1.000     1.000     1.000       300

      accuracy                          0.999      1500
     macro avg      0.999     0.999     0.999      1500
  weighted avg      0.999     0.999     0.999      1500
```

## Modelo: mlp
- accuracy: 1.0000
- f1_macro: 1.0000
- f1_weighted: 1.0000
- inferencia: 0.0000 ms/muestra

```
                precision    recall  f1-score   support

          fist      1.000     1.000     1.000       300
     open_palm      1.000     1.000     1.000       300
pointing_index      1.000     1.000     1.000       300
     thumbs_up      1.000     1.000     1.000       300
       victory      1.000     1.000     1.000       300

      accuracy                          1.000      1500
     macro avg      1.000     1.000     1.000      1500
  weighted avg      1.000     1.000     1.000      1500
```
