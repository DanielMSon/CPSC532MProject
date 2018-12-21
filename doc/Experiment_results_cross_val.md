# Comprison of ensemble and base models results

**cross validation is enabled**
**shuffle data is enabled**

**Error type**: `rmsle`

Please note that training on NN will be much longer with `cross_val = True`

## `train_sig_features.csv`
### base models

```python
base model: Lasso (L2-loss with L1-reg)
training error: 0.0202131
validation error: 0.0202666

base model: ElasticNet (L2-loss with L1-reg and L2-reg)
training error: 0.0201344
validation error: 0.020214

base model: Ridge (L2-loss with L2_reg)
training error: 0.0089159
validation error: 0.00943031

base model: KNN regression
training error: 0.0180156
validation error: 0.0198337

base model: lightGBM
training error: 0.00612227
validation error: 0.0103963

base model: Random Forest Regressor
training error: 0.00427721
validation error: 0.0109499

base model: Neural Net
Finished training
Finished training
Finished training
Finished training
training error: 0.0389578
validation error: 0.0389011
```

### averaging

```python
training error: 0.011379
validation error: 0.0131146
```

### stacking

```python
training error: 0.00400108
validation error: 0.0110578
```

## `train_preprocessed.csv`
### base model
```python
base model: Lasso (L2-loss with L1-reg)
training error: 0.0201954
validation error: 0.0203207

base model: ElasticNet (L2-loss with L1-reg and L2-reg)
training error: 0.0201249
validation error: 0.0202717

base model: Ridge (L2-loss with L2_reg)
training error: 0.00746117
validation error: 0.0099562

base model: KNN regression
training error: 0.0199748
validation error: 0.022034

base model: lightGBM
training error: 0.00550191
validation error: 0.0101759

base model: Random Forest Regressor
training error: 0.00438192
validation error: 0.0113828

base model: Neural Net
Finished training
Finished training
Finished training
Finished training
training error: 0.0311183
validation error: 0.0311207
```

### averaging
```python
training error: 0.011373
validation error: 0.0134416
```

### stacking
```python
training error: 0.00409294
validation error: 0.0109057
```

## `train_anova_features.csv`
### base model
```python
base model: Lasso (L2-loss with L1-reg)
training error: 0.0202079
validation error: 0.0202833

base model: ElasticNet (L2-loss with L1-reg and L2-reg)
training error: 0.0201262
validation error: 0.0201519

base model: Ridge (L2-loss with L2_reg)
training error: 0.0104678
validation error: 0.0111663

base model: KNN regression
training error: 0.0177023
validation error: 0.0197469

base model: lightGBM
training error: 0.00629764
validation error: 0.0108658

base model: Random Forest Regressor
training error: 0.00462652
validation error: 0.011538

base model: Neural Net
Finished training
Finished training
Finished training
Finished training
training error: 0.0308377
validation error: 0.0307177
```

### averaging
```python
training error: 0.0116966
validation error: 0.013615
```

### stacking
```python
training error: 0.00382858
validation error: 0.0119691
```

## `train_xgb_features.csv`
### base model
```python
base model: Lasso (L2-loss with L1-reg)
training error: 0.0202056
validation error: 0.0202753

base model: ElasticNet (L2-loss with L1-reg and L2-reg)
training error: 0.0201343
validation error: 0.0201692

base model: Ridge (L2-loss with L2_reg)
training error: 0.00947945
validation error: 0.010213

base model: KNN regression
training error: 0.0199921
validation error: 0.0218417

base model: lightGBM
training error: 0.0057785
validation error: 0.0104195

base model: Random Forest Regressor
training error: 0.00452184
validation error: 0.0115467

base model: Neural Net
Finished training
Finished training
Finished training
Finished training
training error: 0.0315353
validation error: 0.0314741
```


### averaging
```python
training error: 0.0117196
validation error: 0.0138925
```

### stacking
```python
training error: 0.0041995
validation error: 0.0112177
```