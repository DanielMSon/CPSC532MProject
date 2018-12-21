
**error type** `rmse`

## Base models `preprocessed.csv`
```python
base model: Lasso (L2-loss with L1-reg)
training error: 0.102975
validation error: 0.126007

base model: ElasticNet (L2-loss with L1-reg and L2-reg)
training error: 0.099545
validation error: 0.12492

base model: Ridge (L2-loss with L2_reg)
training error: 0.0960509
validation error: 0.131437

base model: KNN regression
training error: 0.258876
validation error: 0.287137

base model: lightGBM
training error: 0.069691
validation error: 0.133933

base model: Random Forest Regressor
training error: 0.0562137
validation error: 0.149847

base model: Neural Net
Finished training
Finished training
Finished training
Finished training
training error: 0.404809
validation error: 0.404062
```

## Base model `sig_features.csv`
```python
base model: Lasso (L2-loss with L1-reg)
training error: 0.117096
validation error: 0.122728

base model: ElasticNet (L2-loss with L1-reg and L2-reg)
training error: 0.116204
validation error: 0.121008

base model: Ridge (L2-loss with L2_reg)
training error: 0.115912
validation error: 0.122282

base model: KNN regression
training error: 0.23413
validation error: 0.256431

base model: lightGBM
training error: 0.0779608
validation error: 0.132304

base model: Random Forest Regressor
training error: 0.0559845
validation error: 0.140036

base model: Neural Net
Finished training
Finished training
Finished training
Finished training
training error: 0.402055
validation error: 0.402791
```

## Base model `anova_features`
```python
base model: Lasso (L2-loss with L1-reg)
training error: 0.136867
validation error: 0.145095

base model: ElasticNet (L2-loss with L1-reg and L2-reg)
training error: 0.136267
validation error: 0.144432

base model: Ridge (L2-loss with L2_reg)
training error: 0.136066
validation error: 0.14405

base model: KNN regression
training error: 0.229458
validation error: 0.258556

base model: lightGBM
training error: 0.0806648
validation error: 0.143969

base model: Random Forest Regressor
training error: 0.0586503
validation error: 0.14907

base model: Neural Net
Finished training
Finished training
Finished training
Finished training
training error: 0.40772
validation error: 0.412814
```

## Base models `xgb_features.csv`
```python
base model: Lasso (L2-loss with L1-reg)
training error: 0.124424
validation error: 0.132998

base model: ElasticNet (L2-loss with L1-reg and L2-reg)
training error: 0.123656
validation error: 0.133395

base model: Ridge (L2-loss with L2_reg)
training error: 0.123486
validation error: 0.132934

base model: KNN regression
training error: 0.260188
validation error: 0.285058

base model: lightGBM
training error: 0.0740654
validation error: 0.137847

base model: Random Forest Regressor
training error: 0.0589216
validation error: 0.150447

base model: Neural Net
Finished training
Finished training
Finished training
Finished training
training error: 0.488344
validation error: 0.48475
```

## averaging ``preprocessed.csv`

```python
num features: 243
training error: 0.147319
validation error: 0.174396
```

## averaging `sig_features.csv`

```python
num features: 37
training error: 0.147826
validation error: 0.169646
```

## averaging `anova_features.csv`

```python
num features: 50
training error: 0.151607
validation error: 0.177616
```

## averaging `xgb_features.csv`

```python
num features: 49
training error: 0.152551
validation error: 0.177966
```

## stacking `preprocessed.csv`
```python
training error: 0.0942197
validation error: 0.105615

Kaggle implementation
training error: 0.0854358
validation error: 0.122935
```

## stacking `sig_features`

```python
training error: 0.104637
validation error: 0.11138

Kaggle implementation
training error: 0.101775
validation error: 0.119572
```

## stacking `anova_features`

```python
training error: 0.1101
validation error: 0.122596

Kaggle implementation
training error: 0.103227
validation error: 0.139682
```

## stacking `xgb_features`

```python
training error: 0.103657
validation error: 0.114712

Kaggle implementation
training error: 0.0994313
validation error: 0.126571