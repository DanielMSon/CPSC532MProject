# Comprison of ensemble and base models results
This is to document the results of base models and ensemble methods

**cross validation is disabled**
**shuffle data is disabled**

## Results with `train_sig_features.csv`
---
**Error type**: `rmsle`
### Base model results

```python
base model: Lasso (L2-loss with L1-reg)
training error: 0.0201946
validation error: 0.0203211

base model: ElasticNet (L2-loss with L1-reg and L2-reg)
training error: 0.0201219
validation error: 0.0203422

base model: Ridge (L2-loss with L2_reg)
training error: 0.00895386
validation error: 0.00936157

base model: KNN regression
training error: 0.0177876
validation error: 0.0200699

base model: lightGBM
training error: 0.006158
validation error: 0.0106854

base model: Random Forest Regressor
training error: 0.00426537
validation error: 0.0113757

base model: Neural Net 
Finished training
training error: 0.0306303
validation error: 0.0313356
```

### Averaging
Base models: `Lasso`, `ENet`, `Ridge`, `KNN`, `lightGBM`, `RandomForest`

Error type: `rmsle`

```python
training error: 0.011357
validation error: 0.0129759
```

### Stacking
Base models: `Lasso`, `ENet`, `Ridge`, `KNN`, `lightGBM`, `RandomForest`

Meta model: `Ridge`

Error type: `rmsle`

```python
training error: 0.00408947
validation error: 0.0106458
```

## Results with `train_preprocessed.csv`
**Error type:** `rmsle`

### base models
```python
base model: Lasso (L2-loss with L1-reg)
training error: 0.0201599
validation error: 0.0203606

base model: ElasticNet (L2-loss with L1-reg and L2-reg)
training error: 0.0200842
validation error: 0.0204571

base model: Ridge (L2-loss with L2_reg)
training error: 0.00745222
validation error: 0.0106175

base model: KNN regression
training error: 0.019817
validation error: 0.0222131

base model: lightGBM
training error: 0.00554193
validation error: 0.0100432

base model: Random Forest Regressor
training error: 0.00437127
validation error: 0.0111306

base model: Neural Net
Finished training
training error: 0.0310979
validation error: 0.0294411
```
### averaging
Base models: `Lasso`, `ENet`, `Ridge`, `KNN`, `lightGBM`, `RandomForest`

```python
training error: 0.011352
validation error: 0.0132176
```
### stacking
Base models: `Lasso`, `ENet`, `Ridge`, `KNN`, `lightGBM`, `RandomForest`

Meta model: `Ridge`

```python
training error: 0.00412168
validation error: 0.0107669
```

## Results with `train_anova_features.csv`

**Error type: **`rmsle`

### base models

```python
base model: Lasso (L2-loss with L1-reg)
training error: 0.020196
validation error: 0.020128

base model: ElasticNet (L2-loss with L1-reg and L2-reg)
training error: 0.0201205
validation error: 0.0201991

base model: Ridge (L2-loss with L2_reg)
training error: 0.0102534
validation error: 0.0118944

base model: KNN regression
training error: 0.0176754
validation error: 0.0198482

base model: lightGBM
training error: 0.00629813
validation error: 0.0109798

base model: Random Forest Regressor
training error: 0.00448805
validation error: 0.0117554

base model: Neural Net
Finished training
training error: 0.0382662
validation error: 0.038512
```

### averaging

```python
training error: 0.0116373
validation error: 0.0135665
```

### stacking

```python
training error: 0.00388359
validation error: 0.0118588
```

## Results with `train_xgb_features.csv`

### base models

```python
base model: Lasso (L2-loss with L1-reg)
training error: 0.0201599
validation error: 0.0203606

base model: ElasticNet (L2-loss with L1-reg and L2-reg)
training error: 0.0200842
validation error: 0.0204571

base model: Ridge (L2-loss with L2_reg)
training error: 0.00922542
validation error: 0.010935

base model: KNN regression
training error: 0.0198266
validation error: 0.0222135

base model: lightGBM
training error: 0.0058971
validation error: 0.010087

base model: Random Forest Regressor
training error: 0.00461
validation error: 0.0108392

base model: Neural Net
Finished training
training error: 0.0342029
validation error: 0.0334209
```

### averaging

```python
training error: 0.0117008
validation error: 0.0133417
```

### stacking

```python
training error: 0.00424704
validation error: 0.0110392
```