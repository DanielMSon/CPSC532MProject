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
