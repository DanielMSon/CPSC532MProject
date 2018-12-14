# CPSC532M Course Project
## Description

## Contributor
[name] [stu#]

## Base Models and Stacking
The following base models will be implemented individually and at the end will be ensembled by stacking

### Base models
1. sklearn.linear_model.Lasso (L2-loss with L1-reg)
   $f(w) = \frac{1}{2*n_{samples}} ||y - Xw||^2_2 + \alpha ||w||_1$
   
   |Param: | Representation in $f$|
   |`alpha` |$\alpha$|

2. sklearn.linear_model.ElasticNet (L2-loss with L1-reg and L2-reg)
   $f(w) = \frac{1}{2*n_{samples}} ||y- Xw||^2_2 + \alpha \lambda_1 ||w||_1 + 0.5\alpha(1-\lambda_1)||w||^2_2 $
   `l1_ratio` = 
