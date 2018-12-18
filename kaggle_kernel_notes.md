# Notes
This is a note for the models and implementations from the source Kaggle kernels. 

Use [Markdown Preview Enhanced](https://marketplace.visualstudio.com/items?itemName=shd101wyy.markdown-preview-enhanced) to preview the equations in this `.md` file with VSCode.

## Source Kaggle Kernel
1. [Stacked Regressions: Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook)

## Base Models and Stacking
The following base models will be implemented individually and at the end will be ensembled by stacking

### Base models
1. sklearn.linear_model.Lasso (L2-loss with L1-reg)
    $f(w) = \frac{1}{2*n_{samples}} ||y - Xw||^2_2 + \alpha ||w||_1$

    |Param:|Representation in $f$|
    |------|---------------------|
    |`alpha` |$\alpha$|

2. sklearn.linear_model.ElasticNet (L2-loss with L1-reg and L2-reg)
    $f(w) = \frac{1}{2*n_{samples}} ||y- Xw||^2_2 + \alpha \lambda_1 ||w||_1 + 0.5\alpha (1-\lambda_1)||w||^2_2 $
    
    |Param:|Representation in $f$|
    |------|---------------------|
    |`alpha`|$\alpha$|
    |`l1_ratio`| $\lambda_1$|

3. sklearn.kernel_ridge.KernelRidge (L2-loss with L2-reg using kernel trick)
    $\min ||y - Xw||^2_2 + \alpha ||w||^2_2$

    |Param:|Representation in $f$|
    |------|---------------------|
    |`alpha` |$\alpha$|

4. sklearn.ensemble.GradientBoostingRegressor
    ensemble of weak estimators [Gradient Bosting](https://www.youtube.com/watch?v=sRktKszFmSk)
    ([regression trees](./doc/regression_tree.pdf))
5. xgboost.XGBRegressor
    ie. regularized gradient boosting

6. lightGBM
    something similar to XGB but varies in detailed implementation and results in faster training while achieving similar performace

### Base models to add:
1. knn regresion?
2. neural net regression?