# House-pricing

*********************************************
Better and improve code will be uploaded soon
*********************************************


## Overview
Code for the House Pricing Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview). Best model ranks top 30% of the competition. I used feature ranking, Bayesian optimization and built an ensemble of 3 models (Lasso, XGB, NN) with L2 normalization.


## Prerequisites
The packages used are tensorflow, sklearn and hyperopt.

## Files
**input**: Hosts the dataset
**01 - Exploration**: Scripts to check missing entries, correlations, etc.
**02 - Train_Test**: Scripts to train models and output predictions

## Models
Models were trained using Bayesian optimization. The models were then ensembled using L2 normalization.

**Lasso** model: LB score 0.14186
**XGBRegressor** model: LB score 0.13166
**NN** model: LB score 0.16797
**Elastic** model: LB score 0.14059

**Ensemble** of the first three models: 
