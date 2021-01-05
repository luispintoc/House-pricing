# House-pricing


## Overview
Code for the House Pricing Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview). Best model (ensemble) ranks top 10% of the competition.


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
