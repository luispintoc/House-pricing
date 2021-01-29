# House-pricing


## Overview
Code for the House Pricing Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview). Best model ranks top 30% of the competition. I used feature ranking, Bayesian optimization and built an ensemble of 3 models (Lasso, XGB, NN) with L2 normalization.


## Prerequisites
The packages used are tensorflow and sklearn

## Files
**input**: Folder storing the dataset

**Exploratory Data Analysis.ipynb**: EDA jupyter notebook to examine the dataset and stablish guidelines for preprocessing

**main.ipynb**: Scripts to perform preprocessing, feature eng, feature selection, train models and output predictions

## Models
Models were trained using Bayesian optimization. The models were then ensembled using L2 normalization.
