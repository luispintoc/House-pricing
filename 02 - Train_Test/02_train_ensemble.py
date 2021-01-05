import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys
from utils import *
from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import *
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, RepeatedKFold
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.exceptions import DataConversionWarning
from hyperopt.pyll.base import scope
from hyperopt import fmin, tpe, hp, anneal, Trials
from classifiers_hyperopt import *
import warnings

# neural networks
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Model, Sequential


warnings.filterwarnings(action='ignore', category=FutureWarning)
seed = 1

# Read the input files
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

print('Shapes: \n', df_train.shape, df_test.shape)
print('Columns: \n', df_train.columns)

# Separate features and label
y_train = df_train['SalePrice']
x_train = df_train.drop(['SalePrice', 'Id'], axis=1)

id_test = df_test['Id']
x_test = df_test.drop(['Id'], axis=1)
columns = x_train.columns

# Convert target to log
y_train = np.log(y_train)

# Manually separate quantitative and categorical features
quant = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
			'1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
			'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
			'3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
categ = [element for element in list(columns) if element not in quant]

# Apply preprocessing to impute and avoid overfitting
x_train, x_test, quant = numerical_feat_preprocessing(x_train, x_test, quant)
x_train, x_test, categ = categorical_feat_preprocessing(x_train, x_test, categ)
print('\nPreprocessing applied')
print('New shapes: \n', x_train.shape, x_test.shape)

# Scaler
scaler = RobustScaler()
x_train[quant] = scaler.fit_transform(x_train[quant])
x_test[quant] = scaler.transform(x_test[quant])

# Obtain relevant features using L1 normalization
importance_xgb = XGBRegressor().fit(x_train, y_train).feature_importances_
imp_feat_xgb = []
for feat,weight in zip(x_train.columns, importance_xgb):
	if abs(weight) > 0: imp_feat_xgb.append(feat)

importance_lasso = Lasso(alpha=0.001).fit(x_train, y_train).coef_
imp_feat_lasso = []
for feat,weight in zip(x_train.columns, importance_lasso):
	if abs(weight) > 0: imp_feat_lasso.append(feat)

# Define metric
def nrmse(y_true, y_pred):
    return -1.0*np.sqrt(np.mean((y_true-y_pred)**2))

neg_rmse = make_scorer(nrmse)

# Get important features
x_train_xgb = x_train[imp_feat_xgb]
x_test_xgb = x_test[imp_feat_xgb]

x_train_lasso = x_train[imp_feat_lasso]
x_test_lasso = x_test[imp_feat_lasso]

feat1_train, feat1_test = train_predict('lasso', x_train_lasso, y_train, x_test_lasso) #LB score 0.14186
feat2_train, feat2_test = train_predict('xgb', x_train_lasso, y_train, x_test_lasso) #LB score 0.13166
feat3_train, feat3_test = train_predict('elastic', x_train_lasso, y_train, x_test_lasso) #LB score 0.14059
feat4_train, feat4_test = train_predict('nn', x_train_xgb, y_train, x_test_xgb) #LB score 0.16797

feat_train = pd.DataFrame({'lasso':feat1_train, 'xgb':feat2_train, 'nn':feat4_train})
feat_test = pd.DataFrame({'lasso':feat1_test, 'xgb':feat2_test, 'nn':feat4_test})

# Ensemble model
# model = Sequential()

# model.add(tf.keras.Input(shape=(feat_train.shape[1],)))
# model.add(Dense(units=20, kernel_initializer='uniform', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003)))
# model.add(Dropout(0.1))
# model.add(Dense(units=1, kernel_initializer='uniform', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003)))
# model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
# history = model.fit(feat_train, y_train, batch_size=64, epochs=200, validation_split=0.2)#, callbacks=[cp_callback])
# pred_train = model.predict(feat_train).reshape(feat_train.shape[0])
# # print(pred_train.reshape(feat_train.shape[0]))
# print('train rms: ', np.sqrt(mean_squared_error(y_train, pred_train)))

ensemble = Ridge(alpha=100).fit(feat_train, y_train)
print(ensemble.coef_)
print('train rms: ', np.sqrt(mean_squared_error(y_train, ensemble.predict(feat_train))))

ss = pd.DataFrame({'Id': id_test, 'SalePrice': ensemble.predict(feat_test)})
ss.to_csv('submit_ensemble.csv', index=False)