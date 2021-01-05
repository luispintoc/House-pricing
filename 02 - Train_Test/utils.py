import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import *

# neural networks
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Model, Sequential

# Recycle functions from the explore script

# Preprocessing of numerical features
def numerical_feat_preprocessing(x_train, x_test, quant):
	# Filling NaN values in numerical features
	# The following are general properties of houses so the NaN values should be replace with 0s (should do the same for x_test)
	general_feat = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'TotRmsAbvGrd']

	for feat in general_feat:
		if hasattr(x_train, feat):
			getattr(x_train, feat).fillna(0, inplace=True)
			getattr(x_test, feat).fillna(0, inplace=True)

	# The missing numerical features related to the porch will be set to 0s
	porch_feat = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
	for feat in porch_feat:
		if hasattr(x_train, feat):
			getattr(x_train, feat).fillna(0, inplace=True)
			getattr(x_test, feat).fillna(0, inplace=True)

	# Initialize dictionary to save median values to later used in the test set
	feat_dict = dict()

	# The following numerical features are related to area and missing entries will be replaced by the median value
	area_feat = ['LotFrontage', 'LotArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GarageArea', 'GrLivArea']

	for feat in area_feat:
		if hasattr(x_train, feat):
			feat_dict[feat] = getattr(x_train,feat).median()

			idx = x_train[x_train[feat].isnull()].index.tolist()
			x_train.loc[idx,[feat]] = feat_dict[feat]

			idx = x_test[x_test[feat].isnull()].index.tolist()
			x_test.loc[idx,[feat]] = feat_dict[feat]


	# The following numerical features are related to years and will be set to the median value
	year_feat = ['YearBuilt', 'YrSold', 'MoSold']
	for feat in year_feat:
		if hasattr(x_train, feat):
			feat_dict[feat] = getattr(x_train,feat).median()

			idx = x_train[x_train[feat].isnull()].index.tolist()
			x_train.loc[idx,[feat]] = feat_dict[feat]

			idx = x_test[x_test[feat].isnull()].index.tolist()
			x_test.loc[idx,[feat]] = feat_dict[feat]

	# Special feature cases: The missing entries will be filled depending on the value of other features

	if hasattr(x_train, 'YearRemodAdd'):
		idx = x_train[x_train['YearRemodAdd'].isnull()].index.tolist()

		x_train.loc[idx,[feat]] = x_train.loc[idx,['YearBuilt']]
		x_test.loc[idx,[feat]] = x_test.loc[idx,['YearBuilt']]

	if hasattr(x_train, 'GarageYrBlt'):
		feat_dict['GarageYrBlt'] = getattr(x_train,'GarageYrBlt').median()

		idx = x_train[x_train['GarageYrBlt'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_train.loc[ID,['GarageFinish']])[0]):
				value = x_train.loc[ID,['YearBuilt']][0]
				x_train.loc[ID,['GarageYrBlt']] = x_train.loc[ID,['YearBuilt']][0]
			else:
				x_train.loc[ID,['GarageYrBlt']] = feat_dict['GarageYrBlt']

		idx = x_test[x_test['GarageYrBlt'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_test.loc[ID,['GarageFinish']])[0]):
				value = x_test.loc[ID,['YearBuilt']][0]
				x_test.loc[ID,['GarageYrBlt']] = x_test.loc[ID,['YearBuilt']][0]
			else:
				x_test.loc[ID,['GarageYrBlt']] = feat_dict['GarageYrBlt']

	if hasattr(x_train, 'PoolArea'):
		feat_dict['PoolArea'] = getattr(x_train,'PoolArea').median()

		idx = x_train[x_train['PoolArea'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_train.loc[ID,['PoolQC']])[0]):
				x_train.loc[ID,['PoolArea']] = 0
			else:
				x_train.loc[ID,['PoolArea']] = feat_dict['PoolArea']

		idx = x_test[x_test['PoolArea'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_test.loc[ID,['PoolQC']])[0]):
				x_test.loc[ID,['PoolArea']] = 0
			else:
				x_test.loc[ID,['PoolArea']] = feat_dict['PoolArea']

	if hasattr(x_train, 'MasVnrArea'):
		feat_dict['MasVnrArea'] = getattr(x_train,'MasVnrArea').median()

		idx = x_train[x_train['MasVnrArea'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_train.loc[ID,['MasVnrType']])[0]):
				x_train.loc[ID,['MasVnrArea']] = 0
			else:
				x_train.loc[ID,['MasVnrArea']] = feat_dict['MasVnrArea']

		idx = x_test[x_test['MasVnrArea'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_test.loc[ID,['MasVnrType']])[0]):
				x_test.loc[ID,['MasVnrArea']] = 0
			else:
				x_test.loc[ID,['MasVnrArea']] = feat_dict['MasVnrArea']

	if hasattr(x_train, 'MiscVal'):
		feat_dict['MiscVal'] = getattr(x_train,'MiscVal').median()

		idx = x_train[x_train['MiscVal'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_train.loc[ID,['MiscFeature']])[0]):
				x_train.loc[ID,['MiscVal']] = 0
			else:
				x_train.loc[ID,['MiscVal']] = feat_dict['MiscVal']

		idx = x_test[x_test['MiscVal'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_test.loc[ID,['MiscFeature']])[0]):
				x_test.loc[ID,['MiscVal']] = 0
			else:
				x_test.loc[ID,['MiscVal']] = feat_dict['MiscVal']

	if hasattr(x_train, 'GarageCars'):
		feat_dict['GarageCars'] = getattr(x_train,'GarageCars').median()

		idx = x_train[x_train['GarageCars'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_train.loc[ID,['GarageFinish']])[0]) or list(x_train.loc[ID,['GarageArea']]) == 0:
				x_train.loc[ID,['GarageCars']] = 0
			else:
				x_train.loc[ID,['GarageCars']] = feat_dict['GarageCars']

		idx = x_test[x_test['GarageCars'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_test.loc[ID,['GarageFinish']])[0]) or list(x_test.loc[ID,['GarageArea']]) == 0:
				x_test.loc[ID,['GarageCars']] = 0
			else:
				x_test.loc[ID,['GarageCars']] = feat_dict['GarageCars']

	if hasattr(x_train, 'TotalBsmtSF'):
		feat_dict['TotalBsmtSF'] = getattr(x_train,'TotalBsmtSF').median()

		idx = x_train[x_train['TotalBsmtSF'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_train.loc[ID,['BsmtFinType1']])[0]) and pd.isna(list(x_train.loc[ID,['BsmtFinType2']])[0]):
				x_train.loc[ID,['TotalBsmtSF']] = 0
			elif list(x_train.loc[ID,['BsmtFinSF1']])[0] != 0 or pd.notna(list(x_train.loc[ID,['BsmtFinType1']])[0]) or list(x_train.loc[ID,['BsmtFinSF2']])[0] != 0 or pd.notna(list(x_train.loc[ID,['BsmtFinType2']])[0]) or list(x_train.loc[ID,['BsmtUnfSF']])[0] != 0:
				if list(x_train.loc[ID,['BsmtFinSF1']])[0] >= 0 and list(x_train.loc[ID,['BsmtFinSF2']])[0] >= 0 and list(x_train.loc[ID,['BsmtUnfSF']])[0] >= 0:
					x_train.loc[ID,['TotalBsmtSF']] = list(x_train.loc[ID,['BsmtFinSF1']])[0]+list(x_train.loc[ID,['BsmtFinSF2']])[0]+list(x_train.loc[ID,['BsmtUnfSF']])[0]
			else:
				x_train.loc[ID,['TotalBsmtSF']] = feat_dict['TotalBsmtSF']

		idx = x_test[x_test['TotalBsmtSF'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_test.loc[ID,['BsmtFinType1']])[0]) and pd.isna(list(x_test.loc[ID,['BsmtFinType2']])[0]):
				x_test.loc[ID,['TotalBsmtSF']] = 0
			elif list(x_test.loc[ID,['BsmtFinSF1']])[0] != 0 or pd.notna(list(x_test.loc[ID,['BsmtFinType1']])[0]) or list(x_test.loc[ID,['BsmtFinSF2']])[0] != 0 or pd.notna(list(x_test.loc[ID,['BsmtFinType2']])[0]) or list(x_test.loc[ID,['BsmtUnfSF']])[0] != 0:
				if list(x_test.loc[ID,['BsmtFinSF1']])[0] >= 0 and list(x_test.loc[ID,['BsmtFinSF2']])[0] >= 0 and list(x_test.loc[ID,['BsmtUnfSF']])[0] >= 0:
					x_test.loc[ID,['TotalBsmtSF']] = list(x_test.loc[ID,['BsmtFinSF1']])[0]+list(x_test.loc[ID,['BsmtFinSF2']])[0]+list(x_test.loc[ID,['BsmtUnfSF']])[0]
			else:
				x_test.loc[ID,['TotalBsmtSF']] = feat_dict['TotalBsmtSF']

	if hasattr(x_train, 'BsmtFinSF1'):
		feat_dict['BsmtFinSF1'] = getattr(x_train,'BsmtFinSF1').median()

		idx = x_train[x_train['BsmtFinSF1'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_train.loc[ID,['BsmtFinType1']])[0]):
				x_train.loc[ID,['BsmtFinSF1']] = 0
			else:
				x_train.loc[ID,['BsmtFinSF1']] = feat_dict['BsmtFinSF1']

		idx = x_test[x_test['BsmtFinSF1'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_test.loc[ID,['BsmtFinType1']])[0]):
				x_test.loc[ID,['BsmtFinSF1']] = 0
			else:
				x_test.loc[ID,['BsmtFinSF1']] = feat_dict['BsmtFinSF1']

	if hasattr(x_train, 'BsmtFinSF2'):
		feat_dict['BsmtFinSF2'] = getattr(x_train,'BsmtFinSF2').median()

		idx = x_train[x_train['BsmtFinSF2'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_train.loc[ID,['BsmtFinType2']])[0]):
				x_train.loc[ID,['BsmtFinSF2']] = 0
			else:
				x_train.loc[ID,['BsmtFinSF2']] = feat_dict['BsmtFinSF2']

		idx = x_test[x_test['BsmtFinSF2'].isnull()].index.tolist()
		for ID in idx:
			if pd.isna(list(x_test.loc[ID,['BsmtFinType2']])[0]):
				x_test.loc[ID,['BsmtFinSF2']] = 0
			else:
				x_test.loc[ID,['BsmtFinSF2']] = feat_dict['BsmtFinSF2']

	if hasattr(x_train, 'BsmtUnfSF'):
		idx = x_train[x_train['BsmtUnfSF'].isnull()].index.tolist()
		for ID in idx:
			if list(x_train.loc[ID,['BsmtFinSF1']])[0] >= 0 and list(x_train.loc[ID,['BsmtFinSF2']])[0] >= 0 and list(x_train.loc[ID,['TotalBsmtSF']])[0] >= 0:
				x_train.loc[ID,['BsmtUnfSF']] = list(x_train.loc[ID,['TotalBsmtSF']])[0] - list(x_train.loc[ID,['BsmtFinSF2']])[0] - list(x_train.loc[ID,['BsmtFinSF1']])[0]

		idx = x_test[x_test['BsmtUnfSF'].isnull()].index.tolist()
		for ID in idx:
			if list(x_test.loc[ID,['BsmtFinSF1']])[0] >= 0 and list(x_test.loc[ID,['BsmtFinSF2']])[0] >= 0 and list(x_test.loc[ID,['TotalBsmtSF']])[0] >= 0:
				x_test.loc[ID,['BsmtUnfSF']] = list(x_test.loc[ID,['TotalBsmtSF']])[0] - list(x_test.loc[ID,['BsmtFinSF2']])[0] - list(x_test.loc[ID,['BsmtFinSF1']])[0]


	# Create new feature and remove one
	x_train['AgeGarage'] = abs(x_train.YrSold - x_train.GarageYrBlt)
	x_train = x_train.drop(['GarageYrBlt'],axis=1)
	x_test['AgeGarage'] = abs(x_test.YrSold - x_test.GarageYrBlt)
	x_test = x_test.drop(['GarageYrBlt'],axis=1)
	quant += ['AgeGarage']
	quant.remove('GarageYrBlt')

	return x_train, x_test, quant


# Preprocessing of categorical features
def categorical_feat_preprocessing(x_train, x_test, categ):

	no_feat = ['Street', 'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
			'GarageQual', 'GarageCond', 'Fence', 'MiscFeature', 'PoolQC']
	rest_categ = [element for element in categ if element not in no_feat]

	# Replace empty spaces for NaN
	x_train.MasVnrType.fillna('None', inplace=True)
	x_train.Exterior2nd.fillna('None', inplace=True)
	x_test.MasVnrType.fillna('None', inplace=True)
	x_test.Exterior2nd.fillna('None', inplace=True)
	for col in no_feat:
		getattr(x_train, col).fillna('NoFeat', inplace=True)
		getattr(x_test, col).fillna('NoFeat', inplace=True)
	for col in rest_categ:
		mode = getattr(x_train, col).mode()[0]
		getattr(x_train, col).fillna(mode, inplace=True)
		getattr(x_test, col).fillna(mode, inplace=True)

	# Get dummy variables
	size = x_train.shape[0]
	df  = x_train.append(x_test , ignore_index = True)
	df = pd.get_dummies(df, columns=categ, drop_first=True)
	x_train = df[:size]
	x_test = df[size:]

	return x_train, x_test, categ


# Function to obtain predictions for ensemble
def train_predict(model, x_train, y_train, x_test):
	if model == 'xgb':
		x1_train, _, y1_train, _ = train_test_split(x_train, y_train, test_size=0.30, random_state=42)
		best = {'learning_rate': 0.041356493620612826, 'max_depth': 3, 'n_estimators': 1000, 'reg_lambda': 0.08820214287735169}
		trained_clf = XGBRegressor().set_params(**best).fit(x1_train, y1_train)
		feature_train = trained_clf.predict(x_train)
		feature_test = trained_clf.predict(x_test)

	if model == 'lasso':
		x1_train, _, y1_train, _ = train_test_split(x_train, y_train, test_size=0.30, random_state=4)
		best = {'alpha': 0.0008538772922010828, 'max_iter': 50000}
		trained_clf = Lasso().set_params(**best).fit(x1_train, y1_train)
		feature_train = trained_clf.predict(x_train)
		feature_test = trained_clf.predict(x_test)

	if model == 'elastic':
		x1_train, _, y1_train, _ = train_test_split(x_train, y_train, test_size=0.30, random_state=41)
		best = {'alpha': 0.0015514158509780324, 'l1_ratio': 0.009220199702920819, 'max_iter': 10000}
		trained_clf = ElasticNet().set_params(**best).fit(x1_train, y1_train)
		feature_train = trained_clf.predict(x_train)
		feature_test = trained_clf.predict(x_test)

	if model == 'nn':
		model = Sequential()
		model.add(tf.keras.Input(shape=(x_train.shape[1],)))
		model.add(Dense(units=40, kernel_initializer='uniform', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003)))
		model.add(Dropout(0.1))
		model.add(Dense(units=80, kernel_initializer='uniform', activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(units=1, kernel_initializer='uniform', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0003)))
		model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
		history = model.fit(x_train, y_train, batch_size=64, epochs=1500, validation_split=0.2)
		feature_train = model.predict(x_train).reshape(x_train.shape[0])
		feature_test = model.predict(x_test).reshape(x_test.shape[0])


	return feature_train, feature_test