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
importance_scaler = RobustScaler()
x_train[quant] = importance_scaler.fit_transform(x_train[quant])
x_test[quant] = importance_scaler.transform(x_test[quant])

# Obtain relevant features using L1 normalization
importance = XGBRegressor().fit(x_train, y_train).feature_importances_
imp_feat = []
for feat,weight in zip(x_train.columns, importance):
	if abs(weight) > 0:
		imp_feat.append(feat)
print('There are %d important features' %(len(imp_feat)))


# Define metric
def nrmse(y_true, y_pred):
    return -1.0*np.sqrt(np.mean((y_true-y_pred)**2))

neg_rmse = make_scorer(nrmse)

# Get important features
x_train = x_train[imp_feat]
x_test = x_test[imp_feat]

checkpoint_path = "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Create nn model
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
print(model.summary())

history = model.fit(x_train, y_train, batch_size=64, epochs=1500, validation_split=0.2, callbacks=[cp_callback])
# model.load_weights('cp.ckpt')

pred_train = model.predict(x_train)
print(pred_train.reshape(x_train.shape[0]))
print('train rms: ', np.sqrt(mean_squared_error(y_train, pred_train)))

y_test = np.exp(model.predict(x_test).reshape(x_test.shape[0]))

# ss = pd.DataFrame({'Id': id_test, 'SalePrice': y_test})
# ss.to_csv('submit_nn.csv', index=False)