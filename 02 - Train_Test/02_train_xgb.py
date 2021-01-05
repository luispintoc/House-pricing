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
importance = Lasso(alpha=0.001).fit(x_train, y_train).coef_
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


# Optimize model
classifier = 'xgb'
cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=100)
train = False

if train:

  # Classifier
  objective, space = classifier_hyperopt(classifier, x_train, y_train, cv, seed)

  # Trials will contain logging information
  trials = Trials()

  best = fmin(fn=objective, # function to optimize
        space=space, 
        algo=anneal.suggest, # optimization algorithm, hyperotp will select its parameters automatically
        max_evals=30, # maximum number of iterations
        trials=trials, # logging
        rstate=np.random.RandomState(seed), # fixing random state for the reproducibility
        return_argmin=False,
        show_progressbar=True,
       )

  print('\n Best: ', best)
  print('Scaler: ',importance_scaler)

  tpe_results=np.array([[
  					 -x['result']['loss'],
                        x['misc']['vals']['learning_rate'][0],
                        x['misc']['vals']['max_depth'][0],
                        x['misc']['vals']['n_estimators'][0],
                        x['misc']['vals']['reg_lambda'][0]
                        ]
                        for x in trials.trials])

  tpe_results_df=pd.DataFrame(tpe_results,
                             columns=['score', 'rate', 'max_depth', 'n_estimators', 'reg_lambda'])

  tpe_results_df.plot(subplots=True,figsize=(10, 10))

  plt.show()

else:
  best = {'learning_rate': 0.041356493620612826, 'max_depth': 3, 'n_estimators': 1000, 'reg_lambda': 0.08820214287735169}

trained_clf = XGBRegressor().set_params(**best).fit(x_train, y_train)
print(trained_clf.predict(x_test))
y_test = np.exp(trained_clf.predict(x_test))

# ss = pd.DataFrame({'Id': id_test, 'SalePrice': y_test})
# ss.to_csv('submit_xgb.csv', index=False)