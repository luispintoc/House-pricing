import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import *
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.svm import *
from sklearn.ensemble import *
from xgboost import XGBRegressor
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.kernel_approximation import *
from sklearn.metrics import roc_curve, auc, roc_auc_score, make_scorer, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from collections import *
from networkx.algorithms.components.connected import connected_components
from scipy import *
import sys, os, warnings, shap, statistics, math
import networkx
from mlxtend.plotting import plot_decision_regions
from hyperopt import fmin, tpe, hp, anneal, Trials
import xgboost as xgb
from hyperopt.pyll.base import scope



def classifier_hyperopt(classifier, x, y, cv, seed):

	if classifier == 'xgb':

		def objective(hyperparams, random_state=seed, cv=cv, x=x, y=y):
			eclf = XGBRegressor(objective ='reg:squarederror') 
			model = eclf.set_params(**hyperparams)
			score = -1.0*cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-2).mean() - 1.0*cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-2).std()

			return score

		space = {
		 'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(1)),
		 'max_depth': scope.int(hp.choice('max_depth', [3, 4])),
 		 'n_estimators': scope.int(hp.choice('n_estimators', [250, 500, 1000])),
		 'reg_lambda': hp.loguniform('reg_lambda', np.log(0.0001), np.log(10)),
		}

	if classifier == 'lasso':

		def objective(hyperparams, random_state=seed, cv=cv, x=x, y=y):
			eclf = Lasso()   
			model = eclf.set_params(**hyperparams)
			score = -1.0*cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-2).mean() - 1.0*cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-2).std()

			return score

		space = {
		 'alpha': hp.loguniform('alpha', np.log(0.0001), np.log(10)),
		 'max_iter': scope.int(hp.choice('max_iter', [10000, 25000, 50000]))
		 }
		

	if classifier == 'elastic':

		def objective(hyperparams, random_state=seed, cv=cv, x=x, y=y):
			eclf = ElasticNet() 
			model = eclf.set_params(**hyperparams)
			score = -1.0*cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-2).mean()

			return score

		space = {
		 'alpha': hp.loguniform('C', np.log(0.0001), np.log(1)),
		 'l1_ratio': hp.uniform('l1_ratio', 0.001, 1),
		 'max_iter': scope.int(hp.choice('max_iter', [10000, 50000]))
		}


	if classifier == 'ridge':

		def objective(hyperparams, random_state=seed, cv=cv, x=x, y=y):
			eclf = KernelRidge(kernel='polynomial') 
			model = eclf.set_params(**hyperparams)
			score = -1.0*cross_val_score(model, x, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-2).mean()

			return score

		space = {
		 'alpha': hp.loguniform('alpha', np.log(0.01), np.log(100)),
		 'degree': hp.choice('degree', [2, 3]),
		 'coef0': hp.uniform('coef0', -10, 10)
		}




	return objective, space



'''

	if classifier == 'LinearSVC':
		eclf = LinearSVC(class_weight='balanced', dual=False, max_iter=3000)
		parameters_grid = {
					'C': (0.05, 0.1, 0.5),
					'penalty': ('l1','l2'),
					# 'loss': ('hinge','squared_hinge'),
					} 

	if classifier == 'ada':
		eclf = AdaBoostClassifier(SVC(gamma='auto', probability=True, decision_function_shape='ovo', kernel='rbf'), random_state=200)
		# eclf = AdaBoostClassifier(SGDClassifier(eta0=1, tol=0.001, loss='hinge', class_weight='balanced', max_iter=3000, n_jobs=-1, shuffle=True), algorithm='SAMME', random_state=200)
		parameters_grid = {
					'n_estimators': (30, 20, 50),
					'learning_rate': (0.01, 1),
					'base_estimator__C': (0.01, 0.1, 1),
					# 'base_estimator__kernel': ('linear', 'poly', 'rbf')
					# 'base_estimator__penalty': ('l1', 'l2'),
					# 'base_estimator__learning_rate': ('optimal', 'adaptive')
					}

	if classifier == 'SVM':
		eclf = SVC(gamma='auto', probability=True, decision_function_shape='ovo', kernel='rbf')#, class_weight=weight)
		parameters_grid = {
					'C': (0.001, 0.01, 0.1, 1),
					 'kernel': ('linear','rbf'),
					 # 'degree': (2,3),
					 # 'gamma': ('auto', 'scale', 1, 10),
					}

	if classifier == 'RF':
		eclf = RandomForestClassifier(class_weight='balanced')
		parameters_grid = {
					'n_estimators': (30, 40, 50, 100),
					'criterion': ('gini', 'entropy'),
					'min_samples_split': (2,3,4,5),
					# 'class_weight': ('balanced', 'balanced_subsample')
					'max_depth': (5,6,7,8)
					}

	if classifier == 'LDA':
		eclf = LinearDiscriminantAnalysis()
		parameters_grid = {
						'solver': ('svd', 'lsqr', 'eigen'),
					}

	if classifier == 'QDA':
		eclf = QuadraticDiscriminantAnalysis()
		parameters_grid = {
						'reg_param': (0.0, 0.1, 0.5),
					}

	if classifier == 'XGB':
		eclf = xgb.XGBClassifier(reg_lambda=1, eta=0.05, max_depth=3, sub_sample=0.8, colsample_bytree=0.5, objective='binary:logistic', eval_metric='auc', booster='gbtree', min_child_weight=10)
		parameters_grid = { 
					'xgb__learning_rate': (0.001, 0.01, 0.1, 1),
					'xgb__gamma' : (0, 0.5, 1, 5),
					}

	if classifier == 'GaussianNB':
		eclf = GaussianNB()
		parameters_grid = { 
					'var_smoothing': (1e-09, 1e-08)
					}

	if classifier == 'SGDClassifier':
		eclf = SGDClassifier(penalty='elasticnet')
		parameters_grid = { 
					# 'penalty': ('l1', 'l2', 'elasticnet'),
					'alpha': (0.001, 0.01, 0.1, 1)
					}

	if classifier == 'GBC':
		eclf = GradientBoostingClassifier(random_state=40, n_estimators=1000)
		parameters_grid = { 
					'learning_rate': (0.05, 0.1, 0.5, 1),
					'subsample': (0.5, 0.7, 1),
					'max_features': (None, 'auto')
					}


	if classifier == 'KNN':
		eclf = KNeighborsClassifier()
		parameters_grid = { 
					'n_neighbors': (3, 5, 8, 10),#, 30, 50),
					'weights': ('uniform', 'distance'),
					}

	if classifier == 'Logistic':
		eclf = LogisticRegression(C=100,  solver='liblinear', penalty='l1')#, class_weight=weight)
		parameters_grid = {
					'penalty': ('l1','l2'),
					'C': (0.01, 0.1, 1)
					}

	if classifier == 'ensemble':
		clf_svm = SVC(gamma='auto', probability=True, decision_function_shape='ovo', class_weight='balanced', kernel='linear')
		clf_rf = RandomForestClassifier(class_weight='balanced')
		clf_lg = LogisticRegression(penalty='l1', class_weight='balanced', random_state=40, solver='liblinear')
		eclf = VotingClassifier(estimators=[('clf_svm',clf_svm), ('clf_rf', clf_rf), ('clf_lg', clf_lg)])
		parameters_grid = { 
					'voting': ('soft','hard'),
					'clf_svm__C' : (0.1, 1, 10),
					'clf_svm__kernel' : ('linear','rbf','poly'),

					'clf_rf__n_estimators' : (80, 200),
					'clf_rf__max_depth': (5,6),

					'clf_lg__penalty': ('l1','l2'),
					'clf_lg__C': (1,10,100)
					}

	return eclf, parameters_grid

'''