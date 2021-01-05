import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Read the input files
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train = df_train.append(df_test , ignore_index = True)

# Inspection of dataframe
print('Shape: \n', df_train.shape)
print('Columns: \n', df_train.columns)

# Separate features and target
y_train = df_train['SalePrice']
x_train = df_train.drop(['SalePrice', 'Id'], axis=1)
columns = x_train.columns

# Manually separate quantitative and categorical features
quant = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
			'1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
			'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
			'3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
categ = [element for element in list(columns) if element not in quant]

# Replace empty spaces for NaN
x_train[quant] = x_train[quant].replace('', np.nan)


##########
## Numerical features
##########

# Check if quant features contain null values
for col in quant:
	NANvalues = x_train[col].isnull().sum()
	if NANvalues != 0:
		pcnt = NANvalues/x_train.shape[0]*100
		print(f'{col}: {NANvalues} -> {pcnt:{3}.{3}}%')
		if pcnt > 10:
			print(f'Removing {col} from features')
			x_train.drop([col], axis=1)

# From this we can see that only LotFrontage has a high % of entries missing (>10%). This feature will be dropped in the future.

# Filling NaN values in numerical features
# The following are general properties of houses so the NaN values should be replace with 0s (should do the same for x_test)
general_feat = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'TotRmsAbvGrd']

for feat in general_feat:
	if hasattr(x_train, feat):
		getattr(x_train, feat).fillna(0, inplace=True)

# The missing numerical features related to the porch will be set to 0s
porch_feat = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
for feat in porch_feat:
	if hasattr(x_train, feat):
		getattr(x_train, feat).fillna(0, inplace=True)

# Initialize dictionary to save median values to later used in the test set
feat_dict = dict()

# The following numerical features are related to area and missing entries will be replaced by the median value
area_feat = ['LotFrontage', 'LotArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GarageArea', 'GrLivArea']

for feat in area_feat:
	if hasattr(x_train, feat):
		feat_dict[feat] = getattr(x_train,feat).median()
		idx = x_train[x_train[feat].isnull()].index.tolist()
		x_train.loc[idx,[feat]] = feat_dict[feat]


# The following numerical features are related to years and will be set to the median value
year_feat = ['YearBuilt', 'YrSold', 'MoSold']
for feat in year_feat:
	if hasattr(x_train, feat):
		feat_dict[feat] = getattr(x_train,feat).median()
		idx = x_train[x_train[feat].isnull()].index.tolist()
		x_train.loc[idx,[feat]] = feat_dict[feat]



# print([element for element in list(quant) if element not in (general_feat+area_feat+porch_feat+year_feat)])

# Special feature cases: The missing entries will be filled depending on the value of other features

if hasattr(x_train, 'YearRemodAdd'):
	idx = x_train[x_train['YearRemodAdd'].isnull()].index.tolist()
	x_train.loc[idx,[feat]] = x_train.loc[idx,['YearBuilt']]

if hasattr(x_train, 'GarageYrBlt'):
	feat_dict['GarageYrBlt'] = getattr(x_train,'GarageYrBlt').median()
	idx = x_train[x_train['GarageYrBlt'].isnull()].index.tolist()
	for ID in idx:
		# print('hvjb')
		if pd.isna(list(x_train.loc[ID,['GarageFinish']])[0]):
			x_train.loc[ID,['GarageYrBlt']] = x_train.loc[ID,['YearBuilt']][0]
		else:
			x_train.loc[ID,['GarageYrBlt']] = feat_dict['GarageYrBlt']

if hasattr(x_train, 'PoolArea'):
	feat_dict['PoolArea'] = getattr(x_train,'PoolArea').median()
	idx = x_train[x_train['PoolArea'].isnull()].index.tolist()
	for ID in idx:
		if pd.isna(list(x_train.loc[ID,['PoolQC']])[0]):
			x_train.loc[ID,['PoolArea']] = 0
		else:
			x_train.loc[ID,['PoolArea']] = feat_dict['PoolArea']

if hasattr(x_train, 'MasVnrArea'):
	feat_dict['MasVnrArea'] = getattr(x_train,'MasVnrArea').median()
	idx = x_train[x_train['MasVnrArea'].isnull()].index.tolist()
	for ID in idx:
		if pd.isna(list(x_train.loc[ID,['MasVnrType']])[0]):
			x_train.loc[ID,['MasVnrArea']] = 0
		else:
			x_train.loc[ID,['MasVnrArea']] = feat_dict['MasVnrArea']
# print(x_train['MasVnrArea'])
if hasattr(x_train, 'MiscVal'):
	feat_dict['MiscVal'] = getattr(x_train,'MiscVal').median()
	idx = x_train[x_train['MiscVal'].isnull()].index.tolist()
	for ID in idx:
		if pd.isna(list(x_train.loc[ID,['MiscFeature']])[0]):
			x_train.loc[ID,['MiscVal']] = 0
		else:
			x_train.loc[ID,['MiscVal']] = feat_dict['MiscVal']

if hasattr(x_train, 'GarageCars'):
	feat_dict['GarageCars'] = getattr(x_train,'GarageCars').median()
	idx = x_train[x_train['GarageCars'].isnull()].index.tolist()
	for ID in idx:
		if pd.isna(list(x_train.loc[ID,['GarageFinish']])[0]) or list(x_train.loc[ID,['GarageArea']]) == 0:
			x_train.loc[ID,['GarageCars']] = 0
		else:
			x_train.loc[ID,['GarageCars']] = feat_dict['GarageCars']

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

if hasattr(x_train, 'BsmtFinSF1'):
	feat_dict['BsmtFinSF1'] = getattr(x_train,'BsmtFinSF1').median()
	idx = x_train[x_train['BsmtFinSF1'].isnull()].index.tolist()
	for ID in idx:
		if pd.isna(list(x_train.loc[ID,['BsmtFinType1']])[0]):
			x_train.loc[ID,['BsmtFinSF1']] = 0
		else:
			x_train.loc[ID,['BsmtFinSF1']] = feat_dict['BsmtFinSF1']

if hasattr(x_train, 'BsmtFinSF2'):
	feat_dict['BsmtFinSF2'] = getattr(x_train,'BsmtFinSF2').median()
	idx = x_train[x_train['BsmtFinSF2'].isnull()].index.tolist()
	for ID in idx:
		if pd.isna(list(x_train.loc[ID,['BsmtFinType2']])[0]):
			x_train.loc[ID,['BsmtFinSF2']] = 0
		else:
			x_train.loc[ID,['BsmtFinSF2']] = feat_dict['BsmtFinSF2']

if hasattr(x_train, 'BsmtUnfSF'):
	idx = x_train[x_train['BsmtUnfSF'].isnull()].index.tolist()
	for ID in idx:
		if list(x_train.loc[ID,['BsmtFinSF1']])[0] >= 0 and list(x_train.loc[ID,['BsmtFinSF2']])[0] >= 0 and list(x_train.loc[ID,['TotalBsmtSF']])[0] >= 0:
			x_train.loc[ID,['BsmtUnfSF']] = list(x_train.loc[ID,['TotalBsmtSF']])[0] - list(x_train.loc[ID,['BsmtFinSF2']])[0] - list(x_train.loc[ID,['BsmtFinSF1']])[0]


# Get cleaned quantitative data
# x_numerical = x_train[quant]
x_train['AgeGarage'] = abs(x_train.YrSold - x_train.GarageYrBlt)
x_train = x_train.drop(['GarageYrBlt'],axis=1)
quant += ['AgeGarage']
quant.remove('GarageYrBlt')

# Visualize correlations to control weights stability
# sns.heatmap(x_numerical.corr(method='pearson').abs(), cmap="Reds", annot=True)#, yticklabels=patient_ids)
# plt.show()

# There are a few highly correlated features, the ones with a correlation > 0.9 will be dropped
def get_top_abs_correlations(df, threshold, print_pairwise_corr):
	pairs = []
	au_corr = df.corr(method='spearman').abs().unstack()
	labels_to_drop = get_redundant_pairs(df)
	au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
	if print_pairwise_corr:
		print(au_corr.where(au_corr > threshold).dropna())
	for x in au_corr.where(au_corr > threshold).dropna().axes[0]:
		pairs.append(list(x))
	return pairs

def get_redundant_pairs(df):
	'''Get diagonal and lower triangular pairs of correlation matrix'''
	pairs_to_drop = set()
	cols = df.columns
	for i in range(0, df.shape[1]):
		for j in range(0, i+1):
			pairs_to_drop.add((cols[i], cols[j]))
	return pairs_to_drop

corr_feat = get_top_abs_correlations(x_train[quant], threshold=0.9, print_pairwise_corr=True)

# Graph the correlated features
for pair in corr_feat:
	plt.scatter(x_numerical[pair[0]], x_numerical[pair[1]])
	plt.show()


##########
## Categorical features
##########

#Print column names
print(categ)

no_feat = ['Street', 'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
			'GarageQual', 'GarageCond', 'Fence', 'MiscFeature', 'PoolQC']
rest_categ = [element for element in categ if element not in no_feat]

# Replace empty spaces for NaN
x_train.MasVnrType.fillna('None', inplace=True)
x_train.Exterior2nd.fillna('None', inplace=True)
for col in no_feat:
	getattr(x_train, col).fillna('NoFeat', inplace=True)
for col in rest_categ:
	getattr(x_train, col).fillna(getattr(x_train, col).mode()[0], inplace=True)


for col in categ:
	NANvalues = x_train[col].isnull().sum()
	if NANvalues != 0:
		pcnt = NANvalues/x_train.shape[0]*100
		print(f'{col}: {NANvalues} -> {pcnt:{3}.{3}}%')
		if pcnt > 10:
			print(f'Removing {col} from features')
			x_train.drop([col], axis=1)


# print(pd.get_dummies(x_train['Street'],drop_first=True))