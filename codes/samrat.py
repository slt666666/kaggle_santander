import numpy as np
import pandas as pd
import time
import gc

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from IPython.display import display

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# check Null values
print("Total Train Features with NaN Values = " + str(train_df.columns[train_df.isnull().sum() != 0].size))
if (train_df.columns[train_df.isnull().sum() != 0].size):
	print("Features with NaN => {}".format(list(train_df.columns[train_df.isnull().sum() != 0])))
	train_df[train_df.columns[train_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)

print("Total Test Features with NaN Values = " + str(test_df.columns[test_df.isnull().sum() != 0].size))
if (test_df.columns[test_df.isnull().sum() != 0].size):
        print("Features with NaN => {}".format(list(test_df.columns[test_df.isnull().sum() != 0])))
        test_df[test_df.columns[test_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)

# check and remove constant columns
colsToRemove = []
for col in train_df.columns:
	if col != 'ID' and col != 'target':
		if train_df[col].std() == 0:
			colsToRemove.append(col)

# remove constant columns in the training set
train_df.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
test_df.drop(colsToRemove, axis=1, inplace=True)

print("Removed '{}' Constant Columns\n".format(len(colsToRemove)))
#print(colsToRemove)

def duplicate_columns(frame):
	start = time.time()
	groups = frame.columns.to_series().groupby(frame.dtypes).groups
	dups = []

	for t, v, in groups.items():
		
		cs = frame[v].columns
		vs = frame[v]
		lcs = len(cs)

		for i in range(lcs):
			ia = vs.iloc[:, i].values
			for j in range(i+1, lcs):
				ja = vs.iloc[:, j].values
				if np.array_equal(ia, ja):
					dups.append(cs[i])
					break
	print(time.time() - start)
	return dups

colsToRemove = duplicate_columns(train_df)
print(colsToRemove)

# remove duplicate columns in the training set
train_df.drop(colsToRemove, axis=1, inplace=True)

# remove duplicate columns in the test set
test_df.drop(colsToRemove, axis=1, inplace=True)

print("Removed '{}' Duplicate Columns\n".format(len(colsToRemove)))

# drop sparse data
def drop_sparse(train, test):
	start = time.time()
	flist = [x for x in train.columns if not x in ['ID', 'target']]
	for f in flist:
		if len(np.unique(train[f])) < 2:
			train.drop(f, axis=1, inplace=True)
			test.drop(f, axis=1, inplace=True)
	print(time.time() - start)
	return train, test

train_df, test_df = drop_sparse(train_df, test_df)

gc.collect()
print("Train set size: {}".format(train_df.shape))
print("Test set size: {}".format(test_df.shape))

# Build Train and Test data
X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)

dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

# LightGBM
def run_lgb(train_X, train_y, val_X, val_y, test_X):
	params = {
		"objective": "regression",
		"metric": "rmse",
		"num_leaves": 40,
		"learning_rate": 0.005,
		"bagging_fraction": 0.6,
		"feature_fraction": 0.6,
		"bagging_frequency": 6,
		"bagging_seed": 42,
		"verbosity": -1,
		"seed": 42
		}

	lgtrain = lgb.Dataset(train_X, label=train_y)
	lgval = lgb.Dataset(val_X, label=val_y)
	evals_result = {}
	model = lgb.train(params, lgtrain, 5000,
			valid_sets=[lgtrain, lgval],
			early_stopping_rounds=100,
			verbose_eval=150,
			evals_result=evals_result)

	pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best.iteration))
	return pred_test_y, model, evals_result

# Training LGB
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
print("LightGBM Training Completed...")