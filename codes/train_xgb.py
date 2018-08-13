import numpy as np
import pandas as pd
import gc
import math
from tqdm import tqdm

from sklearn import model_selection
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


# A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_pred, y):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# check Null values
print("Total Train Features with NaN Values = " + str(train_df.columns[train_df.isnull().sum() != 0].size))
if (train_df.columns[train_df.isnull().sum() != 0].size):
    print("Features with NaN => {}".format(list(train_df.columns[train_df.isnull().sum() != 0])))
    train_df[train_df.columns[train_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending=False)

print("Total Test Features with NaN Values = " + str(test_df.columns[test_df.isnull().sum() != 0].size))
if (test_df.columns[test_df.isnull().sum() != 0].size):
    print("Features with NaN => {}".format(list(test_df.columns[test_df.isnull().sum() != 0])))
    test_df[test_df.columns[test_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending=False)

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
# print(colsToRemove)


def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v, in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:, i].values
            for j in range(i + 1, lcs):
                ja = vs.iloc[:, j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
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
    flist = [x for x in train.columns if not x in ['ID', 'target']]
    for f in flist:
        if len(np.unique(train[f])) < 2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train, test


train_df, test_df = drop_sparse(train_df, test_df)

gc.collect()
print("Train set size: {}".format(train_df.shape))
print("Test set size: {}".format(test_df.shape))

print(train_df.head())

# Build Train and Test data
X_train = train_df.drop(["ID", "target"], axis=1)

y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)

all_params = {'max_depth': [3, 10],
            'learning_rate': [0.1],
            'min_child_weight': [3],
            'n_estimators': [10000],
            'colsample_bytree': [0.6, 0.8],
            'colsample_bylevel': [0.6, 0.8],
            'reg_alpha': [0.1],
            'max_delta_step': [0.1],
            'seed': [0],
            }

min_score = 100
min_params = None

for params in tqdm(list(ParameterGrid(all_params))):

    list_rmsle_score = []

    dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    clf = xgb.sklearn.XGBRegressor(**params)
    clf.fit(dev_X,
            dev_y,
            eval_set=[(val_X, val_y)],
            early_stopping_rounds=100,
            eval_metric=rmsle
            )

    pred = clf.predict(val_X, ntree_limit=clf.best_ntree_limit)
    print(pred.shape)
    sc_rmsle = rmsle(pred, val_y)

    list_rmsle_score.append(sc_rmsle)

    if min_score > sc_rmsle:
        print("min_score:{}".format(sc_rmsle))
        min_score = sc_rmsle
        min_params = params

clf = xgb.sklearn.XGBRegressor(**min_params)
clf.fit(X_train, y_train)
pred = clf.predict(val_X, ntree_limit=clf.best_ntree_limit)[:, 1]

# submission dataset
sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = pred

print(sub.head())
sub.to_csv('sub_lgb_xgb_cat.csv', index=False)
