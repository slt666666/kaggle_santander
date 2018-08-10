import numpy as np
import pandas as pd
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
