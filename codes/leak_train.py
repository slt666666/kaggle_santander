import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb


# Feature Scoring using XGBoost with the leak feature
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** .5


def feature_check(_f):
    score = 0
    for trn_, val_ in fold_idx:
        reg.fit(
            data[['log_leak', _f]].iloc[trn_], target.iloc[trn_],
            eval_set=[(data[['log_leak', _f]].iloc[val_], target.iloc[val_])],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        score += rmse(target.iloc[val_], reg.predict(data[['log_leak', _f]].iloc[val_], ntree_limit=reg.best_ntree_limit)) / folds.n_splits
    return (_f, score)


if __name__ == "__main__":
    data = pd.read_csv('../input/train.csv')
    target = np.log1p(data['target'])

    # add train leak
    leak = pd.read_csv('../input/train_leak.csv')
    data['leak'] = leak['compiled_leak'].values
    data['log_leak'] = np.log1p(leak['compiled_leak'].values)

    id_list = ['e677c32f6', 'e9830321d', '1e3045c18', '500d02a95', 'e724ee2df',\
       '6aedd2ab2', '92c008ce8', '6fe308206', 'ceaeaab73', '459f7cf2c',\
       'd49a3bd3e', '5244e20bb', 'a69e918f6', '3c50adbbd', 'dfa739303',\
       '5a1ca6faa', '567a57827', '18d443c1c', '5f36db06e', 'c91a8b64e',\
       'eb850ef06', '9593c7453', '26846ac4f', '3b4e4b837', '8e8c910ce',\
       '3de34e632', 'f4e465c9e', '3bd8c8e64', 'fa2c3a369', '7c9e3d6f3',\
       '8ab31c3fb', '226c12d98', '39d5ead1c', '9e1898905', '36a44ee69',\
       '16a02e67a', '8deda4158', 'fba64a995', '73967117c', '73ff07930',\
       '963164f28', '6db4bc9a3', '883aa1994', '4e56fb41c', '59f52b75a',\
       '35b338a94', '376d7dced', '3e7b597ce', 'dd1d4cfb6', '46d18f7a2',\
       '4d7de0cce', '19e6e7f62', 'bd0e6071c', '3540b1ba4', '78f1570a3',\
       '437dc8330', '67c967a26', '08d27660b', '2e8ee92a7', '4618a456e',\
       '1a4fdc864', '52b53691c', '720936c42', '20c0383b4', '61759c2fa',\
       '6d44b1a24', '3f48ded44', '361b567d3', '674f03576', 'aac41aea0',\
       '5c28d45af', 'a97e61e60', 'effb2dde7', '859a58804', '51ef81700',\
       '67480f556', 'cd6e1d1a8', '19624675a', '944220048', '87bb51c8f',\
       '237a22a48', '9f390e6c6', '97dc5416f', '4c8501cc8', '097b5abba',\
       'fa93b66c1', '78060769a', '41553f9b6', '2d6fff5cc', 'a174b2d58',\
       '912a65190', '87daaf67b', '36ec31a95', '5ebfe2bc7', '8469f3db5',\
       '15defc71c', '4fa3167f8', '76908e43d', 'cf8cffe3e', '96d194d7e',\
       'a32d11963', 'bd6df221e', '1ded1e01e', '695a2b76f', '3f8870739',\
       'bf6c2d1ef', '6cec39e52', '7c26cb92c', '5370157bb', '0a6e926ca',\
       '9707e5877', '3dcfa3171', '21f574bbd', '38168399c', 'cf874b631',\
       '00fc78888', '89e157744', 'f0f780ca0', '7d1b89277', '3499dd5b0',\
       '6b889a750', '5bc038a90', 'ecd092724', '102bd04d8', 'b688e11af',\
       'de7c7c229', '8ab2bbf5e', 'e8e4f7325', '7a6169245', '26b4eaa5f',\
       'd8e48b069', 'e43756eb8', '6f80c2adf', '3f77a6bb0', 'b39db0d8f',\
       'a98906daa', 'fdf7658f0', 'cafed6bd3', 'e399e620f', 'c3f756a14',\
       'a8d84534c', '7862786dc', '077787b5d', 'cd963b675', '4b7b98488',\
       '0a26a4be2', 'c9b674211', 'c1ed3742b', '457945253', '4eb8437ce',\
       '0a48a49fa', 'c5e859554', '975915c85', 'a11d9b610', '311ddb270',\
       '2e1ce6102', 'f2b1766be', 'caa5e3db8', '42e8026b9', '1a2fcf1b6',\
       '917a29ba4', '012800ace', '16f48066a', '3d0d0348b', '18f550ac1',\
       'd98ae1f3d', 'b28bda093', '97ebb131c', '4a0a538e2', '4d0bc1992',\
       'c74b684e2', 'f1b2a29d9', 'f080bc65d', 'ba9c3a57e', '1b28e8107',\
       'be33e29b0', 'aed384d2b', 'ba5867b7e', '20eb12e4d', 'cc6c312e1',\
       '10e824dbb', '44aab2c6c', '52bfe707e', '113a0df16', 'e375d6c21',\
       'c431b0384', '95d12be90', 'f0a57697d', '22cd3d13a', '7ff66b22b',\
       '207d02e64', '505ab4508', '124807db0', '1b9611c35', '9288da1da',\
       '630408acf', '32fad0bb2', '96e9102e5', 'dbf23d64e', '227d78eaa',\
       '800c32a64', '797f9a515', '1653b322d', '8beb86ad8', 'c06f18630',\
       '4d91e39a3', 'cab0a7649', 'c0761cafc', '6ab247a6c', '22ad66f80',\
       '2e5e4e336', '18768ec17', 'a46af09b5', 'a42f80cda', '18516f2bb',\
       '9792e90d9', 'b7f09c81c', '0ff5ed246', 'e55eccee8', '6d486ea36',\
       '41f72b99d', 'bf55872c2', '030edce8a', 'db9ddeb12', '955bd6742',\
       'd102acc8e', '12b150758', 'a18e26b8e', 'a338559d8', 'de542d990',\
       '1df3ca92e', 'bafa5d85e', '9df7abe72', '9943726ae', '9b1674589',\
       'ccf7284d8', '480b43b1b', 'bbdb05bab', '385aef014', '9b169cee7',\
       '5049af5d6', '817f7bc97', 'e012e104f', '815281bbe', 'd0a52a9a5',\
       '6726fff18', '08799037c', '0cba9fad9', 'b7e681371', '7810d478c',\
       '448efbb28', '9db0f2cd6', 'c114630ea', '38d55f2c6', '7463cf8a3',\
       'f65e3fb69', 'e23a3e19f', '4d4f73a50', '8095db48d', '3f5a7464e',\
       '7af6415fd', '7ba151fb4', 'd8c97dd5d', '4a0d1739a', '3fb54ba3f',\
       '7f1320729', 'c155bcbb9', '7a5cd9bc4', '4b8754c3f', 'e6139f9c2',\
       '78c27fe37', '042cdc62a', '41a27f10e', '1ef05eb96', '874c9af39',\
       'cb7691896', 'cece97691', '307098375', 'd345ae5dd', '7a6a7a5a8',\
       'bb15a6577', 'a9b88188e', 'c55ed082a', '0cc925146', 'aec6a89a6',\
       'a5c7e6ca0', '263cd411e', '5fd4a3728', 'ac3bd1fa0', 'd0530c654',\
       '46344ce39', 'fef33cb02', '1d778bbac', 'c7d7d2865', '0b625d947',\
       '777e4c57c', '67f95d718', '9ef23b6e2', '491fa71a0', 'ce1794ace',\
       '8bb3b1dc2', 'e645db7a5', '79bb2501c', '984de132a', '2c3a4c8e4',\
       'c83d8952f', '5181084e0', '687a7ddb9', '360890554', '70131f3dd',\
       '6a359cc69', 'cdda19c21', '5ee96ad16', '74a7bd925', 'acb8c5936',\
       'df9fb5c82', '5e01a4774', 'd245d6f13', '1ae64cc48', '6409b6d6b',\
       '5b17bbba4', 'c8246807f', '350fd9a22', '41008080f', '4abd2e2ce',\
       'bb02ff774', 'a384b6ad1', 'b4db86aca', '20b56724d', 'c0b1bda7c',\
       '04b3efeb9', 'b84d43fc6', 'cf9c54e4f', 'f1a0883e9', '1902ba327',\
       '9b2633b79', 'fa5bb9ea6', '1e62607ab', '42e231555', '52ceee213',\
       'ae11f987d', 'b3133eae9', 'e190b24ea', '5a90094e1', '722e73831',\
       '7c44fba26', '5fefa441c', 'bc525dc95', 'c159aac41', '1b27b561b',\
       '24da6a83d', 'ce4411dcc', 'f4b169f75', '013842698', '26119f3aa',\
       '90143986a', 'b99f10b72', '0161ebeea', '1b7d3d6b5', '383d82870',\
       'f6c6b3960', 'fe808da78', '2585fcac9', '58a7b0b9b', '7e147b81f',\
       '8eda8b2e2', '98eb49a01', 'c216db16c']
    for id in id_list:
        data = data[data['ID'] != id]
    data.drop(['ID', 'target'], axis=1, inplace=True)

    reg = XGBRegressor(n_estimators=1000)
    folds = KFold(4, True, 134259)
    fold_idx = [(trn_, val_) for trn_, val_ in folds.split(data)]
    scores = []

    nb_values = data.nunique(dropna=False)
    nb_zeros = (data == 0).astype(np.uint8).sum(axis=0)

    # features = [f for f in data.columns if f not in ['log_leak', 'leak', 'target', 'ID']]
    # pool = mp.Pool(8)
    # scores = pool.map(feature_check, features)
    # pool.close()
    #
    # report = pd.DataFrame(scores, columns=['feature', 'rmse']).set_index('feature')
    # report['nb_zeros'] = nb_zeros
    # report['nunique'] = nb_values
    # report.sort_values(by='rmse', ascending=True, inplace=True)
    # report.to_csv('feature_report.csv', index=True)
    #
    # # select some features (threshold is not optimized)
    # good_features = report.loc[report['rmse'] <= 0.7925].index
    # rmses = report.loc[report['rmse'] <= 0.7925, 'rmse'].values

    test = pd.read_csv('../input/test.csv')

    # add leak to test
    tst_leak = pd.read_csv('../input/test_leak.csv')
    test['leak'] = tst_leak['compiled_leak']
    test['log_leak'] = np.log1p(tst_leak['compiled_leak'])

    # train LightGBM
    folds = KFold(n_splits=5, shuffle=True, random_state=1)

    # Use all features for stats
    features = [f for f in data if f not in ['ID', 'leak', 'log_leak', 'target']]
    data.replace(0, np.nan, inplace=True)
    data['log_of_mean'] = np.log1p(data[features].replace(0, np.nan).mean(axis=1))
    data['mean_of_log'] = np.log1p(data[features]).replace(0, np.nan).mean(axis=1)
    data['log_of_median'] = np.log1p(data[features].replace(0, np.nan).median(axis=1))
    data['nb_nans'] = data[features].isnull().sum(axis=1)
    data['the_sum'] = np.log1p(data[features].sum(axis=1))
    data['the_std'] = data[features].std(axis=1)
    data['the_kur'] = data[features].kurtosis(axis=1)

    test.replace(0, np.nan, inplace=True)
    test['log_of_mean'] = np.log1p(test[features].replace(0, np.nan).mean(axis=1))
    test['mean_of_log'] = np.log1p(test[features]).replace(0, np.nan).mean(axis=1)
    test['log_of_median'] = np.log1p(test[features].replace(0, np.nan).median(axis=1))
    test['nb_nans'] = test[features].isnull().sum(axis=1)
    test['the_sum'] = np.log1p(test[features].sum(axis=1))
    test['the_std'] = test[features].std(axis=1)
    test['the_kur'] = test[features].kurtosis(axis=1)

    # Only use good features, log leak and stats for training
    # features = good_features.tolist()
    features = ['6eef030c1', 'ba42e41fa', '703885424', 'eeb9cd3aa', '3f4a39818', '371da7669', 'b98f3e0d7', 'fc99f9426', '2288333b4', '324921c7b', '66ace2992', '84d9d1228', '491b9ee45', 'de4e75360', '9fd594eec', 'f190486d6', '62e59a501', '20aa07010', 'c47340d97', '1931ccfdd', 'c2dae3a5a', 'e176a204a'] + ['log_leak', 'log_of_mean', 'mean_of_log', 'log_of_median', 'nb_nans', 'the_sum', 'the_std', 'the_kur']
    dtrain = lgb.Dataset(data=data[features],
                         label=target, free_raw_data=False)
    test['target'] = 0

    dtrain.construct()
    oof_preds = np.zeros(data.shape[0])

    for trn_idx, val_idx in tqdm(folds.split(data)):
        lgb_params = {
            'objective': 'regression',
            'num_leaves': 58,
            'subsample': 0.6143,
            'colsample_bytree': 0.6453,
            'min_split_gain': np.power(10, -2.5988),
            'reg_alpha': np.power(10, -2.2887),
            'reg_lambda': np.power(10, 1.7570),
            'min_child_weight': np.power(10, -0.1477),
            'verbose': -1,
            'seed': 3,
            'boosting_type': 'gbdt',
            'max_depth': -1,
            'learning_rate': 0.05,
            'metric': 'l2',
        }

        clf = lgb.train(
            params=lgb_params,
            train_set=dtrain.subset(trn_idx),
            valid_sets=dtrain.subset(val_idx),
            num_boost_round=10000,
            early_stopping_rounds=100,
            verbose_eval=0
        )

        oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])
        test['target'] += clf.predict(test[features]) / folds.n_splits
        print(mean_squared_error(target.iloc[val_idx],
                                 oof_preds[val_idx]) ** .5)

    data['predictions'] = oof_preds
    data.loc[data['leak'].notnull(), 'predictions'] = np.log1p(data.loc[data['leak'].notnull(), 'leak'])

    print('OOF SCORE : %9.6f' % (mean_squared_error(target, oof_preds) ** .5))
    print('OOF SCORE with LEAK : %9.6f' % (mean_squared_error(target, data['predictions']) ** .5))

    test['target'] = np.expm1(test['target'])
    test.loc[test['leak'].notnull(), 'target'] = test.loc[test['leak'].notnull(), 'leak']
    test[['ID', 'target']].to_csv('leaky_submission.csv', index=False, float_format='%.2f')
