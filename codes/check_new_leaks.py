from tqdm import tqdm
import numpy as np
import pandas as pd
import gc
import multiprocessing as mp
from sklearn import linear_model
import sklearn

leak_data = pd.read_csv('../input/test_leak.csv')
test_data = pd.read_csv('../input/test.csv')

cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']
# print(len(cols))
test_data = test_data[cols]
index = ~leak_data["compiled_leak"].isnull()
sample_set = test_data.loc[index, :]
rest_set = test_data.loc[leak_data["compiled_leak"].isnull(), :]
leak = leak_data["compiled_leak"]
leak = leak[index]

df_concat = pd.concat([leak, sample_set], axis=1)


def check_base_list(i):
    # cehck features similar
    for k in range(26):
        base = df_concat.iloc[i, 26-k:41-k].values
        base = pd.DataFrame([base])
        base = base.apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
        for j in range(26-k):
            # normal
            comp_base = rest_set.iloc[:, j:j+15].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
            if pd.merge(comp_base, base, on='key').shape[0] > 0:
                with open('check_new_leak.txt','a') as add_i:
                    add_i.write("{}\n".format(i))
                print("base:{0}..{1}_{2}, rest:{3}_{4}".format(i, 26-k, 41-k, j, j+15))
    print("check: {} finished".format(i))


pool = mp.Pool(8)
scores = pool.map(check_base_list, list(range(df_concat.shape[0])))
pool.close()
