import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import lightgbm as lgb
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import mode, skew, kurtosis, entropy
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

import dask.dataframe as dd
from dask.multiprocessing import get

from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm_notebook)

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

transact_cols = [f for f in train.columns if f not in ["ID", "target"]]
y = np.log1p(train["target"]).values

test["target"] = train["target"].mean()

all_df = pd.concat([train, test]).reset_index(drop=True)
all_df.columns = all_df.columns.astype(str)
print(all_df.shape)

cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']

colgroups = [
    ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'],
    ['266525925', '4b6dfc880', '2cff4bf0c', 'a3382e205', '6488c8200', '547d3135b', 'b46191036', '453128993', '2599a7eb7', '2fc60d4d9', '009319104', 'de14e7687', 'aa31dd768', '2b54cddfd', 'a67d02050', '37aab1168', '939cc02f5', '31f72667c', '6f951302c', '54723be01', '4681de4fd', '8bd53906a', '435f27009', 'f82167572', 'd428161d9', '9015ac21d', 'ec4dc7883', '22c7b00ef', 'd4cc42c3d', '1351bf96e', '1e8801477', 'b7d59d3b5', 'a459b5f7d', '580f5ff06', '39b3c553a', '1eec37deb', '692c44993', 'ce8ce671e', '88ef1d9a8', 'bf042d928'],
    ['9d5c7cb94', '197cb48af', 'ea4887e6b', 'e1d0e11b5', 'ac30af84a', 'ba4ceabc5', 'd4c1de0e2', '6d2ece683', '9c42bff81', 'cf488d633', '0e1f6696a', 'c8fdf5cbf', 'f14b57b8f', '3a62b36bd', 'aeff360c7', '64534cc93', 'e4159c59e', '429687d5a', 'c671db79e', 'd79736965', '2570e2ba9', '415094079', 'ddea5dc65', 'e43343256', '578eda8e0', 'f9847e9fe', '097c7841e', '018ab6a80', '95aea9233', '7121c40ee', '578b81a77', '96b6bd42b', '44cb9b7c4', '6192f193d', 'ba136ae3f', '8479174c2', '64dd02e44', '4ecc3f505', 'acc4a8e68', '994b946ad'],
    ['f1eeb56ae', '62ffce458', '497adaff8', 'ed1d5d137', 'faf7285a1', 'd83da5921', '0231f07ed', '7950f4c11', '051410e3d', '39e1796ab', '2e0148f29', '312832f30', '6f113540d', 'f3ee6ba3c', 'd9fc63fa1', '6a0b386ac', '5747a79a9', '64bf3a12a', 'c110ee2b7', '1bf37b3e2', 'fdd07cac1', '0872fe14d', 'ddef5ad30', '42088cf50', '3519bf4a4', 'a79b1f060', '97cc1b416', 'b2790ef54', '1a7de209c', '2a71f4027', 'f118f693a', '15e8a9331', '0c545307d', '363713112', '73e591019', '21af91e9b', '62a915028', '2ab5a56f5', 'a8ee55662', '316b978cd'],
    ['b26d16167', '930f989bf', 'ca58e6370', 'aebe1ea16', '03c589fd7', '600ea672f', '9509f66b0', '70f4f1129', 'b0095ae64', '1c62e29a7', '32a0342e2', '2fc5bfa65', '09c81e679', '49e68fdb9', '026ca57fd', 'aacffd2f4', '61483a9da', '227ff4085', '29725e10e', '5878b703c', '50a0d7f71', '0d1af7370', '7c1af7bbb', '4bf056f35', '3dd64f4c4', 'b9f75e4aa', '423058dba', '150dc0956', 'adf119b9a', 'a8110109e', '6c4f594e0', 'c44348d76', 'db027dbaf', '1fcba48d0', '8d12d44e1', '8d13d891d', '6ff9b1760', '482715cbd', 'f81c2f1dd', 'dda820122'],
    ['c928b4b74', '8e4d0fe45', '6c0e0801a', '02861e414', 'aac52d8d9', '041c5d0c9', 'd7875bb6c', 'e7c0cfd0f', 'd48c08bda', '0c9462c08', '57dd44c29', 'a93118262', '850027e38', 'db3839ab0', '27461b158', '32174174c', '9306da53f', '95742c2bf', '5831f4c76', '1e6306c7c', '06393096a', '13bdd610a', 'd7d314edc', '9a07d7b1f', '4d2671746', '822e49b95', '3c8a3ced0', '83635fb67', '1857fbccf', 'c4972742d', 'b6c0969a2', 'e78e3031b', '36a9a8479', 'e79e5f72c', '092271eb3', '74d7f2dc3', '277ef93fc', 'b30e932ba', '8f57141ec', '350473311'],
    ['06148867b', '4ec3bfda8', 'a9ca6c2f4', 'bb0408d98', '1010d7174', 'f8a437c00', '74a7b9e4a', 'cfd55f2b6', '632fed345', '518b5da24', '60a5b79e4', '3fa0b1c53', 'e769ee40d', '9f5f58e61', '83e3e2e60', '77fa93749', '3c9db4778', '42ed6824a', '761b8e0ec', 'ee7fb1067', '71f5ab59f', '177993dc6', '07df9f30c', 'b1c5346c4', '9a5cd5171', 'b5df42e10', 'c91a4f722', 'd93058147', '20a325694', 'f5e0f4a16', '5edd220bc', 'c901e7df1', 'b02dfb243', 'bca395b73', '1791b43b0', 'f04f0582d', 'e585cbf20', '03055cc36', 'd7f15a3ad', 'ccd9fc164'],
    ['df838756c', '2cb73ede7', '4dcf81d65', '61c1b7eb6', 'a9f61cf27', '1af4d24fa', 'e13b0c0aa', 'b9ba17eb6', '796c218e8', '37f57824c', 'd1e0f571b', 'f9e3b03b7', 'a3ef69ad5', 'e16a20511', '04b88be38', '99e779ee0', '9f7b782ac', '1dd7bca9f', '2eeadde2b', '6df033973', 'cdfc2b069', '031490e77', '5324862e4', '467bee277', 'a3fb07bfd', '64c6eb1cb', '8618bc1fd', '6b795a2bc', '956d228b9', '949ed0965', 'a4511cb0b', 'b64425521', '2e3c96323', '191e21b5f', 'bee629024', '1977eaf08', '5e645a169', '1d04efde3', '8675bec0b', '8337d1adc'],
    ['a1cd7b681', '9b490abb3', 'b10f15193', '05f54f417', 'a7ac690a8', 'ed6c300c2', 'd0803e3a1', 'b1bb8eac3', 'bd1c19973', 'a34f8d443', '84ec1e3db', '24018f832', '82e01a220', '4c2064b00', '0397f7c9b', 'ba42e41fa', '22d7ad48d', '9abffd22c', 'dbfa2b77f', '2c6c62b54', '9fa38def3', 'ecb354edf', '9c3154ae6', '2f26d70f4', '53102b93f', 'a36b95f78', '1fa0f78d0', '19915a6d3', 'c944a48b5', '482b04cba', '2ce77a58f', '86558e595', 'c3f400e36', '20305585c', 'f8ccfa064', 'dd771cb8e', '9aa27017e', 'cd7f0affd', '236cc1ff5', 'a3fc511cd'],
    ['920a04ee2', '93efdb50f', '15ea45005', '78c57d7cd', '91570fb11', 'c5dacc85b', '145c7b018', '590b24ab1', 'c283d4609', 'e8bd579ae', '7298ca1ef', 'ce53d1a35', 'a8f80f111', '2a9fed806', 'feb40ad9f', 'cfd255ee3', '31015eaab', '303572ae2', 'cd15bb515', 'cb5161856', 'a65b73c87', '71d64e3f7', 'ec5fb550f', '4af2493b6', '18b4fa3f5', '3d655b0ed', '5cc9b6615', '88c0ec0a6', '8722f33bb', '5ed0c24d0', '54f26ee08', '04ecdcbb3', 'ade8a5a19', 'd5efae759', 'ac7a97382', 'e1b20c3a6', 'b0fcfeab8', '438b8b599', '43782ef36', 'df69cf626'],
    ['50603ae3d', '48282f315', '090dfb7e2', '6ccaaf2d7', '1bf2dfd4a', '50b1dd40f', '1604c0735', 'e94c03517', 'f9378f7ef', '65266ad22', 'ac61229b6', 'f5723deba', '1ced7f0b4', 'b9a4f06cd', '8132d18b8', 'df28ac53d', 'ae825156f', '936dc3bc4', '5b233cf72', '95a2e29fc', '882a3da34', '2cb4d123e', '0e1921717', 'c83d6b24d', '90a2428a5', '67e6c62b9', '320931ca8', '900045349', 'bf89fac56', 'da3b0b5bb', 'f06078487', '56896bb36', 'a79522786', '71c2f04c9', '1af96abeb', '4b1a994cc', 'dee843499', '645b47cde', 'a8e15505d', 'cc9c2fc87'],
    ['b6daeae32', '3bdee45be', '3d6d38290', '5a1589f1a', '961b91fe7', '29c059dd2', 'cfc1ce276', '0a953f97e', '30b3daec2', 'fb5f5836e', 'c7525612c', '6fa35fbba', '72d34a148', 'dcc269cfe', 'bdf773176', '469630e5c', '23db7d793', 'dc10234ae', '5ac278422', '6cf7866c1', 'a39758dae', '45f6d00da', '251d1aa17', '84d9d1228', 'b98f3e0d7', '66146c12d', 'd6470c4ce', '3f4a39818', 'f16a196c6', 'b8f892930', '6f88afe65', 'ed8951a75', '371da7669', '4b9540ab3', '230a025ca', 'f8cd9ae02', 'de4e75360', '540cc3cd1', '7623d805a', 'c2dae3a5a'],
    ['d0d340214', '34d3715d5', '9c404d218', 'c624e6627', 'a1b169a3a', 'c144a70b1', 'b36a21d49', 'dfcf7c0fa', 'c63b4a070', '43ebb15de', '1f2a670dd', '3f07a4581', '0b1560062', 'e9f588de5', '65d14abf0', '9ed0e6ddb', '0b790ba3a', '9e89978e3', 'ee6264d2b', 'c86c0565e', '4de164057', '87ba924b1', '4d05e2995', '2c0babb55', 'e9375ad86', '8988e8da5', '8a1b76aaf', '724b993fd', '654dd8a3b', 'f423cf205', '3b54cc2cf', 'e04141e42', 'cacc1edae', '314396b31', '2c339d4f2', '3f8614071', '16d1d6204', '80b6e9a8b', 'a84cbdab5', '1a6d13c4a'],
    ['a9819bda9', 'ea26c7fe6', '3a89d003b', '1029d9146', '759c9e85d', '1f71b76c1', '854e37761', '56cb93fd8', '946d16369', '33e4f9a0e', '5a6a1ec1a', '4c835bd02', 'b3abb64d2', 'fe0dd1a15', 'de63b3487', 'c059f2574', 'e36687647', 'd58172aef', 'd746efbfe', 'ccf6632e6', 'f1c272f04', 'da7f4b066', '3a7771f56', '5807de036', 'b22eb2036', 'b77c707ef', 'e4e9c8cc6', 'ff3b49c1d', '800f38b6b', '9a1d8054b', '0c9b00a91', 'fe28836c3', '1f8415d03', '6a542a40a', 'd53d64307', 'e700276a2', 'bb6f50464', '988518e2d', 'f0eb7b98f', 'd7447b2c5'],
    ['87ffda550', '63c094ba4', '2e103d632', '1c71183bb', 'd5fa73ead', 'e078302ef', 'a6b6bc34a', 'f6eba969e', '0d51722ca', 'ce3d7595b', '6c5c8869c', 'dfd179071', '122c135ed', 'b4cfe861f', 'b7c931383', '44d5b820f', '4bcf15776', '51d4053c7', '1fe5d56b9', 'ea772e115', 'ad009c8b9', '68a945b18', '62fb56487', 'c10f31664', 'cbb673163', 'c8d582dd2', '8781e4b91', 'bd6da0cca', 'ca2b906e8', '11e12dbe8', 'bb0ce54e9', 'c0d2348b7', '77deffdf0', 'f97d9431e', 'a09a238d0', '935ca66a9', '9de83dc23', '861076e21', 'f02ecb19c', '166008929'],
    ['f3cf9341c', 'fa11da6df', 'd47c58fe2', '0d5215715', '555f18bd3', '134ac90df', '716e7d74d', 'c00611668', '1bf8c2597', '1f6b2bafa', '174edf08a', 'f1851d155', '5bc7ab64f', 'a61aa00b0', 'b2e82c050', '26417dec4', '53a550111', '51707c671', 'e8d9394a0', 'cbbc9c431', '6b119d8ce', 'f296082ec', 'be2e15279', '698d05d29', '38e6f8d32', '93ca30057', '7af000ac2', '1fd0a1f2a', '41bc25fef', '0df1d7b9a', '88d29cfaf', '2b2b5187e', 'bf59c51c3', 'cfe749e26', 'ad207f7bb', '11114a47a', '341daa7d1', 'a8dd5cea5', '7b672b310', 'b88e5de84'],
    ['ced6a7e91', '9df4daa99', '83c3779bf', 'edc84139a', 'f1e0ada11', '73687e512', 'aa164b93b', '342e7eb03', 'cd24eae8a', '8f3740670', '2b2a10857', 'a00adf70e', '3a48a2cd2', 'a396ceeb9', '9280f3d04', 'fec5eaf1a', '5b943716b', '22ed6dba3', '5547d6e11', 'e222309b0', '5d3b81ef8', '1184df5c2', '2288333b4', 'f39074b55', 'a8b721722', '13ee58af1', 'fb387ea33', '4da206d28', 'ea4046b8d', 'ef30f6be5', 'b85fa8b27', '2155f5e16', '794e93ca6', '070f95c99', '939f628a7', '7e814a30d', 'a6e871369', '0dc4d6c7d', 'bc70cbc26', 'aca228668'],
    ['81de0d45e', '18562fc62', '543c24e33', '0256b6714', 'd6006ff44', '6a323434b', 'e3a38370e', '7c444370b', '8d2d050a2', '9657e51e1', '13f3a3d19', 'b5c839236', '70f3033c6', 'f4b374613', '849125d91', '16b532cdc', '88219c257', '74fb8f14c', 'fd1102929', '699712087', '22501b58e', '9e9274b24', '2c42b0dce', '2c95e6e31', '5263c204d', '526ed2bec', '01f7de15d', 'cdbe394fb', 'adf357c9b', 'd0f65188c', 'b8a716ebf', 'ef1e1fac8', 'a3f2345bf', '110e4132e', '586b23138', '680159bab', 'f1a1562cd', '9f2f1099b', 'bf0e69e55', 'af91c41f0'],
    ['8677d6620', '75b846f12', '3a01b4018', '23d6be31e', '52695ed4a', 'ba9f3a42c', '135091a07', '19537e282', 'd5d4f936e', '578a07608', '63df94487', '169875559', 'b6ae5f5ca', '315b44e13', '5150b1a17', 'c8c6fe1a0', 'd918835ca', '8768af50f', '2cc11689d', '51c9aee7e', '188a6e279', '649d727e1', 'a8e878643', '8d4f4c571', 'f990bddac', '5719bbfc3', '12d3a67b0', '5f76b9c2f', 'c33a4095a', 'aac0c81ba', '2ba3b18ee', 'be90775f4', '651124842', '51d5e73a8', '8016f08af', 'f80259ab3', '3685524f4', '532740e5d', '30347e683', '806dfdd51'],
    ['e20edfcb8', '842415efb', '300d6c1f1', '720f83290', '069a2c70b', '87a91f998', '611151826', '74507e97f', '504e4b156', 'baa95693d', 'cb4f34014', '5239ceb39', '81e02e0fa', 'dfdf4b580', 'fc9d04cd7', 'fe5d62533', 'bb6260a44', '08d1f69ef', 'b4ced4b7a', '98d90a1d1', 'b6d206324', '6456250f1', '96f5cf98a', 'f7c8c6ad3', 'cc73678bf', '5fb85905d', 'cb71f66af', '212e51bf6', 'd318bea95', 'b70c62d47', '11d86fa6a', '3988d0c5e', '42cf36d73', '9f494676e', '1c68ee044', 'a728310c8', '612bf9b47', '105233ed9', 'c18cc7d3d', 'f08c20722'],
    ['9fa984817', '3d23e8abd', '1b681c3f0', '3be4dad48', 'dcfcddf16', 'b25319cb3', 'b14026520', 'c5cb7200e', 'ede70bfea', 'e5ddadc85', '07cb6041d', 'df6a71cc7', 'dc60842fb', '3a90540ab', '6bab7997a', 'c87f4fbfb', '21e0e6ae3', '9b39b02c0', '5f5cfc3c0', '35da68abb', 'f0aa40974', '625525b5d', 'd7978c11c', '2bbcbf526', 'bc2bf3bcd', '169f6dda5', '4ceef6dbd', '9581ec522', 'd4e8dd865', 'bf8150471', '542f770e5', 'b05eae352', '3c209d9b6', 'b2e1308ae', '786351d97', 'e5a8e9154', '2b85882ad', 'dc07f7e11', '14c2463ff', '14a5969a6'],
    ['1ad24da13', '8c5025c23', 'f52a82e7f', 'c0b22b847', 'd75793f21', '4cffe31c7', '6c2d09fb1', 'fb42abc0d', '206ba1242', '62f61f246', '1389b944a', 'd15e80536', 'fa5044e9e', 'a0b0a7dbf', '1ff6be905', '4e06c5c6d', '1835531cd', '68b647452', 'c108dbb04', '58e8e2c82', 'f3bfa96d5', 'f2db09ac3', '4e8196700', '8cd9be80e', '83fc7f74c', 'dbc48d37c', '2028e022d', '17e160597', 'eb8cbd733', 'addb3f3eb', '460744630', '9108ee25c', 'b7950e538', 'a7da4f282', '7f0d863ba', 'b7492e4eb', '24c41bd80', 'fd7b0fc29', '621f71db3', '26f222d6d'],
    ['1d9078f84', '64e483341', 'a75d400b8', '4fe8154c8', '29ab304b9', '20604ed8f', 'bd8f989f1', 'c1b9f4e76', '4824c1e90', '4ead853dc', 'b599b0064', 'd26279f1a', '58ed8fb53', 'ff65215db', '402bb0761', '74d7998d4', 'c7775aabf', '9884166a7', 'beb7f98fd', 'fd99c18b5', 'd83a2b684', '18c35d2ea', '0c8063d63', '400e9303d', 'c976a87ad', '8a088af55', '5f341a818', '5dca793da', 'db147ffca', '762cbd0ab', 'fb5a3097e', '8c0a1fa32', '01005e5de', '47cd6e6e4', 'f58fb412c', 'a1db86e3b', '50e4f96cf', 'f514fdb2e', '7a7da3079', 'bb1113dbb'],
    ['5030aed26', 'b850c3e18', '212efda42', '9e7c6b515', '2d065b147', '49ca7ff2e', '37c85a274', 'ea5ed6ff7', 'deabe0f4c', 'bae4f747c', 'ca96df1db', '05b0f3e9a', 'eb19e8d63', '235b8beac', '85fe78c6c', 'cc507de6c', 'e0bb9cf0b', '80b14398e', '9ca0eee11', '4933f2e67', 'fe33df1c4', 'e03733f56', '1d00f511a', 'e62cdafcf', '3aad48cda', 'd36ded502', '92b13ebba', 'f30ee55dd', '1f8754c4e', 'db043a30f', 'e75cfcc64', '5d8a55e6d', '6e29e9500', 'c5aa7c575', 'c2cabb902', 'd251ee3b4', '73700eaa4', '8ab6f5695', '54b1c1bc0', 'cbd0256fb'],
    ['b33e83cdc', 'ab8a614fa', 'bf6e38e39', 'eb7981dd4', '30a47af70', 'f7eee8212', '9847e14d8', '1998aa946', '850e01a62', 'ecd4c66ec', '56a21fe66', '3f382323a', 'b0b1c81ac', 'b47be7e76', 'd8ea347e9', 'ccc9ba695', '2e55d0383', 'f471e9e82', '56ec098a1', '172a58959', '809a511d0', 'a5e0d3ddb', '945dad481', 'd66bbb5ed', 'c98c2d3c0', '94ecf4c83', 'bec7c48dd', 'ea18d720e', 'bee71cf84', '2f92a1a45', '3be79d4a5', 'a388d3605', '36cde3ce8', '937854db6', '76e092b8c', '1d744ff92', 'a43c53c45', '6045a2949', '3af1785ee', 'f926a4cb4'],
    ['4302b67ec', '75b663d7d', 'fc4a873e0', '1e9bdf471', '86875d9b0', '8f76eb6e5', '3d71c02f0', '05c9b6799', '26df61cc3', '27a7cc0ca', '9ff21281c', '3ce93a21b', '9f85ae566', '3eefaafea', 'afe8cb696', '72f9c4f40', 'be4729cb7', '8c94b6675', 'ae806420c', '63f493dba', '5374a601b', '5291be544', 'acff85649', '3690f6c26', '26c68cede', '12a00890f', 'dd84964c8', 'a208e54c7', 'fb06e8833', '7de39a7eb', '5fe3acd24', 'e53805953', '3de2a9e0d', '2954498ae', '6c3d38537', '86323e98a', 'b719c867c', '1f8a823f2', '9cc5d1d8f', 'd3fbad629'],
    ['fec5644cf', 'caa9883f6', '9437d8b64', '68811ba58', 'ef4b87773', 'ff558c2f2', '8d918c64f', '0b8e10df6', '2d6565ce2', '0fe78acfa', 'b75aa754d', '2ab9356a0', '4e86dd8f3', '348aedc21', 'd7568383a', '856856d94', '69900c0d1', '02c21443c', '5190d6dca', '20551fa5b', '79cc300c7', '8d8276242', 'da22ed2b8', '89cebceab', 'f171b61af', '3a07a8939', '129fe0263', 'e5b2d137a', 'aa7223176', '5ac7e84c4', '9bd66acf6', '4c938629c', 'e62c5ac64', '57535b55a', 'a1a0084e3', '2a3763e18', '474a9ec54', '0741f3757', '4fe8b17c2', 'd5754aa08'],
    ['0f8d7b98e', 'c30ff7f31', 'ac0e2ebd0', '24b2da056', 'bd308fe52', '476d95ef1', '202acf9bd', 'dbc0c19ec', '06be6c2bb', 'd8296080a', 'f977e99dc', '2191d0a24', '7db1be063', '1bc285a83', '9a3a1d59b', 'c4d657c5b', 'a029667de', '21bd61954', '16bf5a9a2', '0e0f8504b', '5910a3154', 'ba852cc7a', '685059fcd', '21d6a4979', '78947b2ad', '1435ecf6b', '3839f8553', 'e9b5b8919', 'fa1dd6e8c', '632586103', 'f016fd549', 'c25ea08ba', '7da54106c', 'b612f9b7e', 'e7c0a50e8', '29181e29a', '395dbfdac', '1beb0ce65', '04dc93c58', '733b3dc47'],
    ['2d60e2f7a', '11ad148bd', '54d3e247f', 'c25438f10', 'e6efe84eb', '964037597', '0196d5172', '47a8de42e', '6f460d92f', '0656586a4', '22eb11620', 'c3825b569', '6aa919e2e', '086328cc6', '9a33c5c8a', 'f9c3438ef', 'c09edaf01', '85da130e3', '2f09a1edb', '76d34bbee', '04466547a', '3b52c73f5', '1cfb3f891', '704d68890', 'f45dd927f', 'aba01a001', 'c9160c30b', '6a34d32d6', '3e3438f04', '038cca913', '504c22218', '56c679323', '002d634dc', '1938873fd', 'd37030d36', '162989a6d', 'e4dbe4822', 'ad13147bd', '4f45e06b3', 'ba480f343'],
    ['86cefbcc0', '717eff45b', '7d287013b', '8d7bfb911', 'aecaa2bc9', '193a81dce', '8dc7f1eb9', 'c5a83ecbc', '60307ab41', '3da5e42a7', 'd8c61553b', '072ac3897', '1a382b105', 'f3a4246a1', '4e06e4849', '962424dd3', 'a3da2277a', '0a69cc2be', '408d191b3', '98082c8ef', '96b66294d', 'cc93bdf83', 'ffa6b80e2', '226e2b8ac', '678b3f377', 'b56f52246', '4fa02e1a8', '2ef57c650', '9aeec78c5', '1477c751e', 'a3c187bb0', '1ce516986', '080cd72ff', '7a12cc314', 'ead538d94', '480e78cb0', '737d43535', 'a960611d7', '4416cd92c', 'd5e6c18b0'],
    ['7ba58c14d', '1fe02bc17', '4672a8299', '8794c72c8', 'cca45417f', '55dbd6bcb', 'e6e2c3779', '3cae817df', '973663d14', 'e8dfb33d5', '9281abeea', '11c01e052', '1520de553', 'edddb1ba5', 'c18b41ac3', '00e87edf2', 'ae72cba0a', 'eb4f2651e', '300398f1c', '6c05550b8', '9b26736c3', '24744410a', '26faf1b2e', '44f09b92d', '19975f6ff', '1bf6240eb', 'e438105db', 'cdc36a26a', '087e01c14', '828b327a6', 'cc62f0df8', '9370aa48d', 'd4815c074', '18321c252', '22fbf6997', 'feed9d437', 'f6c9661fc', '55f2b3d34', '69fe81b64', '1074273db'],
    ['7f72c937f', '79e55ef6c', '408d86ce9', '7a1e99f69', '736513d36', '0f07e3775', 'eb5a2cc20', '2b0fc604a', 'aecd09bf5', '91de54e0a', '66891582e', '20ef8d615', '8d4d84ddc', 'dfde54714', '2be024de7', 'd19110e37', 'e637e8faf', '2d6bd8275', 'f3b4de254', '5cebca53f', 'c4255588c', '23c780950', 'bc56b26fd', '55f4891bb', '020a817ab', 'c4592ac16', '542536b93', '37fb8b375', '0a52be28f', 'bd7bea236', '1904ce2ac', '6ae9d58e0', '5b318b659', '25729656f', 'f8ee2386d', '589a5c62a', '64406f348', 'e157b2c72', '0564ff72c', '60d9fc568'],
    ['ccc7609f4', 'ca7ea80a3', 'e509be270', '3b8114ab0', 'a355497ac', '27998d0f4', 'fa05fd36e', '81aafdb57', '4e22de94f', 'f0d5ffe06', '9af753e9d', 'f1b6cc03f', '567d2715c', '857020d0f', '99fe351ec', '3e5dab1e3', '001476ffa', '5a5eabaa7', 'cb5587baa', '32cab3140', '313237030', '0f6386200', 'b961b0d59', '9452f2c5f', 'bcfb439ee', '04a22f489', '7e58426a4', 'a4c9ea341', 'ffdc4bcf8', '1a6d866d7', 'd7334935b', '298db341e', '08984f627', '8367dfc36', '5d9f43278', '7e3e026f8', '37c10d610', '5a88b7f01', '324e49f36', '99f466457'],
    ['48b839509', '2b8851e90', '28f75e1a5', '0e3ef9e8f', '37ac53919', '7ca10e94b', '4b6c549b1', '467aa29ce', '74c5d55dc', '0700acbe1', '44f3640e4', 'e431708ff', '097836097', 'd1fd0b9c2', 'a0453715a', '9e3aea49a', '899dbe405', '525635722', '87a2d8324', 'faf024fa9', 'd421e03fd', '1254b628a', 'a19b05919', '34a4338bc', '08e89cc54', 'a29c9f491', 'a0a8005ca', '62ea662e7', '5fe6867a4', '8b710e161', '7ab926448', 'd04e16aed', '4e5da0e96', 'ff2c9aa8f', 'b625fe55a', '7124d86d9', '215c4d496', 'b6fa5a5fd', '55a7e0643', '0a26a3cfe'],
    ['bb6a5b6e2', '30d424f24', 'eea698cf2', '8a158bbb8', 'acd43607d', '0019109c4', '776e9945e', '67ddf8bdd', '025172af5', '2123a2089', 'd40eb2705', '1b20c5c27', '7bde71e2f', '8ba7eacbb', '932b61d77', 'e3fd6fa46', '53bba91b7', 'd24a55c98', '93f686d09', 'fc5690e51', '0ac076350', '18e3e1563', 'd3ff41260', 'c40750aed', 'f2c0fa7cf', '3c9f7809d', 'c65ab9cb9', '6e738ec87', '3475c6ad7', '5964f1856', 'a6bf610b3', '7f9f72202', 'f57ebfed7', '3dd4cc7a8', '8ec06d490', '99fc30923', '71b203550', '09bf8b0cf', '5c1f412ce', '236910072'],
    ['5bf913a56', 'e6c050854', 'edc3f10a1', '3607eabff', '5cec9a2fc', '68153d35e', '193b90919', '5bca7197d', 'da2a2f42d', '0f2b86f4a', '280898a2f', '1c6c0ffb1', 'ec2a9147d', '1ba077222', 'f115e74c0', '34b2a678e', 'cc0045289', 'c00356999', '09184c121', '799625b2f', '5b714cd7a', 'd14ac08a8', '5ef415428', 'f51378159', 'd5dcaa04a', 'e8522c145', '7610d0f28', '20ff37b40', '5b9e32dbe', 'dd84674d0', '587a5d8c3', '2c1ed7d88', '86f0ede14', '05e427fe8', '45226872a', '003da5628', 'fbbd5f5ae', 'a8b6710d0', '99197edf2', 'a1995906f'],
    ['51c141e64', '0e348d340', '64e010722', '55a763d90', '13b54db14', '01fdd93d3', '1ec48dbe9', 'cf3841208', 'd208491c8', '90b0ed912', '633e0d42e', '9236f7b22', '0824edecb', '71deb9468', '1b55f7f4d', '377a76530', 'c47821260', 'bf45d326d', '69f20fee2', 'd6d63dd07', '5ab3be3e1', '93a31829f', '121d8697e', 'f308f8d9d', '0e44d3981', 'ecdef52b2', 'c69492ae6', '58939b6cc', '3132de0a3', 'a175a9aa4', '7166e3770', 'abbde281d', '23bedadb2', 'd4029c010', 'fd99222ee', 'bd16de4ba', 'fb32c00dc', '12336717c', '2ea42a33b', '50108b5b5'],
    ['a5f8c7929', '330006bce', 'b22288a77', 'de104af37', '8d81c1c27', 'd7285f250', '123ba6017', '3c6980c42', '2d3296db7', '95cdb3ab7', '05527f031', '65753f40f', '45a400659', '1d5df91e2', '233c7c17c', '2a879b4f7', 'c3c633f64', 'fdae76b2c', '05d17ab7a', 'c25078fd7', 'e209569b2', '3fd2b9645', '268b047cd', '3d350431d', '5fb9cabb1', 'b70c76dff', '3f6246360', '89e7dcacc', '12122f265', 'fcc17a41d', 'c5a742ee4', '9e711a568', '597d78667', '0186620d7', '4c095683e', '472cd130b', 'b452ba57e', '2ce2a1cdb', '50c7ea46a', '2761e2b76'],
    ['3b843ae7e', 'c8438b12d', 'd1b9fc443', '19a45192a', '63509764f', '6b6cd5719', 'b219e3635', '4b1d463d7', '4baa9ff99', 'b0868a049', '3e3ea106e', '043e4971a', 'a2e5adf89', '25e2bcb45', '3ac0589c3', '413bbe772', 'e23508558', 'c1543c985', '2dfea2ff3', '9dcdc2e63', '1f1f641f1', '75795ea0a', 'dff08f7d5', '914d2a395', '00302fe51', 'c0032d792', '9d709da93', 'cb72c1f0b', '5cf7ac69f', '6b1da7278', '47b5abbd6', '26163ffe1', '902c5cd15', '45bc3b302', '5c208a931', 'e88913510', 'e1d6a5347', '38ec5d3bb', 'e3d64fcd7', '199d30938'],
    ['9d4428628', '37f11de5d', '39549da61', 'ceba761ec', '4c60b70b8', '304ebcdbc', '823ac378c', '4e21c4881', '5ee81cb6e', 'eb4a20186', 'f6bdb908a', '6654ce6d8', '65aa7f194', '00f844fea', 'c4de134af', 'a240f6da7', '168c50797', '13d6a844f', '7acae7ae9', '8c61bede6', '45293f374', 'feeb05b3f', 'a5c62af4a', '22abeffb6', '1d0aaa90f', 'c46028c0f', '337b3e53b', 'd6af4ee1a', 'cde3e280a', 'c83fc48f2', 'f99a09543', '85ef8a837', 'a31ba11e6', '64cabb6e7', '93521d470', '46c525541', 'cef9ab060', '375c6080e', '3c4df440f', 'e613715cc'],
    ['ec5764030', '42fdff3a0', 'fa6e76901', '6e76d5df3', '1c486f8dd', '2daf6b624', '9562ce5c8', 'cbf236577', '8e1822aa3', 'fd9968f0d', 'ed1f680d4', '6bd9d9ae3', '896d1c52d', 'b41a9fc75', 'a60974604', '9d6b84f39', '5661462ee', '186b87c05', 'e5ac02d3c', '0c4bf4863', '1fba6a5d5', '4f2f6b0b3', 'cd8048913', 'e17f1f07c', '707f193d9', '8ca08456c', '3adf5e2b5', 'a60027bb4', 'e7071d5e3', 'c7ae29e66', '50780ec40', 'f8b733d3f', '8485abcab', '994b4c2ac', '6af8b2246', 'dd85a900c', 'ccb68477c', '715fa74a4', 'adadb9a96', '77eb013ca'],
    ['a3e023f65', '9126049d8', '6eaea198c', '5244415dd', '0616154cc', '2165c4b94', 'fc436be29', '1834f29f5', '9d5af277d', 'c6850e7db', '6b241d083', '56f619761', '45319105a', 'fcda960ae', '07746dcda', 'c906cd268', 'c24ea6548', '829fb34b8', '89ebc1b76', '22c019a2e', '1e16f11f3', '94072d7a3', '59dfc16da', '9886b4d22', '0b1741a7f', 'a682ef110', 'e26299c3a', '5c220a143', 'ac0493670', '8d8bffbae', '68c7cf320', '3cea34020', 'e9a8d043d', 'afb6b8217', '5780e6ffa', '26628e8d8', '1de4d7d62', '4c53b206e', '99cc87fd7', '593cccdab'],
    ['2135fa05a', 'e8a3423d6', '90a438099', '7ad6b38bd', '60e45b5ee', '2b9b1b4e2', 'd6c82cd68', '923114217', 'b361f589e', '04be96845', 'ee0b53f05', '21467a773', '47665e3ce', 'a6229abfb', '9666bfe76', '7dcc40cda', '17be6c4e7', 'a89ab46bb', '9653c119c', 'cc01687d0', '60e9cc05b', 'ffcec956f', '51c250e53', '7344de401', 'a15b2f707', 'a8e607456', 'dbb8e3055', '2a933bcb8', 'b77bc4dac', '58d9f565a', '17068424d', '7453eb289', '027a2206a', '343042ed9', 'c8fb3c2d8', '29eddc376', '1c873e4a6', '588106548', '282cfe2ad', '358dc07d0'],
    ['4569d5378', '22f05c895', '5fad07863', 'f32763afc', '9bb02469c', '61063fa1c', '4a93ad962', 'fa1efdadd', '4ef309fc3', 'ed0860a34', '6ae0787f3', 'ffd50f0bf', '704e2dc55', '1b1a893f5', 'b19e65a65', '8d4b52f9a', '85dcc913d', '92ba988e1', '6d46740f1', '0aab2f918', '6610f90f1', 'a235f5488', 'c5c073bb0', '13f7f9c70', 'fb6da0420', '73361d959', '783ee6e9a', '635fbbd2c', '60cd556c9', '150504397', 'f3b6dabf7', 'd92ea0b2a', 'b904b8345', '78bc2558b', '4e1a8f6eb', 'c89ae4ce0', 'f2af9300f', 'ca25aad9f', '9d435a85b', '8d035d41e'],
    ['0d7692145', '62071f7bc', 'ab515bdeb', 'c30c6c467', 'eab76d815', 'b6ee6dae6', '49063a8ed', '4cb2946ce', '6c27de664', '772288e75', 'afd87035a', '44f2f419e', '754ace754', 'e803a2db0', 'c70f77ef2', '65119177e', '3a66c353a', '4c7768bff', '9e4765450', '24141fd90', 'dc8b7d0a8', 'ba499c6d9', '8b1379b36', '5a3e3608f', '3be3c049e', 'a0a3c0f1b', '4d2ca4d52', '457bd191d', '6620268ab', '9ad654461', '1a1962b67', '7f55b577c', '989d6e0f5', 'bc937f79a', 'e059a8594', '3b74ac37b', '555265925', 'aa37f9855', '32c8b9100', 'e71a0278c'],
    ['5b465f819', 'a2aa0e4e9', '944e05d50', '4f8b27b6b', 'a498f253f', 'c73c31769', '025dea3b3', '616c01612', 'f3316966c', '83ea288de', '2dbeac1de', '47b7b878f', 'b4d41b335', '686d60d8a', '6dcd9e752', '7210546b2', '78edb3f13', '7f9d59cb3', '30992dccd', '26144d11f', 'a970277f9', '0aea1fd67', 'dc528471e', 'd51d10e38', 'efa99ed98', '48420ad48', '7f38dafa6', '1af4ab267', '3a13ed79a', '73445227e', '971631b2d', '57c4c03f6', '7f91dc936', '0784536d6', 'c3c3f66ff', '052a76b0f', 'ffb34b926', '9d4f88c7b', '442b180b6', '948e00a8d'],
    ['9a2b0a8be', '856225035', 'f9db72cff', '709573455', '616be0c3e', '19a67cb97', '9d478c2ae', 'cf5b8da95', '9c502dcd9', '2f7b0f5b5', 'd50798d34', '56da2db09', 'c612c5f8f', '08c089775', '7aaefdfd7', '59cb69870', '37c0a4deb', 'fb9a4b46d', 'b4eaa55ea', '304633ac8', '99f22b12d', '65000b269', '4bffaff52', '4c536ffc0', '93a445808', 'e8b513e29', 'a2616a980', '97d5c39cf', '71aae7896', '62d0edc4f', 'c2acc5633', 'c8d5efceb', 'e50c9692b', '2e1287e41', '2baea1172', 'af1e16c95', '01c0495f8', 'b0c0f5dae', '090f3c4f2', '33293f845'],
    ['c13ee1dc9', 'abb30bd35', 'd2919256b', '66728cc11', 'eab8abf7a', 'cc03b5217', '317ee395d', '38a92f707', '467c54d35', 'e8f065c9d', '2ac62cba5', '6495d8c77', '94cdda53f', '13f2607e4', '1c047a8ce', '28a5ad41a', '05cc08c11', 'b0cdc345e', '38f49406e', '773180cf6', '1906a5c7e', 'c104aeb2e', '8e028d2d2', '0dc333fa1', '28a785c08', '03ee30b8e', '8e5a41c43', '67102168f', '8b5c0fb4e', '14a22ab1a', '9fc776466', '4aafb7383', '8e1dfcb94', '55741d46d', '8f940cb1b', '758a9ab0e', 'fd812d7e0', '4ea447064', '6562e2a2c', '343922109'],
    ['63be1f619', '36a56d23e', '9e2040e5b', 'a00a63886', '4edc3388d', '5f11fbe33', '26e998afd', 'f7faf2d9f', '992b5c34d', 'f7f553aea', '7e1c4f651', 'f5538ee5c', '711c20509', '55338de22', '374b83757', 'f41f0eb2f', 'bf10af17e', 'e2979b858', 'd3ed79990', 'fe0c81eff', '5c0df6ac5', '82775fc92', 'f1c20e3ef', 'fa9d6b9e5', 'a8b590c6e', 'b5c4708ad', 'c9aaf844f', 'fe3fe2667', '50a6c6789', '8761d9bb0', 'b6403de0b', '2b6f74f09', '5755fe831', '91ace30bd', '84067cfe0', '15e4e8ee5', 'd01cc5805', '870e70063', '2bd16b689', '8895ea516'],
    ['509e911f0', '9c36a77b3', '50aaba7f1', 'ed5af35f0', 'ffd2f9409', 'd6a122efd', '30768bc79', '9161061c9', '1fbbd4edf', '9a179ed71', '6a055c4fb', '61efa1e29', 'e171bccbe', 'd7cdd8aef', 'd168174c7', 'b791ce9aa', '1a82869a6', '3696a15a7', '7b31055f1', 'a76ad8513', '82ba7a053', '37426563f', 'ba5bbaffc', 'd3022e2f1', '0ccd5ff1c', '31a3f920c', '86eb6ec85', '38df6c628', 'f1fbe249b', '6d0d72180', '22dbe574a', '5860d7fa9', '455f29419', 'f269ec9c8', '75aad4520', '18c0b76e9', 'dae4d14b4', '0cad4d7af', '1e1cb47f3', '9d6410ef5'],
]

pattern_1964666 = pd.read_csv('../input/pattern-found/pattern_1964666.66.csv')
pattern_1166666 = pd.read_csv('../input/pattern-found/pattern_1166666.66.csv')
pattern_812666 = pd.read_csv('../input/pattern-found/pattern_812666.66.csv')
pattern_2002166 = pd.read_csv('../input/pattern-found/pattern_2002166.66.csv')
pattern_3160000 = pd.read_csv('../input/pattern-found/pattern_3160000.csv')
pattern_3255483 = pd.read_csv('../input/pattern-found/pattern_3255483.88.csv')

pattern_1964666.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
pattern_1166666.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
pattern_812666.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
pattern_2002166.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
pattern_3160000.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
pattern_3255483.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)

pattern_1166666.rename(columns={'8.50E+43': '850027e38'},inplace=True)

l=[]
l.append(pattern_1964666.columns.values.tolist())
l.append(pattern_1166666.columns.values.tolist())
l.append(pattern_812666.columns.values.tolist())
l.append(pattern_2002166.columns.values.tolist())
l.append(pattern_3160000.columns.values.tolist())
l.append(pattern_3255483.columns.values.tolist())

ss = l + colgroups

def _get_leak(df, cols,extra_feats, lag=0):
    f1 = cols[:((lag+2) * -1)]
    f2 = cols[(lag+2):]
    for ef in extra_feats:
        f1 += ef[:((lag+2) * -1)]
        f2 += ef[(lag+2):]

    d1 = df[f1].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d1.to_csv('extra_d1.csv')
    d2 = df[f2].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})

    d2['pred'] = df[cols[lag]]
#     d2.to_csv('extra_d2.csv')
    #d2 = d2[d2.pred != 0] ### to make output consistent with Hasan's function
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    d4 = d1[~d1.duplicated(['key'], keep=False)]
    d5 = d4.merge(d3, how='inner', on='key')

    d6 = d1.merge(d5, how='left', on='key')
    d6.to_csv('extra_d6.csv')

    return d1.merge(d5, how='left', on='key').pred.fillna(0)

def compiled_leak_result():

    max_nlags = len(cols)-2
    train_leak = train[["ID", "target"] + cols]
    train_leak["compiled_leak"] = 0
    train_leak["nonzero_mean"] = train[transact_cols].apply(
        lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
    )
    scores = []
    leaky_value_counts = []
    leaky_value_corrects = []
    leaky_cols = []

    for i in range(max_nlags):
        c = "leaked_target_"+str(i)
        print('Processing lag', i)
        #train_leak[c] = _get_leak(train, cols,l, i)
        train_leak[c] = _get_leak(train, cols,ss, i)

        leaky_cols.append(c)
        train_leak = train.join(
            train_leak.set_index("ID")[leaky_cols+["compiled_leak", "nonzero_mean"]],
            on="ID", how="left"
        )[["ID", "target"] + cols + leaky_cols+["compiled_leak", "nonzero_mean"]]
        zeroleak = train_leak["compiled_leak"]==0
        train_leak.loc[zeroleak, "compiled_leak"] = train_leak.loc[zeroleak, c]
        leaky_value_counts.append(sum(train_leak["compiled_leak"] > 0))
        _correct_counts = sum(train_leak["compiled_leak"]==train_leak["target"])
        leaky_value_corrects.append(_correct_counts*1.0/leaky_value_counts[-1])
        print("Leak values found in train", leaky_value_counts[-1])
        print(
            "% of correct leaks values in train ",
            leaky_value_corrects[-1]
        )
        tmp = train_leak.copy()
        tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, "nonzero_mean"]
        print('Na count',tmp.compiled_leak.isna().sum())
        scores.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))
        print(
            'Score (filled with nonzero mean)',
            scores[-1]
        )
    result = dict(
        score=scores,
        leaky_count=leaky_value_counts,
        leaky_correct=leaky_value_corrects,
    )
    return train_leak, result

train_leak, result = compiled_leak_result()

result = pd.DataFrame.from_dict(result, orient='columns')

result.to_csv('train_leaky_stat.csv', index=False)

best_score = np.min(result['score'])
best_lag = np.argmin(result['score'])
print('best_score', best_score, '\nbest_lag', best_lag)

def rewrite_compiled_leak(leak_df, lag):
    leak_df["compiled_leak"] = 0
    for i in range(lag):
        c = "leaked_target_"+str(i)
        zeroleak = leak_df["compiled_leak"]==0
        leak_df.loc[zeroleak, "compiled_leak"] = leak_df.loc[zeroleak, c]
    return leak_df

leaky_cols = [c for c in train_leak.columns if 'leaked_target_' in c]
train_leak = rewrite_compiled_leak(train_leak, best_lag)

def compiled_leak_result_test(max_nlags):
    test_leak = test[["ID", "target"] + cols]
    test_leak["compiled_leak"] = 0
    test_leak["nonzero_mean"] = test[transact_cols].apply(
        lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
    )

    scores = []
    leaky_value_counts = []
    # leaky_value_corrects = []
    leaky_cols = []

    for i in range(max_nlags):
        c = "leaked_target_"+str(i)

        print('Processing lag', i)
        #test_leak[c] = _get_leak(test_leak, cols, i)
        test_leak[c] = _get_leak(test, cols,ss, i)

        leaky_cols.append(c)
        test_leak = test.join(
            test_leak.set_index("ID")[leaky_cols+["compiled_leak", "nonzero_mean"]],
            on="ID", how="left"
        )[["ID", "target"] + cols + leaky_cols+["compiled_leak", "nonzero_mean"]]
        zeroleak = test_leak["compiled_leak"]==0
        test_leak.loc[zeroleak, "compiled_leak"] = test_leak.loc[zeroleak, c]
        leaky_value_counts.append(sum(test_leak["compiled_leak"] > 0))
        #_correct_counts = sum(train_leak["compiled_leak"]==train_leak["target"])
        #leaky_value_corrects.append(_correct_counts/leaky_value_counts[-1])
        print("Leak values found in test", leaky_value_counts[-1])
        #print(
        #    "% of correct leaks values in train ",
        #    leaky_value_corrects[-1]
        #)
        #tmp = test_leak.copy()
        #tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, "nonzero_mean"]
        #scores.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))
        #print(
        #    'Score (filled with nonzero mean)',
        #    scores[-1]
        #)
    result = dict(
        # score=scores,
        leaky_count=leaky_value_counts,
        # leaky_correct=leaky_value_corrects,
    )
    return test_leak, result

test_leak, test_result = compiled_leak_result_test(max_nlags=38)

test_result = pd.DataFrame.from_dict(test_result, orient='columns')

test_result.to_csv('test_leaky_stat.csv', index=False)

best_lag = 37

test_leak = rewrite_compiled_leak(test_leak, best_lag)
test_leak[['ID']+leaky_cols+['compiled_leak']].head()

test_res = test_leak[leaky_cols+['compiled_leak']].replace(0.0, np.nan)
test_res.to_csv('test_leak.csv', index=False)

test_leak.loc[test_leak["compiled_leak"]==0, "compiled_leak"] = test_leak.loc[test_leak["compiled_leak"]==0, "nonzero_mean"]

#submission
sub = test[["ID"]]
sub["target"] = test_leak["compiled_leak"]
sub.to_csv(f"baseline_sub_lag_{best_lag}.csv", index=False)
print(f"baseline_sub_lag_{best_lag}.csv saved")
