import os
import datetime
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

my_dir = os.getcwd()
df_train = pd.read_csv(my_dir+'/Homesite/Data/train.csv')
df_test = pd.read_csv(my_dir+'/Homesite/Data/test.csv')

n_train = df_train.shape[0]
y = df_train['QuoteConversion_Flag'].values
df_train.drop('QuoteConversion_Flag', axis=1, inplace=True)
ids = df_test['QuoteNumber']

# combine train and test
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# drop ids
df_all.drop('QuoteNumber', axis=1, inplace=True)

# preprocess dates
oqd = np.vstack(df_all['Original_Quote_Date'].apply(lambda x: list(map(int, x.split('-')))).values)
df_all['oqd_year'] = oqd[:, 0]
df_all['oqd_month'] = oqd[:, 1]
df_all['oqd_day'] = oqd[:, 2]
df_all.drop('Original_Quote_Date', axis=1, inplace=True)
oqd_weekday = []
for i in range(df_all.shape[0]):
  oqd_weekday += [datetime.date(oqd[i, 0], oqd[i, 1], oqd[i, 2]).weekday()]

df_all['oqd_weekday'] = oqd_weekday

df_all.replace(-1, np.nan, inplace=True)

# OHE
for f in df_all.columns:
	if df_all[f].dtype=='object':
		print(f)
		df_all_dum = pd.get_dummies(df_all[f], prefix=f)
		df_all.drop(f, axis=1, inplace=True)
		df_all = pd.concat([df_all, df_all_dum], axis=1)

# replace NaN with -1
df_all.fillna(-1, inplace=True)

X_all = df_all.values
X = X_all[:n_train, :]

# train test split
# specify random seed
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)

## fit by xgb
xg_train = xgb.DMatrix(X_train, y_train)
xg_val = xgb.DMatrix(X_val, y_val)

# specify parameters for xgb
# no num_class!
param = {}
param['booster'] = "gbtree"
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'auc'
param['nthread'] = 20
param['silent'] = 1

param['colsample_bytree'] = 0.8
param['subsample'] = 0.8
param['eta'] = 0.01

num_round = 10000

evallist  = [(xg_train,'train'), (xg_val,'eval')]
bst = xgb.train(param, xg_train, num_round, evallist, early_stopping_rounds=100)

X_test = X_all[n_train:, :]
xg_test = xgb.DMatrix(X_test)

y_pred = bst.predict(xg_test, ntree_limit=bst.best_iteration)
sub = pd.DataFrame(data={'QuoteNumber':ids, 'QuoteConversion_Flag':y_pred}, 
	columns=['QuoteNumber', 'QuoteConversion_Flag'])
my_dir = os.getcwd()+'/Homesite/Subs/'
sub.to_csv(my_dir+'sub.csv', index=False)
