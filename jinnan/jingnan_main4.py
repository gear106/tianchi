# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:17:12 2019

@author: gear
github:  https://github.com/gear106
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import xgboost as xgb
import lightgbm as lgb
import warnings
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor
from scipy import sparse
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

root = 'data/'

train = pd.read_csv(root + 'jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv(root + 'jinnan_round1_testA_20181227.csv', encoding='gb18030')

# 删除类别唯一的特征
for df in [train, test]:
    df.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)
    
# 删除某一类别占比超过90%的列
good_cols = list(train.columns)
for col in train.columns:
    rate = train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.9:
        good_cols.remove(col)
        print(col,rate)
        
# 暂时不删除，后面构造特征需要
good_cols.append('A1')
good_cols.append('A3')
good_cols.append('A4')

# 删除异常值
train = train[train['收率']>0.88]
        
train = train[good_cols]
good_cols.remove('收率')
test  = test[good_cols]

# 合并数据集
target = train['收率']
del train['收率']
data = pd.concat([train,test],axis=0,ignore_index=True)
data = data.fillna(-1)

def timeTranSecond(t):
    try:
        t,m,s=t.split(":")
    except:
        if t=='1900/1/9 7:00':
            return 7*3600/3600
        elif t=='1900/1/1 2:30':
            return (2*3600+30*60)/3600
        elif t==-1:
            return -1
        else:
            return 0
    
    try:
        tm = (int(t)*3600+int(m)*60+int(s))/3600
    except:
        return (30*60)/3600
    
    return tm
for f in ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']:
    try:
        data[f] = data[f].apply(timeTranSecond)
    except:
        print(f,'应该在前面被删除了！')

def getDuration(se):
    try:
        sh,sm,eh,em=re.findall(r"\d+\.?\d*",se)
    except:
        if se == -1:
            return -1 
        
    try:
        if int(sh)>int(eh):
            tm = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600 + 24
        else:
            tm = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600
    except:
        if se=='19:-20:05':
            return 1
        elif se=='15:00-1600':
            return 1
    
    return tm
for f in ['A20','A28','B4','B9','B10','B11']:
    data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)
    
    
data['样本id'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))

categorical_columns = [f for f in data.columns if f not in ['样本id']]
numerical_columns = [f for f in data.columns if f not in categorical_columns]

data['b14/a1_a3_a4_a19_b1_b12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])

numerical_columns.append('b14/a1_a3_a4_a19_b1_b12')

del data['A1']
del data['A3']
del data['A4']
categorical_columns.remove('A1')
categorical_columns.remove('A3')
categorical_columns.remove('A4')

data['S1'] = data['A7'] - data['A5']
data['S2'] = data['A11'] - data['A9']
data['S3'] = data['A25'] - data['A21']
data['S4'] = data['B5'] - data['B6']
#data['S5'] = data['B10'] * data['B9']
#data['S6'] = data['B11'] * data['B10']
numerical_columns.append('S1')
numerical_columns.append('S2')
numerical_columns.append('S3')
numerical_columns.append('S4')
#numerical_columns.append('S5')
#numerical_columns.append('S6')

#label encoder
for f in categorical_columns:
    data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
train = data[:train.shape[0]]
test  = data[train.shape[0]:]
print(train.shape)
print(test.shape)

train_X = train.values
test = test.values


train_Y = target.values


ridge = Ridge(alpha=0.01, normalize=True, max_iter=1500, random_state=2019)
lrege = LinearRegression()
bayes = BayesianRidge(alpha_1=1e-7, n_iter=5000)

myRFR = RandomForestRegressor(n_estimators=2000, max_depth=10, min_samples_leaf=10, min_samples_split=0.001,
                              max_features='auto', max_leaf_nodes=30, min_weight_fraction_leaf=0.001, random_state=10)

mylgb = lgb.LGBMModel(boosting_type='gbdt', num_leaves=40, max_depth=7, max_bin=233, learning_rate=0.03, n_estimator=10,
                                                   subsample_for_bin=300, objective='regression', min_split_gain=0.0, 
                                                   min_child_weight=0.1, min_child_samples=20, subsample=1.0, verbose=0,
                                                   subsample_freq=1, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                                                   random_state=None, n_jobs=-1, silent=True)

params = {'learning_rate': 0.005, 'n_estimators': 3000, 'max_depth': 9, 'min_child_weight': 1, 'seed': 0,
                'max_delta_step':0.1, 
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.001, 'reg_alpha': 0, 'reg_lambda': 1}
myxgb = xgb.XGBRegressor(**params)

stack = StackingCVRegressor(regressors=[myxgb, myRFR, mylgb], meta_regressor=ridge,
                             use_features_in_secondary=True, cv=5)

train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.1, random_state=2019)

stack.fit(train_X, train_Y)
pred_Y = stack.predict(test_X)
mse = mean_squared_error(test_Y, pred_Y)
print('mse: %.10f' % mse)

#folds = KFold(n_splits=7, shuffle=True, random_state=2019)
#
#mean = []
#for fold, (i, j) in enumerate(folds.split(train_X, train_Y)):
#    print("fold {}".format(fold+1))
#    trn_X, trn_Y = train_X[i], train_Y[i]
#    tsn_X, tsn_Y = train_X[j], train_Y[j]
#    
#    stack = stack
#    stack.fit(trn_X, trn_Y)
#    pred_Y = stack.predict(tsn_X)
#    mse = mean_squared_error(tsn_Y, pred_Y)
#    print('mse: %.10f' % mse)
#    mean.append(mse)
#print('mean_cv5_error: %.10f' % (sum(mean) / 7))

