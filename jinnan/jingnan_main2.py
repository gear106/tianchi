# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 21:05:14 2019

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

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

root = 'data/'

train = pd.read_csv(root + 'jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv(root + 'jinnan_round1_testA_20181227.csv', encoding='gb18030')

# 删除类别唯一的特征
for df in [train, test]:
    df.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)
    
# 删除缺失率超过90%的列
good_cols = list(train.columns)
for col in train.columns:
    rate = train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.9:
        good_cols.remove(col)
        
train = train[good_cols]
good_cols.remove('收率')
test  = test[good_cols]

# 删除异常值
train = train[train['收率'] > 0.89]

# 合并数据集
target = train['收率']
del train['收率']
data = pd.concat([train,test],axis=0,ignore_index=True)
data = data.fillna(-1)

# 日期中有些输入错误和遗漏
def t2s(t):
    try:
        t,m,s=t.split(":")
    except:
        if t=='1900/1/9 7:00':
            return 7*3600
        elif t=='1900/1/1 2:30':
            return 2*3600+30*60
        elif t==-1:
            return -1
        else:
            return 0
    
    try:
        tm = int(t)*3600+int(m)*60+int(s)
    except:
        return 30*60
    
    return tm
for f in ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']:
    data[f] = data[f].apply(t2s)

def getDuration(se):
    try:
        sh,sm,eh,em=re.split("[:,-]",se)
    except:
        if se=='14::30-15:30':
            return 3600
        elif se=='13；00-14:00':
            return 3600
        elif se=='21:00-22；00':
            return 3600
        elif se=='22"00-0:00':
            return 7200
        elif se=='2:00-3;00':
            return 3600
        elif se=='1:30-3;00':
            return 5400
        elif se=='15:00-1600':
            return 3600
        elif se==-1:
            return -1
        else:
            return 30*60
        
    try:
        tm = abs(int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)
    except:
        if se=='19:-20:05':
            return 3600
        else:
            return 30*60
    
    return tm

for f in ['A20','A28','B4','B9','B10','B11']:
    data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)
    
def sample(s):
    s, n=s.split("_")
    
    return int(n)
   
data['样本id'] = data.apply(lambda df: sample(df['样本id']), axis=1)
    
    
    
#cate_columns = [f for f in data.columns if f != '样本id']
#data = data[cate_columns]
scaler1 = StandardScaler().fit(data)
scaler2 = MinMaxScaler().fit(data)

train_X = scaler1.transform(data[:train.shape[0]])
test = scaler1.transform(data[train.shape[0]:])

#train_X = scaler2.transform(data[:train.shape[0]])
#test = scaler2.transform(data[train.shape[0]:])

train_Y = target.values

train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.1, random_state=1)
        

##############################--Ridge--########################################
ridge = Ridge(alpha=0.01, normalize=True, max_iter=1500, random_state=2019)

###############################--RFR--#########################################
myRFR = RandomForestRegressor(n_estimators=2000, max_depth=10, min_samples_leaf=10, min_samples_split=0.001,
                              max_features='auto', max_leaf_nodes=30, min_weight_fraction_leaf=0.001, random_state=10)
stack = myRFR
stack.fit(train_X, train_Y)
Y_pred = stack.predict(test_X)
print(mean_squared_error(test_Y, Y_pred))

################################--GBR--##########################################
myGBR = GradientBoostingRegressor(alpha=0.8, learning_rate=0.01, loss='huber', n_estimators=2000, max_depth=10,
                                  max_features='sqrt', max_leaf_nodes=20,
                                  random_state=20, subsample=0.8, verbose=0,
                                  warm_start=False)

#myGBR.get_params                 # 获取模型的所有参数

stack = myGBR
stack.fit(train_X, train_Y)
Y_pred = stack.predict(test_X)
print(mean_squared_error(test_Y, Y_pred))

##############################--lightgbm--####################################
mylgb = lgb.LGBMModel(boosting_type='gbdt', num_leaves=40, max_depth=7, max_bin=233, learning_rate=0.03, n_estimator=10,
                                                   subsample_for_bin=300, objective='regression', min_split_gain=0.0, 
                                                   min_child_weight=0.1, min_child_samples=20, subsample=1.0, verbose=0,
                                                   subsample_freq=1, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                                                   random_state=None, n_jobs=-1, silent=True)
stack = mylgb
stack.fit(train_X, train_Y)
Y_pred = stack.predict(test_X)
print(mean_squared_error(test_Y, Y_pred))

###############################--xgboost--######################################

cv_params = {'n_estimators': [2000]}
other_params = {'learning_rate': 0.005, 'n_estimators': 500, 'max_depth': 8, 'min_child_weight': 1, 'seed': 0,
                'max_delta_step':0.1, 
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.001, 'reg_alpha': 0, 'reg_lambda': 1}

model = xgb.XGBRegressor(**other_params)
mgb = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1)
mgb.fit(train_X, train_Y)
print('参数的最佳取值：{0}'.format(mgb.best_params_))
print('最佳模型得分:{0}'.format(-mgb.best_score_))
myxgb = mgb.best_estimator_

stack = myxgb
stack.fit(train_X, train_Y)
Y_pred = stack.predict(test_X)
print(mean_squared_error(test_Y, Y_pred))

###############################--模型融合--######################################
stack = StackingCVRegressor(regressors=[myxgb, myRFR, mylgb, myGBR], meta_regressor=ridge,
                             use_features_in_secondary=True, cv=5)
                
stack.fit(train_X, train_Y)
Y_pred = stack.predict(test_X)
print(mean_squared_error(test_Y, Y_pred))
pred_Y = stack.predict(test)

sub_df = pd.read_csv(root + 'jinnan_round1_submit_20181227.csv', header=None)
sub_df[1] = pred_Y
sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))
sub_df.to_csv("2019.csv", index=False, header=None)