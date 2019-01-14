# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 19:13:34 2019

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

# 删除异常值
train = train[train['收率']>0.87]
        
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

#label encoder
for f in categorical_columns:
    data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
train = data[:train.shape[0]]
test  = data[train.shape[0]:]
print(train.shape)
print(test.shape)

#train['target'] = list(target) 
train['target'] = target
train['intTarget'] = pd.cut(train['target'], 5, labels=False)
train = pd.get_dummies(train, columns=['intTarget'])
li = ['intTarget_0.0','intTarget_1.0','intTarget_2.0','intTarget_3.0','intTarget_4.0']
mean_columns = []
for f1 in categorical_columns:
    cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
    if cate_rate < 0.90:
        for f2 in li:
            col_name = 'B14_to_'+f1+"_"+f2+'_mean'
            mean_columns.append(col_name)
            order_label = train.groupby([f1])[f2].mean()
            train[col_name] = train['B14'].map(order_label)
            miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
            if miss_rate > 0:
                train = train.drop([col_name], axis=1)
                mean_columns.remove(col_name)
            else:
                test[col_name] = test['B14'].map(order_label)
                
train.drop(li+['target'], axis=1, inplace=True)
print(train.shape)
print(test.shape)

data = pd.concat([train,test],axis=0,ignore_index=True)
data = data.fillna(-1)

##cate_columns = [f for f in data.columns if f != '样本id']
##data = data[cate_columns]
scaler1 = StandardScaler().fit(data)
scaler2 = MinMaxScaler().fit(data)

#train_X1 = scaler1.transform(data[:train.shape[0]])
#test = scaler1.transform(data[train.shape[0]:])

train_X1 = scaler2.transform(data[:train.shape[0]])
test = scaler2.transform(data[train.shape[0]:])

train_Y1 = target.values

train_X, test_X, train_Y, test_Y = train_test_split(train_X1, train_Y1, test_size=0.1, random_state=1)

#train_X = train_X.reshape(-1, 4, 8, 1)
#test_X = test_X.reshape(-1, 4, 8, 1)

#
######################################CNN-model##################################
#from keras.layers import Conv2D, Input, Dense, Flatten, Dropout, BatchNormalization
#from keras.layers import Add, Activation
#from keras.models import Model
#
#k_units = 256
#
#
#def identify_block(inputs):
#    
#    X_left = inputs
#    X_right = inputs
#    
#    X_left = Dense(units=k_units, activation='relu')(X_left)
#    X_left = BatchNormalization()(X_left)
#    X_left = Dense(units=k_units, activation='relu')(X_left)
#    X_left = BatchNormalization()(X_left)
#    
##    X_right = Dense(units=125, activation='relu')(X_right)
##    X_right = Dense(units=125, activation='relu')(X_right)
##    X_right = BatchNormalization()(X_right)
#    
#    X = Dense(units=k_units, activation='relu')(inputs)
#    X = BatchNormalization()(X) 
#    X = Dense(units=k_units, activation='relu')(X)
#    X = BatchNormalization()(X) 
#    X = Dense(units=k_units, activation='relu')(X)
#    X = BatchNormalization()(X)  
#    X = Add()([X, X_left])
#    X = Activation('relu')(X)
#   
#    return X
#    
#
#input_shape = train_X.shape[1:]
#inputs = Input(shape=input_shape)
#
## 第01层网络
#X = Dense(units=k_units, activation='tanh')(inputs)
#X = BatchNormalization()(X)
##X = Dropout(rate=0.5)(X)
#
#X = Dense(units=k_units, activation='tanh')(X)
#X = BatchNormalization()(X)
#X = Dense(units=k_units, activation='tanh')(X)
#X = BatchNormalization()(X)
#X = Dense(units=k_units, activation='tanh')(X)
#X = BatchNormalization()(X)
#X = Dense(units=k_units, activation='tanh')(X)
#X = BatchNormalization()(X)
#X = Dense(units=k_units, activation='tanh')(X)
#X = BatchNormalization()(X)
#X = Dense(units=k_units, activation='tanh')(X)
#X = BatchNormalization()(X)
#
#X = Dense(units=k_units, activation='relu')(X)
#
#X = Dense(units=1, name='dense1')(X)
#
#model = Model(inputs=inputs, outputs=X)
#
#model.compile(optimizer='adam', loss='mse')
#
#model.fit(train_X, train_Y, epochs=1000, batch_size=139, verbose=2, validation_data=(test_X, test_Y))
#
#pred_Y = model.predict(test_X)
#print(mean_squared_error(test_Y, pred_Y))


#ridge = Ridge(alpha=0.01, normalize=True, max_iter=1500, random_state=2019)
#lrege = LinearRegression()
#bayes = BayesianRidge(alpha_1=1e-7, n_iter=5000)
#
#myRFR = RandomForestRegressor(n_estimators=2000, max_depth=5, min_samples_leaf=10, min_samples_split=0.001,
#                              max_features='auto', max_leaf_nodes=30, min_weight_fraction_leaf=0.001, random_state=10)
#stack = myRFR
#stack.fit(train_X, train_Y)
#pred_Y = stack.predict(test_X)
#mse = mean_squared_error(test_Y, pred_Y)
#print('mse: %.10f' % mse)

mylgb = lgb.LGBMModel(boosting_type='gbdt', num_leaves=50, max_depth=7, max_bin=233, learning_rate=0.03, n_estimator=10,
                                                   subsample_for_bin=300, objective='regression', min_split_gain=0.0, 
                                                   min_child_weight=0.1, min_child_samples=20, subsample=1.0, verbose=0,
                                                   subsample_freq=1, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                                                   random_state=None, n_jobs=-1, silent=True)


stack = mylgb
stack.fit(train_X, train_Y)
pred_Y = stack.predict(test_X)
mse = mean_squared_error(test_Y, pred_Y)
print('mse: %.10f' % mse)
#params = {'learning_rate': 0.005, 'n_estimators': 3000, 'max_depth': 9, 'min_child_weight': 1, 'seed': 0,
#                'max_delta_step':0.1, 
#                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.001, 'reg_alpha': 0, 'reg_lambda': 1}
#myxgb = xgb.XGBRegressor(**params)
#
#stack = StackingCVRegressor(regressors=[myxgb, myRFR, mylgb], meta_regressor=ridge,
#                             use_features_in_secondary=True, cv=5)
#
##train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.1, random_state=2019)
#
#stack.fit(train_X, train_Y)
#pred_Y = stack.predict(test_X)
#mse = mean_squared_error(test_Y, pred_Y)
#print('mse: %.10f' % mse)

