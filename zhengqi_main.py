# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:45:18 2019

@author: GEAR
"""

import pandas as pd 
import numpy as np
#import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


root = 'data/'
train = pd.read_csv(root + 'zhengqi_train.txt', sep='\t')
test = pd.read_table(root + 'zhengqi_test.txt')

features = train.drop(['target'], axis=1).values
labels = train.target.values

scaler1 = StandardScaler()
features = scaler1.fit_transform(features)

train_X, test_X, train_Y, test_Y = train_test_split(features, labels, test_size=0.3, random_state=1)

#myGBR = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#                                  learning_rate=0.03, loss='ls', max_depth=4,
#                                  max_features='sqrt', max_leaf_nodes=None,
#                                  min_impurity_decrease=0.0, min_impurity_split=None,
#                                  min_samples_leaf=1, min_samples_split=2,
#                                  min_weight_fraction_leaf=0.0, n_estimators=1000,
#                                  presort='auto', random_state=10, subsample=0.8, verbose=0,
#                                  warm_start=False)
#myGBR.fit(train_X, train_Y)
#pred_Y = myGBR.predict(test_X)
#print(mean_squared_error(test_Y, pred_Y))

myGBR = GradientBoostingRegressor()
myGBR.get_params                 # 获取模型的所有参数

params = {
          'learning_rate': [0.03],
          'max_features': ['sqrt'],
          'random_state': [10],
          'max_depth':[5],
          'm_estimators': [300, 500, 1000, 1500]
          }
clf = GridSearchCV(estimator=myGBR, param_grid=params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
clf.fit(train_X, train_Y)
print('最佳参数：', clf.best_params_)
print('mse：', - clf.best_score_)


