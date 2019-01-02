# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:45:18 2019

@author: GEAR
"""

import pandas as pd 
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor



root = 'data/'
train = pd.read_csv(root + 'zhengqi_train.txt', sep='\t')
test = pd.read_table(root + 'zhengqi_test.txt')

features = train.drop(['target'], axis=1).values
labels = train.target.values

scaler1 = StandardScaler()
features = scaler1.fit_transform(features)

train_X, test_X, train_Y, test_Y = train_test_split(features, labels, test_size=0.2, random_state=1)
##############################--Ridge--########################################
ridge = Ridge(random_state=2019)

##############################--SVR--##########################################
mySVR = SVR(C=1.8, epsilon=0.2, gamma=0.005)

##############################--GBR--##########################################
myGBR = GradientBoostingRegressor(learning_rate=0.03, loss='ls', n_estimators=1000, max_depth=4,
                                  max_features='sqrt', max_leaf_nodes=None,
                                  random_state=10, subsample=0.8, verbose=0,
                                  warm_start=False)

#myGBR.get_params                 # 获取模型的所有参数

##############################--xgboost--##########################################

cv_params = {'n_estimators': [700]}
other_params = {'learning_rate': 0.05, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

model = xgb.XGBRegressor(**other_params)
mgb = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1)
mgb.fit(train_X, train_Y)
print('参数的最佳取值：{0}'.format(mgb.best_params_))
print('最佳模型得分:{0}'.format(-mgb.best_score_))
myxgb = mgb.best_estimator_

##############################--模型融合--######################################
stack = StackingCVRegressor(regressors=[mySVR, myGBR, myxgb], meta_regressor=LinearRegression(),
                             use_features_in_secondary=True, cv=5)
                
stack.fit(train_X, train_Y)
pred_Y = stack.predict(test_X)
print(mean_squared_error(test_Y, pred_Y))

Y_pred = stack.predict(test.values)
results = pd.DataFrame(Y_pred, columns=['target'])
results.to_csv("results.txt", index=False, header=False)
print("over")

