# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 21:46:41 2019

@author: gear
github:  https://github.com/gear106
"""
import pandas as pd 
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import warnings

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import pca
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor


color = sns.color_palette()  # sns调色板
sns.set_style('dark')  # 设置主题模式
# 1、数据获取
# 2、数据处理
# 3、模型训练
# 4、结果预测
warnings.filterwarnings('ignore')  # 忽略警告（看到一堆警告比较恶心）

'''数据获取'''
root = 'data/'
df = pd.read_table(root + "zhengqi_train.txt")
df_test = pd.read_table(root + "zhengqi_test.txt")

'''数据处理'''
# 找出相关程度
#plt.figure(figsize=(20, 16))  # 指定绘图对象宽度和高度
#colnm = df.columns.tolist()[:39]  # 列表头
#mcorr = df[colnm].corr()  # 相关系数矩阵，即给出了任意两个变量之间的相关系数
#mask = np.zeros_like(mcorr, dtype=np.bool)  # 构造与mcorr同维数矩阵 为bool型
#mask[np.triu_indices_from(mask)] = True  # 角分线右侧为True
#cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象
#g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）
#plt.show()

# 画箱式图
#colnm = df.columns.tolist()[:39]  # 列表头
#fig = plt.figure(figsize=(10, 10))  # 指定绘图对象宽度和高度
#for i in range(38):
#    plt.subplot(13, 3, i + 1)  # 13行3列子图
#    sns.boxplot(df[colnm[i]], orient="v", width=0.5)  # 箱式图
#    plt.ylabel(colnm[i], fontsize=12)
#plt.show()

# 画正太分布图
#sns.distplot(df['target'], fit=norm)
#(mu, sigma) = norm.fit(df['target'])
#print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
#plt.ylabel('Frequency')
#fig = plt.figure()
#res = stats.probplot(df['target'], plot=plt)
#plt.show()

# 查看是否有缺失值（无缺失值 无需补植）
train_data_missing = (df.isnull().sum() / len(df)) * 100
#print(train_data_missing)

# 分析特性与目标值的相关性 热力图
#corrmat = df.corr()
#f, ax = plt.subplots(figsize=(20, 9))
#sns.heatmap(corrmat, vmax=.8, annot=True)
#plt.show()

# 对训练数据处理，分离出特征和标签
X = df.values[:, 0:-1]
Y = df.values[:, -1]
X1_test = df_test.values

# PCA数据处理
pca = pca.PCA(n_components=0.95)
pca.fit(X)
X_pca = pca.transform(X)
X1_pca = pca.transform(X1_test)

'''模型训练'''
# 分离出训练集和测试集，并用梯度提升回归训练
train_X, test_X, train_Y, test_Y = train_test_split(X_pca, Y, test_size=0.3, random_state=1)


#train_X, test_X, train_Y, test_Y = train_test_split(features, labels, test_size=0.2, random_state=1)
##############################--Ridge--########################################
ridge = Ridge(alpha=0.01, normalize=True, max_iter=1500, random_state=2019)

##############################--SVR--##########################################
#myRFR = RandomForestRegressor(n_estimators=3000, max_depth=10, min_samples_leaf=5, min_samples_split=0.001,
#                              max_features='auto', max_leaf_nodes=1000, min_weight_fraction_leaf=0.001, random_state=1)
#stack = myRFR
#stack.fit(train_X, train_Y)
#Y_pred = stack.predict(test_X)
#print(mean_squared_error(test_Y, Y_pred))

###############################--lightgbm--##########################################
mylgb = lgb.LGBMModel(boosting_type='gbdt', num_leaves=80, max_depth=15, max_bin=400, learning_rate=0.05, n_estimator=20,
                                                   subsample_for_bin=500, objective='regression', min_split_gain=0.0, 
                                                   min_child_weight=0.2, min_child_samples=20, subsample=1.0, verbose=0,
                                                   subsample_freq=1, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                                                   random_state=None, n_jobs=-1, silent=True)
stack = mylgb
stack.fit(train_X, train_Y)
Y_pred = stack.predict(test_X)
print(mean_squared_error(test_Y, Y_pred))

###############################--GBR--##########################################
#myGBR = GradientBoostingRegressor(alpha=0.7, learning_rate=0.03, loss='huber', n_estimators=2000, max_depth=10,
#                                  max_features='sqrt', max_leaf_nodes=10,
#                                  random_state=20, subsample=0.8, verbose=0,
#                                  warm_start=False)
#
##myGBR.get_params                 # 获取模型的所有参数
#stack = myGBR
#stack.fit(train_X, train_Y)
#Y_pred = stack.predict(test_X)
#print(mean_squared_error(test_Y, Y_pred))
#
#
################################--xgboost--#######################################
cv_params = {'n_estimators': [3000]}
other_params = {'learning_rate': 0.005, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

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


################################--模型融合--######################################
stack = StackingCVRegressor(regressors=[myxgb, mylgb], meta_regressor=LinearRegression(),
                             use_features_in_secondary=True, cv=5)
                
stack.fit(train_X, train_Y)
pred_Y = stack.predict(test_X)
print(mean_squared_error(test_Y, pred_Y))

Y_pred = stack.predict(X1_pca)
results = pd.DataFrame(Y_pred, columns=['target'])
results.to_csv("results.txt", index=False, header=False)
print("over")

