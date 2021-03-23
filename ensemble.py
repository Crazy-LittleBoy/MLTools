# -*- coding:utf-8 -*-
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from cross_validation import KFoldCrossValidate
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from deepforest import CascadeForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 读取训练数据
start_time = time.time()
train = pd.read_csv('./data/train_data_clean.csv',
                    # nrows=30000,
                    low_memory=False,
                    )
print(train.shape)
# train = train.sample(frac=1).reset_index(drop=True)
print(train.head(3))
feature_names = train.columns.tolist()[1:-1]
train = train.values
train_x, train_y = train[:, 1:-1], train[:, -1]
print('[Info]: train data shape: ', train_x.shape, train_y.shape)

# 读取测试数据
test = pd.read_csv('./data/test_data_clean.csv',
                   low_memory=False,
                   )
print(test.shape)
print(test.head(3))
test_result = test[['ID']]
test = test.values
test_x, test_y = test[:, 1:], test[:, -1]
print('[Info]: test data shape: ', test_x.shape, test_y.shape)

# 载入模型参数
with open('./config/parameters.json', 'r') as jfile:
    model_config = json.load(jfile)
print(model_config)

model_dict = {'xgb_model': XGBClassifier,
              'lgb_model': LGBMClassifier,
              'rf_model': RandomForestClassifier,
              'gbdt_model': GradientBoostingClassifier,
              'adaboost_model': AdaBoostClassifier,
              }

# K折交叉验证
cv_results = []
train_results = []
k_folds = 7
for model_name in model_dict.keys():
    print('[Info]: %s training...' % model_name)
    if 'xgb' in model_name or 'lgb' in model_name:
        with_early_stopping = True
    else:
        with_early_stopping = False
    kfcv = KFoldCrossValidate(k=k_folds,
                              estimator_name=model_name,
                              estimator=model_dict[model_name],
                              model_params=model_config[model_name]['model_params'],
                              fit_params=model_config[model_name]['fit_params'],
                              metric=roc_auc_score,
                              with_early_stopping=with_early_stopping,
                              )
    kfcv.fit(train_x, train_y)
    train_pred = kfcv.predict(train_x, ensemble_way='average')
    train_results.append(train_pred)
    test_pred = kfcv.predict(test_x, ensemble_way='average')
    cv_results.append(test_pred)
    print('%s train score: %s' % (model_name, roc_auc_score(train_y, train_pred)))
    print(classification_report(train_y, (train_pred > 0.5) * 1, zero_division=0))

# 模型融合(多个模型预测结果取平均)
train_results = np.mean(train_results, axis=0)
print('all train models score: %s' % roc_auc_score(train_y, train_results))
print(classification_report(train_y, (train_results > 0.5) * 1, zero_division=0))
cv_results = np.mean(cv_results, axis=0)
test_result['TARGET'] = cv_results
print(test_result.head())
test_result.to_csv('./output/submission_ensemble.cv7.csv', index=False)
