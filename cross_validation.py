# -*- coding:utf-8 -*-
import os
import time
import copy
import inspect
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split
import joblib
import json
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from deepforest import CascadeForestClassifier
from sklearn.linear_model import LogisticRegression


def makedirs(path):
    # noinspection PyBroadException
    try:
        os.makedirs(path)
    except:
        pass


class KFoldCrossValidate(object):
    def __init__(self, k, estimator_name, estimator, model_params, fit_params, metric, with_early_stopping=False, shuffle=True, random_state=2021):
        self.k = k
        self.estimator_name = estimator_name
        self.estimator = estimator
        self.model_params = model_params
        self.fit_params = fit_params
        self.metric = metric
        self.with_early_stopping = with_early_stopping
        self.cv_score = None
        self.models = []
        self.random_state = random_state
        self.shuffle = shuffle

    def predict_probability(self, model, X):
        if 'lgb' in self.estimator_name:
            probability = model.predict_proba(X, num_iteration=model.best_iteration_)
        elif 'xgb' in self.estimator_name:
            probability = model.predict_proba(X, ntree_limit=model.best_ntree_limit)
        else:
            probability = model.predict_proba(X)
        return probability

    def fit(self, X, y):
        pred_probs = []
        y_true = []
        splits = KFold(n_splits=self.k, shuffle=self.shuffle, random_state=self.random_state)
        for index, (train_index, valid_index) in enumerate(splits.split(X, y)):
            print('[Info]: %s fold:%s/%s %s' % ('-' * 50, index + 1, self.k, '-' * 50))
            train_x, valid_x, train_y, valid_y = X[train_index], X[valid_index], y[train_index], y[valid_index]
            if self.with_early_stopping:
                sss = StratifiedShuffleSplit(n_splits=1, random_state=self.random_state, train_size=0.7, test_size=0.3)
                for index_, (train_index_, valid_index_) in enumerate(sss.split(train_x, train_y)):
                    train_split_x, valid_split_x, train_split_y, valid_split_y = train_x[train_index_], train_x[valid_index_], train_y[train_index_], train_y[valid_index_]
                model = self.estimator(**self.model_params).fit(train_split_x, train_split_y, eval_set=[(valid_split_x, valid_split_y)], **self.fit_params)
            else:
                model = self.estimator(**self.model_params).fit(train_x, train_y)
            self.models.append(model)
            pred_prob = self.predict_probability(model, valid_x)
            if pred_prob.shape[1] > 1:
                pred_prob = pred_prob[:, -1]  # 取预测为1的概率
            pred_probs.append(pred_prob)
            y_true.append(valid_y)
        y_score = np.concatenate(pred_probs).ravel()
        y_pred = (y_score > 0.5) * 1
        y_true = np.concatenate(y_true).ravel()
        if 'y_score' in inspect.getfullargspec(self.metric).args:
            self.cv_score = self.metric(y_true, y_score)
        else:
            self.cv_score = self.metric(y_true, y_pred)
        print('[Info]: cv score: %s' % self.cv_score)
        print(classification_report(y_true, y_pred, zero_division=0))

    def predict(self, X, ensemble_way='average'):
        pred_probs = []
        for model in self.models:
            pred_prob = self.predict_probability(model, X)
            pred_probs.append(pred_prob[:, -1])
        if ensemble_way in ['average', 'avg', 'mean']:
            y_pred = np.mean(pred_probs, axis=0).ravel()
        elif ensemble_way in ['voting', 'vote']:
            y_pred = (np.array(pred_probs) > 0.5) * 1
            y_shape = y_pred.shape
            y_pred = (np.sum(y_pred, axis=0) > y_shape[0] / 2.0) * 1
        else:
            return pred_probs
        return y_pred


if __name__ == '__main__':
    # # 读取数据
    #
    # start_time = time.time()
    # data = pd.read_csv('./data/train_data_clean.csv',
    #                    low_memory=False,
    #                    # nrows=20000,
    #                    )
    # print(data.shape)
    # # data = data.sample(frac=1).reset_index(drop=True)
    # print(data.head(3))
    # feature_names = data.columns.tolist()[1:-1]
    # data = data.values
    # # 划分数据集
    # train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(data[:, 1:-1], data[:, -1].ravel(), test_size=0.3)
    # # valid_set_x, test_set_x, valid_set_y, test_set_y = train_test_split(test_set_x, test_set_y, test_size=0.5)
    # print(train_set_x.shape, test_set_x.shape)
    # # print(train_set_y.shape, valid_set_y.shape, test_set_y.shape)
    #
    # # 调好的参数
    # with open('./config/parameters.v0.json', 'r') as jfile:
    #     model_params = json.load(jfile)
    # print(model_params)
    #
    # model_dict = {'xgb_model': XGBClassifier,
    #               'lgb_model': LGBMClassifier,
    #               'rf_model': RandomForestClassifier,
    #               'gbdt_model': GradientBoostingClassifier,
    #               'adaboost_model': AdaBoostClassifier,
    #               }
    #
    # # CV
    # k_folds = 5
    # model_name = 'xgb_model'
    # kfcv = KFoldCrossValidate(k=k_folds,
    #                           estimator_name=model_name,
    #                           estimator=model_dict[model_name],
    #                           model_params=model_params[model_name]['model_params'],
    #                           fit_params=model_params[model_name]['fit_params'],
    #                           metric=roc_auc_score,
    #                           with_early_stopping=True,
    #                           )
    # kfcv.fit(train_set_x, train_set_y)
    # test_pred = kfcv.predict(test_set_x, ensemble_way='average')
    # print('test score:', roc_auc_score(test_set_y, test_pred))
    # print(min(test_pred), max(test_pred), )
    # print(classification_report(test_set_y, (test_pred > 0.5) * 1, zero_division=0))

    # ----------------------------------------------------------------------------------------------#
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

    # 调好的参数
    with open('./config/parameters.v0.json', 'r') as jfile:
        model_params = json.load(jfile)
    print(model_params)

    model_dict = {'xgb_model': XGBClassifier,
                  'lgb_model': LGBMClassifier,
                  'rf_model': RandomForestClassifier,
                  'gbdt_model': GradientBoostingClassifier,
                  'adaboost_model': AdaBoostClassifier,
                  }

    # CV
    k_folds = 5
    model_name = 'xgb_model'
    kfcv = KFoldCrossValidate(k=k_folds,
                              estimator_name=model_name,
                              estimator=model_dict[model_name],
                              model_params=model_params[model_name]['model_params'],
                              fit_params=model_params[model_name]['fit_params'],
                              metric=roc_auc_score,
                              with_early_stopping=True,
                              )
    kfcv.fit(train_x, train_y)
    test_pred = kfcv.predict(test_x, ensemble_way='average')
    test_result['TARGET'] = test_pred
    print(test_result.head())
    test_result.to_csv('./output/submission_xgb_cv_5.csv', index=False)

