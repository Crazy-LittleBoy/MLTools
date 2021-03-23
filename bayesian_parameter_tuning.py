# -*- coding:utf-8 -*-
import os
import time
import copy
import inspect
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from deepforest import CascadeForestClassifier
from sklearn.linear_model import LogisticRegression
from logger import write_print_to_file
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('max_colwidth', 100)
pd.set_option('display.width', 1000)

max_eval = 0
last_max_eval = 0


def makedirs(path):
    # noinspection PyBroadException
    try:
        os.makedirs(path)
    except:
        pass


def threshold_absolute_error(x, threshold, ratio=0.0):
    """
    带有阈值的绝对损失
    当输入的值小于阈值时损失为:阈值减输入值
    当输入的值大于等于阈值时损失为:输入值减阈值再乘以一个损失比率(通常取0)
    Args:
        x: 优化指标
        threshold: 期望达到的优化指标值
        ratio: 惩罚因子
    Returns:
        绝对损失
    """
    return (x < threshold) * (threshold - x) + (x >= threshold) * (x - threshold) * ratio


def surrogate_loss_func_for_auc(y_true, y_score, expect_precision=None, expect_recall=None):
    """
    AUC代理损失函数
    Args:
        y_true: 真实标签
        y_score: 预测概率
        expect_recall: 无意义，为保持形式一直
        expect_precision: 无意义，为保持形式一直
    Returns:
        loss
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_score)
    auc_score = auc(false_positive_rate, true_positive_rate)
    loss = 1 - auc_score
    return loss


def surrogate_loss_func_for_roc_auc(y_true, y_score, expect_precision=None, expect_recall=None):
    """
    AUC代理损失函数
    Args:
        y_true: 真实标签
        y_score: 预测概率
        expect_recall: 无意义，为保持形式一直
        expect_precision: 无意义，为保持形式一直
    Returns:
        loss
    """
    roc_auc_score_ = roc_auc_score(y_true, y_score)
    loss = 1 - roc_auc_score_
    return loss


def surrogate_loss_func_for_acc(y_true, y_pred, expect_precision=None, expect_recall=None):
    """
    AUC代理损失函数
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        expect_recall: 无意义，为保持形式一直
        expect_precision: 无意义，为保持形式一直
    Returns:
        loss
    """
    acc_score = accuracy_score(y_true, y_pred)
    loss = 1 - acc_score
    return loss


def surrogate_loss_func_for_pr1(y_true, y_pred, expect_precision, expect_recall):
    """
    对类别1的查准查全代理损失
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        expect_precision: 期望达到的查准率
        expect_recall: 期望达到的查全率

    Returns:
        loss
    """
    precision_1 = precision_score(y_true, y_pred, average='weighted', labels=[1], zero_division=0)  # 类别1的查准率
    recall_1 = recall_score(y_true, y_pred, average='weighted', labels=[1], zero_division=0)  # 类别1的查准率
    ratio = (expect_precision / expect_recall)
    loss = threshold_absolute_error(precision_1, expect_precision, 0.0) + threshold_absolute_error(recall_1, expect_recall, 0.0) * ratio  # 代理损失函数
    return loss


# def surrogate_loss_func_for_pr0(y_true, y_pred, expect_precision, expect_recall):
#     """
#     对类别0的查准查全代理损失
#     Args:
#         y_true: 真实标签
#         y_pred: 预测标签
#         expect_precision: 期望达到的查准率
#         expect_recall: 期望达到的查全率
#
#     Returns:
#         loss
#     """
#     precision_0 = precision_score(y_true, y_pred, average='weighted', labels=[0], zero_division=0)  # 类别0的查准率
#     recall_0 = recall_score(y_true, y_pred, average='weighted', labels=[0], zero_division=0)  # 类别0的查准率
#     loss = threshold_absolute_error(precision_0, expect_precision, 0.0) + threshold_absolute_error(recall_0, expect_recall, 0.0) * (expect_precision / expect_recall)  # 代理损失函数
#     return loss


def tuning(dict_model, tuning_params, fit_params, other_params, surrogate_loos_func, label_dim, train_set_x, train_set_y, test_set_x, test_set_y):
    """
    贝叶斯调参主函数
    Args:
        dict_model: dict like: {'model_name': model_object}，model_object必须实现fit方法和predict方法
        tuning_params: 可调参数
        fit_params: 模型参数
        other_params: 其它参数
        surrogate_loos_func: 代理损失函数
        label_dim: 标签维度(标签类别数)
        train_set_x: 训练集x
        train_set_y: 训练集y
        test_set_x: 测试集x
        test_set_y: 测试集y
    Returns:
        loss 代理损失
    """
    global max_eval, last_max_eval
    # print('[Info]: training %s ...' % list(dict_model.keys())[0])
    train_x, train_y, test_x, test_y = train_set_x, train_set_y, test_set_x, test_set_y
    test_probs = []
    model_name = list(dict_model.keys())[0]
    model = dict_model[model_name](**tuning_params)
    if 'xgb' in model_name or 'lgb' in model_name:
        # 使用早停策略
        sss = StratifiedShuffleSplit(n_splits=1, random_state=tuning_params['random_state'], train_size=0.7, test_size=0.3)
        for index, (train_index, valid_index) in enumerate(sss.split(train_x, train_y)):
            train_split_x, valid_split_x, train_split_y, valid_split_y = train_x[train_index], train_x[valid_index], train_y[train_index], train_y[valid_index]
            model.fit(train_split_x, train_split_y, eval_set=[(valid_split_x, valid_split_y)], **fit_params)
            if 'lgb' in model_name:
                test_prob = model.predict_proba(test_x, num_iteration=model.best_iteration_)  # 使用最佳训练步数的模型预测
            elif 'xgb' in model_name:
                test_prob = model.predict_proba(test_x, ntree_limit=model.best_ntree_limit)
            test_probs.append(test_prob)
        test_probs = np.vstack(test_probs)
    else:
        model.fit(train_x, train_y)
        # noinspection PyBroadException
        try:
            test_probs = model.predict_proba(test_x)
        except Exception as e:
            print('[Error]: %s' % str(e))
            test_probs = model.predict(test_x)
    # print('test_probs.shape:', test_probs.shape)
    # print('np.unique(test_probs):', np.unique(test_probs))
    if test_probs.shape[1] > 1:
        test_preds = np.argmax(test_probs, axis=1)  # 概率转为0,1
        test_probs = test_probs[:, 1]  # 取预测为1的概率
    else:
        test_preds = (test_probs > 0.5) * 1
    test_probs = test_probs.ravel()
    y_true, y_pred = test_y.ravel(), test_preds.ravel()
    if 'y_score' in inspect.getfullargspec(surrogate_loos_func).args:
        loss = surrogate_loos_func(y_true=y_true,
                                   y_score=test_probs,
                                   expect_precision=other_params['expect_precision'],
                                   expect_recall=other_params['expect_recall'])
    else:
        loss = surrogate_loos_func(y_true=y_true,
                                   y_pred=y_pred,
                                   expect_precision=other_params['expect_precision'],
                                   expect_recall=other_params['expect_recall'])
    last_max_eval = max_eval
    max_eval = max(max_eval, 1 - loss)
    # 若代理损失下降则打印当前最佳模型评估指标
    if max_eval > last_max_eval:
        print('[Metric]: %s' % (1 - loss))
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
        print('[Parameters]:\n\t', tuning_params)
        try:
            feature_importance = pd.DataFrame(np.array([other_params['feature_names'], model.feature_importances_]).T, columns=['feature_name', 'feature_importance'])
            feature_importance = feature_importance.sort_values('feature_importance', ascending=False)
            print(feature_importance)
        except Exception as e:
            print('[Error]: %s' % str(e))
        # print(y_true[:100])
        # print(y_pred[:100])
        # print(test_probs[:1000])
        print('-' * 100, '\n')
    return loss


if __name__ == '__main__':
    # 读取数据
    start_time = time.time()
    data = pd.read_csv('./data/train_data_clean.csv',
                       # nrows=1000,
                       low_memory=False,
                       )
    print(data.shape)
    # data = data.drop(columns=['zeros'])
    data = data.sample(frac=1,random_state=2021).reset_index(drop=True)
    print(data.head(3))
    feature_names = data.columns.tolist()[1:-1]
    data = data.values
    # 划分数据集
    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(data[:, 1:-1], data[:, -1].ravel(), test_size=0.3)
    # valid_set_x, test_set_x, valid_set_y, test_set_y = train_test_split(test_set_x, test_set_y, test_size=0.5)
    print(train_set_x.shape, test_set_x.shape)
    # print(train_set_y.shape, valid_set_y.shape, test_set_y.shape)

    # 训练调参

    # --------------------------------------------------------------XGB Model--------------------------------------------------------------- #
    # dict_model = {'xgb_model': XGBClassifier}
    # makedirs('./logs/%s/' % list(dict_model.keys())[0])
    # write_print_to_file('./logs/%s/' % list(dict_model.keys())[0])
    #
    #
    # def xgb_objective(params):
    #     """
    #     目标函数
    #     """
    #     loss = tuning(dict_model=dict_model, tuning_params=params, fit_params=fit_params, other_params=other_params, surrogate_loos_func=surrogate_loss_func_for_roc_auc, label_dim=1, train_set_x=train_set_x, train_set_y=train_set_y, test_set_x=test_set_x, test_set_y=test_set_y)
    #     return {'loss': loss, 'params': params, 'status': STATUS_OK}
    #
    #
    # other_params = {'expect_precision': 0.65,
    #                 'expect_recall': 0.1,
    #                 'feature_names': feature_names,
    #                 }
    #
    # fit_params = {'early_stopping_rounds': 10,
    #               'eval_metric': 'auc',
    #               'verbose': False,
    #               }
    #
    # xgb_params = {'num_class': 1,
    #               'n_estimators': hp.choice('n_estimators', range(50, 10000, 50)),
    #               'max_depth': hp.choice('max_depth', range(3, 15, 1)),
    #               'learning_rate': hp.uniform('learning_rate', 0.1, 0.5),
    #               'objective': 'binary:logistic',
    #               'booster': hp.choice('booster', ['gbtree', 'dart']),  # ['gbtree', 'gblinear', 'dart']
    #               'n_jobs': 8,
    #               'gamma': hp.uniform('gamma', 1e-6, 10.0),
    #               # 'min_child_weight ': hp.uniform('min_child_weight', 1e-6, 10.0),
    #               # 'max_delta_step ': hp.uniform('max_delta_step', 1e-6, 10.0),
    #               # 'subsample ': hp.uniform('subsample', 0.5, 1.0),
    #               'colsample_bytree': hp.uniform('colsample_bytree', 1e-6, 1.0),
    #               'colsample_bylevel': hp.uniform('colsample_bylevel', 1e-6, 1.0),
    #               'colsample_bynode': hp.uniform('colsample_bynode', 1e-6, 1.0),
    #               'reg_alpha': hp.uniform('reg_alpha', 1e-6, 10.0),
    #               'reg_lambda': hp.uniform('reg_lambda', 1e-6, 10.0),
    #               'random_state': 2021,
    #               # 'verbosity ': 0,
    #               }
    # num_tuning_rounds = 5000  # 调参次数
    # best = fmin(fn=xgb_objective, space=xgb_params, algo=partial(tpe.suggest, n_startup_jobs=24), max_evals=num_tuning_rounds, verbose=False)
    # print(space_eval(xgb_params, best))
    # print('All time is: %s minute(s)' % ((time.time() - start_time) / 60))

    # --------------------------------------------------------------LGB Model--------------------------------------------------------------- #
    # dict_model = {'lgb_model': LGBMClassifier}
    # makedirs('./logs/%s/' % list(dict_model.keys())[0])
    # write_print_to_file('./logs/%s/' % list(dict_model.keys())[0])
    #
    # other_params = {'expect_precision': 0.65,
    #                 'expect_recall': 0.1,
    #                 'feature_names': feature_names,
    #                 }
    #
    #
    # def lgb_objective(params):
    #     """
    #     目标函数
    #     """
    #     loss = tuning(dict_model=dict_model,
    #                   tuning_params=params,
    #                   fit_params=fit_params,
    #                   other_params=other_params,
    #                   surrogate_loos_func=surrogate_loss_func_for_auc,
    #                   label_dim=1,
    #                   train_set_x=train_set_x,
    #                   train_set_y=train_set_y,
    #                   test_set_x=test_set_x,
    #                   test_set_y=test_set_y,
    #                   )
    #     return {'loss': loss, 'params': params, 'status': STATUS_OK}
    #
    #
    # fit_params = {'early_stopping_rounds': 15,
    #               'eval_metric': 'auc',
    #               'verbose': False,
    #               'feature_name': feature_names,
    #               }
    # lgb_params = {'num_class': 1,
    #               'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'rf']),  # ['gbdt', 'dart', 'goss', 'rf']
    #               'num_leaves': hp.choice('num_leaves', range(4, 2 ** 12, 1)),
    #               'max_depth': hp.choice('max_depth', range(3, 12, 1)),
    #               'learning_rate': hp.uniform('learning_rate', 0.05, 0.3),
    #               'n_estimators': hp.choice('n_estimators', range(50, 10000, 50)),
    #               'subsample_for_bin': hp.choice('subsample_for_bin', range(100, 100000, 10)),
    #               'objective': 'binary',
    #               'class_weight': None,
    #               # 'is_unbalance': True,
    #               'min_split_gain': hp.uniform('min_split_gain', 1e-7, 0.1),
    #               'min_child_weight': hp.uniform('min_child_weight', 1e-4, 0.1),
    #               'min_child_samples': hp.choice('min_child_samples', range(10, 1000, 1)),
    #               'subsample': hp.uniform('subsample', 0.5, 1.0),
    #               'subsample_freq': hp.choice('subsample_freq', range(1, 1000, 1)),
    #               'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    #               'reg_alpha': hp.uniform('reg_alpha', 0.01, 10.0),
    #               'reg_lambda': hp.uniform('reg_lambda', 0.01, 10.0),
    #               'random_state': 2021,
    #               'n_jobs': 6,
    #               'silent': True,
    #               'importance_type': 'gain',
    #               }
    # num_tuning_rounds = 5000  # 调参次数
    # best = fmin(fn=lgb_objective, space=lgb_params, algo=partial(tpe.suggest, n_startup_jobs=24), max_evals=num_tuning_rounds, verbose=False)
    # print(space_eval(lgb_params, best))
    # print('All time is: %s minute(s)' % ((time.time() - start_time) / 60))

    # --------------------------------------------------------------RF Model--------------------------------------------------------------- #
    # dict_model = {'rf_model': RandomForestClassifier}
    # makedirs('./logs/%s/' % list(dict_model.keys())[0])
    # write_print_to_file('./logs/%s/' % list(dict_model.keys())[0])
    #
    # other_params = {'expect_precision': 0.65,
    #                 'expect_recall': 0.1,
    #                 'feature_names': feature_names,
    #                 }
    #
    #
    # def rf_objective(params):
    #     """
    #     目标函数
    #     """
    #     loss = tuning(dict_model=dict_model,
    #                   tuning_params=params,
    #                   fit_params=fit_params,
    #                   other_params=other_params,
    #                   surrogate_loos_func=surrogate_loss_func_for_auc,
    #                   label_dim=1,
    #                   train_set_x=train_set_x,
    #                   train_set_y=train_set_y,
    #                   test_set_x=test_set_x,
    #                   test_set_y=test_set_y,
    #                   )
    #     return {'loss': loss, 'params': params, 'status': STATUS_OK}
    #
    #
    # fit_params = {}
    # rf_params = {'n_estimators': hp.choice('n_estimators', range(2, 1024, 1)),
    #              'criterion': 'gini',  # hp.choice('criterion', ['gini', 'entropy']),
    #              'max_depth': hp.choice('max_depth', range(2, 16, 1)),
    #              'min_samples_split': 2,  # hp.uniform('min_samples_split', 0.0, 1.0),
    #              'min_samples_leaf': 1,  # hp.uniform('min_samples_leaf', 0.0, 0.5),
    #              'min_weight_fraction_leaf': 0.0,  # hp.uniform('min_weight_fraction_leaf', 0.0, 0.5),
    #              'max_features': 'auto',  # hp.choice('max_features', ['auto', 'sqrt', 'log2']),
    #              'max_leaf_nodes': None,
    #              'min_impurity_decrease': 0.0,  # hp.uniform('min_impurity_decrease', 0.0, 0.5),
    #              'bootstrap': True,
    #              'oob_score': False,
    #              'n_jobs': 6,
    #              'random_state': 2021,
    #              'verbose': False,
    #              'max_samples': None,  # hp.uniform('max_samples', 0.0, 1.0),
    #              }
    # num_tuning_rounds = 5000  # 调参次数
    # best = fmin(fn=rf_objective, space=rf_params, algo=partial(tpe.suggest, n_startup_jobs=24), max_evals=num_tuning_rounds, verbose=2)
    # print(space_eval(rf_params, best))
    # print('All time is: %s minute(s)' % ((time.time() - start_time) / 60))

    # --------------------------------------------------------------AdaBoost Model--------------------------------------------------------------- #
    # dict_model = {'adaboost_model': AdaBoostClassifier}
    # makedirs('./logs/%s/' % list(dict_model.keys())[0])
    # write_print_to_file('./logs/%s/' % list(dict_model.keys())[0])
    #
    # other_params = {'expect_precision': 0.65,
    #                 'expect_recall': 0.1,
    #                 'feature_names': feature_names,
    #                 }
    #
    # def adaboost_objective(params):
    #     """
    #     目标函数
    #     """
    #     loss = tuning(dict_model=dict_model,
    #                   tuning_params=params,
    #                   fit_params=fit_params,
    #                   other_params=other_params,
    #                   surrogate_loos_func=surrogate_loss_func_for_auc,
    #                   label_dim=1,
    #                   train_set_x=train_set_x,
    #                   train_set_y=train_set_y,
    #                   test_set_x=test_set_x,
    #                   test_set_y=test_set_y,
    #                   )
    #     return {'loss': loss, 'params': params, 'status': STATUS_OK}
    #
    #
    # fit_params = {}
    # adaboost_params = {'n_estimators': hp.choice('n_estimators', range(2, 1024, 1)),
    #                    'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
    #                    'algorithm': hp.choice('algorithm', ['SAMME', 'SAMME.R']),
    #                    'random_state': 2021,
    #                    }
    # num_tuning_rounds = 5000  # 调参次数
    # best = fmin(fn=adaboost_objective, space=adaboost_params, algo=partial(tpe.suggest, n_startup_jobs=24), max_evals=num_tuning_rounds, verbose=2)
    # print(space_eval(adaboost_params, best))
    # print('All time is: %s minute(s)' % ((time.time() - start_time) / 60))

    # # --------------------------------------------------------------GBDT Model--------------------------------------------------------------- #
    # dict_model = {'gbdt_model': GradientBoostingClassifier}
    # makedirs('./logs/%s/' % list(dict_model.keys())[0])
    # write_print_to_file('./logs/%s/' % list(dict_model.keys())[0])
    #
    # other_params = {'expect_precision': 0.65,
    #                 'expect_recall': 0.1,
    #                 'feature_names': feature_names,
    #                 }
    #
    # def gbdt_objective(params):
    #     """
    #     目标函数
    #     """
    #     loss = tuning(dict_model=dict_model,
    #                   tuning_params=params,
    #                   fit_params=fit_params,
    #                   other_params=other_params,
    #                   surrogate_loos_func=surrogate_loss_func_for_auc,
    #                   label_dim=1,
    #                   train_set_x=train_set_x,
    #                   train_set_y=train_set_y,
    #                   test_set_x=test_set_x,
    #                   test_set_y=test_set_y,
    #                   )
    #     return {'loss': loss, 'params': params, 'status': STATUS_OK}
    #
    #
    # fit_params = {}
    # gbdt_params = {'loss': hp.choice('loss', ['deviance', 'exponential']),
    #                'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
    #                'n_estimators': hp.choice('n_estimators', range(2, 1024, 1)),
    #                'subsample': hp.uniform('subsample', 0.5, 1.0),
    #                'criterion': 'friedman_mse',  # hp.choice('criterion', ['friedman_mse', 'mse', 'mae']),
    #                'min_samples_split': 2,
    #                'min_samples_leaf': 1,
    #                'min_weight_fraction_leaf': 0.0,
    #                'max_depth': hp.choice('max_depth', range(2, 16, 1)),
    #                'min_impurity_decrease': 0.0,
    #                'random_state': 2021,
    #                'max_features': 'auto',  # hp.choice('max_features', ['auto', 'sqrt', 'log2']),
    #                'verbose': 0,
    #                'max_leaf_nodes': None,
    #                'validation_fraction': 0.2,
    #                'n_iter_no_change': 10,
    #                'tol': 1e-4,
    #                'ccp_alpha': 0.0,
    #                }
    # num_tuning_rounds = 5000  # 调参次数
    # best = fmin(fn=gbdt_objective, space=gbdt_params, algo=partial(tpe.suggest, n_startup_jobs=24), max_evals=num_tuning_rounds, verbose=2)
    # print(space_eval(gbdt_params, best))
    # print('All time is: %s minute(s)' % ((time.time() - start_time) / 60))

    # --------------------------------------------------------------DF21 Model--------------------------------------------------------------- #
    dict_model = {'df21_model': CascadeForestClassifier}
    makedirs('./logs/%s/' % list(dict_model.keys())[0])
    write_print_to_file('./logs/%s/' % list(dict_model.keys())[0])

    other_params = {'expect_precision': 0.65,
                    'expect_recall': 0.1,
                    'feature_names': feature_names,
                    }

    def df21_objective(params):
        """
        目标函数
        """
        loss = tuning(dict_model=dict_model,
                      tuning_params=params,
                      fit_params=fit_params,
                      other_params=other_params,
                      surrogate_loos_func=surrogate_loss_func_for_auc,
                      label_dim=1,
                      train_set_x=train_set_x,
                      train_set_y=train_set_y,
                      test_set_x=test_set_x,
                      test_set_y=test_set_y,
                      )
        return {'loss': loss, 'params': params, 'status': STATUS_OK}


    fit_params = {}
    df21_params = {'n_bins': hp.choice('n_bins', range(16, 256, 2)),
                   'bin_subsample': 2e5,
                   'max_layers': hp.choice('max_layers', range(4, 32, 2)),
                   'n_estimators': hp.choice('n_estimators', range(2, 32, 1)),
                   'n_trees': hp.choice('n_trees', range(2, 64, 256)),
                   'max_depth': None,  # hp.choice('max_depth', range(2, 16, 1)),
                   'min_samples_leaf': 1,
                   'use_predictor': False,
                   'predictor': hp.choice('predictor', ['forest', 'xgboost', 'lightgbm']),
                   'n_tolerant_rounds': 2,
                   'delta': 1e-5,
                   'partial_mode': False,
                   'n_jobs': 8,
                   'random_state': 2021,
                   'verbose': 2,
                   }
    num_tuning_rounds = 5000  # 调参次数
    best = fmin(fn=df21_objective, space=df21_params, algo=partial(tpe.suggest, n_startup_jobs=24), max_evals=num_tuning_rounds, verbose=2)
    print(space_eval(df21_params, best))
    print('All time is: %s minute(s)' % ((time.time() - start_time) / 60))
