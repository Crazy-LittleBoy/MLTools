# -*- coding:utf-8 -*-
import numpy as np
import json
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, auc


class Stacking(object):
    """
        多模型stacking融合
        建议使用sklearn接口形式的算法(实现了fit和predict方法)
    Args:
        n_splits: 第1层模型k折交叉验证训练集划分份数
        base_models：第1层模型(多模型)，必须实现fit和predict方法
        stacker: 第2层模型(单模型)，必须实现fit和predict方法
    Returns:
        ...
    """

    def __init__(self, n_splits, dict_models, stacker, fit_params, random_state=2021):
        self.n_splits = n_splits
        self.stacker = stacker
        self.dict_models = dict_models
        # self.metric = metric
        self.fit_params = fit_params
        self.random_state = random_state

    def fit_predict(self, input_trainset_x, input_trainset_y, input_validset_x, input_validset_y, input_testset_x):
        trainset_x = np.array(input_trainset_x)
        trainset_y = np.array(input_trainset_y)
        validset_x = np.array(input_validset_x)
        validset_y = np.array(input_validset_y)
        testset_x = np.array(input_testset_x)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state).split(trainset_x, trainset_y))
        stack_trainsets_x = []
        stack_trainsets_y = []
        stack_testsets_x = []
        for i, model_name in enumerate(self.dict_models.keys()):
            model = self.dict_models[model_name]
            stack_trainset_x = []
            stack_trainset_y = []
            stack_testset_x = []
            for j, (train_idx, test_idx) in enumerate(folds):
                train_x = trainset_x[train_idx]
                train_y = trainset_y[train_idx]
                holdout_x = trainset_x[test_idx]
                holdout_y = trainset_y[test_idx]
                print("[Info]: fitting [%s]-->%d/%d [fold]-->%d/%d" % (model_name, i + 1, len(self.dict_models.keys()), j + 1, self.n_splits))
                if 'xgb' in model_name:
                    model.fit(train_x, train_y, eval_set=[(validset_x, validset_y)], **self.fit_params[model_name])
                    pred_holdout_x = model.predict_proba(holdout_x, ntree_limit=model.best_ntree_limit)
                    pred_testset_x = model.predict_proba(testset_x, ntree_limit=model.best_ntree_limit)
                elif 'lgb' in model_name:
                    model.fit(train_x, train_y, eval_set=[(validset_x, validset_y)], **self.fit_params[model_name])
                    pred_holdout_x = model.predict_proba(holdout_x, num_iteration=model.best_iteration_)
                    pred_testset_x = model.predict_proba(testset_x, num_iteration=model.best_iteration_)
                else:
                    model.fit(train_x, train_y, **self.fit_params[model_name])
                    pred_holdout_x = model.predict_proba(holdout_x)
                    pred_testset_x = model.predict_proba(testset_x)
                print('[Info]: pred_holdout_x.shape:%s, holdout_y.shape:%s, pred_testset_x.shape:%s' % (pred_holdout_x.shape, holdout_y.shape, pred_testset_x.shape))
                if pred_holdout_x.shape[1] > 1:
                    pred_holdout_x = pred_holdout_x[:, -1].ravel()
                    pred_testset_x = pred_testset_x[:, -1].ravel()
                stack_trainset_x.append(pred_holdout_x)
                stack_trainset_y.append(holdout_y)
                stack_testset_x.append(pred_testset_x)
            stack_trainset_x = np.concatenate(stack_trainset_x)
            stack_trainsets_x.append(stack_trainset_x)
            stack_trainset_y = np.concatenate(stack_trainset_y)
            stack_trainsets_y.append(stack_trainset_y)
            stack_testset_x = np.mean(np.array(stack_testset_x), axis=0)
            # stack_testset_x = np.mean(np.hstack(stack_testset_x), axis=1, keepdims=True)
            print('[Info]: stack_trainset_x.shape:%s, stack_testset_x.shape: %s' % (str(stack_trainset_x.shape), str(stack_testset_x.shape)))
            stack_testsets_x.append(stack_testset_x)
        stack_trainsets_x = np.array(stack_trainsets_x).T
        stack_trainsets_y = np.array(stack_trainsets_y).T
        stack_testsets_x = np.array(stack_testsets_x).T
        print("[Info]: stack_trainsets_x.shape: %s, stack_trainsets_y.shape: %s, stack_testsets_x.shape: %s" % (stack_trainsets_x.shape,
                                                                                                                stack_trainsets_y.shape,
                                                                                                                stack_testsets_x.shape)
              )
        # pd.DataFrame(stack_trainsets_x).to_csv('./data/stack_trainsets_x.csv', index=False)
        # pd.DataFrame(stack_trainsets_y).to_csv('./data/stack_trainsets_y.csv', index=False)
        # pd.DataFrame(stack_testsets_x).to_csv('./data/stack_testsets_x.csv', index=False)

        self.stacker.fit(stack_trainsets_x, stack_trainsets_y[:, -1].ravel())
        predict = self.stacker.predict_proba(stack_testsets_x)

        return predict


if __name__ == '__main__':
    # # stacking
    # data = pd.read_csv('./data/train_data_clean.csv',
    #                    # nrows=40000,
    #                    low_memory=False,
    #                    )
    # # data['label'] = (data['label'] < 8) * 1
    # print(data.shape)
    # print(data.head(3))
    # feature_names = data.columns.tolist()[1:-1]
    # # 划分数据集
    # data = data.sample(frac=1, random_state=2020).reset_index(drop=True).values
    # train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(data[:, 1:-1], data[:, -1].ravel(), test_size=0.3)
    # valid_set_x, test_set_x, valid_set_y, test_set_y = train_test_split(test_set_x, test_set_y, test_size=0.5)
    # # print(train_set_x.shape, test_set_x.shape)
    # print(train_set_y.shape, valid_set_y.shape, test_set_y.shape)
    #
    # with open('./config/parameters.json', 'r') as jfile:
    #     model_config = json.load(jfile)
    # print(model_config)
    #
    # fit_params = {'lgb_model': {'early_stopping_rounds': 15,
    #                             'eval_metric': 'auc',
    #                             'verbose': False,
    #                             'feature_name': feature_names,
    #                             },
    #               'xgb_model': {'early_stopping_rounds': 15,
    #                             'eval_metric': 'auc',
    #                             'verbose': False,
    #                             },
    #               'gbdt_model': {},
    #               'rf_model': {},
    #               'adaboost_model': {},
    #               }
    #
    # lgb_model = LGBMClassifier(**model_config['lgb_model']['model_params'])
    # xgb_model = XGBClassifier(**model_config['xgb_model']['model_params'])
    # gbdt_model = GradientBoostingClassifier(**model_config['gbdt_model']['model_params'])
    # rf_model = RandomForestClassifier(**model_config['rf_model']['model_params'])
    # adaboost_model = AdaBoostClassifier(**model_config['adaboost_model']['model_params'])
    #
    # stack = Stacking(n_splits=5,
    #                  dict_models={'lgb_model': lgb_model,
    #                               'xgb_model': xgb_model,
    #                               'gbdt_model': gbdt_model,
    #                               'rf_model': rf_model,
    #                               'adaboost_model': adaboost_model,
    #                               },
    #                  stacker=LogisticRegression(),
    #                  fit_params=fit_params,
    #                  )
    # stacking_result = stack.fit_predict(input_trainset_x=train_set_x,
    #                                     input_trainset_y=train_set_y,
    #                                     input_validset_x=valid_set_x,
    #                                     input_validset_y=valid_set_y,
    #                                     input_testset_x=test_set_x,
    #                                     )
    # print(stacking_result[:10])
    # print('[Metric]: %s' % roc_auc_score(test_set_y, stacking_result[:, -1].ravel()))
    # print(classification_report(y_true=test_set_y, y_pred=(stacking_result[:, -1] > 0.5).ravel() * 1))

    ### 预测
    # 读取训练数据
    data = pd.read_csv('./data/train_data_clean.csv',
                       # nrows=4000,
                       low_memory=False,
                       )
    # data['label'] = (data['label'] < 8) * 1
    print(data.shape)
    print(data.head(3))
    feature_names = data.columns.tolist()[1:-1]
    # 划分数据集
    data = data.sample(frac=1, random_state=2020).reset_index(drop=True).values
    # train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(data[:, 1:-1], data[:, -1].ravel(), test_size=0.3)
    # # valid_set_x, test_set_x, valid_set_y, test_set_y = train_test_split(valid_set_x, valid_set_y, test_size=0.5)
    # print(train_set_x.shape, valid_set_x.shape)
    # # print(train_set_y.shape, valid_set_y.shape, test_set_y.shape)

    # 读取测试数据
    test = pd.read_csv('./data/test_data_clean.csv',
                       low_memory=False,
                       )
    print(test.shape)
    print(test.head(3))
    test_result = test[['ID']]
    test = test.values
    test_x = test[:, 1:]
    print('[Info]: test data shape: ', test_x.shape)

    with open('./config/parameters.json', 'r') as jfile:
        model_config = json.load(jfile)
    print(model_config)

    fit_params = {'lgb_model': {'early_stopping_rounds': 15,
                                'eval_metric': 'auc',
                                'verbose': False,
                                'feature_name': feature_names,
                                },
                  'xgb_model': {'early_stopping_rounds': 15,
                                'eval_metric': 'auc',
                                'verbose': False,
                                },
                  'gbdt_model': {},
                  'rf_model': {},
                  'adaboost_model': {},
                  }

    lgb_model = LGBMClassifier(**model_config['lgb_model']['model_params'])
    xgb_model = XGBClassifier(**model_config['xgb_model']['model_params'])
    gbdt_model = GradientBoostingClassifier(**model_config['gbdt_model']['model_params'])
    rf_model = RandomForestClassifier(**model_config['rf_model']['model_params'])
    adaboost_model = AdaBoostClassifier(**model_config['adaboost_model']['model_params'])

    stack = Stacking(n_splits=7,
                     dict_models={'lgb_model': lgb_model,
                                  'xgb_model': xgb_model,
                                  'gbdt_model': gbdt_model,
                                  'rf_model': rf_model,
                                  'adaboost_model': adaboost_model,
                                  },
                     stacker=LogisticRegression(),
                     fit_params=fit_params,
                     )

    stacking_results = []
    trainset_x, trainset_y = data[:, 1:-1], data[:, -1]
    n_splits = 7
    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=2021).split(trainset_x, trainset_y))
    for i, (train_idx, valid_idx) in enumerate(splits):
        print('\n')
        print('-' * 50 + ' training for rounds:  %s/%s ' % (i + 1, n_splits) + '-' * 50)
        train_set_x, valid_set_x, train_set_y, valid_set_y = trainset_x[train_idx], trainset_x[valid_idx], trainset_y[train_idx], trainset_y[valid_idx],
        stacking_result = stack.fit_predict(input_trainset_x=train_set_x,
                                            input_trainset_y=train_set_y,
                                            input_validset_x=valid_set_x,
                                            input_validset_y=valid_set_y,
                                            input_testset_x=test_x,
                                            )
        print(stacking_result[:10])
        stacking_results.append(stacking_result[:, 1].ravel())
    stacking_results = np.mean(stacking_results, axis=0)
    test_result['TARGET'] = stacking_results
    print(test_result.head())
    test_result.to_csv('./output/submission_cv_stacking.cv7.csv', index=False)
