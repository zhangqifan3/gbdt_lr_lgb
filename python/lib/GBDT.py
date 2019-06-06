import sys
import os
import gc
from os.path import dirname, abspath
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)


import lightgbm as lgb
from sklearn.externals import joblib
import numpy as np

MODEL_DIR = os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'model')
print(MODEL_DIR)
from lib.read_conf import Config
from lib.tf_dataset import TF_Data



class GBDT_spr(object):
    '''
    GBDT_spr class
    GBDT模型训练，生成离散特征
    '''
    def __init__(self, data_file):
        self._data_file = data_file
        self._Tf_Data = TF_Data(self._data_file)
        self._conf = Config()
        self.dataset_train = self._Tf_Data.gbdt_input()
        self.dataset_trans = self._Tf_Data.gbdt_input()
        self.dataset_pred = self._Tf_Data.gbdt_input()
        self.gbdt_conf = self._conf.read_model_conf()['gbdt_conf']
        self.model_conf = self._conf.read_model_conf()['model_conf']

    def gbdt_model(self, mode):
        '''
        gbdt模型训练，生成离散特征
        :param
            mode: ‘train’ or  ‘pred’
        :return:
            transformed_training_matrix：gbdt生成的离散特征
            batch_y：对应数据的label
        '''

        params = {
            'task': 'train',
            'boosting_type': self.gbdt_conf['boosting_type'],
            'objective': 'binary',
            'metric': {'binary_logloss'},
            'num_leaves': int(self.gbdt_conf['num_leaves']),
            # 'num_trees': 60,
            'min_data_in_leaf': int(self.gbdt_conf['min_data_in_leaf']),
            'learning_rate': float(self.gbdt_conf['learning_rate']),
            'feature_fraction': float(self.gbdt_conf['feature_fraction']),
            'bagging_fraction': float(self.gbdt_conf['bagging_fraction']),
            # 'bagging_freq': 5,
            'verbose': -1
        }

        if mode == 'train':
            if self.model_conf['batch_size'] == '0':
                print('TODO')
            else:
                i = 0
                while True:
                    try:
                        dataset = next(self.dataset_train)
                        batch_X = dataset[0]
                        batch_y = dataset[1]
                        lgb_train = lgb.Dataset(batch_X, batch_y)
                        if i == 0:
                            gbm = lgb.train(params, lgb_train,valid_sets=lgb_train,keep_training_booster=True)
                            i += 1
                        else:
                            gbm = lgb.train(params, lgb_train, valid_sets=lgb_train,keep_training_booster=True,
                                            init_model='/home/zhangqifan/LightGBM_model.txt')
                            i += 1
                        gbm.save_model('/home/zhangqifan/LightGBM_model.txt')
                        del (dataset)
                        del (batch_y)
                        del (batch_X)
                        gc.collect()
                    except StopIteration:
                        break

                joblib.dump(gbm, os.path.join(MODEL_DIR, "gbdt_model.m"))

                while True:
                    try:
                        dataset = next(self.dataset_trans)
                        batch_X = dataset[0]
                        batch_y = dataset[1]
                        gbm_trans = joblib.load(os.path.join(MODEL_DIR, "gbdt_model.m"))
                        y_pred = gbm_trans.predict(batch_X, pred_leaf=True)
                        transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[1]) * int(self.gbdt_conf['num_leaves'])],
                                                               dtype=np.int64)  # N * num_tress * num_leafs
                        for m in range(0, len(y_pred)):
                            # temp表示在每棵树上预测的值所在节点的序号（0,64,128,...,6436 为100棵树的序号，中间的值为对应树的节点序号）
                            temp = np.arange(len(y_pred[0])) * int(self.gbdt_conf['num_leaves']) + np.array(y_pred[m])
                            # 构造one-hot 训练数据集
                            transformed_training_matrix[m][temp] += 1
                        del (dataset)
                        del (batch_X)
                        gc.collect()
                        yield transformed_training_matrix, batch_y
                    except StopIteration:
                        break

        else:
            while True:
                try:
                    dataset = next(self.dataset_pred)
                    gbm_trans = joblib.load(os.path.join(MODEL_DIR, "gbdt_model.m"))
                    batch_X = dataset[0]

                    batch_y = dataset[1]
                    y_pred = gbm_trans.predict(batch_X, pred_leaf=True)
                    transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[1]) * int(self.gbdt_conf['num_leaves'])],
                                                           dtype=np.int64)  # N * num_tress * num_leafs
                    for m in range(0, len(y_pred)):
                        # temp表示在每棵树上预测的值所在节点的序号（0,64,128,...,6436 为100棵树的序号，中间的值为对应树的节点序号）
                        temp = np.arange(len(y_pred[0])) * int(self.gbdt_conf['num_leaves']) + np.array(y_pred[m])
                        # 构造one-hot 训练数据集
                        transformed_training_matrix[m][temp] += 1
                    yield transformed_training_matrix, batch_y
                except StopIteration:
                    break


if __name__ == '__main__':
    train_new_feature2  = GBDT_spr('/home/zhangqifan/data/rawdata/20190520/part_3.csv').gbdt_model(mode = 'train')
    for i, dataset in enumerate(train_new_feature2):
        print(dataset)
        print(len(dataset[0]))
        print(len(dataset[0][0]))
     #   print(dataset[0])