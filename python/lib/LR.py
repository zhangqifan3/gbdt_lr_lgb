import sys
import os
from os.path import dirname, abspath
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import gc
from sklearn.externals import joblib
from lib.read_conf import Config
from lib.GBDT import GBDT_spr
from lib.tf_dataset import TF_Data

MODEL_DIR = os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'model')

class LR(object):
    '''
    LR class
    LR模型训练，预测
    '''
    def __init__(self, data_file, mode):
        self._conf = Config()
        self._data_file = data_file
        self._Tf_Data = TF_Data(self._data_file)
        self.dataset_train = self._Tf_Data.gbdt_input()
        self.lr_conf = self._conf.read_model_conf()['lr_conf']

        self._mode = mode
        self._gbdt_spr = GBDT_spr(self._data_file).gbdt_model(self._mode)

    def lr_model(self):
        '''
        lr模型训练及预测
        :return: AUC
        '''
        if self._mode == 'train':
            grd_lm = SGDClassifier(penalty = self.lr_conf['penalty'],
                                   loss='log',warm_start=True)
            i = 0
            while True:
                try:
                    dataset = next(self._gbdt_spr)
                    batch_X = dataset[0]
                    batch_y = dataset[1]
                    print('start training LR epochs_%d' % i)
                    grd_lm = grd_lm.partial_fit(batch_X, batch_y, classes=[0, 1])
                    i += 1
                    del(dataset)
                    del(batch_y)
                    del(batch_X)
                    gc.collect()
                except StopIteration as e:
                    print('Generator return value:', e.value)
                    break
            joblib.dump(grd_lm, os.path.join(MODEL_DIR, "lr_model.m"))
        else:
            y_all_label = []
            y_all_pred_grd_lm = []
            pred_all_res = []
            grd_lm = joblib.load(os.path.join(MODEL_DIR, "lr_model.m"))
            while True:
                try:
                    dataset = next(self._gbdt_spr)
                    gbdt_features = dataset[0]
                    y_label = dataset[1]
                    y_pred_grd_lm = grd_lm.predict_proba(gbdt_features)[:, 1]
                    pred_res = grd_lm.predict(gbdt_features)
                    y_all_label.extend(y_label)
                    y_all_pred_grd_lm.extend(y_pred_grd_lm)
                    pred_all_res.extend(pred_res)
                    del (dataset)
                    del (gbdt_features)
                    gc.collect()
                except StopIteration as e:
                    print('Generator return value:', e.value)
                    break
            accuracy_score = metrics.accuracy_score(y_all_label, pred_all_res)
            fpr_grd_lm, tpr_grd_lm, _ = metrics.roc_curve(y_all_label, y_all_pred_grd_lm)
            roc_auc = metrics.auc(fpr_grd_lm, tpr_grd_lm)
            AUC_Score = metrics.roc_auc_score(y_all_label, y_all_pred_grd_lm)
            return accuracy_score, AUC_Score
if __name__ == '__main__':
    train1 = LR('/home/zhangqifan/data/rawdata/20190520/part_1.csv',mode = 'trai').lr_model()
    print(train1)


