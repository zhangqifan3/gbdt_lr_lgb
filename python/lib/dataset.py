import sys
import os
import csv
import numpy as np
import pandas as pd
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config


class DataSet(object):
    '''
    DataSet class
    处理输入数据
    '''
    def __init__(self, data_file):
        self._conf = Config()
        self._data_file = data_file
        self._feature_conf_dic = self._conf.read_feature_conf()[0]
        self._feature_used = self._conf.read_feature_conf()[1]
        self._all_features = self._conf.read_schema_conf()
        self.model_conf = self._conf.read_model_conf()['model_conf']
        self._csv_defaults = self._column_to_csv_defaults()

    def _column_to_csv_defaults(self):
        '''
        定义输入数据类型，获取数据特征名
        :return:
            all_columns：数据每一列对应的名称 type：list
            csv_defaults：csv默认数据类型 ['feature name': [''],...]
        '''
        features = []
        for i in range(1, len(self._all_features) + 1):
            features.append(self._all_features[str(i)])
        all_columns = ['label'] + features
        csv_defaults = {}
        csv_defaults['label'] = np.int
        for f in self._all_features.values():
            if f in self._feature_used:
                conf = self._feature_conf_dic[f]
                if conf['type'] == 'category':
                    if conf['transform'] == 'identity':
                        csv_defaults[f] = np.int
                    else:
                        csv_defaults[f] = np.str
                else:
                    csv_defaults[f] = np.float
            else:
                csv_defaults[f] = np.str
        return all_columns, csv_defaults

    def iter_minibatches(self):
        '''
        迭代器,给定文件流（比如一个大文件），每次输出minibatch_size行
        :return:
            将输出转化成dataframe输出
        '''

        cur_line_num = 0
        dataset = []
        csvfile = open(self._data_file, 'rt',encoding="utf-8")
        reader = csv.reader(csvfile, delimiter=' ')
        all_columns, csv_defaults = self._csv_defaults
        for line in reader:
            dataset.append(line)
            cur_line_num += 1
            if cur_line_num >= int(self.model_conf['batch_size']):
                dataset = pd.DataFrame(dataset, columns=all_columns)
                dataset = dataset.astype(csv_defaults)
                yield dataset
                dataset = []
                cur_line_num = 0
        dataset = pd.DataFrame(dataset, columns=all_columns)
        dataset = dataset.astype(csv_defaults)
        yield dataset
        csvfile.close()

    def input_fn(self):
        '''
        读取csv文件，转化为dataframe，填充nan值
        :return:
            dataset
        '''
        all_columns, csv_defaults = self._csv_defaults
        dataset = pd.read_csv(self._data_file, sep=' ',names =all_columns, dtype = csv_defaults)
        dataset = dataset.fillna('-')
        return dataset



