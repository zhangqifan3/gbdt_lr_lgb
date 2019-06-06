import sys
import os
import csv
import numpy as np
import pandas as pd
from collections import OrderedDict
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config
import tensorflow as tf


class TF_Data(object):
    def __init__(self, data_file):
        self._conf = Config()
        self._data_file = data_file
        self._feature_conf_dic = self._conf.read_feature_conf()[0]
        self._feature_used = self._conf.read_feature_conf()[1]
        self._all_features = self._conf.read_schema_conf()
        self.model_conf = self._conf.read_model_conf()['model_conf']
        self._csv_defaults = self._column_to_csv_defaults()

    def _normalizer_fn_builder(self, scaler, normalization_params):
        """normalizer_fn builder"""
        if scaler == 'min_max':

            return lambda x: (x - normalization_params[0]) / (
                max(normalization_params[1] - normalization_params[0], 0.001))
        elif scaler == 'standard':
            return lambda x: (x - normalization_params[0]) / normalization_params[1]
        else:
            return lambda x: tf.log(x)


    def _column_to_csv_defaults(self):
        """
        定义csv文件中各个特征默认的数据类型
        :return:
            OrderedDict {'feature name': [''],...}
        """
        csv_defaults = OrderedDict()
        csv_defaults['label'] = [0]
        for f in self._all_features.values():
            if f in self._feature_used:
                conf = self._feature_conf_dic[f]
                if conf['type'] == 'category':
                    if conf['transform'] == 'identity':
                        csv_defaults[f] = [0]
                    else:
                        csv_defaults[f] = ['']
                else:
                    csv_defaults[f] = [0.0]
            else:
                csv_defaults[f] = ['']
        return csv_defaults

    def _parse_csv(self, field_delim=' ', na_value='-'):
        """
        csv数据的解析函数
        :param field_delim: csv字段分隔符
        :param na_value: 使用csv默认值填充na_value
        :return:
            feature dict: {feature: Tensor ... }
        """
        csv_defaults = self._csv_defaults
        def decode_csv(value):
            parsed_line = tf.decode_csv(value, record_defaults = list(csv_defaults.values()), field_delim=field_delim, na_value = na_value)
            features = dict(zip(self._csv_defaults.keys(), parsed_line))
            for f in self._all_features.values():
                if f not in self._feature_used:
                    features.pop(f)
                    continue
            for f, tensor in features.items():
                if f == 'tag':
                    features[f] = tf.string_split([tensor], ',').values
                if f == 'main_actor':
                    features[f] = tf.string_split([tensor], ',').values
            label = features.pop('label')
            return features, label
        return decode_csv

    def input_fn(self):
        """
        生成dataset（tensor）
        :return:
            generator
        """
        dataset = tf.data.TextLineDataset(self._data_file)
        dataset = dataset.map(self._parse_csv())  # Decode each line

        # Shuffle, repeat, and batch the examples.
        # dataset = dataset.shuffle(10).repeat(1)
        padding_dic = {k: () for k in self._feature_used}
        padding_dic['tag'] = [None]
        # padding_dic['main_actor'] = [None]
        padded_shapes = (padding_dic, ())
        dataset = dataset.padded_batch(int(self.model_conf['batch_size']), padded_shapes=padded_shapes)

        # Return the read end of the pipeline.
        return dataset.make_one_shot_iterator().get_next()

    def feat_column(self):
        """
        特征列处理
        :return:
            wide_columns
        """
        wide_columns = []
        wide_dim = 0
        for feature, conf in self._feature_conf_dic.items():
            f_type, f_tran, f_param = conf["type"], conf["transform"], conf["parameter"]
            if feature == 'tag' or feature == 'main_actor':
                col = tf.feature_column.categorical_column_with_vocabulary_file(feature,
                                                                                vocabulary_file=f_param)
                col = tf.feature_column.indicator_column(col)
                wide_columns.append(col)
                wide_dim += int(conf["dim"])
            else:
                if f_type == 'category':
                    if f_tran == 'hash_bucket':
                        hash_bucket_size = int(f_param)
                        col = tf.feature_column.categorical_column_with_hash_bucket(feature,
                                                                                    hash_bucket_size=hash_bucket_size,
                                                                                    dtype=tf.string)
                        col = tf.feature_column.indicator_column(col)
                        wide_columns.append(col)
                        wide_dim += hash_bucket_size
                    elif f_tran == 'vocab':
                        col = tf.feature_column.categorical_column_with_vocabulary_list(feature,
                                                                                        vocabulary_list=list(map(str, f_param)),
                                                                                        dtype=None,
                                                                                        default_value=-1,
                                                                                        num_oov_buckets=0)
                        col = tf.feature_column.indicator_column(col)
                        wide_columns.append(col)
                        wide_dim += len(f_param)
                    elif f_tran == 'identity':
                        num_buckets = f_param
                        col = tf.feature_column.categorical_column_with_identity(feature,
                                                                                 num_buckets=num_buckets,
                                                                                 default_value=0)
                        col = tf.feature_column.indicator_column(col)
                        wide_columns.append(col)
                        wide_dim += num_buckets
                else:
                    normalizer_fn = self._normalizer_fn_builder(scaler = f_tran, normalization_params = tuple([0, 1]))
                    col = tf.feature_column.numeric_column(feature,
                                                           shape=(1,),
                                                           default_value=0,
                                                           dtype=tf.float32)
                                            #               normalizer_fn=normalizer_fn)
             #       col = tf.feature_column.indicator_column(col)
                    wide_columns.append(col)
                    wide_dim += 1
        return wide_columns

    def gbdt_input(self):
        """
        将特征列处理后的数据转化为array输出
        :return:
            process_data：训练或预估数据集； type：array
            label：数据集对应的标签； type：array
        """
        tensor = tf.feature_column.input_layer(self.input_fn()[0], self.feat_column())
        label_element = self.input_fn()[1]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            while True:
                try:
                    process_data = sess.run(tensor)
                    label = sess.run(label_element)
                    yield process_data, label
                except tf.errors.OutOfRangeError:
                    break


class FileListGenerator(object):
    """
    按天读取数据（每天的数据分为多个part）
    """
    def __init__(self, data_path):
        self._data_path = data_path

    def list_one_folder(self, folder_path):
        file_list = [folder_path + '/' + file_name for file_name in tf.gfile.ListDirectory(folder_path) if
                         not file_name.startswith('.')]
        return file_list

    def generate(self):
        yield self.list_one_folder(self._data_path)

if __name__ == '__main__':
    # datagen = TF_Data('/home/zhangqifan/data/rawdata/20190520/test.csv')
    # dataset1 = datagen.input_fn()
    # tensor = tf.feature_column.input_layer(dataset, datagen.feat_column())
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.tables_initializer())
    #     process_data = sess.run(tensor)
    #     print(process_data)

    generator_data =  FileListGenerator('/home/zhangqifan/data/testdata/pred_data')
    file_set = generator_data.generate()
    print(file_set)
    for i, file in enumerate(file_set):
        print(file)


    # m = datagen.gbdt_input()
    # for i,dataset in enumerate(m):
    #
    #     print(i)
    #     print(dataset)
    #     print(len(dataset[0]))
    #     print(len(dataset[1]))





























# conf = Config()
# feature_conf_dic, feature_used = conf.read_feature_conf()
# All_features = conf.read_schema_conf()
#
# from collections import OrderedDict
# csv_defaults = OrderedDict()
# csv_defaults['label'] = [0]
# for f in All_features.values():
#     if f in feature_used:
#         conf = feature_conf_dic[f]
#         if conf['type'] == 'category':
#             if conf['transform'] == 'identity':
#                 csv_defaults[f] = [0]
#             else:
#                 csv_defaults[f] = ['']
#         else:
#             csv_defaults[f] = [0.0]
#     else:
#         csv_defaults[f] = ['']
# record_defaults_1 = list(csv_defaults.values())
#
#
# def decode_csv(line):
#     parsed_line = tf.decode_csv(line, record_defaults=record_defaults_1, field_delim=' ')
#     #   feature_names = [i for i in list(test.columns)]
#     features = dict(zip(csv_defaults.keys(), parsed_line))
#     for f in All_features.values():
#         if f not in feature_used:
#             features.pop(f)
#             continue
#     for f, tensor in features.items():
#         if f == 'tag':
#             features[f] = tf.string_split([tensor], ',').values
#         if f == 'main_actor':
#             features[f] = tf.string_split([tensor], ',').values
#
#     label = features.pop('label')
#
#     return features, label
#
# def input_fn(tex):
#     """An input function for training"""
#     # Convert the inputs to a Dataset.
#     dataset = tf.data.TextLineDataset(tex)
#     dataset = dataset.map(decode_csv)  # Decode each line
#
#     # Shuffle, repeat, and batch the examples.
#     dataset = dataset.shuffle(10).repeat(1)
#     padding_dic = {k:() for k in feature_used}
#     #padding_dic['tag'] = [None]
#     #padding_dic['main_actor'] = [None]
#     padded_shapes = (padding_dic, ())
#     dataset = dataset.padded_batch(5, padded_shapes=padded_shapes)
#
#
#     # Return the read end of the pipeline.
#     return dataset.make_one_shot_iterator().get_next()
#
# datatest, label = input_fn('/home/zhangqifan/data/rawdata/20190520/test.csv')
#
#
# def normalizer_fn_builder(scaler, normalization_params):
#     """normalizer_fn builder"""
#     if scaler == 'min_max':
#         return lambda x: (x - normalization_params[0]) / (max(normalization_params[1] - normalization_params[0], 0.001))
#     elif scaler == 'standard':
#         return lambda x: (x - normalization_params[0]) / normalization_params[1]
#     else:
#         return lambda x: tf.log(x)
#
#
# wide_columns = []
# wide_dim = 0
# normalizer_scaler = 'min_max'
# for feature, conf in feature_conf_dic.items():
#     f_type, f_tran, f_param = conf["type"], conf["transform"], conf["parameter"]
#     if feature == 'tag' or feature == 'main_actor':
#         col = tf.feature_column.categorical_column_with_vocabulary_file(feature,
#                                                                         vocabulary_file=f_param)
#         col = tf.feature_column.indicator_column(col)
#         wide_columns.append(col)
#         wide_dim += int(conf["dim"])
#     else:
#         if f_type == 'category':
#             if f_tran == 'hash_bucket':
#                 hash_bucket_size = int(f_param)
#                 col = tf.feature_column.categorical_column_with_hash_bucket(feature,
#                                                                             hash_bucket_size=hash_bucket_size,
#                                                                             dtype=tf.string)
#                 col = tf.feature_column.indicator_column(col)
#                 wide_columns.append(col)
#                 wide_dim += hash_bucket_size
#             elif f_tran == 'vocab':
#                 col = tf.feature_column.categorical_column_with_vocabulary_list(feature,
#                                                                                 vocabulary_list=list(map(str, f_param)),
#                                                                                 dtype=None,
#                                                                                 default_value=-1,
#                                                                                 num_oov_buckets=0)
#                 col = tf.feature_column.indicator_column(col)
#                 wide_columns.append(col)
#                 wide_dim += len(f_param)
#             elif f_tran == 'identity':
#                 num_buckets = f_param
#                 col = tf.feature_column.categorical_column_with_identity(feature,
#                                                                          num_buckets=num_buckets,
#                                                                          default_value=0)
#                 col = tf.feature_column.indicator_column(col)
#                 wide_columns.append(col)
#                 wide_dim += num_buckets
#         else:
#             normalizer_fn = normalizer_fn_builder(normalizer_scaler, tuple([0, 1]))
#             col = tf.feature_column.numeric_column(feature,
#                                                    shape=(1,),
#                                                    default_value=0,
#                                                    dtype=tf.float32,
#                                                    normalizer_fn=normalizer_fn)
#      #       col = tf.feature_column.indicator_column(col)
#             wide_columns.append(col)
#             wide_dim += 1
#
#
# tf.logging.info('Build total {} wide columns'.format(len(wide_columns)))
# for col in wide_columns:
#     tf.logging.debug('Wide columns: {}'.format(col))
# tf.logging.info('Wide input dimension is: {}'.format(wide_dim))
#
# tensor = tf.feature_column.input_layer(datatest, wide_columns)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.tables_initializer())
#     res = sess.run([tensor])
#
# print(res)
# print(res[0])
# print(len(res[0]))