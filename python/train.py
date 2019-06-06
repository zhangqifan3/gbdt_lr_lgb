import time
import tensorflow as tf
from lib.LR import LR
from lib.read_conf import Config
from lib.tf_dataset import FileListGenerator
def main():
    CONFIG = Config()
    model_conf = CONFIG.read_model_conf()['model_conf']
    traindata_list = FileListGenerator(model_conf['data_dir_train']).generate()
    testdata_list = FileListGenerator(model_conf['data_dir_pred']).generate()

    if model_conf['mode'] == 'train':
        traindata = next(traindata_list)
        tf.logging.info('Start training {}'.format(traindata))
        t0 = time.time()
        train1 = LR(traindata, mode='train').lr_model()
        t1 = time.time()
        tf.logging.info('Finish training {}, take {} mins'.format(traindata, float((t1 - t0) / 60)))

    else:
        testdata = next(testdata_list)
        tf.logging.info('Start evaluation {}'.format(testdata))
        t0 = time.time()
        Accuracy, AUC = LR(testdata, mode='pred').lr_model()
        t1 = time.time()
        tf.logging.info('Finish evaluation {}, take {} mins'.format(testdata, float((t1 - t0) / 60)))
        print("LR_Accuracy: %f" % Accuracy)
        print("LR_AUC: %f" % AUC)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()