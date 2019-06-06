import pandas as pd
import numpy as np

dataset = pd.read_csv('/home/zhangqifan/data/rawdata/20190520/part_0.csv', sep=' ')
testdf = dataset.head(11)
testdf.to_csv(r"/home/zhangqifan/data/rawdata/20190520/test.csv", header=False, index=False,sep=' ')