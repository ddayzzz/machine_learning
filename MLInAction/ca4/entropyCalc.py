import loaddata
import numpy as np

def calcEntropy(D, v):
    """
    计算 D^v 的 Ent
    :param D:
    :param v:
    :return:
    """
    colums = D[:, v]
    l = set(colums.tolist())  # 计算 D^i 的所有取值
    for value in l:
        all_attr_value = D[D[:, v] == value]
        pos_attr_x = all_attr_value[all_attr_value[:,-1] == '是']
        neg_attr_x = all_attr_value[all_attr_value[:, -1] == '否']
        pos_x_m = pos_attr_x.shape[0]
        neg_x_m = neg_attr_x.shape[0]
        total_x_m = all_attr_value.shape[0]
        ent = float(pos_attr_x) / total_x_m

data, labels = loaddata.read2_0()
data = np.array(data)
l = np.array(labels)
calcEntropy(data, 1)
