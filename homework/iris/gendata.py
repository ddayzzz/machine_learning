"""
这个文件用来产生数据：训练集、测试集合
"""
import numpy as np
import time


def loadDataset(filename):
    """
    加载数据集
    :return: 返回数据的列表：前四个是属性，后一个是所属的鸢尾花的名称
    """
    datMatrix = []
    with open(filename) as fp:
        for line in fp.readlines():
            linearr = line.strip().split(',')
            datMatrix.append([float(linearr[0]), float(linearr[2]), linearr[-1]])
    return np.array(datMatrix)

def loadFullDataset(filename):
    """
    加载数据集(包含所有的属性和标签)
    :return: 返回数据的列表：前四个是属性，后一个是所属的鸢尾花的名称
    """
    datMatrix = []
    with open(filename) as fp:
        for line in fp.readlines():
            linearr = line.strip().split(',')
            datMatrix.append([float(linearr[0]), float(linearr[1]), float(linearr[2]),float(linearr[3]),linearr[-1]])
    return np.array(datMatrix)


def generateBinaryClassification(orgDataMat, posClass, classIndex=-1):
    """
    得到适用于二分类的标签属性。
    :param dataMat:
    :param posClass: 正例的属性名称
    :return: 返回一个矩阵，最后一列的为1.0代表正例
    """
    dataMat = orgDataMat.copy()
    for i in range(dataMat.shape[0]):
        if dataMat[i][classIndex] == posClass:
            dataMat[i][classIndex] = 1
        else:
            dataMat[i][classIndex] = 0
    # 转换为浮点数的矩阵。
    return dataMat.astype(np.float64)


def generateUniqueRandom(beg, end, size):
    """
    产生无重复的随机数 [beg, end) 以及差集
    :param beg: 起始
    :param end: 终止
    :param size: 长度
    :return:
    """
    np.random.seed(seed=int(time.time()))
    total = [x for x in range(beg, end)]
    trainDataIndices = []
    while len(trainDataIndices) < size:
        i = np.random.randint(beg, end)
        if i not in trainDataIndices:
            trainDataIndices.append(i)
    return trainDataIndices, list(set(total) - set(trainDataIndices))

def divideDataset(dataMatrix, trainDataProportion):
    """
    将数据分为 trainDataProportion 的训练数据
    :param dataMatrix: 数据的矩阵
    :param trainDataProportion: 训练数据的比例
    :return: 返回训练和测试的数据矩阵
    """
    m = dataMatrix.shape[0]
    sizePerClass = int(m * trainDataProportion / 3)
    train_indices1, test_indices1 = generateUniqueRandom(0, 50, sizePerClass)
    train_indices2, test_indices2= generateUniqueRandom(50, 100, sizePerClass)
    train_indices3, test_indices3= generateUniqueRandom(100, 150, sizePerClass)
    return np.take(dataMatrix, train_indices1 + train_indices2 + train_indices3, axis=0), np.take(dataMatrix, test_indices1+test_indices2 +test_indices3, axis=0)



