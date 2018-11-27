# coding=utf-8
"""
逻辑回归
"""
from numpy import *
import random

def loadDataset():
    """
    加载数据集
    :return: 返回 1.0，x，y 向量和标记的值
    """
    datmat = []
    labelmat = []
    with open('testset.txt') as fp:
        for line in fp.readlines():
            linearr = line.strip().split()
            datmat.append([1.0, float(linearr[0]), float(linearr[1])])
            labelmat.append(int(linearr[2]))
    return datmat, labelmat

def sigmoid(inx):
    return 1.0/ (1 + exp(-inx))

def gradAscent(datamatin, classlabels):
    datamatrix = mat(datamatin)
    labelmat = mat(classlabels).transpose()
    m, n = shape(datamatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(datamatrix * weights)
        error = (labelmat - h)
        weights = weights + alpha * datamatrix.transpose() * error
    return weights


def stocGradAscent1(datamatrix, classlabels, numIter=150):
    """
    随机梯度上升，可以适用于不能线性可分的数据集
    :param datamatrix: 数据矩阵。列向量是 [1,x,y]
    :param classlabels: 标记的类型
    :param numIter: 迭代次数
    :return:
    """
    m, n = shape(datamatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # j 是迭代次数，i 是样本点下标
            randIndex = int(random.uniform(0, len(dataIndex)))

            h = sigmoid(sum(datamatrix[randIndex] * weights))
            error = classlabels[randIndex] - h
            weights = weights + alpha * error * datamatrix[randIndex]
            del dataIndex[randIndex]
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    d, l = loadDataset()
    dataArr = array(d)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(l[i]) == 1:
            xcord1.append(dataArr[i ,1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 创建子图
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x  = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    dataArr, labelMat = loadDataset()
    weights = gradAscent(dataArr, labelMat)
    plotBestFit(weights.getA())