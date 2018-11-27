# coding=utf-8
__doc__ = '加载数据集'


def loadDataSet(fileName):
    """
    加载数据集
    :param fileName:
    :return: 返回数据(list) , 标签(-1和1,list)
    """
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
