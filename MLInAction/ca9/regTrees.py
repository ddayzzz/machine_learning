# coding=utf-8
"""
CART 算法
有所修改：https://blog.csdn.net/PIPIXIU/article/details/78127793
"""
from numpy import *


def loadDataset(filename):
    datamat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        # fltline = map(float, curline)  # 作为一个映射，访问 curline[i] 就是访问 float(curline[i])
        datamat.append([float(x) for x in curline])
    return datamat


def binSplitDataset(dataset, feature, value):
    mat0 = dataset[nonzero(dataset[:, feature] > value)[0], :]  # 返回 datatest 中>value 的行，从 dataset 中选择出指定的行，(这个行中对应的feature列是包含非0值的)。返回的是第一个
    mat1 = dataset[nonzero(dataset[:, feature] <= value)[0], :]
    # nonzero的非0元素。https://blog.csdn.net/roler_/article/details/42395393
    # 第一个tuple，指定行上的非0元素的索引
    # 第二个tuple 是列上的
    return mat0, mat1

def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])  # 返沪数据最后一列的均值

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]  # 计算最后一列的方差 * 个数。这个就是方差总值


def chooseBestSplit(dataset, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataset[:, -1].T.tolist()[0])) == 1:  # 如果最后一个划分结果是一致（属于同一类）
        return None, leafType(dataset)  # 构建叶子节点
    m, n = shape(dataset)  # 获取唯独信息
    S = errType(dataset)  # 获取方差总值
    bestS = inf  # 定义无穷大
    bestIndex = 0  # 最佳切分的属性索引
    bestValue = 0  # 最佳切分的值
    for featIndex in range(n-1):  # 按照列
        for splitVal in set((dataset[:, featIndex].T.A.tolist())[0]):  # 这个属性所有可能值
            mat0, mat1 = binSplitDataset(dataset, featIndex, splitVal)  # 划分中分别满足>和<=条件的行
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 如果选取的临界值分解的行数量小于最小切分样本数
                continue
            newS = errType(mat0) + errType(mat1)  # 计算两个的总方差（数量*方差）
            if newS < bestS:
                bestIndex = featIndex  # 最好切分的属性索引
                bestValue = splitVal  # 最好切分的临界值
                bestS = newS  # 我也不知道这个是那个公式的（Gini？）
    if (S - bestS) < tolS:  # 如果误差减小不是很大
        return None, leafType(dataset)  # 返回叶节点
    mat0, mat1 = binSplitDataset(dataset, bestIndex, bestValue)  # 获取最好划分的样本（每一个样本）
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataset)  # 如果切分的太小，直接设置为叶子节点
    return bestIndex, bestValue

def createTree(dataset, leafType=regLeaf, errType=regErr, ops=(1,4)):
    """

    :param dataset:
    :param leafType:
    :param errType:
    :param ops:
    :return:
    """
    feat, val = chooseBestSplit(dataset, leafType, errType, ops)  # 选择一个最好的分配和临界值
    if feat == None:
        return val  # 这是一个叶子
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val  # 设置参数
    lSet, rSet = binSplitDataset(dataset, feat, val)  # 获取分解的行
    retTree['left'] = createTree(lSet, leafType, errType, ops)  # 继续递归分解
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  # 将子树返回

# 剪枝处理
def isTree(obj):
    return isinstance(obj, dict)


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0  # 返回这两个叶子的平均值


def prune(tree, testData):
    """
    后剪枝处理
    :param tree:待剪枝的树
    :param testData:  测试数据集
    :return:
    """
    if shape(testData)[0] == 0:
        return getMean(tree)  # 如果没有可以测试的数据集
    if isTree(tree['right']) or isTree(tree['left']):  # 如果还没有到叶子节点
        lSet, rSet = binSplitDataset(testData, tree['spInd'], tree['spVal'])  # 分割为左子树和右子树
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)  # 继续剪枝
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):  # 直到左子树和右子树不是叶子
        lSet, rSet = binSplitDataset(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))  # 不需要剪枝的时候的一个评判标准的值
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))  # 需要剪枝的时候的一个评判标准的值
        if errorMerge < errorNoMerge:  # 是否需要剪枝
            print('Merging')
            return treeMean
        else:
            return tree
    else:
        return tree


def linearSolve(dataset):
    m, n = shape(dataset)
    X = mat(ones((m, n)))
    Y = mat(ones((m ,1)))
    X[:, 1:n] = dataset[:, 0:n-1]
    Y = dataset[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('这个矩阵是奇异矩阵')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataset):
    ws, X, Y = linearSolve(dataset)
    return ws

def modelErr(dataset):
    ws, X, Y = linearSolve(dataset)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


if __name__ == '__main__':
    mymat = mat(loadDataset('ex00.txt'))
    m = createTree(mymat)
    print(m)