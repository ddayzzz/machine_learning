# coding=utf-8
"""
决策树构造
"""
from math import log
import operator


def calcShannonEntropy(dataSet):
    """
    计算香农信息熵
    :param dataSet:  数据集：是一个list。list[i] 表示的是一个实例
    :return:
    """
    numEntries = len(dataSet)  # 获取样本数量
    labCounts = {}  # 列表 label->样本数
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labCounts.keys():
            labCounts[currentLabel] = 0
        labCounts[currentLabel] += 1  # 计算 0 或 1 的个数
    shannotEnt = 0.0  # 默认的熵
    for key in labCounts:  # 默认输出 keys
        prob = float(labCounts[key]) / numEntries
        shannotEnt -= prob * log(prob, 2)  # 内信息熵
    return shannotEnt


def splitDataSet(dataset, axis, value):
    """
    划分数据集
    :param dataset: 数据集
    :param axis: 轴，按照哪一个属性进行划分
    :param value: 只要 axis 对应的值是 value，就属于一类
    :return: 返回一类，且删除掉axis 属性
    """
    retdataset = []
    for featvec in dataset:
        if featvec[axis] == value:
            reducedFeatVec = featvec[:axis]
            reducedFeatVec.extend(featvec[axis+1:])
            retdataset.append(reducedFeatVec)
    return retdataset


def chooseBestFeatureToSplit(dataset):
    """
    选择合适的特征
    :param dataset: 数据集
    :return: 返回适合的特征
    """
    numfeatures = len(dataset[0]) - 1  # 特性的数量
    baseEntropy = calcShannonEntropy(dataset)
    bestInfoGain = 0.0 # 信息增益
    bestFeature = -1  # 选择的属性索引
    for i in range(numfeatures):
        featList = [example[i] for example in dataset]  # axis=i 对应属性的列
        uniqueVals = set(featList)  # 去重
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset, i, value)  # 获取所有在属性 i 有取值 value 的样本个数
            prob = len(subDataSet) / float(len(dataset))
            newEntropy += prob * calcShannonEntropy(subDataSet)  # 这个算出所有axis=i 的取值的信息熵之和
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature


def majorityCnt(classList):
    """
    返回指定标签中做多的标签名
    :param classList: 所有的标签名。C = {是，否}
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():  # 如果没有现在标签还没有填入到 是/否 -> 对应的个数
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClasssCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)  # 按照字典中每个 key-value 对中的 value (循环中每一个item是一个 tuple )按照降序排列
    return sortedClasssCount[0][0]  # 最大的类别名


def createTree(dataset, labels):
    """
    创建一个决策树
    :param dataset: 数据集 R^{m x n}
    :param labels: 样本的类型标签
    :return:
    """
    classList = [example[-1] for example in dataset]  # 获取分类的情况，yes 和 no
    if classList.count(classList[0]) == len(classList):  # 如果 yes 或 no 的数量等于所有样本的类别名的数量
        # 如果类别完全相同就停止分类
        return classList[0]
    if len(dataset[0]) == 1:  # 如果没有属性列可用了（注意标签）
        return majorityCnt(classList)  # 当遍历完所有的特征的时候，返回出现频率最多的标签名。
    bestFeat = chooseBestFeatureToSplit(dataset)  # 选择增益最大的属性编号
    bestFeatLabel = labels[bestFeat]  # 指定的信息增益最大的属性名字
    mytree = {bestFeatLabel:{}}  # 一棵树, 新建一颗 最优标签的子树
    del(labels[bestFeat])  # 这个是子树的标签，所以删除根节点使用的标签
    featValues = [example[bestFeat] for example in dataset]  # 需要获取所有可能的取值
    uniqueVals = set(featValues)  # 注意去重
    for value in uniqueVals:  # 对于最优属性的所有可能的取值
        subLabels = labels[:]  # 创建一个新的列表
        mytree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value), subLabels)  # 子树包含总是在 bestFeatlabel 取值未 value 的所有节点
    return mytree


def classify(inputTree, featLabels, testVec):
    """
    根据输入的决策树做分类
    :param inputTree:
    :param featLabels: 样本标签
    :param testVec: 测试的样本
    :return:
    """
    firstStr = tuple(inputTree.keys())[0]  # 取子树的开始的根节点的属性名
    secondDict = inputTree[firstStr]  # 获取可能的子树
    featIndex = featLabels.index(firstStr)  # 获取根节点所属类型在测试样本中属性的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:  # 在测试样本的属性的值与
            if isinstance(secondDict[key], dict):  # 这个节点还是子树
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel