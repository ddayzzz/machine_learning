# coding=utf-8
"""
8.1 标准回归
"""
from numpy import *

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) -1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print('这个矩阵是奇异矩阵，没有逆矩阵')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    # 转换为矩阵对象
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]  # x 的秩
    # 特征矩阵
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print('奇异矩阵，没有逆矩阵')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def lwlr_draw(k):
    xArr, yArr = loadDataSet('ex0.txt')
    yHat = lwlrTest(xArr, xArr, yArr, k)
    xMat = mat(xArr)
    strInd = xMat[:,1].argsort(0)
    xSort = xMat[strInd][:,0,:]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[strInd])
    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()

# 预测鲍鱼年龄
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

if __name__ == '__main__':
    lwlr_draw(0.003)