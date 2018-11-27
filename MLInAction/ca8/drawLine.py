# coding=utf-8
__doc__ = "用于绘制图8-2拟合的线"


from ca8 import regression as regre
import matplotlib.pyplot as plt
from numpy import mat


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 读取数据
    xArr, yArr = regre.loadDataSet('ex0.txt')
    ws = regre.standRegres(xArr, yArr)
    xMat = mat(xArr)
    yMat = mat(yArr)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1], yHat)
    plt.show()