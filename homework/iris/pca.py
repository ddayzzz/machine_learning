"""
PCA测试的代码
主要采用深度学习钟的描述。D对应于周志华的W
"""
import gendata
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import itertools
import binaryClassifier as bc


def plotData3D(X):
    # 可能没有设计好，这个把数据的范围写死了。隔50个数据就是一个特征的范围
    classes = ['r', 'g', 'b']
    colors = [classes[x // 50] for x in range(X.shape[0])]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('Iris 数据集（PCA处理）', size=10)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors)

    ax.set_xlabel('第一个特征向量')
    ax.set_ylabel('第二个特征向量')
    ax.set_zlabel('第三个特征向量')
    ax.w_xaxis.set_ticklabels(())
    ax.w_yaxis.set_ticklabels(())
    ax.w_zaxis.set_ticklabels(())
    plt.show()

def plotData2D(X):
    """
    在2D平面中的点和绘制第一主成分
    :param X: 降维后的数据
    :return:
    """
    # 可能没有设计好，这个把数据的范围写死了。隔50个数据就是一个特征的范围
    X = np.array(X)  # 必须转换为 array： https://stackoverflow.com/questions/44224076/python-3-scatter-plot-gives-valueerror-masked-arrays-must-be-1-d-even-though?rq=1

    classes = ['r', 'g', 'b']

    colors = [classes[x // 50] for x in range(X.shape[0])]

    fig = plt.figure()
    ax = fig.add_subplot('111')
    ax.set_title('Iris 数据集（PCA降到2维）', size=10)
    ax.scatter(X[:, 0], X[:, 1], c=colors)
    ax.set_xlabel('第一个特征向量')
    ax.set_ylabel('第二个特征向量')

    plt.show()


def splitToDatasetWithoutLabel(dataMat):
    return dataMat[:,0:-1].astype(np.float64)

def zeroBasedCenterlization(datMat):
    means = np.mean(datMat, axis=0)
    return datMat - means, means  # means 也需要是因为需要给重构r使用

def getCov(dataMat):
    cov = np.cov(dataMat, rowvar=0)  # 每一行是一个样本
    return cov

def eigenVectorsOrderedBySizeOfEigenValue(covMat, t):
    """
    按照对应的特征值的大小降序排列特征项向量
    :param covMat: 协方差矩阵
    :return:
    """
    eigVal, eigVec = np.linalg.eig(np.mat(covMat))
    new_eigValsIndices = np.argsort(eigVal) # 按照升序排序,返回的是下标
    # 选择最大的前l个。
    sortedVals = np.argsort(eigVal)[-1::-1]  # 按照降序排序
    valuesSum = sum(sortedVals)
    tmp_sum = 0
    d = 0
    for lamb in sortedVals:
        tmp_sum += lamb
        d += 1
        if tmp_sum >= valuesSum * t:
            break
    # d是最佳选择的特征值数量。周志华写的是d。深度学习中是l
    l_EigValDesc = new_eigValsIndices[-1:-(d+1):-1]
    l_EigVecDesc = eigVec[:,l_EigValDesc]
    return l_EigVecDesc



def pca(data):

    dataMat = splitToDatasetWithoutLabel(data)
    zero_dataMat, zero_mean = zeroBasedCenterlization(dataMat)  # 0中心化
    cov = getCov(zero_dataMat)
    eigVec = eigenVectorsOrderedBySizeOfEigenValue(cov, 0.16)  # 0.16 降到 2维；0.6 降到3维
    print("D'D=(应该是I)\n", eigVec.T * eigVec)
    print("选择的特征向量：\n",eigVec)
    # 降维后的数据
    low_dataMat = zero_dataMat * eigVec
    # 重构的数据r
    ref_dataMat = low_dataMat * eigVec.T + zero_mean
    print(ref_dataMat)
    # 这个是有3个特征向量的情况。
    plotData2D(low_dataMat)
    return low_dataMat, ref_dataMat

def transportToBinaryClassifier():
    data = gendata.loadFullDataset('iris.data')
    low, ref = pca(data)
    # 重组为带有类型的数据
    # dataWithLabels = np.array(np.column_stack((low, data[:,-1])))
    # bc.logistic(0.9, dataWithLabels)


if __name__ == '__main__':
    transportToBinaryClassifier()
