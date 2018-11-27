"""
SVM 特别的绘制函数
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn import svm
import numpy as np
from MLInAction.ca6 import kernels



def plotAllData(dataMat, labelMat, title, classnames, two_featuresname, smoObj, resolution=0.02):

    pos_classes = labelMat == -1.0
    neg_classes = labelMat == 1.0
    fig = plt.figure()
    ax = fig.add_subplot(111)

    p1 = ax.scatter(dataMat[pos_classes, 0], dataMat[pos_classes, 1], s=20, c='r')
    p2 = ax.scatter(dataMat[neg_classes, 0], dataMat[neg_classes, 1], s=20, c='b')
    svIndices = np.nonzero(smoObj.alphas.A > 0)[0]  # 支持向量的索引
    svs = smoObj.dataMat[svIndices]  # 支持向量(行向量形式)
    svLabels = smoObj.labelsMat[svIndices]  # 标记
    # 圈出支持向量和决策边界
    for i in range(svs.shape[0]):
        sv = svs[i, :].T
        circle = Circle(sv, 0.05, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)

    # 根据核函数的类型绘制决策边界
    p3 = smoObj.kernel.plotDecisionBoundary(ax=ax, X=smoObj.dataMat, y=smoObj.labelsMat, alphas=smoObj.alphas, b=smoObj.b, svs=svs, svLabels=svLabels, svIndices=svIndices)
    # 调整坐标轴
    # ax.axis([x1min + 1, x1max + 1, x2min + 1, x2max + 1])
    ax.legend(handles=[p1, p2, p3], labels=classnames + ['决策边界'], loc='best')
    plt.xlabel(two_featuresname[0])
    plt.ylabel(two_featuresname[1])
    plt.title(title)
    plt.show()

