from sklearn import svm
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Circle


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
    return np.array(dataMat), np.array(labelMat)


X, y = loadDataSet('dataset_train_rbf.txt')  # 加载数据集
# 画出数据点
fig1 = plt.figure()
ax1 = plt.subplot(111)  # 添加子图
indices_c1 = y == 1
indices_c2 = y == -1
h1 = ax1.scatter(X[indices_c1, 0], X[indices_c1, 1])
h2 = ax1.scatter(X[indices_c2, 0], X[indices_c2, 1])
#
clf = svm.SVC(kernel='rbf')
clf.fit(X, y)
print(clf.support_vectors_)
svs = [Circle(sv, 0.03, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5) for sv in clf.support_vectors_]  # 添加 SV 的圆圈
lastSV = None  # 最后一个圆圈用于图例
for sv in svs:
    lastSV = ax1.add_patch(sv)
# 绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
XM, YM = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))  # 获取采样点网格 设 x 维度为 m，设 y 的维度 n。 meshgrid(x, y) 将 x 按照行赋值为n行。 y 转置赋值为 m 列。构成一个  nxm 的矩阵
P = np.c_[XM.ravel(), YM.ravel()]  # 全部扁平化, 相当于 x 和 y 做笛卡儿积的点对（个人理解）
y_dis = clf.decision_function(P).reshape(XM.shape)  # 计算样本点到超平面的距离 重新转换到采样点的形式

h3 = plt.contour(XM, YM, y_dis, [0],cmap=plt.cm.winter, alpha=0.5)  # 等高线图，显示第一层
# plt.contour(XM, YM, pre, [0], cmap=plt.cm.winter, alpha=0.2)
plt.title('SVM-RBF')
plt.legend(handles=[h1, h2, h3, lastSV], labels=['类1', '类2', '决策边界', "支持向量"],loc='best')
plt.show()