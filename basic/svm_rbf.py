from sklearn import svm
from MLInAction.ca6.loadDataset import loadDataSet
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Circle


X, y = loadDataSet('MLInAction/ca6/dataset_train_rbf.txt')
X, y = np.array(X), np.array(y)
# 画出数据点
fig1 = plt.figure()
ax1 = plt.subplot(111)
indices_c1 = y == 1
indices_c2 = y == -1
h1 = ax1.scatter(X[indices_c1, 0], X[indices_c1, 1])
h2 = ax1.scatter(X[indices_c2, 0], X[indices_c2, 1])
#
clf = svm.SVC(kernel='rbf')
clf.fit(X, y)
print(clf.support_vectors_)
svs = [Circle(sv, 0.03, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5) for sv in clf.support_vectors_]
lastSV = None
for sv in svs:
    lastSV = ax1.add_patch(sv)
# 绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
XM, YM = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
P = np.c_[XM.ravel(), YM.ravel()]
pre = clf.predict(P).reshape(XM.shape)
y_desc = clf.decision_function(P).reshape(XM.shape)

h3 = plt.contour(XM, YM, y_desc, [0],cmap=plt.cm.winter, alpha=0.5)
plt.contour(XM, YM, pre, [0], cmap=plt.cm.winter, alpha=0.2)
plt.title('SVM-RBF')
plt.legend(handles=[h1, h2, h3, lastSV], labels=['类1', '类2', '决策边界', "支持向量"],loc='best')
plt.show()