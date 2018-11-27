from sklearn import svm
from homework.iris.gendata import loadDataset
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Circle


iris_X = loadDataset('homework/iris/iris.data')
iris_X = np.vstack([iris_X[0:50, 0:-1].astype(np.float), iris_X[50:100,0:-1].astype(np.float)])  # 去掉导入的标签和转换为 float 型数据, 并且选择第一类和第二类 iris
iris_y_num = np.hstack((np.zeros([50], np.int), np.zeros([50], np.int) - 1))  # 标签设置, [+1 -1]
# 画出数据点
fig1 = plt.figure()
ax1 = plt.subplot(111)
c1 = ax1.scatter(iris_X[:50, 0], iris_X[:50, 1])
c2 = ax1.scatter(iris_X[50:100, 0], iris_X[50:100, 1])
#
clf = svm.SVC(kernel='linear')
clf.fit(iris_X, iris_y_num)
print(clf.support_vectors_)

svs = [Circle(sv, 0.1, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5) for sv in clf.support_vectors_]
lastSV = None
for sv in svs:
    lastSV = ax1.add_patch(sv)
plt.title('SVM 二分类')
plt.legend(handles=[lastSV, c1, c2], labels=["支持向量", "类1", "类2"], loc='best')
plt.show()