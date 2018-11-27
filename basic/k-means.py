"""
K-means 聚类
"""
import matplotlib.pyplot as plt
import numpy as np


def distance(x, mu):
    return np.sqrt(np.sum(np.square(x - mu)))
# 记录每一个类型有多少个
batch_size = 1000
# 分类的个数
K = 3
# 产生随即数据
X1 = np.random.normal(0, 1.5, [batch_size, 2])
X2 = np.random.normal(6, 1.5, [batch_size, 2])
X3 = np.random.normal(12, 1.5, [batch_size, 2])
# 输出图像
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.scatter(X1[:,0], X1[:,1], label='类1')
ax1.scatter(X2[:,0], X2[:,1], label='类2')
ax1.scatter(X3[:,0], X3[:,1], label='类3')
ax1.set_title('原始样本')
ax1.legend(loc='best')
# 样本集合
X = np.vstack([X1, X2, X3])
M, N = X.shape
# 随机选择, 反正是随机生成的
indices = np.arange(0, M)
np.random.shuffle(indices)
means_vector = X[indices[:K],:]
max_iter = 10000
# 建立一个矩阵
C = np.zeros([M, 1], dtype=np.int)  # 最后一个维度记录所属的簇号
# 打开交互模式
plt.ion()
plt.show()
while True:
    for j in range(M):
        minD = np.inf
        minL = -1
        for i in range(K):
            dis = distance(X[j], means_vector[i])
            if dis < minD:
                minL = i
                minD = dis
        # 记录X[j]所属的簇号
        C[j] = minL
    # 画每一个簇的样本变化的情况
    if True:  # 每一次迭代都显示
        # 画图
        ax2.cla()
        ## 每一类的结果
        c1 = X[C[:, -1] == 0]
        c2 = X[C[:, -1] == 1]
        c3 = X[C[:, -1] == 2]

        h1 = ax2.scatter(c1[:, 0], c1[:, 1], c='r')
        h2 = ax2.scatter(c2[:, 0], c2[:, 1], c='y')
        h3 = ax2.scatter(c3[:, 0], c3[:, 1], c='darkorange')
        ax2.set_title('聚类的结果')
        # 绘制各个类的类型
        h4 = ax2.scatter(means_vector[0, 0], means_vector[0, 1], marker='x', c='k', s=80)
        h5 = ax2.scatter(means_vector[1, 0], means_vector[1, 1], marker='x', c='k', s=80)
        h6 = ax2.scatter(means_vector[2, 0], means_vector[2, 1], marker='x', c='k', s=80)
        ax2.legend(handles=[h1, h2, h3, h4], labels=['类1', '类2', '类3', '均值向量'],
                   loc='best')
        plt.pause(0.3)
    # changed = False
    for i in range(K):
        i_Xs = X[C[:, -1] == i]
        new_means_vector = np.mean(i_Xs, axis=0)
        means_vector[i] = new_means_vector
    max_iter = max_iter - 1
    if max_iter == 0:
        break
plt.ioff()




