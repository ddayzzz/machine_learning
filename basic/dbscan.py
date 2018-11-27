import matplotlib.pyplot as plt
import numpy as np
from queue import Queue

# 每一类的样本个数
random_nums = 100
# 不同类样本数
classes = 3
# 超参:
MinPts = 2
EPSILON = 0.8

class DBSCAN(object):

    """
    DBSCAN 聚类
    """

    def __init__(self, epsilon, MinPts):
        pass

def generate_dataset(num=3, ratio=1.0):
    # 产生随机的圆型数据
    XY = np.arange(random_nums)
    np.random.shuffle(XY)
    # 半径
    radius = 1.0
    # 先做一次
    XX = []
    # 继续添加数据
    for i in range(num):
        XX.append(np.column_stack((radius * np.sin(XY), radius * np.cos(XY))))
        radius += radius * ratio
        np.random.shuffle(XY)
    return np.row_stack(XX)  # 按照行连接



# 记录每一个类型有多少个
X = generate_dataset(ratio=1.0, num=classes)
M, N = X.shape

NEG = np.zeros((M, 1))
NEG_POSINTS = dict()


def distance(x, mu):
    return np.sqrt(np.sum(np.square(x - mu)))


def neig(x, mu):
    if distance(x, mu) < EPSILON:
        return True
    else:
        return False

for j in range(M):
    # 确定领域
    for i in range(M):
        if neig(X[i], X[j]):
            if not NEG_POSINTS.get(j):
                NEG_POSINTS[j] = set()
            NEG_POSINTS[j].add(i)  # 将满足距离的点添加到集合
            NEG[j, 0] += 1  # 集合数加1

Omega_ = np.array(np.nonzero(NEG[:, 0] >= MinPts)[0])  # 获取 Omega 中 X_j 的编号
total_omega = Omega_.shape[0]
Omega = np.column_stack((Omega_, X[Omega_], np.zeros([total_omega]).T))  # 每一个条目: [在样本X中的编号j,样本向量X_j,标记] 标记:0=未访问;1=已经访问
k = 0  # 记录簇数量
Gamma = np.column_stack((np.arange(M).T, X, np.zeros([M]).T))  # 每一个条目: [样本自身的编号j, X_j, 标记]
CK = dict()  # CK[k] 表示为 k 的簇
while np.sum(Omega[:,-1]) != total_omega:
    Gamma_old = Gamma[Gamma[:, -1] != 1].copy()  # 所有未标记(未访问的样本)
    # 选择目标的 o. 由于附带条件以及需要"假删除", 所以记录编号
    target_o_index = np.random.randint(high=total_omega, low=0)
    while Omega[target_o_index, -1] != 0:
        target_o_index = np.random.randint(high=total_omega, low=0)
    # 随机选取了 o
    Gamma[target_o_index, -1] = 1  # 在 Gamma 中删除这个点(标记未1)
    o = Gamma[target_o_index]  # 获取这个 o 对象

    Q = Queue()
    Q.put(o)
    while not Q.empty():
        q = Q.get(True)
        qindex = int(q[0])
        if NEG[qindex] >= MinPts:
            Delta = NEG_POSINTS[qindex] & set(Gamma[Gamma[:, -1] != 1][:, 0])  # 所有在 Gamma 中未访问的点的样本号与X[qindex] 样本的范围内的交集
            for qq in Delta:
                qqindex = int(qq)  # 索引,用于在 Gamma 中删除
                x = Gamma[qqindex]
                Q.put(x)  # 加入队列
                Gamma[qqindex, -1] = 1
    k = k + 1  # 已经找到了一个簇
    gamma = set(Gamma[Gamma[:, -1] != 1][:, 0])  # Gamma 中未删除的元素对应的样本号
    gamma_old = set(Gamma_old[:, 0])  # Gamma_old 的未标记节点的样本号
    gamma_indices = np.array(list(gamma_old - gamma), np.int32)  # 已经删除的样本的索引, 也就是 Ck
    CK[k] = X[gamma_indices]  # 加入簇
    for i in gamma_indices:
        Omega[Omega[:, 0] == i,-1] = 1  # 在 Omega 中标记删除

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for start in range(0, random_nums * classes, random_nums):
    XX = X[start: start + random_nums]
    ax1.scatter(XX[:, 0], XX[:, 1], label="类%d" % ((1 + start) // random_nums))
ax1.legend(loc='best')
ax1.set_title('原始数据')

print(CK)
for k, v in CK.items():
    ax2.scatter(v[:, 0], v[:, 1], label="聚类-%d" % k, marker='x')
ax2.legend(loc='best')
ax2.set_title('DBSCAN 聚类')
plt.show()




