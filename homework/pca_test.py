"""
PCA 测试
# SVD 的V矩阵对应的最大2个特征向量与手动计算的存在符号级别的差异.
"""
import numpy as np
from sklearn.decomposition import PCA


test2 = [
    [149.5,69.5,38.5],
    [162.5,77,55.5],
    [162.7,78.5,50.8],
    [162.2,87.5,65.5],
    [156.5,74.5,49.0],
    [156.1,74.5,45.5],
    [172.0,76.5,51.0],
    [173.2,81.5,59.5],
    [159.5,74.5,43.5],
    [157.7,79.0,53.5]
]

# 产生随机数据
M = 100
N = 10
X = np.random.random_sample([M, N])
# 替换
X = np.array(test2)
M, N = X.shape
print(M, N)
# 计算平均数
means = np.mean(X, axis=0)  # 按照行求平均
mX = X - means
# 协方差矩阵, 每一个行是一个样本
conv = np.cov(mX, rowvar=False)  # XX^T
eigx, eigv = np.linalg.eig(conv)
#

indices = np.argsort(eigx)[-1::-1]  # 降序

print("得到的特征值:", eigx)
print("特征值降序的索引:", indices)
print("得到的特征向量:", eigv[:,indices[:2]].T)  # d' = 2, 注意 eig 返回的是列向量

pca = PCA(n_components=2)
nx = pca.fit(X)  # 降维后的数据
print('sklearn 特征值', nx.explained_variance_)  # 打印sklearn中的特征值. nx == pca , 这是流式操作
print('sklearn 主成分', nx.components_)  # 打印sklearn中的主成分
print('sklearn 奇异值', nx.singular_values_)
U, Sigma, V = np.linalg.svd(mX)
# sigma 是矩阵Sigma对角线上的奇异值元素.
# 按照降序排列
svd_simgma_indices = np.argsort(Sigma)[-1::-1]
svd_eigvec = V[svd_simgma_indices[:2], :]
print("SVD: Sigma=", Sigma)
print("SVD: V[:2]=", svd_eigvec)  # d'=2, V是 X*X^T的特征向量构成的矩阵
print("SVD: svd_eigvec==sklearn.PCA: components_", pca.components_ == svd_eigvec)

# 数据显示
W = eigv[:,indices[:2]]
X_HAT = np.dot(W.T ,mX.T)
SKLEARN = pca.fit_transform(X)
print("X_HAT", X_HAT)
print("X_HAT_SKLEARN", SKLEARN)

import matplotlib.pyplot as plt
plt.scatter(X_HAT[:])
