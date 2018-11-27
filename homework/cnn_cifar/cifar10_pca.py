import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

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
    dts = []
    for lamb in sortedVals:
        tmp_sum += lamb
        d += 1
        dt = tmp_sum / valuesSum
        if dt >= t:
            break
        dts.append(dt)
    # d是最佳选择的特征值数量。周志华写的是d。深度学习中是l
    l_EigValDesc = new_eigValsIndices[-1:-(d+1):-1]
    l_EigVecDesc = eigVec[:,l_EigValDesc]
    plt.plot(dts)
    plt.show()
    return l_EigVecDesc

f = open('data/data_batch_1.bin', 'rb')
datadict = cPickle.load(f,encoding='latin1')
f.close()
X = datadict["data"]
Y = datadict['labels']
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)


# plt.imshow(X[1])
gray = rgb2gray(X[4])
plt.imshow(gray, cmap='Greys_r')
plt.show()  # 灰度图


means = np.mean(gray, axis=0)  # 按照行求平均
mX = gray - means
# 协方差矩阵, 每一个行是一个样本
conv = np.cov(mX, rowvar=False)  # XX^T

W = eigenVectorsOrderedBySizeOfEigenValue(conv, 0.5)
print(W.shape)
plt.imshow(np.dot(W.T, mX.T).astype(np.uint8), cmap='Greys_r')
plt.show()


from sklearn.decomposition import PCA
pca = PCA(n_components=32)
XX = pca.fit_transform(gray)
plt.imshow(XX.clip(0,255).astype(np.uint8), cmap='Greys_r')
plt.show()