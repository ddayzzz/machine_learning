"""
核函数的一些相关的定义
"""
import numpy as np
from sklearn import svm
from MLInAction.ca6 import smo
import matplotlib.pyplot as plt

class KernelFunction(object):

    """
    核函数对象的定义
    """

    def __init__(self, kernelName):
        self.kernelName = kernelName

    def getGramMatrix(self, *args, **kwargs):
        """
        生成Gram 矩阵
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError('没有实现 getGramMatrix')

    def getGramVector(self, *args, **kwargs):
        raise NotImplementedError('没有实现 getGramVector')

    def predict(self, X, alphas, b, svs, svIndices, svLabels):
        raise NotImplementedError('没有实现 predict')

    def plotDecisionBoundary(self, ax, X, y, alphas, b, svs, svIndices, svLabels, **kwargs):
        raise NotImplementedError('没有实现 plotDecisionBoundary')


class RBFKernel(KernelFunction):

    def __init__(self, sigma):
        super(RBFKernel, self).__init__(kernelName='RBF')
        self.sigma = sigma


    def getGramVector(self, X, A, **kwargs):
        sigma = kwargs.get('sigma', self.sigma)  # 径向基函数的参数 sigma. 可以运行时指定
        m, n = np.shape(X)
        K = np.mat(np.zeros((m, 1)))
        for j in range(m):
            deltaRow = X[j, :] - A  # 每一行元素(x_i 和 x_j)之间的差距 delta。其中 x_j = A。R_{1xn}
            K[j] = deltaRow * deltaRow.T  # 求内积。注意是 deltaRow 是 1xn. 保存到K[j]是因为i一定一个列向量.总是保存到K矩阵中.见书128
        K = np.exp(K / (-1 * sigma ** 2))  # Gram 矩阵（列向量部分）。K的对角线元素是0。如果x\inR_{1xn}都是随机变量那么K也是协方差矩阵
        return K

    def getGramMatrix(self, X):
        m = np.shape(X)[0]
        G = np.mat(np.zeros((m, m)))
        # K 就是格拉姆矩阵 https://zh.wikipedia.org/wiki/%E6%A0%BC%E6%8B%89%E5%A7%86%E7%9F%A9%E9%98%B5
        # 以及 https://stackoverflow.com/questions/26962159/how-to-use-a-custom-svm-kernel/26962861#26962861
        for i in range(m):
            G[:, i] = self.getGramVector(X, X[i,:])  # 生成Gram 矩阵
        return G

    def predict(self, X, alphas, b, svs, svIndices, svLabels):
        """
        预测
        :param X: 样本
        :param alphas: SMO得到的alpha
        :param b: bias
        :param svs: 支持向量
        :param svIndices: 支持向量的
        :param svLabels: 支持向量的标签
        :return: 返回预测的值,是一个向量 R_{m}
        """
        #　参见西瓜书公式 P127页 6.24
        m = np.shape(X)[0]
        ret = np.zeros((m, 1))
        for i in range(m):
            kernelEval = self.getGramVector(svs, X[i, :])
            p = kernelEval.T * np.multiply(svLabels, alphas[svIndices]) + b
            ret[i] = p
        return ret

    def plotDecisionBoundary(self, ax, X, y, alphas, b, svs, svIndices, svLabels, **kwargs):
        """
        绘制 RBF 的决策边界
        :param ax: 子图对象
        :param X: 数据矩阵
        :param y:
        :param alphas:
        :param b:
        :param svs:
        :param svIndices:
        :param svLabels:
        :param kwargs:
        :return:
        """
        x1min, x1max, x2min, x2max = np.min(X[:, 0]), np.max(X[:, 0]), np.min(X[:, 1]), np.max(X[:, 1])
        x1 = np.transpose(np.linspace(x1min, x1max, 100).reshape(1, -1))  # 最后一个的维度不变化
        x2 = np.transpose(np.linspace(x2min, x2max, 100).reshape(1, -1))
        X1, X2 = np.meshgrid(x1, x2)
        vals = np.zeros(X1.shape)
        for i in range(X1.shape[1]):
            this_X = np.hstack((X1[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1)))
            pre = self.predict(this_X, alphas=alphas, b=b, svs=svs, svIndices=svIndices, svLabels=svLabels)
            vals[:, i] = pre.T
        p3 = ax.contour(X1, X2, vals, [0])
        return p3


class PolynomialKernel(KernelFunction):

    def __init__(self, degree, remain):
        super(PolynomialKernel, self).__init__(kernelName='Polynomial')
        self.d = degree
        self.r = remain


    def getGramVector(self, X, A):
        m, n = np.shape(X)
        K = np.mat(np.zeros((m, 1)))
        for j in range(m):
            K[j, :] = (self.r + np.dot(A, X[j, :].T)) ** self.d  # 由于数据是横向的，所以需要考虑内积转置的位置
        return K

    def getGramMatrix(self, X):
        m = np.shape(X)[0]
        G = np.mat(np.zeros((m, m)))
        for i in range(m):
            G[:, i] = self.getGramVector(X, X[i, :])
        return G

    def predict(self, X, alphas, b, svs, svIndices, svLabels):
        """
        预测
        :param X: 样本
        :param alphas: SMO得到的alpha
        :param b: bias
        :param svs: 支持向量
        :param svIndices: 支持向量的
        :param svLabels: 支持向量的标签
        :return: 返回预测的值,是一个向量 R_{m}
        """
        #　参见西瓜书公式 P127页 6.24
        m = np.shape(X)[0]
        ret = np.zeros((m, 1))
        for i in range(m):
            kernelEval = self.getGramVector(svs, X[i, :])
            p = kernelEval.T * np.multiply(svLabels, alphas[svIndices]) + b
            ret[i] = p
        return ret

    def plotDecisionBoundary(self, ax, X, y, alphas, b, svs, svIndices, svLabels, **kwargs):
        """
        绘制 RBF 的决策边界
        :param ax: 子图对象
        :param X: 数据矩阵
        :param y:
        :param alphas:
        :param b:
        :param svs:
        :param svIndices:
        :param svLabels:
        :param kwargs:
        :return:
        """
        x1min, x1max, x2min, x2max = np.min(X[:, 0]), np.max(X[:, 0]), np.min(X[:, 1]), np.max(X[:, 1])
        x1 = np.transpose(np.linspace(x1min, x1max, 100).reshape(1, -1))  # 最后一个的维度不变化
        x2 = np.transpose(np.linspace(x2min, x2max, 100).reshape(1, -1))
        X1, X2 = np.meshgrid(x1, x2)
        vals = np.zeros(X1.shape)
        for i in range(X1.shape[1]):
            this_X = np.hstack((X1[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1)))
            pre = self.predict(this_X, alphas=alphas, b=b, svs=svs, svIndices=svIndices, svLabels=svLabels)
            vals[:, i] = pre.T
        p3 = ax.contour(X1, X2, vals, [0])
        return p3

class LinearKernel(KernelFunction):

    def __init__(self):
        super(LinearKernel, self).__init__(kernelName='LIN')

    def getGramVector(self, X, A, **kwargs):
        return X * A.T

    def getGramMatrix(self, X):
        m = np.shape(X)[0]
        G = np.mat(np.zeros((m, m)))
        for i in range(m):
            G[:, i] = self.getGramVector(X, X[i, :])  # 生成Gram 矩阵. 注意在线性核和书上的线性核公式不一样,x_I 就是参数1, 因为一次求一行
        return G

    def plotDecisionBoundary(self, ax, X, y, alphas, b, svs, svIndices, svLabels, **kwargs):
        # 线性核函数
        m, n = X.shape
        w = np.zeros((n, 1))
        x1min, x1max, x2min, x2max = np.min(X[:, 0]), np.max(X[:, 0]), np.min(X[:, 1]), np.max(X[:, 1])
        for i in range(m):
            w += np.multiply(alphas[i] * y[i], X[i, :].T)
        x1 = np.arange(x1min, x1max, 0.1)
        x2 = (-w[0] * x1 - b[0, 0]) / w[1]
        p3 = ax.plot(x1, x2)
        return p3
