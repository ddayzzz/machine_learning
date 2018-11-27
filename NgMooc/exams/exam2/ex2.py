"""
吴恩达网课，逻辑回归。
优化程序：https://blog.csdn.net/csdn_inside/article/details/81558079
思路：https://www.jianshu.com/p/82370a35dc22
第二题:https://blog.csdn.net/u012759262/article/details/73105519
https://github.com/arturomp/coursera-machine-learning-in-python/blob/master/mlclass-ex2-004/mlclass-ex2/mapFeature.py
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def loadDataset(filename):
    """
    加载数据集
    :return: 返回数据
    """
    datMatrix = []
    with open(filename) as fp:
        for line in fp.readlines():
            linearr = line.strip().split(',')
            datMatrix.append([float(x) for x in linearr])
    return datMatrix


def sigmoid(z):
    """
    Sigmoid 函数
    :param z: 参数z
    :return:
    """
    return 1.0 / (1 + np.exp(-z))


def plotData(theta, X, Y):
    """
    绘制图像，并不绘制决策边界
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot('111')
    # 数据散点
    if len(Y.shape) <= 1:
        posIndices = np.nonzero(Y)  # 正例的坐标
    else:
        posIndices = np.nonzero(Y[:, 0])  # 正例的坐标
    px = X[posIndices]
    nx = np.delete(X, posIndices,0)
    ax.scatter(px[:,1], px[:,2], s=30, c='r', marker='x')
    ax.scatter(nx[:, 1], nx[:, 2], s=30, c='g')
    # 绘制反例子

    if X.shape[1] <= 3:
        # 这个是不带有正则项的Logistic 回归
        plotX = np.linspace(np.min(X[:,1]) - 2, np.max(X[:,1] )+ 2, 2)
        plotY = ((-1 / theta[2]) * (theta[1] * plotX + theta[0]))
        ax.plot(plotX, plotY)
    else:
        # 网格形
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        # 计算z = theta * x
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = np.dot(mapFeatures(np.array([u[i]]), np.array([v[j]])), theta)
        z = np.transpose(z)
        p3 = plt.contour(u, v, z,[0])
    plt.show()


def costFunction(theta, x, y):
    m = np.shape(y)[0]
    # 这里的转置再相乘扮演的是求和功能
    # numpy 的转置对于一维向量不起作用
    J = (-(np.dot(y.T,np.log(sigmoid(x.dot(theta)))))
         - np.dot((1 - y).T, np.log(1-sigmoid(x.dot(theta))))) / m
    return J


def gradient(theta, x, y):
    m, n = np.shape(x)
    # 梯度。
    # x可以表示x_j
    theta = theta.reshape((n, 1))
    grad = np.dot(x.T, sigmoid(x.dot(theta)) - y) / m
    return grad.flatten()


def mapFeatures(X1, X2):
    """
    这个将生成 [1,x1,x2,x1^2,x1 x2, x2^2,...x2^degree]^T
    :return:
    """
    degree = 6
    out = np.ones((X1.shape[0], sum(range(degree + 2))))  # could also use ((degree+1) * (degree+2)) / 2 instead of sum
    curr_column = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out[:, curr_column] = np.power(X1, i - j) * np.power(X2, j)
            curr_column += 1
    return out


def costFunctionWithReg(theta, X, y, rlambda):
    """
    加入正则化的logisitic 回归
    :param theta:
    :param X:
    :param Y:
    :return:
    """
    J = 0
    m = y.shape[0]
    one = y * np.transpose(np.log(sigmoid(np.dot(X, theta))))
    two = (1 - y) * np.transpose(np.log(1 - sigmoid(np.dot(X, theta))))
    reg = (float(rlambda) / (2 * m)) * np.power(theta[1:theta.shape[0]], 2).sum()
    J = -(1. / m) * (one + two).sum() + reg
    return J

def gradientReg(theta, X, y, rlambda):
    m,_ = np.shape(X)
    # 对于 j>=1 的情况
    grad = (1. / m) * np.dot(sigmoid(np.dot(X, theta)).T - y, X).T
    regTerm = (float(rlambda) / m) * theta
    grad += regTerm

    # the case of j = 0 (recall that grad is a n+1 vector)
    # since we already have the whole vectorized version, we use that
    grad_no_regularization = (1. / m) * np.dot(sigmoid(np.dot(X, theta)).T - y, X).T

    # and then assign only the first element of grad_no_regularization to grad
    grad[0] = grad_no_regularization[0]
    return grad.flatten()


def minimize_cost(theta, x, y):
    xopt = opt.minimize(costFunction, x0=theta, args=(x,y), jac=gradient, method='TNC')
    return xopt['x']

def minimize_costReg(theta, x, y, rlambda):
    xopt = opt.minimize(costFunctionWithReg, x0=theta, args=(x,y, rlambda),
                        jac=gradientReg, method='TNC')
    # xopt = opt.fmin_bfgs(costFunctionWithReg, x0=theta, args=(x,y, rlambda))
    return xopt['x']


def prepareData(filename):
    """
    预备数据，X,Y
    :return:
    """
    dat = loadDataset(filename)
    # m个样本，n个特征
    m = len(dat)
    datMat = np.array(dat)
    X = datMat[:, 0:2]  # 列向量
    Y = datMat[:,2]
    Y = Y.reshape((m, 1))  # 不是横向量
    return X, Y


def initialData(X):
    m, n = X.shape
    initial_theta = np.zeros(n + 1)  # theta_0 默认为0
    ones = np.ones((m, 1))
    X = np.column_stack((ones, X))  # 与 theta_0 相乘的部分
    return X, initial_theta


def logistic():
    X, Y = prepareData('ex2data1.txt')
    X, initial_theta = initialData(X)
    result = minimize_cost(initial_theta, X, Y)
    print(result)
    plotData(result, X, Y)

def logisticReg():
    dat = loadDataset('NgMooc/exams/exam2/ex2data2.txt')
    # m个样本，n个特征
    datMat = np.array(dat)
    X = mapFeatures(datMat[:,0], datMat[:,1])
    Y = datMat[:,2]
    m, n = X.shape
    initial_theta = np.zeros((n, 1))
    # cost = costFunctionWithReg(initial_theta, X, Y, rlambda=1)
    result = minimize_costReg(initial_theta, X, Y, rlambda=1)  # 不进行正则化
    # result = minimize_costReg(initial_theta, X, Y, rlambda=1)
    print(result)
    plotData(result, X, Y)

logisticReg()