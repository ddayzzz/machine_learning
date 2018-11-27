"""
分类器：使用logistics 进行多分类
"""
import numpy as np
import gendata
import scipy.optimize as opt
import matplotlib.pyplot as plt


classes = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
alpha = 0.008
epsilon = 1e-8
max_iter = 30000
M = 5

plotClassBoundaryAndLossFunc = False  # 是否画出决策边界和损失函数的图像
plotAllThreeData = False  # 是否打印所有的三种类型的鸢尾花的图像


def plot(theta, data, posClasss, xylabel, trainPosX, trainPosY):
    fig = plt.figure()

    # 获取原始的数据
    X = data.copy()
    # 数据散点
    posIndices = np.where(X[:,-1] == posClasss)
    negIndices = np.where(X[:, -1] != posClasss)
    # 替换数据
    X[posIndices,-1] = 1
    X[negIndices,-1] = 0
    X = X.astype(np.float64)
    # 正例的 X
    px = X[posIndices]
    # 反例
    nx = X[negIndices]

    p = plt.scatter(px[:, 0], px[:, 1], s=30, c='r', marker='x')
    n = plt.scatter(nx[:, 0], nx[:, 1], s=30, c='y')
    train = plt.scatter(trainPosX[:,1], trainPosX[:, 2], s=45, c='', marker='o',edgecolors='b')
    # 这个是不带有正则项的Logistic 回归
    plotX = np.linspace(np.min(X[:, 0]) - 2, np.max(X[:, 0]) + 2, 2)
    plotY = ((-1 / theta[2]) * (theta[1] * plotX + theta[0]))
    plt.plot(plotX, plotY)
    # 图例
    plt.xlabel(xylabel[0])
    plt.ylabel(xylabel[1])
    plt.legend(handles = [p,n, train], labels = ['正例-%s' % posClasss, '反例', '训练的点'], loc = 'best')
    # 标题
    plt.title('二分类')
    plt.show()


def plotAllData(data):
    pclass1 = data[np.where(data[:, 2] == classes[0])]
    pclass2 = data[np.where(data[:, 2] == classes[1])]
    pclass3 = data[np.where(data[:, 2] == classes[2])]
    fig = plt.figure()
    pclass1 = pclass1[:,0:2].astype(np.float)
    pclass2 = pclass2[:, 0:2].astype(np.float)
    pclass3 = pclass3[:, 0:2].astype(np.float)
    p1 = plt.scatter(pclass1[:, 0], pclass1[:, 1], s=30, c='r')
    p2 = plt.scatter(pclass2[:, 0], pclass2[:, 1], s=30, c='y')
    p3 = plt.scatter(pclass3[:, 0], pclass3[:, 1], s=30, c='b')
    plt.legend(handles=[p1, p2, p3], labels=classes, loc='best')
    plt.title('所有的Iris 类别(两个属性)')
    plt.show()

def sigmoid(Z):
    """
    Sigmoid 函数
    :param Z: 参数z
    :return:
    """
    return 1.0 / (1 + np.exp(-Z))


def costFunction(theta, X, Y):
    m = np.shape(Y)[0]
    # 这里的转置再相乘扮演的是求和功能
    J = (-(np.dot(Y.T, np.log(sigmoid(X.dot(theta))))) - np.dot((1 - Y).T, np.log(1-sigmoid(X.dot(theta))))) / m
    return J


def gradient(theta, X, y):
    m, n = np.shape(X)
    # 梯度。
    # x可以表示x_j
    theta = theta.reshape((n, 1))
    grad = np.dot(X.T, sigmoid(X.dot(theta)) - y) / m
    return grad.flatten()


def gradient_mini_batch(theta, X, y, lower):
    n = np.shape(X)[1]
    # 梯度。
    # x可以表示x_j
    theta = theta.reshape((n, 1))
    grad = np.dot(X[lower:lower + M].T, sigmoid(X[lower:lower + M].dot(theta)) - y[lower:lower + M]) / M
    return grad.flatten()


def gradient_descent(initial_theta, X, y):
    J = []
    thetas = []
    for i in range(max_iter):
        # 计算 减去的项
        grad = gradient(initial_theta, X,y)

        initial_theta = initial_theta - alpha * grad
        # 获取J的值
        JK_1 = costFunction(initial_theta, X,y)
        if i != 0 and abs(J[-1] - JK_1) < epsilon:
            break
        J.append(JK_1)
        thetas.append(initial_theta)

    return J, thetas



def gradient_descent_mini_batch(initial_theta, X, y):
    J = []
    thetas = []
    mini_batch_bound = range(0, X.shape[0], M) # mini_batch 的范围，确定了batch 的数量，每一组的长度由M确定
    for k in range(max_iter):
        for i in mini_batch_bound:
            # 计算 减去的项
            grad = gradient_mini_batch(initial_theta, X, y, i)
            initial_theta = initial_theta - alpha * grad
        # 获取J的值
        JK_1 = costFunction(initial_theta, X, y)
        # if k != 0 and abs(J[-1] - JK_1) < epsilon:
        #     break
        J.append(JK_1)
        thetas.append(initial_theta)
    return J, thetas






def minimize_cost(theta, X, Y):
    """
    这个 scipy 带的优化函数
    :param theta:
    :param X:
    :param Y:
    :return:
    """
    xopt = opt.minimize(costFunction, x0=theta, args=(X,Y), jac=gradient, method='CG')
    return xopt


def logistic(trainDataProportion, data):
    # 处理数据
    # 所有的类

    train, test = gendata.divideDataset(data, trainDataProportion)  # test 数据集就是整个样本

    # 测试数据对应于classes 的可能性
    test_possibilities = np.zeros((test.shape[0],len(classes)))
    train_possibilities = np.zeros((train.shape[0], len(classes)))
    for i, posClass in enumerate(classes):
        train_pos = gendata.generateBinaryClassification(train, posClass, -1)
        # test_pos = gendata.generateBinaryClassification(test, posClass, -1)
        X = train_pos[:, 0:-1]  # R^{m * 4}
        Y = train_pos[:, -1]  # R^{m*1}
        # X 与 \theta_0 相乘那一个部分必须设置位1（也就是第一列）。
        m, n = X.shape
        ones = np.ones(m)
        X = np.column_stack((ones, X))
        # 处理 Y
        Y = Y.reshape((m, 1))  # 转换为行向量
        # initial_theta = np.zeros((n + 1,1))  # R^{n+1}  # matrix
        initial_theta = np.zeros(n + 1)  # R^{n+1}  # vector
        # result = minimize_cost(theta=initial_theta, X=X, Y=Y)
        # 获取训练的参数
        # final_theta = result['x']
        # 处理的是自己写的梯度下降
        J, thetas = gradient_descent(initial_theta, X, Y)  # thetas是训练的次数对应的theta参数
        iter = len(J)
        # 调整 theta 转换为矩阵
        final_theta = np.array(thetas[-1])
        # 设置测试的例子
        test_m = test.shape[0]
        test_ones = np.ones(test_m)
        # 测试用例的X矩阵
        # test_X = np.column_stack((test_ones, test[:,0:-1].astype(np.float64)))
        # 将测试集合和训练集合合并
        train_possibilities[:, i] = sigmoid(X.dot(final_theta))
        test_X = np.column_stack((test_ones, test[:,0:-1].astype(np.float64)))
        test_possibilities[:, i] = sigmoid(test_X.dot(final_theta))
        print("正例 %s：[θ0,..,θ2]=" % posClass, final_theta, '迭代次数：%d' % iter)

        if plotClassBoundaryAndLossFunc:
            # 绘制图像
            plot(final_theta, data, posClass, ('sepal length', 'petal length'), X, Y)
            # 绘制损失函数
            plt.figure()
            plt.plot([x for x in range(1, iter + 1)], J)
            plt.show()
    # 对每个测试数据进行了分类。输出测试的信息
    # error = 0
    # for item in range(possibilities.shape[0]):
    #     argmax = possibilities[item].argmax()
    #     if data[item][-1] != classes[argmax]:
    #         error = error + 1
    # print("准确率：%.2f%%" % (100 * (1.0 - float(error) / data.shape[0])))
    # print("错误预测的个数: %d" % error)
    # 针对测试集和训练集的准确率
    test_error, train_error = 0, 0
    for item in range(test_possibilities.shape[0]):
        argmax = test_possibilities[item].argmax()
        if test[item][-1] != classes[argmax]:
            test_error += 1
    for item in range(train_possibilities.shape[0]):
        argmax = train_possibilities[item].argmax()
        if train[item][-1] != classes[argmax]:
            train_error += 1
    print('训练集的准确率:%.2f, 错误个数:%d' % ((1 - train_error / train.shape[0]) * 100, train_error))
    print('测试集的准确率:%.2f, 错误个数:%d' % ((1 - test_error / test.shape[0]) * 100, test_error))



if __name__ == '__main__':
    data = gendata.loadDataset("iris.data")
    proportions = (0.5, 0.7, 0.9)
    if plotAllThreeData:
        plotAllData(data)
    for p in proportions:
        print("%d%%的样本作为训练集：" % (p * 100))
        logistic(p, data)


