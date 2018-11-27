#coding=utf-8
"""
SMO 的实现
建议的和优化的SMO 算法的原理，参见：https://zh.wikipedia.org/wiki/%E5%BA%8F%E5%88%97%E6%9C%80%E5%B0%8F%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95
公式:https://www.jianshu.com/p/eef51f939ace
"""
import random
from MLInAction.ca6 import kernels
from MLInAction.ca6 import plotData
from MLInAction.ca6.loadDataset import loadDataSet
import numpy as np


class SimpleSMO(object):
    """
    最简单的SMO实现
    """


    @staticmethod
    def selectJrandomly(i, m):
        """
        随机选择
        :param i: 第一个aplha 的下标
        :param m: 范围是[0,m)
        :return:
        """
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j

    @staticmethod
    def clipAlpha(aj, H, L):
        """
        调整取值,类似于 clip_by_min_max
        :param aj:
        :param H:
        :param L:
        :return:
        """
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def getWeights(self):
        """
        对于线性的分类器, 求出超平面的 w
        :return:
        """
        X = np.mat(self.dataMatIn)
        y = np.mat(self.classLabels).T
        m, n = X.shape
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(self.alphas[i] * y[i], X[i, :].T)
        return w

    def __init__(self, dataMatIn, classLabels, test_dataMatIn, test_classLabels, slack_C, tolerance, maxIteration):
        """
        Ctor
        :param dataMatIn: 输入的数据，非矩阵形式
        :param classLabels: 标签，向量
        :param test_classLabels: 测试集的样本标签
        :param test_dataMatIn: 测试集的样本
        :param slack_C: 软间隔的松弛变量
        :param tolerance: 允许的误差
        :param maxIteration: 最大的迭代次数
        """
        self.dataMatIn = dataMatIn
        self.classLabels = classLabels
        self.slack_C = slack_C
        self.tolerance = tolerance
        self.maxIteration = maxIteration
        self.alphas = None
        self.b = None
        self.kernel = None
        self.test_dataMatIn = test_dataMatIn
        self.test_classLabels = test_classLabels

    def __call__(self, *args, **kwargs):
        """
        重载了调用运算符，转换为可调用对象
        :param args:
        :param kwargs:
        :return:
        """
        self.alphas = self.b = None
        # 数据处理
        dataMat = np.mat(self.dataMatIn)
        labelMat = np.mat(self.classLabels).T
        b = 0
        m, n = dataMat.shape
        alphas = np.mat(np.zeros((m, 1)))  # 初始的广义拉格朗日引入的 alphas 参数
        # 外循环:是否结束迭代
        iter = 0
        while iter < self.maxIteration:
            alphaPairsChanged = 0  # 用来记录 alpha 对是否已经修改
            # 所有数据集
            for i in range(m):
                fXi = float(np.multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + b  # 预测的类别
                Ei = fXi - float(labelMat[i])  # 与真实类别的误差 E_i=f(x_i) - y_i.
                if (labelMat[i] * Ei < -self.tolerance and alphas[i] < self.slack_C) or (labelMat[i] * Ei > self.tolerance and alphas[i] > 0):
                    # 误差很大的话，需要优化
                    j = self.selectJrandomly(i, m)  # 随机选择的普通策略
                    fXj = float(np.multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T)) + b
                    Ej = fXj - float(labelMat[j])
                    alphas_i_old = alphas[i].copy()  # alpha_i^{old}
                    alphas_j_old = alphas[j].copy()  # 同理,注意不要使用引用.深拷贝
                    if labelMat[i] != labelMat[j]:
                        # y_1 * y_2 == -1 的情况. 也就是不相等
                        U = max(0, alphas[j] - alphas[i])
                        V = min(self.slack_C, self.slack_C + alphas[j] - alphas[i])
                    else:
                        # 相同
                        U = max(0, alphas[j] + alphas[i] - self.slack_C)
                        V = min(self.slack_C, alphas[j] + alphas[i])
                    if U == V:
                        print('警告: U==V')
                        continue
                    # U V 确定的约束使得 a_i^{new} 和 a_j^{new}在矩形区域 [0,C]X[0,C]中
                    K = 2.0 * dataMat[i, :] * dataMat[j, :].T - dataMat[i,:] * dataMat[i,:].T - dataMat[j,:] * dataMat[j,:].T
                    if K >= 0:
                        print('警告: K>=0')
                        continue
                    alphas[j] -= labelMat[j] * (Ei - Ej) / K
                    alphas[j] = self.clipAlpha(alphas[j], V, U)  # alpha_2^{new}. 必须满足直线 a_1*y_1 + a_2*y_2 = \gama 的定义域的取值.在[U,V]
                    if abs(alphas[j] - alphas_j_old) < 0.00001:
                        print('警告: alpha_j 没有足够的移动')
                    alphas[i] += labelMat[j] * labelMat[i] * (alphas_j_old - alphas[j])  # alpha_1^{new}
                    b1 = b - Ei - labelMat[i] * (alphas[i] - alphas_i_old) * \
                         dataMat[i,:] * dataMat[i,:].T - \
                         labelMat[j] * (alphas[j] - alphas_j_old) *\
                         dataMat[i,:] * dataMat[j,:].T
                    b2 = b - Ej - labelMat[i] * (alphas[i] - alphas_i_old) * \
                         dataMat[i, :] * dataMat[j, :].T - \
                         labelMat[j] * (alphas[j] - alphas_j_old) * \
                         dataMat[j, :] * dataMat[j, :].T
                    # 偏移b的修改:
                    if 0 < alphas[i] and self.slack_C > alphas[i]:
                        b = b1
                    elif 0 < alphas[j] and self.slack_C > alphas[j]:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alphaPairsChanged += 1
                    print('第{iter}迭代，{count}α对被优化'.format(iter=iter, count=alphaPairsChanged))
            if alphaPairsChanged == 0:
                iter += 1
            else:
                iter = 0
            print('迭代次数：%d' % iter)
        self.alphas = alphas
        self.b = b
        return b, alphas

class OptimalSMO(SimpleSMO):

    """
    优化版本的SMO算法
    """

    def __init__(self, dataMatIn, classLabels, test_dataMatIn, test_classLabels, slack_C, tolerance, maxIteration, kernel):
        super(OptimalSMO, self).__init__(dataMatIn=dataMatIn,
                                         classLabels=classLabels,
                                         slack_C=slack_C,
                                         tolerance=tolerance,
                                         maxIteration=maxIteration, test_classLabels=test_classLabels,
                                         test_dataMatIn=test_dataMatIn)  # 最后的参数在外循环需要，可以在构造的时候指定
        self.m = np.shape(dataMatIn)[0]  # 样本数量
        self.alphas = np.mat(np.zeros((self.m, 1)))  # 全0向量
        self.b = 0.0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # 误差缓存。eCache[:,0] 表示是否有效，eCache[:,1] 实际存储的E值
        # 存储矩阵形式的数据
        self.dataMat = np.mat(dataMatIn)
        self.labelsMat = np.mat(classLabels).T
        self.testDataMat = np.mat(test_dataMatIn)
        self.testLabelsMat = np.mat(test_classLabels).T
        # 核函数相关
        self.K = kernel.getGramMatrix(self.dataMat)
        self.kernel = kernel

    def calcEk(self, k):
        fXk = float(np.multiply(self.alphas, self.labelsMat).T * self.K[:, k]) + self.b
        Ek = fXk - float(self.labelsMat[k])
        return Ek

    def selectJ(self, Ei, i):
        """
        启发式地搜索。使得 abs(E_1 - E_2) 最大的向量
        :param Ei: 误差地期望。f(\vec{alpha_i}) - label_i
        :param i:
        :return:
        """
        maxK = -1
        maxDeltaE = 0  # 尽可能地
        Ej = 0
        self.eCache[i] = [1, Ei]  # 标记计算的Ei有效
        validEcachedList = np.nonzero(self.eCache[:, 0].A)[0]  # A: mat -> array(asarray(self))。注意，返回的是 alpha_i 的下标 i
        if len(validEcachedList) > 1:
            for k in validEcachedList:
                if k == i:
                    continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = self.selectJrandomly(i, self.m)
            Ej = self.calcEk(j)
        return j, Ej

    def updateEk(self, k):
        """
        更新缓存，将误差保存
        :param k:
        :return:
        """
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]

    def innerLoop(self, i):
        """
        SMO 算法的内循环部分
        :param i:
        :return:
        """
        alphas = self.alphas
        labelMat = self.labelsMat
        dataMat = self.dataMat
        # 以上是为了简化，直接复用简易版本的SMO算法
        Ei = self.calcEk(i)  # 计算误差
        if (labelMat[i] * Ei < self.tolerance and alphas[i] < self.slack_C) or \
            (labelMat[i] * Ei > self.tolerance and alphas[i] > 0):
            j, Ej = self.selectJ(Ei, i)
            alphas_i_old = alphas[i].copy()  # alpha_i^{old}
            alphas_j_old = alphas[j].copy()  # 同理,注意不要使用引用.深拷贝
            if labelMat[i] != labelMat[j]:
                # y_1 * y_2 == -1 的情况. 也就是不相等
                U = max(0, alphas[j] - alphas[i])
                V = min(self.slack_C, self.slack_C + alphas[j] - alphas[i])
            else:
                # 相同
                U = max(0, alphas[j] + alphas[i] - self.slack_C)
                V = min(self.slack_C, alphas[j] + alphas[i])
            if U == V:
                print('警告: U==V')
                return 0
            # U V 确定的约束使得 a_i^{new} 和 a_j^{new}在矩形区域 [0,C]X[0,C]中
            # K = 2.0 * dataMat[i, :] * dataMat[j, :].T - dataMat[i, :] * dataMat[i, :].T - dataMat[j, :] * dataMat[j,
            #                                                                                               :].T
            # 引入核函数
            K = 2.0 * self.K[i,j] - self.K[i,i] - self.K[j,j]
            if K >= 0:
                print('警告: K>=0')
                return 0
            alphas[j] -= labelMat[j] * (Ei - Ej) / K
            alphas[j] = self.clipAlpha(alphas[j], V,
                                       U)  # alpha_2^{new}. 必须满足直线 a_1*y_1 + a_2*y_2 = \gama 的定义域的取值.在[U,V]
            # 更新 Ej 到缓存
            self.updateEk(j)
            if abs(alphas[j] - alphas_j_old) < 0.00001:
                print('警告: alpha_j 没有足够的移动')
                return 0
            alphas[i] += labelMat[j] * labelMat[i] * (alphas_j_old - alphas[j])  # alpha_1^{new}
            self.updateEk(i)  # 由于 alpha_j 的修改，要计算新的 E_i 到缓存
            # b1 = self.b - Ei - labelMat[i] * (alphas[i] - alphas_i_old) * \
            #      dataMat[i, :] * dataMat[i, :].T - \
            #      labelMat[j] * (alphas[j] - alphas_j_old) * \
            #      dataMat[i, :] * dataMat[j, :].T
            # b2 = self.b - Ej - labelMat[i] * (alphas[i] - alphas_i_old) * \
            #      dataMat[i, :] * dataMat[j, :].T - \
            #      labelMat[j] * (alphas[j] - alphas_j_old) * \
            #      dataMat[j, :] * dataMat[j, :].T
            # 引入核函数
            b1 = self.b - Ei - labelMat[i] * (alphas[i] - alphas_i_old) * \
                 self.K[i,i] - labelMat[j] * (alphas[j] - alphas_j_old) * self.K[i, j]
            b2 = self.b - Ej - labelMat[i] * (alphas[i] - alphas_i_old) * \
                 self.K[i, j] - labelMat[j] * (alphas[j] - alphas_j_old) * self.K[j, j]
            # 偏移b的修改:
            if 0 < alphas[i] and self.slack_C > alphas[i]:
                self.b = b1
            elif 0 < alphas[j] and self.slack_C > alphas[j]:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def __call__(self, *args, **kwargs):
        """
        外循环的定义
        :param args:
        :param kwargs:
        :return:
        """
        iter = 0
        wholeSet = True
        alphaPairsChanged = 0
        while iter < self.maxIteration and (alphaPairsChanged > 0 or wholeSet):  # 退出条件，达到最大跌倒或者是遍历整个数据集但是没对任何的alpha修改
            alphaPairsChanged = 0
            if wholeSet:
                for i in range(self.m):
                    alphaPairsChanged += self.innerLoop(i)
                    print('对于整个数据集，迭代：%d次，i：%d，%d 对 alphas 改变' % (iter, i, alphaPairsChanged))
                iter += 1
            else:
                nonBoundIs = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.slack_C))[0]  # 在边界的值，在超平面垂直的2C的范围或者就在边界上.
                # 相当于布尔矩阵对应位置相乘。注意是nonzero。确定了不在边界范围的位置
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerLoop(i)
                    print('没有在边界，迭代：%d次，i：%d，%d 对 alphas 改变' % (iter, i, alphaPairsChanged))
                iter += 1
            if wholeSet: # 控制切换不在边界和这个数据集
                wholeSet = False
            elif alphaPairsChanged == 0:
                wholeSet = True
            print('外循环迭代次数：%d' % iter)
        return self.b, self.alphas

    def testOnTrainDataSet(self):
        """
        在训练集上训练,求出误差
        :return:
        """
        svIndices = np.nonzero(self.alphas.A > 0)[0]  # alpha 大于零对应的数据的索引
        svs = self.dataMat[svIndices]  # 支持向量
        svLabels = self.labelsMat[svIndices] # 支持向量的标签
        print('支持向量:%d' % svIndices.shape[0])
        for i in range(svIndices.shape[0]):
            print('索引号={i}, x_i={data}, y_i={label}'.format(i=svIndices[i], data=svs[i], label=svLabels[i]))
        m, n = np.shape(self.dataMat)
        error = 0
        predict = self.kernel.predict(self.dataMat, self.alphas, self.b, svs, svIndices, svLabels)
        for i in range(m):
            if np.sign(predict[i]) != np.sign(self.labelsMat[i]):
                error += 1
        print("训练集的正确率:%.2f" % (1.0 - float(error) / m))

    def testOnTestDataSet(self):
        """
        在测试集上训练
        :return:
        """
        svIndices = np.nonzero(self.alphas.A > 0)[0]
        svs = self.dataMat[svIndices]
        svLabels = self.labelsMat[svIndices]
        # print('支持向量:%d' % svIndices.shape[0])
        # for i in range(svIndices.shape[0]):
        #     print('索引号={i}, x_i={data}, y_i={label}'.format(i=svIndices[i], data=svs[i], label=svLabels[i]))
        m, n = np.shape(self.dataMat)
        error = 0
        predict = self.kernel.predict(self.testDataMat, self.alphas, self.b, svs, svIndices, svLabels)
        for i in range(m):
            if np.sign(predict[i]) != np.sign(self.testLabelsMat[i]):
                error += 1
        print("训练集的正确率:%.2f" % (1.0 - float(error) / m))







if __name__ == '__main__':
    data, labels = loadDataSet('dataset_train_rbf.txt')
    tdata, tlabels = loadDataSet('dataset_test_rbf.txt')
    kernel_rbf = kernels.RBFKernel(1.4)  # RBF 核函数
    kernel_poly = kernels.PolynomialKernel(degree=11, remain=0)  # 多项式核函数,还有问题
    kernel_linear = kernels.LinearKernel()  # 线性核函数
    smo = OptimalSMO(data, labels, slack_C=200, tolerance=0.001, maxIteration=10000, kernel=kernel_rbf, test_dataMatIn=tdata, test_classLabels=tlabels)  # C的取值越大，效果不一定越好。算法会试图找到最大的间隔使他们分开


    b, alphas = smo()  # SMO 优化对象
    plotData.plotAllData(np.array(data), np.array(labels), title='数据点和支持向量', two_featuresname=['1', '2'], classnames=['C1', 'C2'],
                         smoObj=smo)
    smo.testOnTrainDataSet()
    smo.testOnTestDataSet()














