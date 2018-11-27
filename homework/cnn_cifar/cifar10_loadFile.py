"""
读取 Cifar 的数据:
数据集描述:
1. Python 版本的是使用 Pickle 序列化的数据
2. data 键关联的是 uint8 的 32x32 的图片的 numpy 矩阵
3. 前1024 的条目包含的是 R 分量, 后面1024依次是 G 和 B。image 是行优先的顺序。32x32x3 = 1024 * 3.总共 3072 的数据
4. 每个 batch  10000 个图像
5. 由于可供训练的数据不多, 需要对数据进行增广
"""
import pickle
import numpy as np
import os

class CifarData(object):

    """
    Cifar 数据记录的对象
    具有一下方法.
    """
    def __init__(self, filename, loadImmediately=False, path_prefix='./data'):
        """
        定义 Cifar10 的输入数据格式
        :param filename: 文件名
        :param path_prefix: 数据保存的前缀目录
        :param loadImmediately: 是否立即加载到内存
        """
        self.filename = os.sep.join((path_prefix, filename))
        if loadImmediately:
            self.loadData()

    @staticmethod
    def reshape_X(images):
        """
        由于 cifar-10 的数据是 [batch 3 32 32] 的格式， 与 tf 默认 cov2d 定义的 [batch 32 32 3] 不同
        :param images: 图像
        :return:
        """
        nx = np.transpose(images, (0, 2, 3, 1))  # BATCH WIDTH HEIGHT CHANELS
        return nx

    def loadData(self):
        """
        加载数据集
        :return:
        """
        with open(self.filename, 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
            print('Loaded: ' + str(dataset.keys()))
            X = dataset['data']
            Y = dataset['labels']
            X = np.reshape(X, (10000, 3, 32, 32))  # 图像数据, 这不是 tensorflow 的格式
            Y = np.array(Y)
            X = self.reshape_X(X)  # 转换为 tensorflow 格式的数据（我不需要别的处理）
            # 处理图像
            # pad_width = ((0, 0), (2, 2), (2, 2), (0, 0))
            # X = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)
            # X = random_crop_and_flip(X, padding_size=2)
            # X = whitening_image(X)
            self.X = X
            self.Y = Y

class DataGenerator(object):

    def generate_augment_batch(self):
        raise NotImplementedError('generate_augment_batch')


class CifarDataGenerator(DataGenerator):
    """
    这个是 Cifar 数据产生。可以产生任何批次的数据。具体的子类包括训练集、测试集的生成
    """

    def __init__(self, testOnly, next_batch_size):
        super(CifarDataGenerator, self).__init__()
        self.next_batch_size = next_batch_size
        if testOnly:
            self.dataMap = {1: CifarData('./data/test_batch', True)}

        else:
            # 有测试的数据
            lists = [CifarData('./data/data_batch_%d' % x, loadImmediately=True) for x in range(1, 6)]
            # 转换为字典
            self.dataMap = dict()
            for i, obj in enumerate(lists):
                self.dataMap[i + 1] = obj
        # 初始化定位的内容
        ## 批次号
        self.batch_no = 1
        ## 每一个批次的下一条数据的起始位置
        self.next_inner_batch_pos = 0

    def next_data(self):
        """
        返回下一次的图像数据、标签。
        :return: 数据样本[next_batch_size, 3, 32, 32]， 标签 [next_batch_size]
        """
        if self.next_inner_batch_pos >= 10000:
            if self.batch_no >= 5:
                # 重新循环
                self.batch_no = 1
                self.next_inner_batch_pos = 0
            else:
                # 加载新的 batch
                # 释放之前的资源
                self.batch_no += 1
                self.next_inner_batch_pos = 0
        # 已经确保数据可用了
        upper = self.next_inner_batch_pos + self.next_batch_size
        X = self.dataMap[self.batch_no].X[self.next_inner_batch_pos: upper,:,:,:]
        y = self.dataMap[self.batch_no].Y[self.next_inner_batch_pos: upper]
        # 随机化
        index = np.arange(self.next_batch_size)
        np.random.shuffle(index)
        X = X[index, :, :, :]
        y = y[index, :]
        # 递增新的坐标
        self.next_inner_batch_pos = upper
        return X, y

class AugmentImageGenerator(object):

    """
    随机选择一组数据，可以认为是增广的
    """
    def __init__(self, testOnly, next_batch_size, path_prefix='./cifar10_data'):
        super(AugmentImageGenerator, self).__init__()
        # 标题
        with open(os.sep.join((path_prefix, 'batches.meta')), 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
            self.label_names = meta['label_names']  # 标签名称
        if testOnly:
            testXY = CifarData('test_batch', loadImmediately=True, path_prefix=path_prefix)
            X = testXY.X
            Y = testXY.Y
        else:
            lists = [CifarData('data_batch_%d' % x, loadImmediately=True, path_prefix=path_prefix) for x in range(1, 6)]

            X = np.vstack([lists[x].X for x in range(5)])  # 将每一批次的数据整合为一个数据集 X
            Y = np.hstack([lists[x].Y for x in range(5)])  # 注意每个 Y 是一个列向量，要行连接

        self.X = X.astype(np.float32)  # 设置为样本矩阵
        self.Y = Y  # 设置标签数据
        self.batch_size = next_batch_size  # 批次的大小
        self._orders = np.arange(0, X.shape[0])  # 维护一个下标的数组，用于形成随机的坐标
        # 测试集合的数据都要测试到位
        self._testStartPos = 0
        self.testOnly =testOnly
        self.length_of_X = X.shape[0]

    def generate_augment_batch(self):
        """
        生成一个 batch_size 大小的数据集
        :return:
        """
        if self.testOnly:
            if self._testStartPos + self.batch_size >= self.length_of_X:
                # 余下的几个
                remain = self.length_of_X - self._testStartPos
                # 计算起始位置开始读取多少个
                remain_to_use = self.batch_size - remain
                # 获取切片的索引
                res = np.vstack((self.X[self._testStartPos:], self.X[:remain_to_use]))
                resy = np.hstack((self.Y[self._testStartPos:], self.Y[:remain_to_use]))
                self._testStartPos = remain_to_use
            else:
                res = self.X[self._testStartPos:self._testStartPos + self.batch_size]
                resy = self.Y[self._testStartPos:self._testStartPos + self.batch_size]
                self._testStartPos = self._testStartPos + self.batch_size
            return res, resy
        else:
            np.random.shuffle(self._orders)  # 打乱顺序
            indices = np.random.choice(self._orders, size=self.batch_size, replace=False)  # 不要替换（重复）
            batch_data = self.X[indices]
            # 可以做一下其他的图像处理，相见 preprocess， 但是感觉差不多
            batch_label = self.Y[indices]
            return batch_data, batch_label

class ResizedAugmentImageGenerator(AugmentImageGenerator):

    def __init__(self, testOnly, next_batch_size, processImageAfterInit, path_prefix='./cifar10_data'):
        super(ResizedAugmentImageGenerator, self).__init__(testOnly=testOnly, next_batch_size=next_batch_size, path_prefix=path_prefix)
        self.processed = False
        if processImageAfterInit:
            self.X = self._processImage(self.X)
            self.processed = True

    def generate_augment_batch(self):
        images, labels = super(ResizedAugmentImageGenerator, self).generate_augment_batch()
        if not self.processed:
            images = self._processImage(images)
        return images, labels

    def _processImage(self, images):
        # 处理图像
        from skimage import transform
        # 注意 X 的格式是 [BACH HEIGHT WIDTH CHANNEL]
        newImage = []
        for i in range(images.shape[0]):
            img = images[i]
            t = transform.resize(img.astype(np.uint8), (224, 224)) * 255
            newImage.append(t)
        # 重新设置
        return np.array(newImage)


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    aug = AugmentImageGenerator(True, 9999)
    for k in range(10):
        X, Y = aug.generate_augment_batch()

        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            img = X[i] / 255.
            I = np.stack((img[:, :, 0], img[:, :, 1], img[:, :, 2]), axis=2)  # 新产生的合并的数据是 第3个维度. 32x32x3
            plt.title('label=' + aug.label_names[Y[i]])
            plt.imshow(I)

        # show the plot
        plt.show()

