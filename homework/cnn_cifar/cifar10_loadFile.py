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

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


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
            self.X = X
            self.Y = Y


class Cifar10DataGenerator(object):

    """
    通用的数据产生器
    """

    def __init__(self, next_batch_size, filenames, path_prefix='./cifar10_data'):
        """
        定义 cifar10 的数据产生器
        :param next_batch_size: 一批的图片样本数量
        :param filenames: 文件名列表
        :param path_prefix: 载入的前缀
        """
        from keras.utils import to_categorical
        # 标题
        with open(os.sep.join((path_prefix, 'batches.meta')), 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
            self.label_names = meta['label_names']  # 标签名称
        # 数据部分
        lists = [CifarData(filename, loadImmediately=True, path_prefix=path_prefix) for filename in filenames]  # 数据列表
        packs = len(lists)
        X = np.vstack([lists[x].X for x in range(packs)])  # 将每一批次的数据整合为一个数据集 X
        Y = np.hstack([lists[x].Y for x in range(packs)])  # 注意每个 Y 是一个列向量，要行连接
        # 数据处理
        X = X / 255.  # 归一
        X_mean = np.mean(X, axis=0)
        X -= X_mean
        self.X = X
        self.Y = to_categorical(Y, 10)  # one-hot
        self.batch_size = next_batch_size

    def num_images(self):
        """
        包含图片的大小
        :return:
        """
        return self.X.shape[0]

    def generate_augment_batch(self):
        """
        测试数据产生， 所有的数据将按照顺序返回. 有限循环 循环  num_images // batch_size
        :return:
        """
        batchs = self.num_images() // self.batch_size
        for start_pos in range(0, self.num_images(), self.batch_size):
            batchs -= 1
            end = start_pos + self.batch_size
            yield self.X[start_pos:end], self.Y[start_pos:end,:]
            if batchs == 0:
                break


class Cifar10TestDataGenerator(Cifar10DataGenerator):

    """
    测试数据集合产生
    """
    def __init__(self, next_batch_size, path_prefix='./cifar10_data'):
        super(Cifar10TestDataGenerator, self).__init__(next_batch_size=next_batch_size, path_prefix=path_prefix, filenames=['test_batch'])


class Cifar10PreprocessedAugmentDataGenerator(Cifar10DataGenerator):

    def __init__(self, next_batch_size, path_prefix='./cifar10_data'):
        """
        使用 Keras 处理的数据, 用于训练集
        :param next_batch_size: 下一批的数据大小
        :param path_prefix: 数据前导路径
        """
        from keras.preprocessing.image import ImageDataGenerator
        super(Cifar10PreprocessedAugmentDataGenerator, self).__init__(next_batch_size=next_batch_size,
                                                                      path_prefix=path_prefix,
                                                                      filenames=['data_batch_%d' % x for x in range(1, 6)])
        # 图像增强
        self.datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=True,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0.2,
            # randomly shift images horizontally
            width_shift_range=0.2,
            # randomly shift images vertically
            height_shift_range=0.2,
            # set range for random shear
            shear_range=0.2,
            # set range for random zoom
            zoom_range=0.2,
            # set range for random channel shifts
            channel_shift_range=0.2,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

    def generate_augment_batch(self):
        """
        产生一批次的数据（可迭代对象, 有限形式）
        :return: 图像 [BATCH SIZE, HEIGHT, WIDTH, CHANNEL], 标签（One-hot）。注意循环仅仅执行 数据长度 // BATCH_SIZE
        """
        batchs = self.num_images() // self.batch_size
        for batchX, batchY in self.datagen.flow(self.X, self.Y, batch_size=self.batch_size):
            batchs -= 1
            yield batchX, batchY
            if batchs == 0:
                break


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    bc = 128
    aug = Cifar10TestDataGenerator(bc)

    for X, Y in aug.generate_augment_batch():
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            img = X[i]
            I = np.stack((img[:, :, 0], img[:, :, 1], img[:, :, 2]), axis=2)  # 新产生的合并的数据是 第3个维度. 32x32x3
            plt.title('label=' + aug.label_names[np.argmax(Y[i])])
            plt.imshow(I)

        # show the plot
        plt.show()


