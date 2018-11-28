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

    def __init__(self, next_batch_size, filenames, path_prefix='/tmp/cifar-10-batches-py'):
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
        self.X = X / 255.  # 归一
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
    def __init__(self, next_batch_size):
        super(Cifar10TestDataGenerator, self).__init__(next_batch_size=next_batch_size, filenames=['test_batch'])


class Cifar10PreprocessedAugmentDataGenerator(Cifar10DataGenerator):

    def __init__(self, next_batch_size):
        """
        使用 Keras 处理的数据, 用于训练集
        :param next_batch_size: 下一批的数据大小
        :param path_prefix: 数据前导路径
        """
        from keras.preprocessing.image import ImageDataGenerator
        super(Cifar10PreprocessedAugmentDataGenerator, self).__init__(next_batch_size=next_batch_size,
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



import tensorflow as tf
from math import sqrt



class CNNNetwork(object):

    """
    CNN 常用的结构
    """

    def __init__(self,
                 batch_size=128,
                 image_width_height=32,
                 conv_strides=[1,1,1,1],
                 pool_kernel_size=[1,3,3,1],
                 pool_strides=[1,2,2,1],
                 num_classes=10,
                 keep_prob=0.5,
                 regularizer_ratio=0.002,
                 init_learning_rate=0.001,
                 num_epoch_per_decay=5,
                 learning_rate_decay=0.1,
                 batch_normalization_epsilon=0.001,
                 tiny_output_logs=True
                 ):
        """
        :param batch_size: 每一次送如神经网络的样本数量，满足 batch_size * epoch = num_examples_per_epoch
        :param image_width_height: 图像的长度和高度，默认是 32， VGG 等其他的长度不一样
        :param conv_strides: 卷积层步长，输入张量的每一个维度上都只是移动1
        :param pool_kernel_size: 池化的核，不需要再 BATCH_SIZE 和 CHANNELS 上采样。所以移动 1
        :param pool_strides: 池化的步长
        :param keep_prob: dropout 的比例
        :param regularizer_ratio: L2 正则化系数
        :param init_learning_rate: 初始的学习率(学习率衰减)
        :param num_epoch_per_decay: 当训练完一批数据的
        :param batch_normalization_epsilon: 对卷积层输出做 batch normalization 的参数 epsilon
        :param learning_rate_decay: 学习率衰减率， num_epoch_per_decay 可以决定衰减的 epoch 间隔
        :param tiny_output_logs: 仅仅输出 loss acc 和 learning rate 的数据（不包含各层权重等）
        """

        self.conv_strides = conv_strides
        self.pool_kernel_size = pool_kernel_size
        self.pool_strides = pool_strides
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.regularizer_ratio = regularizer_ratio
        self.batch_normalization_epsilon = batch_normalization_epsilon

        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.num_epoch_per_decay = num_epoch_per_decay
        self.learning_rate_decay = learning_rate_decay
        self.tiny_output_logs = tiny_output_logs
        self.image_width_height = image_width_height
        # 记录(非训练过程)卷积核的输出信息，注意需要保存输出图像的大小，后面几层仅过了激活要 / 2
        # 由 tf collection 保存



class TensorflowNetwork(CNNNetwork):

    def __init__(self, training, num_examples_per_epoch, **kwargs):
        """
        定义基于 Tensorflow 的神经网络结构
        :param training: 是否是训练状态
        :param num_examples_per_epoch: epoch 包含的样本数量（cifar10一般的训练集就是50000）
        :param kwargs: 参数
        """
        self.training = training
        self.num_examples_per_epoch = num_examples_per_epoch
        super(TensorflowNetwork, self).__init__(**kwargs)

    def inference(self, images, **kwargs):
        raise NotImplementedError('没有实现前向传播网络')

    def _add_activated_summary(self, tensor):
        """
        为一个激活后的卷积层输出的张量添加 summary
        :param tensor: 张量
        :return: None
        """
        if not self.tiny_output_logs:
            tensor_name = tensor.op.name  # relu 激活的操作
            tf.summary.histogram(tensor_name + '/activations', tensor)
            tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor))  # 统计0的比例反映稀疏度

    def put_kernels_on_grid(self, kernel, grid_Y, grid_X, pad=1):
        """
        参考：
        将 kernel 输出为一张图片中，主要进行 filter 各个channel的排列
        :param kernel: 卷积核 4-D 张量 [Y, X, NumChannels, NumKernels]
        :param grid_Y: 输出小卷积核的长 满足 NumKernels == grid_Y * grid_X
        :param grid_X: 输出小卷积的宽 满足 NumKernels == grid_Y * grid_X
        :param pad: 小卷积核的间隔像素
        :return: 4-D 张量 [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
        """
        # 求卷积核的最大和最小值
        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        # 归一化
        kernel1 = (kernel - x_min) / (x_max - x_min)

        # 填充 X 和 Y 的部分
        x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

        # 添加填充的 X Y 维度
        Y = kernel1.get_shape()[0] + 2 * pad  #
        X = kernel1.get_shape()[1] + 2 * pad
        # 输入的维度
        channels = kernel1.get_shape()[2]

        # 把输出的通道数的维度放到第一维上
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # 在 Y 轴调整 grid
        x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))  # 3

        # 对调 X Y 轴
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # 在 X 轴调整 grid
        x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))  # 3

        # 调整为正常顺序
        x6 = tf.transpose(x5, (2, 1, 3, 0))
        #
        # # summary 图像格式 [batch_size, height, width, channels],
        # #  只有一个图像：输出的是 batch_size
        # x7 = tf.transpose(x6, (3, 0, 1, 2))
        #
        # # scale to [0, 255] and convert to uint8
        return x6

    def _add_conv_output_image(self, scope, kernel, conv_output, conv_output_size, filter_out_channels):
        """
        https://gist.github.com/panmari/4622b78ce21e44e2d69c
        添加 卷积核的输出
        :param conv_output: 卷积核输出
        :return:
        """

        # 拆分 grid 的行列 是的行列相乘等于 kernel 的输出 channel
        # 求解最大的因子
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1:
                        print('不要用质数作为 filter 输出的通道数')
                    return (i, int(n / i))

        out_channel = kernel.get_shape().as_list()[3]
        grid_y, grid_x = factorization(out_channel)
        grid = self.put_kernels_on_grid(kernel=kernel, grid_X=grid_x, grid_Y=grid_y)
        tf.summary.image(scope.name + '/filter', grid, max_outputs=out_channel)

        # images
        # out_channels = filter_out_channels
        # layer_image1 = tf.transpose(conv_output, perm=[3,0,1,2])
        # tf.summary.image(scope.name + "/filtered_images", layer_image1, max_outputs=out_channel)
        ## Prepare for visualization
        # # Take only convolutions of first image, discard convolutions for other images.
        # V = tf.slice(conv_output, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input')
        # V = tf.reshape(V, (conv_output_size, conv_output_size, filter_out_channels))
        #
        # # Reorder so the channels are in the first dimension, x and y follow.
        # V = tf.transpose(V, (2, 0, 1))
        # # Bring into shape expected by image_summary
        # V = tf.reshape(V, (-1, conv_output_size, conv_output_size, 1))
        # tf.summary.image(tensor=V, max_outputs=1, name=scope.name +'/img_out')
        # 把 kernel 放缩到 [0,1] 的浮点
        # x_min = tf.reduce_min(kernel)
        # x_max = tf.reduce_max(kernel)
        # kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
        # # 转换 kernel 到 tf 的输入样本格式 [batch_size, height, width, channels]
        # kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
        # # 显示随机的从 卷积层输出的三个过滤器
        # # tf.summary.image(scope.name + '/filter', kernel_transposed, max_outputs=3)
        # # 随机选取 16 张输出处理后的图片显示
        # layer_image1 = conv_output[0:1, :, :, 0:16]
        # layer_image1 = tf.transpose(layer_image1, perm=[3, 1, 2, 0])
        # tf.summary.image(scope.name + '/filted_image', layer_image1, max_outputs=16)
        # tf.summary.image(scope.name + '/conv_output', conv_output, max_outputs=3)

    def _create_or_get_variable(self, name, shape, initializer=tf.contrib.layers.xavier_initializer()):
        """
        获取一个变量, 如果这个变量不存在，则需会自动添加。通常是 FC 中 w 和 b 以及 CNN 中的过滤器
        :param name: 变量名
        :param shape: 维度
        :param initializer: 如果首次获取，就使用初始化器
        :return: 返回一个张量， 维度=shape
        """
        regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_ratio)
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, regularizer=regularizer)
        return var

    def _add_batch_normalization_for_fc(self, input, channel_dim, name="batch_normalization"):
        """
        添加 Batch normalization 层
        :param input: 输入的变量
        :param channel_dim: 通道的维度
        :return: batch normalization 处理的结果
        """
        mean, variance = tf.nn.moments(input, axes=[0])
        beta = tf.get_variable('beta', channel_dim, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', channel_dim, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        bn = tf.nn.batch_normalization(input, mean, variance, beta, gamma, self.batch_normalization_epsilon, name=name)
        return bn

    def _add_batch_normalization_for_tensor_input(self, input, channel_dim, name="batch_normalization"):
        """
        添加 Batch normalization 层，针对的是 4-D张量输入
        :param input: 输入的变量
        :param channel_dim: 通道的维度
        :return: batch normalization 处理的结果
        """
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2])
        beta = tf.get_variable('beta', channel_dim, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', channel_dim, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        bn = tf.nn.batch_normalization(input, mean, variance, beta, gamma, self.batch_normalization_epsilon, name=name)
        return bn

    def _reshape_to_flatten(self, input, op_name):
        """
        在 conv 输出后转换为全连接层的输入张量
        :param input: 卷积层的输出
        :param op_name: 操作命名
        :return: 张量
        """
        shape = input.get_shape().as_list()  # 将第四层的输出的维度保存为列表。维度意义与输入张量相同。第一维是batch
        nodes = shape[1] * shape[2] * shape[3]  # weight * height * channels. 全连接层输出的张量，转换为向量。一个行向量是一个样本。输入 FC 做分类
        fc_reshape = tf.reshape(input, (shape[0], nodes), name=op_name)
        return fc_reshape

    def _add_flatten_layer(self, input, output_nodes_num, layerId):
        """
        添加一个conv 之后的一个全连接层
        :param input: 输入张量
        :param output_nodes_num: 输出的节点数量
        :param layerId: 层次编号
        :return:
        """
        fc_reshape = self._reshape_to_flatten(input, 'flatten_op')
        nodes = fc_reshape.get_shape().as_list()[-1]
        with tf.variable_scope('flatten%d' % layerId) as scope:

            weights = self._create_or_get_variable(name='weights', shape=[nodes, output_nodes_num], initializer=tf.uniform_unit_scaling_initializer(1.0))
            biases = self._create_or_get_variable(name='biases', shape=[output_nodes_num], initializer=tf.zeros_initializer())
            results = tf.matmul(fc_reshape, weights) + biases
            fc_bn = self._add_batch_normalization_for_fc(results, output_nodes_num)
            activated = tf.nn.relu(fc_bn, name=scope.name)
            self._add_activated_summary(activated)
        return activated

    def _add_full_connected_layer(self, input, output_nodes_num, input_node_num, layerId):
        """
        添加全连接层
        :param input: 输入
        :param first: 是否是第一层（连接在最后一个卷积层后，需要 flatten 操作）
        :param output_nodes_num: 输出的节点数量（中间层、第一层需要）
        :param input_node_num: 输入的节点数量（中间层、最后一层需要）
        :param layerId: 层ID
        :return: 输出一个张量
        """
        with tf.variable_scope('fc%d' % layerId) as scope:
            weights = self._create_or_get_variable(name='weights', shape=[input_node_num, output_nodes_num], initializer=tf.uniform_unit_scaling_initializer())
            biases = self._create_or_get_variable(name='biases', shape=[output_nodes_num], initializer=tf.zeros_initializer())
            results = tf.matmul(input, weights) + biases
            fc_bn = self._add_batch_normalization_for_fc(results, output_nodes_num)
            activated = tf.nn.relu(fc_bn, name=scope.name)
            self._add_activated_summary(activated)
        return activated

    def _add_linear_softmax(self, input, input_node_num, layerId):
        """
        最后一层进行 softmax
        :param input: 输入
        :param input_node_num:输入的节点个数
        :param layerId: 层次ID
        :return: 张量 [BATCH_SIZE N]
        """
        # 最后一层是 softmax 的输出
        with tf.variable_scope('softmax_linear%d' % layerId) as scope:
            weights = self._create_or_get_variable(name='weights', shape=[input_node_num, self.num_classes], initializer=tf.uniform_unit_scaling_initializer())
            biases = self._create_or_get_variable(name='biases', shape=[self.num_classes], initializer=tf.zeros_initializer())
            results = tf.matmul(input, weights) + biases
            softmax_linear = self._add_batch_normalization_for_fc(results, self.num_classes, name=scope.name)
            self._add_activated_summary(softmax_linear)
        return softmax_linear

    def _add_droupout(self, input, op_id):
        """
        添加 dropout 层
        :param input: 输入
        :param op_id: 操作编号
        :return: 张量
        """
        if self.training:
            drop = tf.nn.dropout(input, self.keep_prob, name='drop%d' % op_id)
            return drop
        return input

    def _add_max_pool(self, input, ksize, strides, padding, op_id):
        """
        添加最大池化层
        :param input: 输入
        :param ksize: 核大小
        :param strides: 步长
        :param padding: 填充的格式: SAME 不足部分补充 0, VALID
        :param op_id:操作编号
        :return:不改变输入的大小的张量
        """
        mpool = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding, name='max_pool%d' % op_id)
        return mpool

    def loss(self, logits, labels):
        """
        将 logits 的预测损失加入到 tf 的集合中
        :param logits: logit 的输出
        :param labels: 标签(One-hot)，维度 [batch_size, num_classes].
        :return: 返回交叉熵的 loss 张量集合操作
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, name='cross_entropy_per_example')  # 每一个样本的交叉熵
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')  # 交叉熵均值
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, cross_entropy_mean)  # 交叉熵的损失
        # 总的损失是交叉熵损失加上所有变量的 L2正则化损失
        return tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    def train(self, total_loss, global_step):
        """
        训练
        :param total_loss: 所有的损失的操添加操作，也就是 self.loss 返回的结果
        :param global_step: 当前的迭代步数
        :return: 返回训练操作, 学习率
        """
        # 学习率衰减：影响学习率的变量.
        num_batches_per_epoch = self.num_examples_per_epoch / self.batch_size  # 每一训练完的一次 epoch 的迭代次数(BATCH_SIZE每一批次)
        decay_steps = int(num_batches_per_epoch * self.num_epoch_per_decay)  # 跑完多少次 epoch 就衰减
        # 学习率随着迭代次数指数衰减
        learning_rate = tf.train.exponential_decay(self.init_learning_rate,
                                                   global_step,  # 计算总的 step
                                                   decay_steps,  # 所有样本训练样本完的迭代次数
                                                   self.learning_rate_decay,  # 衰减率
                                                   staircase=True)  # 阶梯状衰减
        # 显示学习率
        tf.summary.scalar('learning_rate', learning_rate)
        # 优化
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(total_loss, global_step)
        return train_op, learning_rate

    def accuracy(self, logits, target_labels):
        """
        计算准确率
        :param logits: 输出各个样本的频率
        :param target_labels: 标签(one-hot) [BATCH SIZE, num_classes]
        :return: 返回正确率的操作，比较的结果，是布尔张量
        """
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(target_labels, 1))  # y 和 y^hat 的误差
        # 将 bool 向量转化为浮点数矩阵，求和就是准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求值为 1.0 的数量
        return accuracy, correct_prediction

    def getConvList(self):
        """
        获取所有的卷积核
        :return: 卷积核列表（按照添加的顺序） 元素为张量
        """
        return tf.get_collection('convs')


class NormalCNNNetwork(TensorflowNetwork):

    def __init__(self, training, num_examples_per_epoch, **kwargs):
        """
        构建普通的神经网络（Cifar-10）
        :param training: 是否是训练状态
        :param num_examples_per_epoch: 总体的样本数量
        """
        super(NormalCNNNetwork, self).__init__(training=training, num_examples_per_epoch=num_examples_per_epoch,**kwargs)

    def _add_signle_conv(self, input, filter_in_channels, filter_out_channels, layerId, activation=tf.nn.relu):
        """
        添加一个简单的卷积层
        :param input: 输入的4-D张量
        :param scope: 所属的变量域
        :param filter_in_channels: 输入的 channel 的维度
        :param filter_out_channels: 输出过滤后的维度
        :param layerId: 层次 id
        :param activation: 激活函数(一元),默认 relu
        :return:
        """
        with tf.variable_scope('layer%d' % layerId) as scope:
            weights = self._create_or_get_variable(name='weights', shape=[5, 5, filter_in_channels, filter_out_channels], initializer=tf.uniform_unit_scaling_initializer())  # 共享权重
            conv = tf.nn.conv2d(input=input, filter=weights, strides=self.conv_strides, padding='SAME', name='conv')
            biases = self._create_or_get_variable(name='biases', shape=[filter_out_channels], initializer=tf.zeros_initializer())  # 偏移
            to_activate = tf.nn.bias_add(conv, biases)  # biases 只能是1-D维度的
            # 进行 batch_norm
            conv = activation(to_activate, name=scope.name)  # 激活
            self._add_activated_summary(conv)
        return conv

    def _add_signle_conv_bn_act(self, input, filter_in_channels, filter_out_channels, conv_output_size, layerId, activation=tf.nn.relu):
        """
        添加一个简单的卷积层：卷积、偏移、batch_norm  和 激活
        :param input: 输入的4-D张量
        :param scope: 所属的变量域
        :param filter_in_channels: 输入的 channel 的维度
        :param filter_out_channels: 输出过滤后的维度
        :param layerId: 层次 id
        :param activation: 激活函数(一元),默认 relu
        :return:
        """
        with tf.variable_scope('layer%d' % layerId) as scope:
            weights = self._create_or_get_variable(name='weights', shape=[5, 5, filter_in_channels, filter_out_channels], initializer=tf.uniform_unit_scaling_initializer())  # 共享权重
            conv = tf.nn.conv2d(input=input, filter=weights, strides=self.conv_strides, padding='SAME', name='conv')

            # self._add_conv_output_image(scope, weights, conv_output=conv, conv_output_size=conv_output_size,
            #                             filter_out_channels=filter_out_channels)
            tf.add_to_collection('convs', conv)
            biases = self._create_or_get_variable(name='biases', shape=[filter_out_channels], initializer=tf.zeros_initializer())  # 偏移
            to_activate = tf.nn.bias_add(conv, biases)  # biases 只能是1-D维度的
            to_activate = self._add_batch_normalization_for_tensor_input(to_activate, channel_dim=filter_out_channels)
            # 进行 batch_norm

            conv = activation(to_activate, name=scope.name)  # 激活

            self._add_activated_summary(conv)
        return conv

    def inference(self, images, **kwargs):
        """
        前向传播过程
        :param images: 输入的图像， 必须是 [BATCH_SIZE, IN_HEIGHT, IN_WIDTH, CHANNELS]
        :return: softmax 的张量（未归一化，也不需要）。
        """
        POOL_KERNEL_SIZE = self.pool_kernel_size
        POOL_STRIDES = self.pool_strides
        # Layer1
        layer1 = self._add_signle_conv_bn_act(input=images, layerId=1, filter_in_channels=3, filter_out_channels=32, conv_output_size=32)
        # Layer2
        layer2 = self._add_max_pool(
            input=self._add_signle_conv_bn_act(input=layer1, layerId=2, filter_in_channels=32, filter_out_channels=32, conv_output_size=32),
            ksize=POOL_KERNEL_SIZE, strides=POOL_STRIDES, padding='SAME', op_id=2)
        # Layer3
        layer3 = self._add_signle_conv_bn_act(input=layer2, layerId=3, filter_in_channels=32, filter_out_channels=64, conv_output_size=16)
        # Layer4
        layer4 = self._add_max_pool(
            input=self._add_signle_conv_bn_act(input=layer3, layerId=4, filter_in_channels=64, filter_out_channels=64, conv_output_size=16),
            ksize=POOL_KERNEL_SIZE, strides=POOL_STRIDES, padding='SAME', op_id=4)
        # Layer5
        layer5 = self._add_flatten_layer(layer4, 384, 5)
        # Layer6
        layer6 = self._add_full_connected_layer(layer5, 192, 384, 6)
        # Layer7
        softmax_linear = self._add_linear_softmax(layer6, 192, 7)
        return softmax_linear

    def __str__(self):
        return 'NormalCNNNetwork'

class VGGNetwork(TensorflowNetwork):

    def __init__(self, training, num_examples_per_epoch, **kwargs):
        """
        VGG19 cifar-10 的实现
        :param num_examples_per_epoch: 样本总量
        """
        super(VGGNetwork, self).__init__(training=training, num_examples_per_epoch=num_examples_per_epoch,pool_strides=[1,2,2,1], pool_kernel_size=[1,2,2,1], **kwargs)

    def _add_vgg_conv_bn_act_layer(self, input, id_conv, in_filter_channels, out_filter_channels, lastConv=False):
        convname = 'conv%d' % id_conv
        with tf.variable_scope(convname) as scope:
            weights = self._create_or_get_variable(name='weights', shape=[3, 3, in_filter_channels, out_filter_channels], initializer=tf.uniform_unit_scaling_initializer())  # 共享权重
            conv = tf.nn.conv2d(input=input, filter=weights, strides=self.conv_strides, padding='SAME', name=convname)
            if lastConv:
                # 添加这个 block 的卷积核输出
                tf.add_to_collection('convs', conv)
            to_activate = self._add_batch_normalization_for_tensor_input(conv, out_filter_channels)
            # 进行 batch_norm
            activated = tf.nn.relu(to_activate)  # 激活
            self._add_activated_summary(activated)
        return activated

    def _add_vgg_block(self, input, id_block, num_conv_repeat, in_filter_channels, out_filter_channels):
        """
        添加 VGG-19 的 block
        :param id_block: block 的编号
        :param num_conv_repeat: 卷积层重复量
        :param in_filter_channels:　过滤器的输入通道
        :param out_filter_channels: 过滤器的输出通道
        :param last_block: 是否是最后一个 block
        :return:
        """
        blockname = 'block%d' % id_block
        with tf.variable_scope(blockname):
            conv = self._add_vgg_conv_bn_act_layer(input, 1, in_filter_channels, out_filter_channels)
            for i in range(2, num_conv_repeat + 1):
                # 添加卷积层
                conv = self._add_vgg_conv_bn_act_layer(conv, i, out_filter_channels, out_filter_channels, lastConv=i == num_conv_repeat)
            conv = tf.nn.max_pool(conv, ksize=self.pool_kernel_size, strides=self.pool_strides, padding='SAME', name='max_pool')
        return conv

    def inference(self, images, **kwargs):
        # vgg-19
        block1 = self._add_vgg_block(images, id_block=1, num_conv_repeat=2,
                                     in_filter_channels=3, out_filter_channels=64)
        block2 = self._add_vgg_block(block1, id_block=2, num_conv_repeat=2,
                                     in_filter_channels=64, out_filter_channels=128)
        block3 = self._add_vgg_block(block2, id_block=3, num_conv_repeat=4,
                                     in_filter_channels=128, out_filter_channels=256)
        block4 = self._add_vgg_block(block3, id_block=4, num_conv_repeat=4,
                                     in_filter_channels=256, out_filter_channels=512)
        block5 = self._add_vgg_block(block4, id_block=5, num_conv_repeat=4,
                                     in_filter_channels=512, out_filter_channels=512)
        flatten = self._reshape_to_flatten(block5, op_name='flatten_op')
        fc1 = self._add_full_connected_layer(flatten, 2048, 512, 1)
        drop1 = self._add_droupout(fc1, 1)
        fc2 = self._add_full_connected_layer(drop1, 4096, 2048, 2)
        drop2 = self._add_droupout(fc2, 2)
        linear_softmax = self._add_linear_softmax(drop2, 4096, 1)
        return linear_softmax

    def __str__(self):
        return 'VGGNetwork_based_on_tensorflow'

class PretrainedVGG19Network(TensorflowNetwork):

    """
    定义的使用 tensornets 的现有的网络结构, 使用 VGG19 需要使用的图片的大小为 224
    继承普通的 CNNNetwork 结构
    """
    def __init__(self, training, num_all_batchs, **kwargs):
        super(PretrainedVGG19Network, self).__init__(training=training, num_examples_per_epoch=num_all_batchs, **kwargs)

    def inference(self, images, **kwargs):
        import tensornets as nets
        logits = nets.VGG19(images, self.training, self.num_classes)
        return logits

    def loss(self, logits, target_labels):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=target_labels, logits=logits)  # 整个预选训练的 Loss 在 GraphKey.Loss 中
        return loss

    def accuracy(self, logits, target_labels):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(target_labels, 1))  # y 和 y^hat 的误差
        # 将 bool 向量转化为浮点数矩阵，求和就是准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求值为 1.0 的数量
        return accuracy, correct_prediction

    def train(self, total_loss, global_step):
        # 学习率衰减：影响学习率的变量.
        num_batches_per_epoch = self.num_examples_per_epoch / self.batch_size  # 每一训练完的一次 epoch 的迭代次数(BATCH_SIZE每一批次)
        decay_steps = int(num_batches_per_epoch * self.num_epoch_per_decay)  # 跑完多少次 epoch 就衰减
        # 学习率随着迭代次数指数衰减
        learning_rate = tf.train.exponential_decay(self.init_learning_rate,
                                                   global_step,  # 计算总的 step
                                                   decay_steps,  # 所有样本训练样本完的迭代次数
                                                   self.learning_rate_decay,  # 衰减率
                                                   staircase=True)  # 阶梯状衰减
        # 显示学习率
        tf.summary.scalar('learning_rate', learning_rate)
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, global_step=global_step), learning_rate

class KerasCNNNetwork(CNNNetwork):

    """
    使用 Keras 做为框架的 CNN 结构
    """

    def __init__(self, **kwargs):
        self._model = None
        super(KerasCNNNetwork, self).__init__(**kwargs)

    def print_summary(self):
        raise NotImplementedError('print_summary')

    def learn_rate_changer(self):
        raise NotImplementedError('learn_rate_changer')

    def loadWeights(self, file):
        """
        加载权重
        :param file: 权重文件 *.h5
        :return:
        """
        self._model.load_weights(file)

    def buildModel(self, loss, optimizer, metrics, **kwargs):
        self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)

    def fit_generator(self, **kwargs):
        self._model.fit_generator(**kwargs)

    def evaluate(self, X, y, verbose=1):
        """
        测试
        :param X: 图片输入
        :param y: 标签
        :param verbose: Keras 输出是否冗余
        :return: loss 和 accuracy
        """
        scores = self._model.evaluate(X, y, verbose=verbose)
        return scores[0], scores[1]

    def inference(self, inputs_shape, **kwargs):
        raise NotImplementedError('没有实现前向传播网络')

class KerasResNetwork(KerasCNNNetwork):

    """
    使用 Keras 的残差网络(ResNet)实现
    """

    def __init__(self, **kwargs):
        super(KerasResNetwork, self).__init__(conv_strides=1, **kwargs)
        self.depth = 20 # 使用 ResNet20
        self.resNetVersion = 2  # ResNet20 版本 2

    def _add_resnet_layer(self, inputs, filter_channels=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
        """
        添加一个残差网络层
        :param inputs: 输入的张量
        :param filter_channels: 输出的过滤器的通道数量
        :param kernel_size: 卷积核大小 kernel_size * kernel_size
        :param strides: 步长(在各个方向 NHWC的方向)
        :param activation: 激活函数
        :param batch_normalization: 是否使用 BN 层
        :param conv_first: 卷积层是否在第一层结构
        :return: 张量输出
        """
        # 按照需求定义, 如果不调用就不会有需求
        from keras.layers import Conv2D, BatchNormalization, Activation
        from keras.regularizers import l2

        conv = Conv2D(filters=filter_channels, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(self.regularizer_ratio))
        X = inputs
        if conv_first:
            X = conv(X)
            if batch_normalization:
                X = BatchNormalization()(X)
            if activation:
                X = Activation(activation)(X)
        else:
            if batch_normalization:
                X = BatchNormalization()(X)
            if activation:
                X = Activation(activation)(X)
            X = conv(X)
        return X


    def inference(self, inputs_shape, **kwargs):
        """
        https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
        定义传播过程
        :param inputs_shape: 样本数据除去 batch 的维度信息
        :param kwargs: 其他的参数
        :return: logits
        """
        import keras
        from keras.layers import Dense, BatchNormalization, Activation
        from keras.layers import AveragePooling2D, Input, Flatten
        from keras.models import Model
        # 定义模型
        in_filler_channel = 16  # 过滤器
        num_resnet_block = (self.depth - 2) // 9  # Resnet 块个数
        # 转换输入
        inputs = Input(inputs_shape)  # 输入的单张图片的维度信息
        X = self._add_resnet_layer(inputs=inputs, filter_channels=in_filler_channel, conv_first=True)
        for stage in range(3):
            for resNetBlock in range(num_resnet_block):
                activation_func = 'relu'  # 激活函数
                using_batch_normalization = True  # 是否使用批正规化
                strides = 1
                if stage == 0:
                    out_filler_channel = in_filler_channel * 4
                    if resNetBlock == 0:
                        # 第一个 Stage 的第一层
                        activation_func = None
                        using_batch_normalization = False
                else:
                    out_filler_channel = in_filler_channel * 2
                    if resNetBlock == 0:
                        strides = 2
                # ResNet 残差单元
                out = self._add_resnet_layer(X, in_filler_channel,
                                             kernel_size=1,
                                             strides=strides,
                                             activation=activation_func,
                                             batch_normalization=using_batch_normalization,
                                             conv_first=False)
                out = self._add_resnet_layer(out, in_filler_channel, conv_first=False)
                out = self._add_resnet_layer(out, out_filler_channel, kernel_size=1, conv_first=False)
                if resNetBlock == 0:
                    # 添加线性投影残差
                    X = self._add_resnet_layer(X, filter_channels=out_filler_channel,
                                               kernel_size=1,
                                               strides=strides,
                                               activation=None,
                                               batch_normalization=False)
                X = keras.layers.add([X, out])
            in_filler_channel = out_filler_channel
        # 添加分类器
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = AveragePooling2D(pool_size=8)(X)
        out = Flatten()(X)
        sf_out = Dense(self.num_classes, activation='softmax', kernel_initializer='he_normal')(out)
        # 实例化模型
        model = Model(inputs=inputs, outputs=sf_out)
        self._model = model
        return model

    def print_summary(self):
        """
        打印模型相关信息
        :return:
        """
        print('ResNet%dv%d:' % (self.depth, self.resNetVersion))
        self._model.summary()

    def learn_rate_changer(self):
        """
        定义的是学习率下降的趋势函数
        :return: 返回一元函数对象
        """
        def changer(epoch):
            lr = self.init_learning_rate
            if epoch > 180:
                lr *= 0.5e-3
            elif epoch > 160:
                lr *= 1e-3
            elif epoch > 120:
                lr *= 1e-2
            elif epoch > 80:
                lr *= 1e-1
            print('Learning rate: ', lr)
            return lr
        return changer


    def __str__(self):
        return 'KerasResNetwork%dv%d' % (self.depth, self.resNetVersion)


from multiprocessing import cpu_count
from pprint import pprint
import os

class Trainer(object):

    """
    自定义的训练器
    """
    def __init__(self, logout_prefix, model_saved_prefix, **kwargs):
        """
        定义一个训练器
        :param logout_prefix: 输出的日志文件目录
        :param model_saved_prefix: 输出的保存的模型目录
        :param kwargs: 其他参数
        """
        self.logout_prefix = logout_prefix
        self.model_saved_prefix = model_saved_prefix
        if not os.path.exists(logout_prefix):
            os.makedirs(logout_prefix)

        if not os.path.exists(model_saved_prefix):
            os.makedirs(model_saved_prefix)

    def train(self, trainDataGenerater, validDataGenerater, **kwargs):
        """
        训练过程
        :param trainDataGenerater: 训练数据产生器
        :param validDataGenerater: 验证数据产生器
        :param kwargs: 其他的参数
        :return:
        """
        raise NotImplementedError('没有实现 train!')

class TensorflowTrainer(Trainer):

    def __init__(self, cnnnet, max_epoch, ):
        """
        TF 训练器
        :param cnnnet: TF 神经网络
        :param max_epoch: 最大 epoch
        """
        if not isinstance(cnnnet, TensorflowNetwork):
            raise ValueError('请使用 TensorflowNetwork 子类作为 CNN 的参数')
        self.cnn = cnnnet
        self.max_epoch = max_epoch
        super(TensorflowTrainer, self).__init__(logout_prefix=os.sep.join(('logouts', str(cnnnet))), model_saved_prefix=os.sep.join(('models', str(cnnnet))))

    def _input_placeholder(self):
        """
        创建输入的张量
        :return:
        """
        self.train_images_input = tf.placeholder(tf.float32, [
            self.cnn.batch_size,  # validation test train 的样本数量 batch_size 并不一致
            self.cnn.image_width_height,  # 图像大小
            self.cnn.image_width_height,
            3  # RGB
        ], name='train_images_input')
        # 输入的是真实标签
        self.train_labels_input = tf.placeholder(tf.float32, [self.cnn.batch_size, self.cnn.num_classes], name='train_labels_input')

    def _build_compute_graph(self):
        """
        生成计算图
        :return:
        """
        self.global_steps = tf.Variable(0, trainable=False)
        # logit 输出
        logits = self.cnn.inference(self.train_images_input)
        # 获取网络中的损失
        self.train_loss = self.cnn.loss(logits, self.train_labels_input)
        # 计算准确率
        self.train_accuracy, _ = self.cnn.accuracy(logits, self.train_labels_input)
        train_op, lr = self.cnn.train(self.train_loss, self.global_steps)

        self.train_op = train_op
        self.learning_rate = lr
        self.logits = logits

    def _check_acc_on_valid(self, imgs, labels, sess):
        valid_acc = sess.run(self.train_accuracy, feed_dict={self.train_images_input: imgs,
                                                             self.train_labels_input: labels})
        return valid_acc

    def train(self, trainDataGenerater, validDataGenerater, **kwargs):
        """
        训练神经网络
        :param trainDataGenerater: 训练数据集的产生对象
        :param validDataGenerater: 验证数据集的对象
        :return:
        """
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.7 config=config
        # 打开一个新的会话
        with tf.Session() as sess:
            self._input_placeholder()
            self._build_compute_graph()
            tf.summary.scalar('train_loss', self.train_loss)
            tf.summary.scalar('train_accuracy', self.train_accuracy)
            tf.summary.scalar('learning_rate', self.learning_rate)
            # 聚合所有的 summary 操作 用于同步更新
            merged = tf.summary.merge_all()
            # 初始化变量
            global_variables = tf.global_variables()
            # 保存到模型文件的操作
            saver = tf.train.Saver(global_variables)
            # 恢复所有的变量
            ckpt = tf.train.get_checkpoint_state(self.model_saved_prefix)
            # 模型描述文件
            model_saved_file = os.sep.join((self.model_saved_prefix, 'saved_model.ckpt'))
            if ckpt and ckpt.model_checkpoint_path:
                startstep = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                print('检查点的训练次数:', startstep)
                pprint('从 “%s” 加载模型' % model_saved_file)
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 修改步数
                tf.assign(self.global_steps, startstep)
            else:
                pprint('没有找到检查点!')
                init = tf.global_variables_initializer()
                sess.run(init)
                #
                if isinstance(self.cnn, PretrainedVGG19Network):
                    sess.run(self.logits.pretrained())
                startstep = 0
            # 初始化所有的变量
            # 定义 summary 输出器
            train_writer = tf.summary.FileWriter(os.sep.join((self.logout_prefix, 'train')), sess.graph)
            # 计算一次 epoch 走过的 step 次数
            batches_per_epoch = self.cnn.num_examples_per_epoch // self.cnn.batch_size  # 每进行一次 epoch 训练， 内循环迭代的次数. 也就是batch 的数量
            # 把验证集上最后的平均结果进行计算
            max_acc = -1.0
            for epoch in range(startstep // batches_per_epoch, self.max_epoch):
                # 跑一次 epoch
                for images_batch, labels_batch in trainDataGenerater.generate_augment_batch():
                    # 训练数据
                    _, loss_, lr, merged_value, global_step_, train_acc = sess.run(
                        [self.train_op, self.train_loss, self.learning_rate, merged, self.global_steps, self.train_accuracy],
                        feed_dict={self.train_images_input: images_batch, self.train_labels_input: labels_batch})
                    # 每100 个 step 输出一次
                    if (global_step_ + 1) % 100 ==0:
                        print('Epoch %d, Global step: %10d, Train accuracy: %.3f, Loss: %.3f, Learning rate: %.7f' % (epoch, global_step_, train_acc, loss_, lr))
                    train_writer.add_summary(merged_value, global_step=global_step_)  # 每一个 step 记录一次
                # 验证集的平均正确率
                valid_accs = []
                for valid_images_batch, valid_labels_batch in validDataGenerater.generate_augment_batch():
                    va = self._check_acc_on_valid(valid_images_batch, valid_labels_batch, sess)
                    valid_accs.append(va)
                # 平均的验证集准确率
                mean_valid_acc = np.mean(np.array(valid_accs))
                if mean_valid_acc > max_acc:
                    print('Epoch {2}, Validation average accuracy changed from {0:.3f} to {1:.3f}, save model.'.format(max_acc, mean_valid_acc, epoch))
                    max_acc = mean_valid_acc
                    # 保存模型
                    saver.save(sess, model_saved_file, global_step=self.global_steps)
                else:
                    print('Epoch {0}, Validation average accuracy {1:.3f} not improve from {2:.3f}, save model.'.format(
                        epoch, mean_valid_acc, max_acc))
                # 继续下一次训练

            train_writer.close()

class KerasTrainer(Trainer):

    """
    正对于 Keras 模型的训练器
    """

    def __init__(self, model, max_epochs):
        if not isinstance(model, KerasCNNNetwork):
            raise ValueError('请使用 KerasCNNNetwork 子类作为 CNN 的参数')
        self.model = model  # Keras 模型
        self.max_epochs = max_epochs
        super(KerasTrainer, self).__init__(logout_prefix=os.sep.join(('logouts', str(model))),
                                                model_saved_prefix=os.sep.join(('models', str(model))))

    def train(self, trainDataGenerater=None, validDataGenerater=None, **kwargs):
        import keras
        from keras.optimizers import Adam
        from keras.callbacks import ModelCheckpoint, LearningRateScheduler
        from keras.callbacks import ReduceLROnPlateau
        from keras.datasets import cifar10
        from keras.preprocessing.image import ImageDataGenerator
        from keras.callbacks import TensorBoard
        # 处理 Tensorboard 的输出
        tensorBoardParams = {'log_dir': self.logout_prefix, 'write_graph': True, 'write_images': True}
        if not self.model.tiny_output_logs:
            tensorBoardParams.update({'write_grads': True})  # histgram 可能可以设置 具体参看 Keras 的回调 https://keras-cn.readthedocs.io/en/latest/other/callbacks/
        tensorBoardCallBack = TensorBoard(**tensorBoardParams)
        # 准备保存的检查点(权重文件)
        checkpoints = ModelCheckpoint(filepath=os.sep.join([self.model_saved_prefix, 'checkpoints.h5']),
                                      monitor='val_acc',
                                      verbose=1,
                                      save_best_only=True)  # 保存最好的验证集误差的权重
        lr_changer = self.model.learn_rate_changer()
        lr_scheduler = LearningRateScheduler(lr_changer)  # 学习率衰减的调用器
        lr_reducer = ReduceLROnPlateau(factor=self.model.learning_rate_decay, cooldown=0, patience=5, min_lr=.5e-6)
        callbacks = [checkpoints, lr_reducer, lr_scheduler, tensorBoardCallBack]  # 回调顺序
        # 定义数据
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        # 均1化数据 float -> 0 ~ 255
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        # 减去像素的均值
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean

        print('x_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)
        # 转换标签
        y_test = keras.utils.to_categorical(y_test, self.model.num_classes)
        y_train = keras.utils.to_categorical(y_train, self.model.num_classes)
        # 数据增强
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
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
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
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
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        # 计算相关的增强属性
        datagen.fit(X_train)
        # 模型, 初始化模型
        self.model.inference(inputs_shape=X_train.shape[1:])
        self.model.buildModel(loss='categorical_crossentropy',
                         optimizer=Adam(lr=lr_changer(0)),
                         metrics=['accuracy'])
        self.model.print_summary()
        self.model.fit_generator(generator=datagen.flow(X_train, y_train, batch_size=self.model.batch_size),
                                 validation_data=(X_test, y_test),
                                 epochs=self.max_epochs,
                                 verbose=2,  # 每一个 epoch 显示一次
                                 workers=cpu_count(),
                                 callbacks=callbacks,
                                 steps_per_epoch=X_train.shape[0] // self.model.batch_size)  # https://stackoverflow.com/questions/43457862/whats-the-difference-between-samples-per-epoch-and-steps-per-epoch-in-fit-g





