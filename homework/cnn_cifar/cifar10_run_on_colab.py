!rm -rf model
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
            X = reshape_X(X)  # 转换为 tensorflow 格式的数据（我不需要别的处理）
            # 处理图像
            # pad_width = ((0, 0), (2, 2), (2, 2), (0, 0))
            # X = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)
            # X = random_crop_and_flip(X, padding_size=2)
            # X = whitening_image(X)
            self.X = X
            self.Y = Y

def reshape_X(images):
    """
    由于 cifar-10 的数据是 [batch 3 32 32] 的格式， 与 tf 默认 cov2d 定义的 [batch 32 32 3] 不同
    :param images: 图像
    :return:
    """
    nx = np.transpose(images, (0, 2, 3, 1))  # BATCH WIDTH HEIGHT CHANELS
    return nx

class AugmentImageGenerator(object):

    """
    随机选择一组数据，可以认为是增广的
    """
    def __init__(self, testOnly, next_batch_size, path_prefix='./data'):
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

import tensorflow as tf
from math import sqrt

from pprint import pprint
import os
from math import sqrt


class CNNNetwork(object):

    # 静态常量
    # NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000  # 训练样本数量
    # NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 10000  # 训练样本数量
    # MOVING_AVERAGE_DECAY = 0.9999  # 滑动平均衰减率
    # NUM_EPOCHS_PER_DECAY = 5  # 每送入30批次就更新一下学习率
    # LEARNING_RATE_DECAY_FACTOR = 0.1  # 学习率的阶梯衰减率
    # INITIAL_LEARNING_RATE = 0.1  # 初始的学习率
    # BATCH_SIZE = 128  # 送入神经网络的样本数量
    # ONLY_OUTPUT_LOSS_ACCURACY = True

    def __init__(self, training, num_examples_per_epoch,
                 batch_size=128,
                 image_width_height=32,
                 conv_strides=[1,1,1,1],
                 pool_kernel_size=[1,3,3,1],
                 pool_strides=[1,2,2,1],
                 num_classes=10,
                 keep_prob=0.5,
                 regularizer_ratio=0.002,
                 init_learning_rate=0.001,
                 moving_average_decay=0.9,
                 num_epoch_per_decay=5,
                 learning_rate_decay=0.1,
                 batch_normalization_epsilon=0.001,
                 tiny_output_logs=True
                 ):
        """
        :param training: 是否是训练状态
        :param num_examples_per_epoch: epoch 包含的样本数量（cifar10一般的训练集就是50000）
        :param batch_size: 每一次送如神经网络的样本数量，满足 batch_size * epoch = num_examples_per_epoch
        :param image_width_height: 图像的长度和高度，默认是 32， VGG 等其他的长度不一样
        :param conv_strides: 卷积层步长，输入张量的每一个维度上都只是移动1
        :param pool_kernel_size: 池化的核，不需要再 BATCH_SIZE 和 CHANNELS 上采样。所以移动 1
        :param pool_strides: 池化的步长
        :param keep_prob: dropout 的比例
        :param regularizer_ratio: L2 正则化系数
        :param init_learning_rate: 初始的学习率(学习率衰减)
        :param moving_average_decay: 滑动平均的衰减率
        :param num_epoch_per_decay: 当训练完一批数据的
        :param batch_normalization_epsilon: 对卷积层输出做 batch normalization 的参数 epsilon
        :param learning_rate_decay: 学习率衰减率， num_epoch_per_decay 可以决定衰减的 epoch 间隔
        :param tiny_output_logs: 仅仅输出 loss acc 和 learning rate 的数据（不包含各层权重等）
        """
        self.training = training
        self.conv_strides = conv_strides
        self.pool_kernel_size = pool_kernel_size
        self.pool_strides = pool_strides
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.regularizer_ratio = regularizer_ratio
        self.batch_normalization_epsilon = batch_normalization_epsilon
        self.num_examples_per_epoch = num_examples_per_epoch
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.moving_average_decay = moving_average_decay
        self.num_epoch_per_decay = num_epoch_per_decay
        self.learning_rate_decay = learning_rate_decay
        self.tiny_output_logs = tiny_output_logs
        self.image_width_height = image_width_height
        # 记录(非训练过程)卷积核的输出信息，注意需要保存输出图像的大小，后面几层仅过了激活要 / 2
        # 由 tf collection 保存

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
        参考：https://gist.github.com/kukuruza/03731dc494603ceab0c5
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
        # 最后一层是 softmax 的输出
        with tf.variable_scope('softmax_linear%d' % layerId) as scope:
            weights = self._create_or_get_variable(name='weights', shape=[input_node_num, self.num_classes], initializer=tf.uniform_unit_scaling_initializer())
            biases = self._create_or_get_variable(name='biases', shape=[self.num_classes], initializer=tf.zeros_initializer())
            results = tf.matmul(input, weights) + biases
            softmax_linear = self._add_batch_normalization_for_fc(results, self.num_classes, name=scope.name)
            self._add_activated_summary(softmax_linear)
        return softmax_linear

    def _add_droupout(self, input, op_id):
        if self.training:
            drop = tf.nn.dropout(input, self.keep_prob, name='drop%d' % op_id)
            return drop
        return input

    def _add_max_pool(self, input, ksize, strides, padding, op_id):
        mpool = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding, name='max_pool%d' % op_id)
        return mpool

    def loss(self, logits, labels):
        """
        将 logits 的预测损失加入到 tf 的集合中
        :param logits: logit 的输出
        :param labels: 标签，维度 [batch_size].
        :return: 返回交叉熵的 loss 张量集合操作
        """
        labels = tf.cast(labels, tf.int64)  # 转换成类别的下标
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                name='cross_entropy_per_example')  # 每一个样本的交叉熵
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
        # EMA
        loss_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, name='ema_average')
        # 更新变量。EMA中有影子变量控制模型更新的速度
        loss_averages_op = loss_averages.apply([total_loss])
        # 优化损失, 先进行 EMA 的计算
        with tf.control_dependencies([loss_averages_op]):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(total_loss, global_step)
        return train_op, learning_rate

    def accuracy(self, logits, target_labels):
        """
        计算准确率
        :param logits: 输出各个样本的频率
        :param target_labels: 标签 [BATCH SIZE]
        :return: 返回正确率的操作，比较的结果，是布尔张量
        """
        correct_prediction = tf.equal(tf.argmax(logits, 1), target_labels)  # y 和 y^hat 的误差
        # 将 bool 向量转化为浮点数矩阵，求和就是准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求值为 1.0 的数量
        return accuracy, correct_prediction

    def getConvList(self):
        """
        获取所有的卷积核
        :return: 卷积核列表（按照添加的顺序） 元素为张量
        """
        return tf.get_collection('convs')

    def inference(self, images):
        raise NotImplementedError('没有实现前向传播网络')

class VGGNetwork(CNNNetwork):

    def __init__(self, training, num_examples_per_epoch, **kwargs):
        """
        VGG19 cifar-10 的实现
        :param num_examples_per_epoch: 样本总量
        """
        super(VGGNetwork, self).__init__(training, num_examples_per_epoch=num_examples_per_epoch,pool_strides=[1,2,2,1], pool_kernel_size=[1,2,2,1], **kwargs)

    def _add_vgg_conv_bn_act_layer(self, input, id_conv, in_filter_channels, out_filter_channels):
        convname = 'conv%d' % id_conv
        with tf.variable_scope(convname) as scope:
            weights = self._create_or_get_variable(name='weights', shape=[3, 3, in_filter_channels, out_filter_channels], initializer=tf.uniform_unit_scaling_initializer())  # 共享权重
            conv = tf.nn.conv2d(input=input, filter=weights, strides=self.conv_strides, padding='SAME', name=convname)
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
                conv = self._add_vgg_conv_bn_act_layer(conv, i, out_filter_channels, out_filter_channels)
            # 添加这个 block 的输出
            tf.add_to_collection('convs', conv)

            conv = tf.nn.max_pool(conv, ksize=self.pool_kernel_size, strides=self.pool_strides, padding='SAME', name='max_pool')
        return conv

    def inference(self, images):
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

TENSORBOARD_OUTPUT_PATH = 'log'
MAX_EPOCHS = 150  # 跑完一次样本的时间
DISPLAY_PRE_GLOBAL_STEPS = 100  # 每一次 epoch 显示的频率
MODEL_PATH = 'model'  # 保存的模型文件夹
MODEL_SAVE_FILE = os.sep.join((MODEL_PATH, 'saved_model.ckpt'))  # 保存的元文件路径

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

if not os.path.exists(TENSORBOARD_OUTPUT_PATH):
    os.mkdir(TENSORBOARD_OUTPUT_PATH)


class Train(object):

    def __init__(self, cnnnet):
        if not isinstance(cnnnet, CNNNetwork):
            raise ValueError('参数错误!')
        self.cnn = cnnnet

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
        self.train_labels_input = tf.placeholder(tf.int64, [self.cnn.batch_size], name='train_labels_input')


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

    def train(self, trainDataGenerater, validDataGenerater):
        """
        训练神经网络
        :param trainDataGenerater: 训练数据集的产生对象
        :param validDataGenerater: 验证数据集的对象
        :return:
        """
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
            ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                startstep = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                print('检查点的训练次数:', startstep)
                pprint('从 “%s” 加载模型' % MODEL_SAVE_FILE)
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 修改步数
                tf.assign(self.global_steps, startstep)
            else:
                pprint('没有找到检查点!')
                init = tf.global_variables_initializer()
                sess.run(init)
                #
                startstep = 0
            # 初始化所有的变量
            # 定义 summary 输出器
            train_writer = tf.summary.FileWriter(os.sep.join((TENSORBOARD_OUTPUT_PATH, 'train')), sess.graph)
            # 计算一次 epoch 走过的 step 次数
            batches_per_epoch = self.cnn.num_examples_per_epoch // self.cnn.batch_size  # 每进行一次 epoch 训练， 内循环迭代的次数. 也就是batch 的数量
            # 验证数据
            images_valid, labels_valid = validDataGenerater.generate_augment_batch()
            for epoch in range(startstep // batches_per_epoch, MAX_EPOCHS):
                # 训练一次
                for batch_counter in range(batches_per_epoch):
                    # 获取训练的数据
                    images_batch, labels_batch = trainDataGenerater.generate_augment_batch()
                    # 训练数据
                    _, loss_, lr, merged_value, global_step_ = sess.run([self.train_op, self.train_loss, self.learning_rate, merged, self.global_steps],
                        feed_dict={self.train_images_input: images_batch, self.train_labels_input: labels_batch})
                    valid_acc = sess.run(self.train_accuracy, feed_dict={self.train_images_input: images_valid, self.train_labels_input: labels_valid})
                    # 这个主要区别的是准确率输出的时机
                    if (global_step_ + 1) % DISPLAY_PRE_GLOBAL_STEPS == 0:
                        ## 检查准确率
                        train_acc = sess.run(self.train_accuracy,
                                             feed_dict={self.train_images_input: images_batch, self.train_labels_input: labels_batch})
                        pprint(
                            'Epoch:{e:3d}, Global Step:{step:5d}, Train Acc:{tacc:.3f}, Validation Acc:{vacc: .3f}, Learning rate:{lr:.5f}, Loss:{loss:.3f}'.format(
                                step=global_step_, tacc=train_acc, lr=lr, loss=loss_, e=epoch, vacc=valid_acc))
                        # 保存模型
                        saver.save(sess, MODEL_SAVE_FILE, global_step=self.global_steps)

                    train_writer.add_summary(merged_value, global_step=global_step_)
            train_writer.close()


cnnnet = VGGNetwork(True, 50000, init_learning_rate=0.001, learning_rate_decay=0.7, num_epoch_per_decay=40)

data = AugmentImageGenerator(False, cnnnet.batch_size, path_prefix='/tmp/cifar-10-batches-py')
testData = AugmentImageGenerator(True, cnnnet.batch_size, path_prefix='/tmp/cifar-10-batches-py')  # 用作 验证集合
trainObj = Train(cnnnet)
trainObj.train(data, testData)
