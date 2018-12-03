import tensorflow as tf
from math import sqrt


class TensorflowNetwork(object):

    def __init__(self, training,
                 batch_size,
                 image_width_height,
                 conv_filter_strides,
                 conv_filter_size,
                 pool_kernel_size,
                 pool_strides,
                 num_classes,
                 keep_prob,
                 regularizer_ratio,
                 batch_normalization_epsilon,
                 num_examples_per_epoch,
                 tiny_output_logs=True,
                 displayKernelOnTensorboard=False,
                 **kwargs):
        """
        定义基于 Tensorflow 的神经网络结构
        :param training: 是否是训练状态, 目前是影响 dropout
        :param batch_size: 每一次送如神经网络的样本数量，满足 batch_size * epoch = num_examples_per_epoch
        :param image_width_height: 图像的长度和高度，默认是 32， VGG 等其他的长度不一样
        :param conv_filter_strides: 卷积层步长，一般是输入张量的每一个维度上都只是移动1
        :param conv_filter_size: 卷积核的大小, 一般是 3x3
        :param pool_kernel_size: 池化的核，不需要再 BATCH_SIZE 和 CHANNELS 上采样。所以移动 1
        :param pool_strides: 池化的步长
        :param keep_prob: dropout 的比例
        :param regularizer_ratio: L2 正则化系数
        :param init_learning_rate: 初始的学习率(学习率衰减)
        :param num_epoch_per_decay: 当训练完一批数据的
        :param batch_normalization_epsilon: 对卷积层输出做 batch normalization 的参数 epsilon
        :param learning_rate_decay: 学习率衰减率， num_epoch_per_decay 可以决定衰减的 epoch 间隔
        :param num_examples_per_epoch: epoch 包含的样本数量（cifar10一般的训练集就是50000）
        :param tiny_output_logs: 是否仅仅显示学习率 loss 和准确率
        :param displayKernelOnTensorboard: 是否显示卷积核中的过滤器的输出
        :param kwargs: 参数
        """
        self.training = training
        self.num_examples_per_epoch = num_examples_per_epoch
        self.displayKernelOnTensorboard = displayKernelOnTensorboard
        self.tiny_output_logs = tiny_output_logs
        self.conv_filter_strides = conv_filter_strides
        self.conv_filter_size = conv_filter_size
        self.pool_kernel_size = pool_kernel_size
        self.pool_strides = pool_strides
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.regularizer_ratio = regularizer_ratio
        self.batch_normalization_epsilon = batch_normalization_epsilon

        self.batch_size = batch_size
        self.image_width_height = image_width_height

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
        参考：https://gist.github.com/kukuruza/03731dc494603ceab0c5
        将 kernel 输出为一张图片中，主要进行 filter 各个channel的排列
        :param kernel: 卷积核 4-D 张量 [Y, X, NumChannels, NumKernels]
        :param grid_Y: 输出小卷积核的长 满足 NumKernels == grid_Y * grid_X
        :param grid_X: 输出小卷积的宽 满足 NumKernels == grid_Y * grid_X
        :param pad: 卷积核通道之间的间隔像素
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
        # x7 = tf.transpose(x6, (3, 0, 1, 2))  # CHWN
        x7 = tf.transpose(x6, (2, 0, 1, 3))  # CHWN
        #
        # # scale to [0, 255] and convert to uint8
        return tf.image.convert_image_dtype(x7, dtype=tf.uint8)

    def _add_conv_output_image(self, scope, kernel, max_output_image):
        """
        https://gist.github.com/panmari/4622b78ce21e44e2d69c
        添加 卷积核的输出
        :param scope: 变量层的名称
        :param kernel: 卷积核(alias 过滤器 权重)
        :param max_output_image: 输出的最终的图片数量(不超过 batch_size)
        :return: 不输出
        """

        # 拆分 grid 的行列 是的行列相乘等于 kernel 的输出 channel
        # 求解最大的因子
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1:
                        print('It is not good at using the prime number as the filter output channel!')
                    return (i, int(n / i))

        out_channel = kernel.get_shape().as_list()[3]
        grid_y, grid_x = factorization(out_channel)
        grid = self.put_kernels_on_grid(kernel=kernel, grid_X=grid_x, grid_Y=grid_y)
        tf.summary.image(scope.name + '/filter', grid, max_outputs=max_output_image)

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

    def train(self, total_loss, global_step, init_learning_rate, num_epoch_per_decay, learning_rate_decay):
        """
        训练
        :param total_loss: 所有的损失的操添加操作，也就是 self.loss 返回的结果
        :param global_step: 当前的迭代步数
        :return: 返回训练操作, 学习率
        """
        # 学习率衰减：影响学习率的变量.
        num_batches_per_epoch = self.num_examples_per_epoch / self.batch_size  # 每一训练完的一次 epoch 的迭代次数(BATCH_SIZE每一批次)
        decay_steps = int(num_batches_per_epoch * num_epoch_per_decay)  # 跑完多少次 epoch 就衰减
        # 学习率随着迭代次数指数衰减
        learning_rate = tf.train.exponential_decay(init_learning_rate,
                                                   global_step,  # 计算总的 step
                                                   decay_steps,  # 所有样本训练样本完的迭代次数
                                                   learning_rate_decay,  # 衰减率
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
        raise NotImplementedError("getConvList")


class NormalCNNNetwork(TensorflowNetwork):

    def __init__(self, training, num_examples_per_epoch, **kwargs):
        """
        构建普通的神经网络（Cifar-10）
        :param training: 是否是训练状态
        :param num_examples_per_epoch: 总体的样本数量
        """
        super(NormalCNNNetwork, self).__init__(training=training,
                                               num_examples_per_epoch=num_examples_per_epoch,
                                               **kwargs)

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
            weights = self._create_or_get_variable(name='weights', shape=[self.conv_filter_size, self.conv_filter_size, filter_in_channels, filter_out_channels], initializer=tf.uniform_unit_scaling_initializer())  # 共享权重
            conv = tf.nn.conv2d(input=input, filter=weights, strides=self.conv_filter_strides, padding='SAME', name='conv')
            biases = self._create_or_get_variable(name='biases', shape=[filter_out_channels], initializer=tf.zeros_initializer())  # 偏移
            to_activate = tf.nn.bias_add(conv, biases)  # biases 只能是1-D维度的
            # 进行 batch_norm
            conv = activation(to_activate, name=scope.name)  # 激活
            self._add_activated_summary(conv)
        return conv

    def _add_signle_conv_bn_act(self, input, filter_in_channels, filter_out_channels, layerId, activation=tf.nn.relu):
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
            weights = self._create_or_get_variable(name='weights', shape=[self.conv_filter_size, self.conv_filter_size, filter_in_channels, filter_out_channels], initializer=tf.uniform_unit_scaling_initializer())  # 共享权重
            conv = tf.nn.conv2d(input=input, filter=weights, strides=self.conv_filter_strides, padding='SAME', name='conv')

            # self._add_conv_output_image(scope, weights, conv_output=conv, conv_output_size=conv_output_size,
            #                             filter_out_channels=filter_out_channels)
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
        layer1 = self._add_signle_conv_bn_act(input=images, layerId=1, filter_in_channels=3, filter_out_channels=32)
        # Layer2
        layer2 = self._add_max_pool(
            input=self._add_signle_conv_bn_act(input=layer1, layerId=2, filter_in_channels=32, filter_out_channels=32),
            ksize=POOL_KERNEL_SIZE, strides=POOL_STRIDES, padding='SAME', op_id=2)
        # Layer3
        layer3 = self._add_signle_conv_bn_act(input=layer2, layerId=3, filter_in_channels=32, filter_out_channels=64)
        # Layer4
        layer4 = self._add_max_pool(
            input=self._add_signle_conv_bn_act(input=layer3, layerId=4, filter_in_channels=64, filter_out_channels=64),
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

    def __init__(self, training, num_examples_per_epoch, conv_filter_size=3, **kwargs):
        """
        VGG19 cifar-10 的实现
        :param num_examples_per_epoch: 样本总量
        :param conv_filter_size: 卷积的过滤器大小, VGG19 cifar10 默认为3
        """
        super(VGGNetwork, self).__init__(training=training,
                                         num_examples_per_epoch=num_examples_per_epoch,
                                         pool_strides=[1,2,2,1],
                                         pool_kernel_size=[1,2,2,1],
                                         batch_size=128,
                                         image_width_height=32,
                                         conv_filter_strides=[1,1,1,1],
                                         conv_filter_size=conv_filter_size,
                                         num_classes=10,
                                         keep_prob=0.5,
                                         regularizer_ratio=0.002,
                                         batch_normalization_epsilon=0.001,
                                         **kwargs)

    def _add_vgg_conv_bn_act_layer(self, input, id_conv, in_filter_channels, out_filter_channels, lastConv=False):
        convname = 'conv%d' % id_conv
        with tf.variable_scope(convname) as scope:
            weights = self._create_or_get_variable(name='weights', shape=[self.conv_filter_size, self.conv_filter_size, in_filter_channels, out_filter_channels], initializer=tf.uniform_unit_scaling_initializer())  # 共享权重

            conv = tf.nn.conv2d(input=input, filter=weights, strides=self.conv_filter_strides, padding='SAME', name=convname)
            if lastConv:
                # 添加这个 block 的卷积核输出
                tf.add_to_collection('convs', conv)  # 添加卷积层
                if self.displayKernelOnTensorboard:
                    self._add_conv_output_image(scope=scope, kernel=weights, max_output_image=1)  # 默认显示一张图片
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

    def getConvList(self):
        """
        获取所有的卷积层
        :return: 卷积层列表（按照添加的顺序） 元素为张量
        """
        return tf.get_collection('convs')


class PretrainedVGG19Network(TensorflowNetwork):

    """
    定义的使用 tensornets 的现有的网络结构, 使用 VGG19 需要使用的图片的大小为 224. 目前存在问题
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

    def train(self, total_loss, global_step, init_learning_rate, num_epoch_per_decay, learning_rate_decay, **kwargs):
        # 学习率衰减：影响学习率的变量.
        num_batches_per_epoch = self.num_examples_per_epoch / self.batch_size  # 每一训练完的一次 epoch 的迭代次数(BATCH_SIZE每一批次)
        decay_steps = int(num_batches_per_epoch * num_epoch_per_decay)  # 跑完多少次 epoch 就衰减
        # 学习率随着迭代次数指数衰减
        learning_rate = tf.train.exponential_decay(init_learning_rate,
                                                   global_step,  # 计算总的 step
                                                   decay_steps,  # 所有样本训练样本完的迭代次数
                                                   learning_rate_decay,  # 衰减率
                                                   staircase=True)  # 阶梯状衰减
        # 显示学习率
        tf.summary.scalar('learning_rate', learning_rate)
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, global_step=global_step), learning_rate


class KerasCNNNetwork(object):

    """
    使用 Keras 做为框架的 CNN 结构
    """

    def __init__(self,
                 batch_size,
                 num_classes,
                 regularizer_ratio,
                 **kwargs):
        """
        使用 Keras 框架的 CNN
        :param init_leanring_rate: 初始学习率
        :param kwargs:
        """
        self._model = None
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.regularizer_ratio = regularizer_ratio

    def print_summary(self):
        raise NotImplementedError('print_summary')

    def learn_rate_changer(self, init_learning_rate):
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

    def __init__(self,
                 batch_size=32,
                 num_classes=10,
                 regularizer_ratio=1e-4,
                 **kwargs):
        super(KerasResNetwork, self).__init__(batch_size=batch_size, num_classes=num_classes, regularizer_ratio=regularizer_ratio, **kwargs)
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
                    # 添加线性投影残差, 减少训练的参数数量
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

    def learn_rate_changer(self, init_learning_rate):
        """
        定义的是学习率下降的趋势函数
        :return: 返回一元函数对象
        """
        def changer(epoch):
            lr = init_learning_rate
            if epoch > 300:
                lr *= 0.5e-3
            elif epoch > 220:
                lr *= 1e-3
            elif epoch > 160:
                lr *= 1e-2
            elif epoch > 100:
                lr *= 1e-1
            print('Learning rate: ', lr)
            return lr
        return changer


    def __str__(self):
        return 'KerasResNetwork%dv%d' % (self.depth, self.resNetVersion)
