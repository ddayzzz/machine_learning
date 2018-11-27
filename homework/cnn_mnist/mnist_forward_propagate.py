# coding=utf-8
import tensorflow as tf

# MNIST 数据集的常数
INPUT_NODE = 784  # 输入层的节点数。对于MNIST数据集，数据等于图片的像素
OUTPUT_NODE = 10  # 输出层节点数。MNIST数据结需要区分0~9着10个数字。随意输出层节点数为10


# 神经网络参数
IMAGE_SIZE = 28
NUM_CHANELS = 1  # 图像的分量，目前只有一个通道，而非[R G B]
NUM_LABELS = 10  # 0-9 的 10 个数据
# CONV-1 的参数
CONV1_DEEP = 32  # 卷积层深度 32
CONV1_SIZE = 5  # 卷积层尺寸 5
# CONV-2 的参数
CONV2_DEEP = 64  # 卷积层深度 64
CONV2_SIZE = 5  # 卷积层尺寸 5
# FC 全连接层参数
FC_SIZE = 512  # 全连接层的节点个数


def forward_propagate(input_tensor, regularizer):
    """
    前向传播过程
    :param input_tensor: 输入的张量，满足 [BATCH HEIGHT WIDTH CHANNELS]
    :param regularizer: 正则化器
    :return: 返回的是 FC 层的输出
    """
    # 第一层输入，1@28x28->32@28x28
    with tf.variable_scope('Layer1-conv1'):
        conv1_weights = tf.get_variable(name='weights',
                                        shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(name='biases', shape=[CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input=input_tensor,
                             filter=conv1_weights,  # 过滤器
                             strides=[1, 1, 1, 1],
                             # 过滤器移动步长为1 在张量[batch, height, width, channels] 的分量移动的步长. batch=channels=1 不跳过任何样本/分量
                             padding='SAME')  # 不足的元素使用 0 填充
        to_activate1 = tf.nn.bias_add(conv1, conv1_biases)
        relu1 = tf.nn.relu(to_activate1)  # RELU 作为激活函数
        tf.summary.histogram('weights', conv1_weights)
        tf.summary.histogram('biases', conv1_biases)
        tf.summary.histogram('to_activate', to_activate1)
    # 第二层，池化层. 32@28x28 -> 32@14x14
    # apis ： https://blog.csdn.net/mao_xiao_feng/article/details/53453926
    with tf.variable_scope('Layer2-pooling'):
        pool1 = tf.nn.max_pool(value=relu1,  # conv1 的输出
                               ksize=[1, 2, 2, 1],  # 大小，选取最大的 [1 height width 1]。不在 batch 和 channels 做池化
                               strides=[1, 2, 2, 1],  # 步长2
                               padding='SAME')

    # 第三层，卷积层。32@14x14 -> 64@14x14
    with tf.variable_scope('Layer3-conv2'):
        conv2_weights = tf.get_variable(name='weights',
                                        shape=[CONV1_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(name='biases', shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(input=pool1,
                             filter=conv2_weights,  # 过滤器
                             strides=[1, 1, 1, 1],  # 过滤器移动步长为1
                             padding='SAME')  # 不足的元素使用 0 填充
        to_activate2 = tf.nn.bias_add(conv2, conv2_biases)
        relu2 = tf.nn.relu(to_activate2)  # RELU 作为激活函数
        tf.summary.histogram('weights', conv2_weights)
        tf.summary.histogram('biases', conv2_biases)
        tf.summary.histogram('to_activate', to_activate2)
    # 第四层，池化层。64@14x14 -> 64@7x7
    with tf.variable_scope('Layer4-pooling'):
        pool2 = tf.nn.max_pool(value=relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')  # 每两个边长就选择最大的值
    # 第五层，全连接层
    pool_shape = pool2.get_shape().as_list()  # 将第四层的输出的维度保存为列表。维度意义与输入张量相同。第一维是batch
    input_fc_nodes_size = pool_shape[1] * pool_shape[2] * pool_shape[3]  # pool2：weight * height * channels
    # input_fc_nodes = tf.reshape(pool2, shape=[pool_shape[0], input_fc_nodes_size])  # 全连接层输出的张量，转换为向量。一个行向量是一个样本。输入 FC 做分类
    input_fc_nodes = tf.reshape(pool2, shape=[-1, input_fc_nodes_size])
    with tf.variable_scope('Layer5-fc1'):
        fc1_weights = tf.get_variable('weights',
                                      shape=[input_fc_nodes_size, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 正则化权重
        if regularizer:
            tf.add_to_collection('losses', value=regularizer(fc1_weights))  # 将 losses->正则化后的权重 加入
        fc1_biases = tf.get_variable('biases', shape=[FC_SIZE],
                                     initializer=tf.constant_initializer(0.1))
        to_activate3 = tf.matmul(input_fc_nodes, fc1_weights) + fc1_biases
        fc1 = tf.nn.relu(to_activate3)  # 激活一下
        tf.summary.histogram('weights', fc1_weights)
        tf.summary.histogram('biases', fc1_biases)
        tf.summary.histogram('to_activate', to_activate3)
    # 第六层，全连接层。R^FC_SIZE->R^NUM_LABELS. 输出作为 softmax 层的输入
    with tf.variable_scope('Layer6-fc2'):
        fc2_weights = tf.get_variable('weights',
                                      shape=[FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 正则化权重
        if regularizer:
            tf.add_to_collection('losses', value=regularizer(fc2_weights))  # 将 losses->正则化后的权重 加入
        fc2_biases = tf.get_variable('biases', shape=[NUM_LABELS],
                                     initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        tf.summary.histogram('weights', fc2_weights)
        tf.summary.histogram('biases', fc2_biases)
        tf.summary.histogram('logit', logit)
    return logit