# coding=utf-8
# MNIST 的训练过程
# 操作 Tensorboard。ref： https://www.cnblogs.com/David-Wei/p/6984898.html?utm_source=itdadao&utm_medium=referral
# 注意：使用 GPU 计算 summary 没有意义，所以删除上下文 tf.device https://stackoverflow.com/questions/45876021/tensorflow-summary-ops-can-assign-to-gpu
import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from pprint import pprint


from homework.cnn_mnist import mnist_forward_propagate  # 导入前向传递过程

# 配置参数
BATCH_SIZE = 100
LEANRING_RATE = 1e-4
REGULARIZATION_RATE = 0.001  # 描述模型复杂度的正则化项在损失函数中的系数
TRANING_STEPS = 3000  # 训练的论数
# 模型保存的路径和文件名
HOME_PATH = '.'
MODEL_SAVE_PATH = os.path.sep.join((HOME_PATH, 'model'))
MODEL_NAME = 'saved_model.ckpt'
# 加载的数据集
DATA_PATH = os.path.sep.join((HOME_PATH, 'data'))
# 每多少次迭代的显示周期
PER_ITERATION = 100
# 测试集大小
TEST_BATCH_SIZE = 100
# 验证集大小
VALIDATION_BATCH_SIZE = 100
# 是否保存训练的模型
SAVE_MODE = False
# 输出结构 log 文件的目录
TENSORBOARD_OUTPUT_PATH = os.sep.join(('.', 'log'))


def reshape_X(X):
    """
    将样本 X \in R^MxN 改为 CNN 的 [BATCH HEIGHT WIDTH CHANELS]的形式
    :param X:
    :return:
    """
    return np.reshape(X, newshape=[-1, mnist_forward_propagate.IMAGE_SIZE,
                                   mnist_forward_propagate.IMAGE_SIZE,
                                   mnist_forward_propagate.NUM_CHANELS])

def train(mnist):
    """
    训练 MNIST 模型
    :param mnist: MNIST 数据集的 tensorflow 对象
    :return:
    """
    # 1. 定义输入的样本和标签
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [
            None,  # validation test train 的样本数量 batch_size 并不一致
            mnist_forward_propagate.IMAGE_SIZE,
            mnist_forward_propagate.IMAGE_SIZE,
            mnist_forward_propagate.NUM_CHANELS
        ], name='X-input')
        ## FC2 前向神经网路的输出
        y = tf.placeholder(tf.float32, [None, mnist_forward_propagate.OUTPUT_NODE], name='y-input')

    # 2. 网路结构
    ## 定义正则化器
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    ## y^hat <- X，根据样本的的输出的预测值 y
    y_output_for_softmax = mnist_forward_propagate.forward_propagate(X, regularizer)
    # 3. 损失函数的相关定义
    with tf.name_scope('loss_function'):
        ## 交叉熵, logits 是前向网络输出的概率, labels 是正确分类的. https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_output_for_softmax, labels=tf.argmax(y, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        ## 使用交叉熵的损失函数 同时加上正则项
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 4. 优化过程，
    with tf.name_scope('optimizer'):
        ## 使用的 Adam 优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=LEANRING_RATE).minimize(loss)
    # 5. 预测的准确率函数
    with tf.name_scope('pred_accuracy'):
        ## 定义预测输出与样本标记的等价函数. 没有使用 softmax 层, 最大就是哪一个类型
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_output_for_softmax, 1))  # y 和 y^hat 的误差
        ## 将 bool 向量转化为浮点数矩阵，求和就是zh
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求值为 1.0 的数量
        ## 计算一下出错的
    # 6.初始化 Tensorflow 模型持久化
    if SAVE_MODE:
        saver = tf.train.Saver()
    # 7. 开启新的会话
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 初始化
        ## 训练的 summaries
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        ## 聚合所有的 summary 操作 用于同步更新
        merged = tf.summary.merge_all()
        ## 定义 summary 输出器
        train_writer = tf.summary.FileWriter(os.sep.join((TENSORBOARD_OUTPUT_PATH, 'train')), sess.graph)
        test_writer = tf.summary.FileWriter(os.sep.join((TENSORBOARD_OUTPUT_PATH, 'test')), sess.graph)
        validation_writer = tf.summary.FileWriter(os.sep.join((TENSORBOARD_OUTPUT_PATH, 'validation')), sess.graph)
        ## 定义不变的测试集、验证集
        validX, validy = mnist.validation.images[:VALIDATION_BATCH_SIZE], mnist.validation.labels[:VALIDATION_BATCH_SIZE]  # 验证集， 用于判断过拟合等
        testX, testy = mnist.test.images[:TEST_BATCH_SIZE], mnist.test.labels[:TEST_BATCH_SIZE]  # 测试集
        #### 转换成张量输入
        reshaped_validX = reshape_X(validX)
        reshaped_testX = reshape_X(testX)
        for i in range(TRANING_STEPS):
            ## 这个主要区别的是准确率输出的时机
            if i % PER_ITERATION == 0:
                ## 检查准确率
                test_merged_value, test_acc = sess.run([merged, accuracy], feed_dict={X: reshaped_testX, y: testy})
                validation_merged_value, validation_acc = sess.run([merged, accuracy], feed_dict={X: reshaped_validX, y: validy})
                ## 保存准确率的输出，到对应的节点
                test_writer.add_summary(test_merged_value, global_step=i)
                validation_writer.add_summary(validation_merged_value, global_step=i)
                ## 输出相关的信息
                pprint('Step:{step:5d}, Test Acc:{tacc:.5f}, Validation Acc:{vacc:.5f}'.format(step=i,
                                                                                               tacc=test_acc,
                                                                                               vacc=validation_acc))
                if SAVE_MODE:
                    ## 保存持久化的模型，其中是各个 epoch 的参数
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
            else:
                ## 针对训练过程的准确率
                trainX, trainy = mnist.train.next_batch(BATCH_SIZE)  # 训练数据：xs 是 图像的数据，ys 是标记 R^10, 对应的数字是1
                reshaped_trainX = reshape_X(trainX)
                ## 这里需要运行优化器
                merged_value, _ = sess.run([merged, optimizer], feed_dict={X: reshaped_trainX, y: trainy})
                train_writer.add_summary(merged_value, global_step=i)
        # 计算错误的分类
        correct_bool_vec, y_output_for_softmax = sess.run([correct_prediction, y_output_for_softmax], feed_dict={X: reshaped_testX, y: testy})
        ## 得到指定的图片（普通的 matplotlib 的形式）, tensorboard 可以参考的是 https://stackoverflow.com/questions/36015170/how-can-i-add-labels-to-tensorboard-images

        error_indices = np.where(correct_bool_vec == False)[0]
        error_test_samples = testX[error_indices]
        error_test_labels = y_output_for_softmax[error_indices,:]
        for i in range(error_test_samples.shape[0]):
            pixels = np.reshape(error_test_samples[i, :], [mnist_forward_propagate.IMAGE_SIZE, mnist_forward_propagate.IMAGE_SIZE])
            plt.gray()
            plt.title('预测标签：{y}, 实际标签：{s}'.format(s=np.argmax(testy[error_indices[i], :]), y=np.argmax(error_test_labels[i, :])))  # testy 需要进一步转换
            plt.imshow(pixels)
            plt.show()
        # 8. 关闭文件流
        train_writer.close()
        test_writer.close()
        validation_writer.close()


def main(*args, **kwargs):
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)  # 避免下载
    train(mnist)


if __name__ == '__main__':
    tf.app.run()