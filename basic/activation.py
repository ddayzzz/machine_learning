import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    """
    添加神经层的函数
    :param inputs:
    :param in_size:
    :param out_size:
    :param activation_function: 激活函数
    :return:
    """
    Weights = tf.Variable(tf.random_uniform([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases  # 未激活的值
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.linspace(start=-1, stop=1, num=300)[:, np.newaxis]  # 定义一个线性空间，在创造的空间中添加一个轴
noise = np.random.normal(0, 0.05, x_data.shape)  # 噪点
y_data = np.square(x_data) - 0.5 + noise
# train_step 输入的值
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)  # 输入的维度是1
# 输出层
prediction = add_layer(l1, 10, 1, activation_function=None)
# loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 训练
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data, y_data)  # 实际的曲线
    plt.ion()  # 不要暂停
    plt.show()

    for i in range(5000):
        sess.run(train_step, feed_dict={xs: x_data, ys:y_data})
        if i % 50 == 0:
            # print(sess.run(loss, feed_dict={xs: x_data, ys:y_data}))
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            try:
                ax.lines.remove(lines[0])  # lines 总是只有一个函数
            except Exception:
                pass
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)

