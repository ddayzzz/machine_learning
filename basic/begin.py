#
import tensorflow as tf
import numpy as np


x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
# 基本 tensorflow 结构
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))  # 初始值
y = Weights * x_data + biases  # 预测的 y
loss = tf.reduce_mean(tf.square(y - y_data))  # 定义的损失，使用均方误差
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 优化器， 减少误差，提升参数的准确率。下一次的误差会更小。0.5 指定学习率
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()  # 初始化结构
# tensorflow 构架
sess = tf.Session()
sess.run(init)

# 训练 NN
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
sess.close()