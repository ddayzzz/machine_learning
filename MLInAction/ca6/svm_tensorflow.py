#导入库
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from MLInAction.ca6.loadDataset import loadDataSet


BATCH_SIZE = 100  # Mini-batch 梯度下降的批量参数。这里的 Mini-batch 的必须保证 m=测试集样本数量=训练姐样本数量。alphas 与 m 有关
LEARNING_RATE = 0.0005
ITERATION = 4000
DISPLAY_STEP = 500
SIGMA = tf.constant(-50.0)
TENSORBOARD_DIR = './log'  # 输出用于 tensorboard 的文件


with tf.device('/gpu:0'):
    with tf.Session() as sess:
        # 第一步: 声明变量和占位符
        X_train = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='X_train')
        y_train_target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y_train_target')  # 训练得到的目标 y
        prediction_meshgrid = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='prediction_func')
        alphas = tf.Variable(tf.random_normal(shape=[1, BATCH_SIZE], name='lag_alphas'))  # 广义拉格朗日的乘子 alpha
        # 第二步: 建立优化的过程
        ## 1. RBF
        l2_norm = tf.reduce_sum(tf.square(X_train), 1)  # 求二范数, 没有开根号,注意 样本是行向量,所以需要行相加
        l2_norm = tf.reshape(l2_norm, [-1, 1])  # 转换成列向量
        X_inner_product = tf.matmul(X_train, tf.transpose(
            X_train))  # X*X^T(样本是行向量), 矩阵乘法. X_inner_product[i,j] = X[i] dot X[j]
        X_inner_product = tf.multiply(2.0, X_inner_product)  # 这个是平方差公式中的2倍的项
        square_l2_xi_minus_xj = tf.add(tf.subtract(l2_norm, X_inner_product), tf.transpose(
            l2_norm))  # (X_i - X_j)^2. 最后的注意是转置, 因为L2(X1)^2 - X1 dot X2 + L2(X2)^2. (可以画出矩阵出来). TF 自动进行广播操作
        # kernel = tf.exp(tf.negative(tf.div(tf.abs(square_l2_xi_minus_xj), tf.multiply(2.0, tf.pow(SIGMA, 2)))), name='rbf_kernel')
        kernel = tf.exp(tf.multiply(SIGMA, tf.abs(square_l2_xi_minus_xj)))
        ## 2. 预测函数fx
        #### 预测值用的 核函数
        row_A = tf.reshape(tf.reduce_sum(tf.square(X_train), 1), [-1, 1])  # 1 表示的是按行求和, 然后转换为列向量, 每一个列向量的值是原 X_i 的 L范数的平方
        row_B = tf.reshape(tf.reduce_sum(tf.square(prediction_meshgrid), 1), [-1, 1])
        square_row_A_minus_row_B_l2_norm = tf.add(
            tf.subtract(row_A, tf.multiply(2.0, tf.matmul(X_train, tf.transpose(prediction_meshgrid)))),
            tf.transpose(row_B))  # L2 范数. L2(row_A - row_B)
        # prediction_kernel = tf.exp(tf.negative(tf.div(tf.abs(square_row_A_minus_row_B_l2_norm), tf.multiply(2.0, tf.pow(SIGMA, 2)))), name='rbf_prediction_kernel')
        prediction_kernel = tf.exp(tf.multiply(SIGMA, tf.abs(square_row_A_minus_row_B_l2_norm)))
        #### 这个的作用还需要细看
        prediction_output = tf.matmul(tf.multiply(tf.transpose(y_train_target), alphas), prediction_kernel)
        prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))  # 求出的预测输出的均值(按照轴的顺序以此求平均数。)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_train_target)), tf.float32))  # squeeze: 删除分量为1的分量(去掉是因为?), cast: 把equal 返回的布尔矩阵转换成 float 类型.
        ## 3. 对偶问题和损失函数, P127 公式 6.24
        model_output = tf.matmul(alphas, kernel)
        sum_alphas = tf.reduce_sum(alphas)
        alphas_innner_product = tf.matmul(tf.transpose(alphas), alphas)
        y_inner_product = tf.matmul(y_train_target, tf.transpose(y_train_target))  # 还是要注意是一行一个标签
        sum_sum_alpha_i_alpha_j_kernel_xi_xj = tf.reduce_sum(
            tf.multiply(kernel, tf.multiply(alphas_innner_product, y_inner_product)))  # 一大堆 sum 的那个部分
        loss_func = tf.negative(
            tf.subtract(sum_alphas, sum_sum_alpha_i_alpha_j_kernel_xi_xj))  # 损失函数, 这个需要最小化. 对偶问题是最大化
        ## 4. 优化过程
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_program = optimizer.minimize(loss_func)
        ## 5. 初始化变量
        initializer = tf.global_variables_initializer()
        sess.run(initializer)  # 初始化全局变量

        ## 6. 加载数据
        X, y = loadDataSet('dataset_train_rbf.txt')
        X = np.mat(X)
        y = np.mat(y).T
        testX, testy = loadDataSet('dataset_test_rbf.txt')
        testX = np.mat(testX)
        testy = np.mat(testy).T
        ## 7. 训练过程
        loss_func_value = []
        batch_gd_accuracy = []
        test_accuracy = []
        for i in range(ITERATION):
            random_indices = np.random.choice(X.shape[0], size=BATCH_SIZE)  # 随机选择 batch 的样本索引
            random_X = X[random_indices, :]
            random_y = y[random_indices, :]
            sess.run(train_program, feed_dict={X_train: random_X, y_train_target: random_y})  # 运行梯度下降的优化函数
            new_loss = sess.run(loss_func, feed_dict={X_train: random_X, y_train_target: random_y})
            loss_func_value.append(new_loss)  #　新的损失

            new_accuracy = sess.run(accuracy, feed_dict={X_train: random_X, y_train_target: random_y, prediction_meshgrid: random_X})  # 准确率
            batch_gd_accuracy.append(new_accuracy)
            ### 测试集的准确率随着迭代的变化
            test_acc = sess.run(accuracy, feed_dict={X_train: testX, y_train_target: testy, prediction_meshgrid: testX})
            test_accuracy.append(test_acc)
            ### 显示输出
            if (i + 1) % DISPLAY_STEP == 0:
                print('Step: {dstep:}, Loss={dloss:.5f}'.format(dstep=i + 1, dloss=new_loss))

        # 创建数据点网格用于后续的数据空间可视化分类
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        XM, YM = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        mershgrid_point = np.column_stack([XM.ravel(), YM.ravel()])
        [mershgrid_predictions] = sess.run(prediction, feed_dict={X_train: random_X, y_train_target: random_y,
                                                                  prediction_meshgrid: mershgrid_point})
        mershgrid_predictions = mershgrid_predictions.reshape(XM.shape)

        # 绘制预测结果
        plt.contourf(XM, YM, mershgrid_predictions, cmap=plt.cm.Paired, alpha=0.7)
        # 将数据点绘制
        class1_indices = np.nonzero(y > 0)[0]
        class2_indices = np.nonzero(y < 0)[0]
        class1_x1 = X[class1_indices, 0]
        class1_x2 = X[class1_indices, 1]
        class2_x1 = X[class2_indices, 0]
        class2_x2 = X[class2_indices, 1]
        plt.plot(class1_x1, class1_x2, 'ro', label='类1')
        plt.plot(class2_x1, class2_x2, 'bx', label='类2')
        plt.legend(loc='best')
        plt.show()
        # 绘制批量结果准确度
        plt.plot(batch_gd_accuracy, 'r-', label='最终的准确率。训练集：{tr:.5f}'.format(tr=batch_gd_accuracy[-1]))
        plt.plot(test_accuracy, 'b-', label='测试集的准确率 测试集合：{te:.5f}'.format(te=test_accuracy[-1]))
        plt.title('准确率')
        plt.xlabel('迭代次数')
        plt.ylabel('准确率')
        plt.legend(loc='best')
        plt.show()

        # 绘制损失函数
        plt.plot(loss_func_value, 'r-')
        plt.title('损失函数随梯度的变化')
        plt.xlabel('迭代次数')
        plt.ylabel('损失误差')
        plt.show()
        # 可视化
        if os.path.exists(TENSORBOARD_DIR):
            writer = tf.summary.FileWriter(TENSORBOARD_DIR, sess.graph)
            writer.close()  # 运行 tensorboard --logdir = ./log







