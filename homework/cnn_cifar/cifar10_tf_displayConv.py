import tensorflow as tf
from homework.cnn_cifar import cifar10_buildNet2
from pprint import pprint
import os
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt


MODEL_PATH = 'tf_model'  # 保存的模型文件夹
MODEL_SAVE_FILE = os.sep.join((MODEL_PATH, 'saved_model.ckpt'))  # 保存的元文件路径

class DisplayConv(object):

    def __init__(self, cnnnet):
        if not isinstance(cnnnet, cifar10_buildNet2.CNNNetwork):
            raise ValueError('参数错误!')
        self.cnn = cnnnet
        # 输出
        out_dir = './out_conv'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        self.out_drir = out_dir

    def _factorization(self, n):
        """
        求 n 的最大的因子
        :return: 最大因子，n和最大因子的整数商
        """
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1:
                    raise ValueError('n 是质数！')
                return (i, n // i)

    def _input_placeholder(self):
        """
        创建输入的张量
        :return:
        """
        self.one_image_input = tf.placeholder(tf.float32, [
            1,  # validation test train 的样本数量 batch_size 并不一致
            32,  # 图像大小
            32,
            3  # RGB
        ], name='one_image_input')
        self.one_label_input = tf.placeholder(tf.int64, [1], name='one_label_input')

    def _build_compute_graph(self):
        """
        生成计算图
        :return:
        """
        # logit 输出
        self.logits = self.cnn.inference(self.one_image_input)
        # 计算准确率
        self.test_accuracy, self.test_equals = self.cnn.accuracy(self.logits, self.one_label_input)

    def display(self, testDataGenerater):
        # 打开一个新的会话
        with tf.Session() as sess:
            self._input_placeholder()
            self._build_compute_graph()  # 这个是必须的操作,用来构建恢复的变量
            variables_in_train_to_restore = tf.all_variables()
            ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                check_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                pprint('从 {0} 加载模型，检查点的训练次数{1}'.format(MODEL_SAVE_FILE, check_step))
                saver = tf.train.Saver(variables_in_train_to_restore)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                pprint('没有找到检查点!')
                return
            # 选取数据
            input_image, input_label = testDataGenerater.generate_augment_batch()  # 获取图片样本
            # 输出原图

            fig1, ax1 = plt.subplots(nrows=1, ncols=1)
            one_image = input_image[0]  # 减少维度
            I = np.stack([one_image[:,:,0], one_image[:,:,1], one_image[:,:,2]], axis=2)  # 拼成像素矩阵，元素是RGB分类值
            ax1.imshow(I / 255.)
            ax1.set_title('原图')
            plt.show(block=False)


            # 画出原图
            # 获取各个卷积层的信息
            convs = self.cnn.getConvList()
            for index, conv in enumerate(convs):
                shape = conv.get_shape().as_list()  # 获取卷积层输出的张量维度
                out_filter_shape = shape[-1] if shape[-1] <= 16 else 16  # 设置最多显示的过滤器的输出的 channel 大小
                conv_out = sess.run(conv, feed_dict={self.one_image_input: input_image, self.one_label_input: input_label})  # 得到的实际的输出(维度为[BATCH_SIZE HEIGHT WIDTH CHANNEL])
                conv_transpose = np.transpose(conv_out, [3, 0, 1, 2])  # 转置，tf 的格式是维度在后. 交换轴后 [CHANNEL BATCH HEIGHT WIDTH]
                # 计算每行和每列多少的图像
                grid_y, grid_x = self._factorization(out_filter_shape)
                # 添加子图信息
                fig2, ax2 = plt.subplots(nrows=grid_x, ncols=grid_y, figsize=(out_filter_shape, 1))
                filter_title_prefix = '{name} {out_channel}x{img_width}x{img_height}'.format(name=conv.name, out_channel=out_filter_shape, img_height=shape[1], img_width=shape[2])
                for row in range(grid_x):
                    for col in range(grid_y):
                        ax2[row][col].imshow(conv_transpose[row * grid_y + col][0])  # 获取每个通道输出的图像， 选择的是第一张图片
                        ax2[row][col].set_title('Channel %d' % (row * grid_y + col))
                plt.title(filter_title_prefix)
                plt.show(block=False)
            plt.pause(500)  # 等待 500 秒, 不让其自动退出


cnnnet_test = cifar10_buildNet2.VGGNetwork(False, num_examples_per_epoch=10000)
from homework.cnn_cifar.cifar10_loadFile import AugmentImageGenerator
data = AugmentImageGenerator(True, 1)  # 显示一张图
testObj = DisplayConv(cnnnet_test)
testObj.display(data)