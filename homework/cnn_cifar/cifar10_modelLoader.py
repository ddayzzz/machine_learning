import tensorflow as tf
from homework.cnn_cifar import cifar10_buildNet2
from pprint import pprint
import os
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt






class ModelLoader(object):

    """
    定义一个加载模型的工具类
    """

    def __init__(self, model_saved_prefix, **kwargs):
        """
        定义一个可视化器
        :param model_saved_prefix: 输出的保存的模型目录
        :param kwargs: 其他参数
        """
        self.model_saved_prefix = model_saved_prefix

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
    @staticmethod
    def _plotAImage(img):
        """
        绘制一张 TF 格式的图片
        :param image:
        :return:
        """
        I = np.stack((img[:, :, 0], img[:, :, 1], img[:, :, 2]), axis=2)  # 新产生的合并的数据是 第3个维度. 32x32x3
        plt.title('An image')
        plt.imshow(I)

    def displayConvLayers(self, **kwargs):
        """
        可视化卷积层
        :param kwargs:
        :return:
        """
        raise NotImplementedError('displayConvLayers')

    def displayConvWeights(self, **kwargs):
        """
        可视化卷积核
        :param kwargs:
        :return:
        """
        raise NotImplementedError('displayConvWeights')

    def evaulateOnTest(self, **kwargs):
        """
        在测试集合上测试
        :param kwargs:
        :return:
        """
        raise NotImplementedError('evaulateOnTest')


class TensorflowModelLoader(ModelLoader):

    def __init__(self, cnnnet):
        if not isinstance(cnnnet, cifar10_buildNet2.TensorflowNetwork):
            raise ValueError('参数错误!')
        self._session = tf.Session()
        self.cnn = cnnnet
        self._input_placeholder()
        self._build_compute_graph()  # 这个是必须的操作,用来构建恢复的变量
        # 新建会话
        super(TensorflowModelLoader, self).__init__(model_saved_prefix=os.sep.join(('models', str(cnnnet))))
        # 恢复保存的模型
        variables_in_train_to_restore = tf.all_variables()
        ckpt = tf.train.get_checkpoint_state(self.model_saved_prefix)
        if ckpt and ckpt.model_checkpoint_path:
            saved_model_file = os.sep.join((self.model_saved_prefix, 'saved_model.ckpt'))
            check_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            pprint('从 {0} 加载模型，检查点的训练次数{1}'.format(saved_model_file, check_step))
            saver = tf.train.Saver(variables_in_train_to_restore)
            saver.restore(self._session, ckpt.model_checkpoint_path)
        else:
            pprint('没有找到检查点!')
            return

    def _input_placeholder(self):
        """
        创建输入的张量
        :return:
        """
        self.test_images_input = tf.placeholder(tf.float32, [self.cnn.batch_size,self.cnn.image_width_height, self.cnn.image_width_height,3], name='test_images_input')
        self.test_labels_input = tf.placeholder(tf.float32, [self.cnn.batch_size, self.cnn.num_classes], name='test_labels_input')

    def _build_compute_graph(self):
        """
        生成计算图
        :return:
        """
        # logit 输出
        self.logits = self.cnn.inference(self.test_images_input)
        # 计算准确率
        self.test_accuracy, self.test_equals = self.cnn.accuracy(self.logits, self.test_labels_input)

    def displayConvLayers(self, testDataGenerater):
        """
        可视化卷积层
        :param testDataGenerater:
        :return:
        """
        # 选取数据
        for input_image, input_label in testDataGenerater.generate_augment_batch():  # 获取图片样本
            # 输出原图

            fig1, ax1 = plt.subplots(nrows=1, ncols=1)
            one_image = input_image[0]  # 减少维度
            I = np.stack([one_image[:, :, 0], one_image[:, :, 1], one_image[:, :, 2]], axis=2)  # 拼成像素矩阵，元素是RGB分类值
            ax1.imshow(I)
            ax1.set_title('Image Input, Label=%s' % (testDataGenerater.label_names[np.argmax(input_label[0])]))
            plt.show(block=False)

            # 画出原图
            # 获取各个卷积层的信息
            convs = self.cnn.getConvList()
            for index, conv in enumerate(convs):
                shape = conv.get_shape().as_list()  # 获取卷积层输出的张量维度
                out_filter_shape = shape[-1]  # 设置最多显示的过滤器的输出的 channel 大小. 这里不限制
                conv_out = self._session.run(conv, feed_dict={self.test_images_input: input_image,
                                                              self.test_labels_input: input_label})  # 得到的实际的输出(维度为[BATCH_SIZE HEIGHT WIDTH CHANNEL])
                conv_transpose = np.transpose(conv_out,
                                              [3, 0, 1, 2])  # 转置，tf 的格式是维度在后. 交换轴后 [CHANNEL BATCH HEIGHT WIDTH]
                # 计算每行和每列多少的图像
                grid_y, grid_x = self._factorization(out_filter_shape)
                # 添加子图信息
                fig2, ax2 = plt.subplots(nrows=grid_x, ncols=grid_y, figsize=(shape[1], 1))

                fig2.subplots_adjust(wspace=0.02, hspace=0.02)
                filter_title_prefix = '{name} {out_channel}x{img_width}x{img_height}'.format(name=conv.name,
                                                                                             out_channel='[Real:%d, Displayed: %d]' % (
                                                                                             shape[-1],
                                                                                             out_filter_shape),
                                                                                             img_height=shape[1],
                                                                                             img_width=shape[2])
                for row in range(grid_x):
                    for col in range(grid_y):
                        ax2[row][col].imshow(conv_transpose[row * grid_y + col][0])  # 获取每个通道输出的图像， 选择的是第一张图片
                        ax2[row][col].set_xticks([])
                        ax2[row][col].set_yticks([])
                        # ax2[row][col].set_title('Channel %d' % (row * grid_y + col))
                fig2.suptitle(filter_title_prefix)
                plt.show(block=False)
            plt.pause(500)  # 等待 500 秒, 不让其自动退出
            break

    @staticmethod
    def _deprocess_image(x):
        x -= x.mean()
        x /= (x.std() - 1e-5)
        x *= 0.1
        x += 0.5
        x = np.clip(x, 0, 1)
        x *= 255
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    def _compute_gradient_ascending(self, images_input, images_input_data, filter_out, filter_index, size=32):
        import numpy as np
        from keras import backend as K
        filter_out = tf.convert_to_tensor(filter_out)
        loss = tf.reduce_mean(filter_out[:, :, :, filter_index])
        grads = tf.gradients(loss, images_input)[0]
        grads /= (K.sqrt(K.mean(K.sqrt(grads))) + 1e-5)
        iterate = K.function([images_input], [loss,grads])
        # 图像
        images_input_data = (np.random.random((1, size, size, 3)) - 0.5) * 20 + 128
        step = 1
        for i in range(40):
            loss_value, grads_value = iterate([images_input_data])
            images_input_data += grads_value * step
        img = images_input_data[0]
        return self._deprocess_image(img)

    def displayConvWeights(self, testDataGenerater):
        """
        可视化卷积核
        :param testDataGenerater:
        :return:
        """
        # 选取数据
        for input_image, input_label in testDataGenerater.generate_augment_batch():  # 获取图片样本
            # 输出原图
            fig1, ax1 = plt.subplots(nrows=1, ncols=1)
            one_image = input_image[0]  # 减少维度
            I = np.stack([one_image[:, :, 0], one_image[:, :, 1], one_image[:, :, 2]], axis=2)  # 拼成像素矩阵，元素是RGB分类值
            ax1.imshow(I)
            ax1.set_title('Image Input, Label=%s' % (testDataGenerater.label_names[np.argmax(input_label[0])]))
            plt.show(block=False)

            # 画出原图
            # 获取各个卷积核的信息
            filters = self.cnn.getWeights()
            for index, filter in enumerate(filters):

                # shape = filter.get_shape().as_list()  # 获取卷积层输出的张量维度 [ksize, ksize, in_channel, out_channel]
                # out_filter_shape = shape[-1] if shape[-1] <= 16 else 16  # 设置最多显示的过滤器的输出的 channel 大小
                # filter_out = self._session.run(filter, feed_dict={self.test_images_input: input_image, self.test_labels_input: input_label})  # 得到的实际的输出(维度为[BATCH_SIZE HEIGHT WIDTH CHANNEL])
                print(filter)
                plt.imshow(self._compute_gradient_ascending(self.test_images_input, input_image, filter, 0))
                plt.show()
                break
                # filter_transpose = np.transpose(filter_out, [3, 0, 1, 2])  # 转置, [out_channel, ksize, ksize, in_channel]
                # # 计算每行和每列多少的图像
                # grid_y, grid_x = self._factorization(out_filter_shape)
                # # 添加子图信息
                # fig2, ax2 = plt.subplots(nrows=grid_x, ncols=grid_y, figsize=(shape[1], 1))
                #
                # # fig2.subplots_adjust(wspace=0.02, hspace=0.02)
                # filter_title_prefix = '{name} {out_channel}x{in_cahnnel}x{ksize}x{ksize}'.format(ksize=shape[2], name=filter.name, out_channel='[Real:%d, Displayed: %d]' % (shape[-1], out_filter_shape), in_cahnnel=shape[1])
                # for row in range(grid_x):
                #     for col in range(grid_y):
                #         ax2[row][col].imshow(filter_transpose[row * grid_y + col][0])  # 获取每个通道输出的图像， 选择的是第一张图片
                #         # ax2[row][col].set_xticks([])
                #         # ax2[row][col].set_yticks([])
                #         # ax2[row][col].set_title('Channel %d' % (row * grid_y + col))
                # fig2.suptitle(filter_title_prefix)
                # plt.show(block=False)
            plt.pause(500)  # 等待 500 秒, 不让其自动退出
            break

    def evaulateOnTest(self, testDataGenerater, **kwargs):
        """
        测试数据集
        :param testDataGenerater: 测试数据集产生器
        :param kwargs: 参数
        :return:
        """
        steps_per_batch = testDataGenerater.num_images() // self.cnn.batch_size
        batch_count = 1
        accuracies = []
        for images in range(steps_per_batch):
            images_test, labels_test = testDataGenerater.generate_augment_batch()
            accuracy = self._session.run(self.test_accuracy, feed_dict={
                self.test_images_input: images_test,
                self.test_labels_input: labels_test
            })
            pprint('Batch %d, Test accuracy: %.3f' % (batch_count, accuracy))
            accuracies.append(accuracy)
            batch_count += 1
        accuracies = np.array(accuracies)
        print('Test accuracy: \nMean: {0:.3f}\nMax: {1:.3f}\nMin: {2:.3f}'.format(np.mean(accuracies), np.max(accuracies), np.min(accuracies)))

    def __del__(self):
        """
        析构函数
        :return:
        """
        print('Close tf Session')
        self._session.close()

class KerasModelLoader(ModelLoader):

    """
    Keras 版本的模型加载器
    """

    def __init__(self, model):
        if not isinstance(model, cifar10_buildNet2.KerasCNNNetwork):
            raise ValueError('参数错误!')
        self.model = model
        super(KerasModelLoader, self).__init__(model_saved_prefix=os.sep.join(('models', str(model))))
        # 载入一些测试数据
        self._init()

    def _init(self):
        """
        载入一些测试的数据
        :return:
        """
        from keras.datasets import cifar10
        from keras.optimizers import Adam
        import keras
        _, (x_test, y_test) = cifar10.load_data()  # 仅仅加载测试的数据
        x_test = x_test.astype('float32') / 255  # 归一化
        y_test = keras.utils.to_categorical(y_test, self.model.num_classes)  # 处理标签
        self.X_test = x_test
        self.y_test = y_test
        self.X_shape = x_test.shape[1:]

        # 初始化网络结构
        lr_changer = self.model.learn_rate_changer()
        self.model.inference(inputs_shape=self.X_shape)
        # 建立模型
        self.model.buildModel(loss='categorical_crossentropy',
                              optimizer=Adam(lr=lr_changer(0)),
                              metrics=['accuracy'])
        # 加载模型的权重
        self.model.loadWeights(os.sep.join([self.model_saved_prefix, 'checkpoints.h5']))

    def displayConv(self, **kwargs):
        """
        显示神经网络中的卷积核的输出
        :param kwargs:
        :return:
        """
        # 选择一张图片
        image = self.X_test[0]
        # 显示出来
        self._plotAImage(img=image)
        # 输出


    def evaulateOnTest(self, **kwargs):
        """
        测试准确率
        :param kwargs:
        :return:
        """
        loss, acc = self.model.evaluate(self.X_test, self.y_test, **kwargs)
        print('Test Loss:{0:.3f}, Test Accuracy:{1:.3f}'.format(loss, acc))



