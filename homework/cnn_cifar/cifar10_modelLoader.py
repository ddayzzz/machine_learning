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

    def __init__(self, logout_prefix, model_saved_prefix, **kwargs):
        """
        定义一个可视化器
        :param model_saved_prefix: 输出的保存的模型目录
        :param logout_prefix: 输出的日志文件目录
        :param kwargs: 其他参数
        """
        self.model_saved_prefix = model_saved_prefix
        self.logout_prefix = logout_prefix

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

    def __init__(self, cnnnet, logdir=None, modelDir=None):
        """
        TF 模型加载器
        :param cnnnet:
        :param logdir： 日志文件的输出目录
        :param modelDir: 模型恢复的文件目录（用于继续训练）
        """
        if not isinstance(cnnnet, cifar10_buildNet2.TensorflowNetwork):
            raise ValueError('参数错误!')
        self._session = tf.Session()
        self.cnn = cnnnet
        self._input_placeholder()
        self._build_compute_graph()  # 这个是必须的操作,用来构建恢复的变量
        # 新建会话
        ## 恢复的模型和 log
        logdir = os.sep.join(('logouts', str(cnnnet))) if not logdir else logdir
        modelDir = os.sep.join(('models', str(cnnnet))) if not modelDir else modelDir
        super(TensorflowModelLoader, self).__init__(logout_prefix=logdir, model_saved_prefix=modelDir)
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

    def displayConvWeights(self, testDataGenerater):
        """
        可视化卷积核
        :param testDataGenerater:
        :return:
        """
        weights_writer = tf.summary.FileWriter(os.sep.join((self.logout_prefix, 'filters_out')), self._session.graph)
        merged = tf.summary.merge_all()
        print(merged)
        # 选取数据
        for input_image, input_label in testDataGenerater.generate_augment_batch():  # 获取图片样本
            # 输出原图
            # fig1, ax1 = plt.subplots(nrows=1, ncols=1)
            # one_image = input_image[0]  # 减少维度
            # I = np.stack([one_image[:, :, 0], one_image[:, :, 1], one_image[:, :, 2]], axis=2)  # 拼成像素矩阵，元素是RGB分类值
            # ax1.imshow(I)
            # ax1.set_title('Image Input, Label=%s' % (testDataGenerater.label_names[np.argmax(input_label[0])]))
            # plt.show(block=False)
            # 绘图
            merged_value = self._session.run(merged, feed_dict={self.test_images_input: input_image, self.test_labels_input: input_label})
            weights_writer.add_summary(merged_value)  # 记录
            print("Please run Tensorboard to view the result")
            break
        weights_writer.close()


    def evaulateOnTest(self, testDataGenerater, **kwargs):
        """
        测试数据集
        :param testDataGenerater: 测试数据集产生器
        :param kwargs: 参数
        :return:
        """
        valid_accs = []
        for index, (images_batch, labels_batch) in enumerate(testDataGenerater.generate_augment_batch()):
            va = self._session.run(self.test_accuracy, feed_dict={self.test_images_input: images_batch, self.test_labels_input: labels_batch})
            valid_accs.append(va)
            print('Batch {0}, Test accuracy: {1}'.format(index +1, va))
        # 平均的验证集准确率
        valid_accs = np.array(valid_accs)
        mean_valid_acc = np.mean(valid_accs)
        max_acc = np.max(valid_accs)
        min_acc = np.min(valid_accs)
        print('Average accuracy: {0:.3f}, Min accuracy: {1:.3f}, Max accuracy :{2:.3f}'.format(mean_valid_acc, min_acc, max_acc))

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

    def __init__(self, modelFile):
        """
        Keras 版本加载器
        :param modelFile: 模型恢复的文件 *.h5（用于加载整个网络结构）
        :param logdir： 日志文件的输出目录

        """
        # 恢复
        super(KerasModelLoader, self).__init__(logout_prefix=None, model_saved_prefix=modelFile)
        # 载入一些测试数据
        self._init()

    def _init(self):
        """
        载入一些测试的数据
        :return:
        """
        from keras.datasets import cifar10
        from keras.models import load_model
        import keras
        print('Load model from %s' % self.model_saved_prefix)
        self.model = load_model(self.model_saved_prefix)
        _, (x_test, y_test) = cifar10.load_data()  # 仅仅加载测试的数据

        x_test = x_test.astype('float32') / 255  # 归一化

        x_test_mean = np.mean(x_test, axis=0)

        x_test -= x_test_mean

        y_test = keras.utils.to_categorical(y_test, self.model.output.get_shape().as_list()[1])  # 处理标签, 需要 softmax 输出的维度
        self.X_test = x_test
        self.y_test = y_test
        self.X_shape = x_test.shape[1:]

        # 初始化网络结构

        # 加载各层的输出
        self.layers = dict([(layer.name, layer) for layer in self.model.layers[1:]])
        self.image_input_placeholder = self.model.input

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

    @staticmethod
    def normalize(x):
        from keras import backend as K
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

    def _compute_gradient_ascending(self, layer_name, size=200):
        import numpy as np
        from keras import backend as K
        kept_filters = []
        for filter_index in range(20):
            # we build a loss function that maximizes the activation
            # of the nth filter of the layer considered
            layer_output = self.layers[layer_name].output
            if K.image_data_format() == 'channels_first':
                loss = K.mean(layer_output[:, filter_index, :, :])
            else:
                loss = K.mean(layer_output[:, :, :, filter_index])

            # we compute the gradient of the input picture wrt this loss
            grads = K.gradients(loss, self.image_input_placeholder)[0]

            # normalization trick: we normalize the gradient
            grads = self.normalize(grads)

            # this function returns the loss and grads given the input picture
            iterate = K.function([self.image_input_placeholder], [loss, grads])

            # step size for gradient ascent
            step = 1.

            # we start from a gray image with some random noise
            if K.image_data_format() == 'channels_first':
                input_img_data = np.random.random((1, 3, size, size))
            else:
                input_img_data = np.random.random((1, size, size, 3))
            input_img_data = (input_img_data - 0.5) * 20 + 128

            # we run gradient ascent for 20 steps
            for i in range(20):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                print('Current loss value:', loss_value)
                if loss_value <= 0.:
                    # some filters get stuck to 0, we can skip them
                    break

            # decode the resulting input image
            if loss_value > 0:
                img = self._deprocess_image(input_img_data[0])
                kept_filters.append((img, loss_value))
        return kept_filters

    def displayConvWeights(self, **kwargs):
        """
        可视化卷积核
        :param kwargs:
        :return:
        """
        # 选取数据
        from keras.preprocessing.image import save_img
        one_image, one_label = self.X_test[0], self.y_test[0]
        # 输出原图
        # fig1, ax1 = plt.subplots(nrows=1, ncols=1)
        # I = np.stack([one_image[:, :, 0], one_image[:, :, 1], one_image[:, :, 2]], axis=2)  # 拼成像素矩阵，元素是RGB分类值
        # ax1.imshow(I)
        # ax1.set_title('Image Input, Label=%s' % (testDataGenerater.label_names[np.argmax(input_label[0])]))
        # plt.show(block=False)
        # 各层卷积核可视化


        kept_filters = self._compute_gradient_ascending('conv2d_22')
        # we will stich the best 64 filters on a 8 x 8 grid.
        n = 3

        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top 64 filters.
        kept_filters.sort(key=lambda x: x[1], reverse=True)
        kept_filters = kept_filters[:n * n]

        # build a black picture with enough space for
        # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
        margin = 5
        img_width = img_height = 200
        width = n * img_width + (n - 1) * margin
        height = n * img_height + (n - 1) * margin
        stitched_filters = np.zeros((width, height, 3))

        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                img, loss = kept_filters[i * n + j]
                width_margin = (img_width + margin) * i
                height_margin = (img_height + margin) * j
                stitched_filters[
                width_margin: width_margin + img_width,
                height_margin: height_margin + img_height, :] = img

        # save the result to disk
        save_img('stitched_filters_%dx%d.png' % (n, n), stitched_filters)

        # plt.pause(500)  # 等待 500 秒, 不让其自动退出




