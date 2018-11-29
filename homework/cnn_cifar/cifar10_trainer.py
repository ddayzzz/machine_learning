import tensorflow as tf
from homework.cnn_cifar import cifar10_buildNet2
from multiprocessing import cpu_count
from pprint import pprint
import os
import numpy as np

class Trainer(object):

    """
    自定义的训练器
    """
    def __init__(self, logout_prefix, model_saved_prefix, init_learning_rate, learning_rate_decay_per_epoch, learning_rate_decay_rate, **kwargs):
        """
        定义一个训练器
        :param logout_prefix: 输出的日志文件目录
        :param model_saved_prefix: 输出的保存的模型目录
        :param init_learning_rate: 初始的学习率
        :param learning_rate_decay_per_epoch: 多少 epoch 就减少学习率
        :param learning_rate_decay_rate: 学习衰减率
        :param kwargs: 其他参数
        """
        self.logout_prefix = logout_prefix
        self.model_saved_prefix = model_saved_prefix
        self.init_learning_rate = init_learning_rate
        self.learning_rate_decay_per_epoch = learning_rate_decay_per_epoch
        self.learning_rate_decay_rate = learning_rate_decay_rate
        if not os.path.exists(logout_prefix):
            os.makedirs(logout_prefix)

        if not os.path.exists(model_saved_prefix):
            os.makedirs(model_saved_prefix)

    def train(self, trainDataGenerater, validDataGenerater, **kwargs):
        """
        训练过程
        :param trainDataGenerater: 训练数据产生器
        :param validDataGenerater: 验证数据产生器
        :param kwargs: 其他的参数
        :return:
        """
        raise NotImplementedError('没有实现 train!')


class TensorflowTrainer(Trainer):

    def __init__(self, cnnnet, max_epoch, init_learning_rate, learning_rate_decay_per_epoch, learning_rate_decay_rate):
        """
        TF 训练器
        :param cnnnet: TF 神经网络
        :param max_epoch: 最大 epoch
        :param init_learning_rate: 初始的学习率
        :param learning_rate_decay_per_epoch: 多少 epoch 就减少学习率
        :param learning_rate_decay_rate: 学习衰减率
        """
        if not isinstance(cnnnet, cifar10_buildNet2.TensorflowNetwork):
            raise ValueError('请使用 TensorflowNetwork 子类作为 CNN 的参数')
        self.cnn = cnnnet
        self.max_epoch = max_epoch
        super(TensorflowTrainer, self).__init__(logout_prefix=os.sep.join(('logouts', str(cnnnet))),
                                                model_saved_prefix=os.sep.join(('models', str(cnnnet))),
                                                init_learning_rate=init_learning_rate,
                                                learning_rate_decay_per_epoch=learning_rate_decay_per_epoch,
                                                learning_rate_decay_rate=learning_rate_decay_rate)

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
        self.train_labels_input = tf.placeholder(tf.float32, [self.cnn.batch_size, self.cnn.num_classes], name='train_labels_input')

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
        train_op, lr = self.cnn.train(self.train_loss,
                                      self.global_steps,
                                      init_learning_rate=self.init_learning_rate,
                                      num_epoch_per_decay=self.learning_rate_decay_per_epoch,
                                      learning_rate_decay=self.learning_rate_decay_rate)

        self.train_op = train_op
        self.learning_rate = lr
        self.logits = logits

    def _check_acc_on_valid(self, imgs, labels, sess):
        valid_acc = sess.run(self.train_accuracy, feed_dict={self.train_images_input: imgs,
                                                             self.train_labels_input: labels})
        return valid_acc

    def train(self, trainDataGenerater, validDataGenerater, **kwargs):
        """
        训练神经网络
        :param trainDataGenerater: 训练数据集的产生对象
        :param validDataGenerater: 验证数据集的对象
        :return:
        """
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.7 config=config
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
            ckpt = tf.train.get_checkpoint_state(self.model_saved_prefix)
            # 模型描述文件
            model_saved_file = os.sep.join((self.model_saved_prefix, 'saved_model.ckpt'))
            if ckpt and ckpt.model_checkpoint_path:
                startstep = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                print('检查点的训练次数:', startstep)
                pprint('从 “%s” 加载模型' % model_saved_file)
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 修改步数
                tf.assign(self.global_steps, startstep)
            else:
                pprint('没有找到检查点!')
                init = tf.global_variables_initializer()
                sess.run(init)
                #
                if isinstance(self.cnn, cifar10_buildNet2.PretrainedVGG19Network):
                    sess.run(self.logits.pretrained())
                startstep = 0
            # 初始化所有的变量
            # 定义 summary 输出器
            train_writer = tf.summary.FileWriter(os.sep.join((self.logout_prefix, 'train')), sess.graph)
            # 计算一次 epoch 走过的 step 次数
            batches_per_epoch = self.cnn.num_examples_per_epoch // self.cnn.batch_size  # 每进行一次 epoch 训练， 内循环迭代的次数. 也就是batch 的数量
            # 把验证集上最后的平均结果进行计算
            max_acc = -1.0
            for epoch in range(startstep // batches_per_epoch, self.max_epoch):
                # 跑一次 epoch
                for images_batch, labels_batch in trainDataGenerater.generate_augment_batch():
                    # 训练数据
                    _, loss_, lr, merged_value, global_step_, train_acc = sess.run(
                        [self.train_op, self.train_loss, self.learning_rate, merged, self.global_steps, self.train_accuracy],
                        feed_dict={self.train_images_input: images_batch, self.train_labels_input: labels_batch})
                    # 每100 个 step 输出一次
                    if (global_step_ + 1) % 100 ==0:
                        print('Epoch %d, Global step: %10d, Train accuracy: %.3f, Loss: %.3f, Learning rate: %.7f' % (epoch, global_step_, train_acc, loss_, lr))
                    train_writer.add_summary(merged_value, global_step=global_step_)  # 每一个 step 记录一次
                # 验证集的平均正确率
                valid_accs = []
                for valid_images_batch, valid_labels_batch in validDataGenerater.generate_augment_batch():
                    va = self._check_acc_on_valid(valid_images_batch, valid_labels_batch, sess)
                    valid_accs.append(va)
                # 平均的验证集准确率
                mean_valid_acc = np.mean(np.array(valid_accs))
                if mean_valid_acc > max_acc:
                    print('Epoch {2}, Validation average accuracy changed from {0:.7f} to {1:.7f}, save model.'.format(max_acc, mean_valid_acc, epoch))
                    max_acc = mean_valid_acc
                    # 保存模型
                    saver.save(sess, model_saved_file, global_step=self.global_steps)
                else:
                    print('Epoch {0}, Validation average accuracy {1:.7f} not improve from {2:.7f}.'.format(
                        epoch, mean_valid_acc, max_acc))
                # 继续下一次训练

            train_writer.close()


class KerasTrainer(Trainer):

    """
    正对于 Keras 模型的训练器
    """

    def __init__(self, model, max_epochs, init_learning_rate, learning_rate_decay_per_epoch, learning_rate_decay_rate, tiny_output_logs=True):
        """
        Keras 训练器
        :param cnnnet: TF 神经网络
        :param max_epoch: 最大 epoch
        :param init_learning_rate: 初始的学习率
        :param learning_rate_decay_per_epoch: 多少 epoch 就减少学习率
        :param learning_rate_decay_rate: 学习衰减率
        :param tiny_output_logs: 是否仅仅显示 train loss 学习率 和 train 准确率
        """
        if not isinstance(model, cifar10_buildNet2.KerasCNNNetwork):
            raise ValueError('请使用 KerasCNNNetwork 子类作为 CNN 的参数')
        self.model = model  # Keras 模型
        self.tiny_output_logs = tiny_output_logs
        self.max_epochs = max_epochs
        super(KerasTrainer, self).__init__(logout_prefix=os.sep.join(('logouts', str(model))),
                                           model_saved_prefix=os.sep.join(('models', str(model))),
                                           init_learning_rate=init_learning_rate,
                                           learning_rate_decay_per_epoch=learning_rate_decay_per_epoch,
                                           learning_rate_decay_rate=learning_rate_decay_rate)

    def train(self, trainDataGenerater=None, validDataGenerater=None, **kwargs):
        import keras
        from keras.optimizers import Adam
        from keras.callbacks import ModelCheckpoint, LearningRateScheduler
        from keras.callbacks import ReduceLROnPlateau
        from keras.datasets import cifar10
        from keras.preprocessing.image import ImageDataGenerator
        from keras.callbacks import TensorBoard
        # 处理 Tensorboard 的输出
        tensorBoardParams = {'log_dir': self.logout_prefix, 'write_graph': True, 'write_images': True}
        if not self.tiny_output_logs:
            tensorBoardParams.update({'write_grads': True})  # histgram 可能可以设置 具体参看 Keras 的回调 https://keras-cn.readthedocs.io/en/latest/other/callbacks/
        tensorBoardCallBack = TensorBoard(**tensorBoardParams)
        # 准备保存的检查点(权重文件)
        checkpoints = ModelCheckpoint(filepath=os.sep.join([self.model_saved_prefix, 'checkpoints.h5']),
                                      monitor='val_acc',
                                      verbose=1,
                                      save_best_only=True)  # 保存最好的验证集误差的权重
        lr_changer = self.model.learn_rate_changer(init_learning_rate=self.init_learning_rate)
        lr_scheduler = LearningRateScheduler(lr_changer)  # 学习率衰减的调用器
        lr_reducer = ReduceLROnPlateau(factor=self.learning_rate_decay_rate, cooldown=0, patience=5, min_lr=.5e-6)
        callbacks = [checkpoints, lr_reducer, lr_scheduler, tensorBoardCallBack]  # 回调顺序
        # 定义数据
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        # 均1化数据 float -> 0 ~ 1
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        # 减去像素的均值
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean

        print('x_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)
        # 转换标签
        y_test = keras.utils.to_categorical(y_test, self.model.num_classes)
        y_train = keras.utils.to_categorical(y_train, self.model.num_classes)
        # 数据增强(官方的设置)
        datagen = ImageDataGenerator(
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        # 计算相关的增强属性
        datagen.fit(X_train)
        # 模型, 初始化模型
        self.model.inference(inputs_shape=X_train.shape[1:])
        self.model.buildModel(loss='categorical_crossentropy',
                         optimizer=Adam(lr=lr_changer(0)),
                         metrics=['accuracy'])
        self.model.print_summary()
        self.model.fit_generator(generator=datagen.flow(X_train, y_train, batch_size=self.model.batch_size),
                                 validation_data=(X_test, y_test),
                                 epochs=self.max_epochs,
                                 verbose=2,  # 每一个 epoch 显示一次
                                 workers=cpu_count(),
                                 callbacks=callbacks,
                                 steps_per_epoch=X_train.shape[0] // self.model.batch_size)  # https://stackoverflow.com/questions/43457862/whats-the-difference-between-samples-per-epoch-and-steps-per-epoch-in-fit-g




