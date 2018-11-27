import tensorflow as tf
from homework.cnn_cifar import cifar10_buildNet2
from pprint import pprint
import os
import numpy as np

class Trainer(object):

    """
    自定义的训练器
    """
    def __init__(self, logout_prefix, model_saved_prefix, **kwargs):
        """
        定义一个训练器
        :param logout_prefix: 输出的日志文件目录
        :param model_saved_prefix: 输出的保存的模型目录
        :param kwargs: 其他参数
        """
        self.logout_prefix = logout_prefix
        self.model_saved_prefix = model_saved_prefix
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

    def __init__(self, cnnnet, max_epoch, display_per_global_steps=100):
        if not isinstance(cnnnet, cifar10_buildNet2.TensorflowNetwork):
            raise ValueError('请使用 TensorflowNetwork 子类作为 CNN 的参数')
        self.cnn = cnnnet
        self.max_epoch = max_epoch
        self.display_per_global_steps = display_per_global_steps
        super(TensorflowTrainer, self).__init__(logout_prefix=os.sep.join(('logouts', str(cnnnet))), model_saved_prefix=os.sep.join(('models', str(cnnnet))))

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
        self.train_labels_input = tf.placeholder(tf.int64, [self.cnn.batch_size], name='train_labels_input')
        # 验证集部分
        self.valid_images_input = tf.placeholder(tf.float32, shape=[self.cnn.batch_size, self.cnn.image_width_height, self.cnn.image_width_height, 3], name="valid_images_input")
        self.valid_labels_input = tf.placeholder(tf.int64, [self.cnn.batch_size], name='valid_labels_input')

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
        train_op, lr = self.cnn.train(self.train_loss, self.global_steps)

        self.train_op = train_op
        self.learning_rate = lr
        self.logits = logits

    def train(self, trainDataGenerater, validDataGenerater, **kwargs):
        """
        训练神经网络
        :param trainDataGenerater: 训练数据集的产生对象
        :param validDataGenerater: 验证数据集的对象
        :return:
        """
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
            # 验证数据
            images_valid, labels_valid = validDataGenerater.generate_augment_batch()
            for epoch in range(startstep // batches_per_epoch, self.max_epoch):
                # 训练一次
                for batch_counter in range(batches_per_epoch):
                    # 获取训练的数据
                    images_batch, labels_batch = trainDataGenerater.generate_augment_batch()
                    # 训练数据
                    _, loss_, lr, merged_value, global_step_ = sess.run([self.train_op, self.train_loss, self.learning_rate, merged, self.global_steps],
                        feed_dict={self.train_images_input: images_batch, self.train_labels_input: labels_batch})
                    valid_acc = sess.run(self.train_accuracy, feed_dict={self.valid_images_input: images_valid, self.valid_labels_input: labels_valid})
                    # 这个主要区别的是准确率输出的时机
                    if (global_step_ + 1) % self.display_per_global_steps == 0:
                        ## 检查准确率
                        train_acc = sess.run(self.train_accuracy,
                                             feed_dict={self.train_images_input: images_batch, self.train_labels_input: labels_batch})
                        pprint(
                            'Epoch:{e:3d}, Global Step:{step:5d}, Train Acc:{tacc:.3f}, Validation Acc:{vacc: .3f}, Learning rate:{lr:.5f}, Loss:{loss:.3f}'.format(
                                step=global_step_, tacc=train_acc, lr=lr, loss=loss_, e=epoch, vacc=valid_acc))
                        # 保存模型
                        saver.save(sess, model_saved_file, global_step=self.global_steps)

                    train_writer.add_summary(merged_value, global_step=global_step_)
            train_writer.close()

class KerasTrainer(Trainer):

    """
    正对于 Keras 模型的训练器
    """

    def __init__(self, model, max_epochs):
        if not isinstance(model, cifar10_buildNet2.KerasCNNNetwork):
            raise ValueError('请使用 KerasCNNNetwork 子类作为 CNN 的参数')
        self.model = model  # Keras 模型
        self.max_epochs = max_epochs
        super(KerasTrainer, self).__init__(logout_prefix=os.sep.join(('logouts', str(model))),
                                                model_saved_prefix=os.sep.join(('models', str(model))))

    def train(self, trainDataGenerater=None, validDataGenerater=None, **kwargs):
        import keras
        from keras.optimizers import Adam
        from keras.callbacks import ModelCheckpoint, LearningRateScheduler
        from keras.callbacks import ReduceLROnPlateau
        from keras.datasets import cifar10
        from keras.preprocessing.image import ImageDataGenerator

        # 准备保存的数据点
        checkpoints = ModelCheckpoint(filepath=os.sep.join([self.model_saved_prefix, 'cifar10_ResNet_checkpoints.h5']),
                                      monitor='val_acc',
                                      verbose=1,
                                      save_best_only=True)
        lr_changer = self.model.learn_rate_changer()
        lr_scheduler = LearningRateScheduler(lr_changer)  # 学习率衰减的调用器
        lr_reducer = ReduceLROnPlateau(factor=self.model.learning_rate_decay, cooldown=0, patience=5, min_lr=.5e-6)
        callbacks = [checkpoints, lr_reducer, lr_scheduler]  # 回调顺序
        # 定义数据
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        # 均1化数据 float -> 0 ~ 255
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
        # 数据增强
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
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
                                 verbose=1,
                                 workers=4,
                                 callbacks=callbacks)




