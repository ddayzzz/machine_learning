import tensorflow as tf
from homework.cnn_cifar import cifar10_buildNet2
from pprint import pprint
import os
import numpy as np
from matplotlib import pyplot as plt

TENSORBOARD_OUTPUT_PATH = 'tf_log'
MODEL_PATH = 'tf_model'  # 保存的模型文件夹
MODEL_SAVE_FILE = os.sep.join((MODEL_PATH, 'saved_model.ckpt'))  # 保存的元文件路径

class Test(object):

    def __init__(self, cnnnet):
        if not isinstance(cnnnet, cifar10_buildNet2.CNNNetwork):
            raise ValueError('参数错误!')
        self.cnn = cnnnet

    def _input_placeholder(self):
        """
        创建输入的张量
        :return:
        """
        self.test_images_input = tf.placeholder(tf.float32, [
            self.cnn.batch_size,  # validation test train 的样本数量 batch_size 并不一致
            32,  # 图像大小
            32,
            3  # RGB
        ], name='test_images_input')
        # 输入的是真实标签
        self.test_labels_input = tf.placeholder(tf.int64, [self.cnn.batch_size], name='test_labels_input')

    def _build_compute_graph(self):
        """
        生成计算图
        :return:
        """
        # logit 输出
        self.logits = self.cnn.inference(self.test_images_input)
        # 计算准确率
        self.test_accuracy, self.test_equals = self.cnn.accuracy(self.logits, self.test_labels_input)


    def test(self, testDataGenerater):
        # 打开一个新的会话
        with tf.Session() as sess:
            # 定义输入的边拉
            self._input_placeholder()
            #　定义计算图
            self._build_compute_graph()
            # 恢复所有的变量
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
            batches_per_epoch = self.cnn.num_examples_per_epoch // self.cnn.batch_size  # 每进行一次 epoch 训练， 内循环迭代的次数. 也就是batch 的数量
            accuarcies = []
            # 测试过程
            for batch_counter in range(batches_per_epoch):
                images_batch, labels_batch = testDataGenerater.generate_augment_batch()
                test_accuarcy, test_equals, logits = sess.run([self.test_accuracy, self.test_equals, self.logits], feed_dict={self.test_images_input: images_batch, self.test_labels_input: labels_batch})
                pprint('Batch no {0}, Test Acc {1}'.format(batch_counter, test_accuarcy))
                # 错误的数据
                accuarcies.append(test_accuarcy)
                # 输出图像
                # error_indices = np.nonzero(test_equals.astype(np.uint8) == 0)[0]
                # error_examples_real_labels = labels_batch[error_indices]
                # error_examples_pred_labels = logits[error_indices]
                # if error_indices.shape[0] >= 5:
                #     for n in range(5):
                #         rlabel = testDataGenerater.label_names[error_examples_real_labels[n]]
                #         plabel = testDataGenerater.label_names[np.argmax(error_examples_pred_labels[n])]
                #
                #         img = images_batch[error_indices[n]] / 255.
                #         I = np.stack((img[:, :, 0], img[:, :, 1], img[:, :, 2]), axis=2)  # 新产生的合并的数据是 第3个维度. 32x32x3
                #         plt.title('预测标签：{0}，实际标签{1}'.format(plabel, rlabel))
                #         plt.imshow(I)
                #         plt.show()
            pprint('Average accuracy: {0:.3f}%'.format(100 * np.mean(np.array(accuarcies))))

cnnnet_test = cifar10_buildNet2.VGGNetwork(False, 10000)
from homework.cnn_cifar.cifar10_loadFile import AugmentImageGenerator
data = AugmentImageGenerator(True, cnnnet_test.batch_size)
testObj = Test(cnnnet_test)
testObj.test(data)