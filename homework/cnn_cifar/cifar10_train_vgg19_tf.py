from homework.cnn_cifar.cifar10_buildNet2 import VGGNetwork
from homework.cnn_cifar.cifar10_loadFile import Cifar10PreprocessedAugmentDataGenerator, Cifar10TestDataGenerator
from homework.cnn_cifar.cifar10_trainer import TensorflowTrainer


cnnnet = VGGNetwork(True, 50000, init_learning_rate=0.001, learning_rate_decay=0.5, num_epoch_per_decay=8)

data = Cifar10PreprocessedAugmentDataGenerator(cnnnet.batch_size)
testData = Cifar10TestDataGenerator(cnnnet.batch_size)  # 用作 验证集合
trainObj = TensorflowTrainer(cnnnet, max_epoch=200)
trainObj.train(data, testData)