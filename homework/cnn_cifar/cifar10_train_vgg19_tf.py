from homework.cnn_cifar.cifar10_buildNet2 import VGGNetwork
from homework.cnn_cifar.cifar10_loadFile import AugmentImageGenerator
from homework.cnn_cifar.cifar10_trainer import TensorflowTrainer


cnnnet = VGGNetwork(True, 50000, init_learning_rate=0.001, learning_rate_decay=0.3, num_epoch_per_decay=10)

data = AugmentImageGenerator(False, cnnnet.batch_size)
testData = AugmentImageGenerator(True, cnnnet.batch_size)  # 用作 验证集合

trainObj = TensorflowTrainer(cnnnet, max_epoch=100)
trainObj.train(data, testData)