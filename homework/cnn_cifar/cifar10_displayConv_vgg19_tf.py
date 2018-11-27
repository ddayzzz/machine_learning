from homework.cnn_cifar.cifar10_loadFile import AugmentImageGenerator
from homework.cnn_cifar.cifar10_buildNet2 import VGGNetwork
from homework.cnn_cifar.cifar10_modelLoader import TensorflowModelLoader


cnnnet_test = VGGNetwork(False, num_examples_per_epoch=10000)
data = AugmentImageGenerator(True, 1)  # 显示一张图
testObj = TensorflowModelLoader(cnnnet_test)
testObj.displayConv(data)