from homework.cnn_cifar.cifar10_loadFile import Cifar10TestDataGenerator
from homework.cnn_cifar.cifar10_buildNet2 import VGGNetwork
from homework.cnn_cifar.cifar10_modelLoader import TensorflowModelLoader


cnnnet_test = VGGNetwork(False, num_examples_per_epoch=10000, displayKernelOnTensorboard=True)
data = Cifar10TestDataGenerator(cnnnet_test.batch_size)  # 显示一张图
testObj = TensorflowModelLoader(cnnnet_test)
testObj.displayConvWeights(data)  # displayConv matplotlib 可能必须要阻塞显示.很奇怪!
# testObj.evaulateOnTest(data)
del testObj  # 显式释放对象(关闭 tf 会话)
