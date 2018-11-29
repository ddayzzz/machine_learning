from homework.cnn_cifar.cifar10_buildNet2 import KerasResNetwork
from homework.cnn_cifar.cifar10_modelLoader import KerasModelLoader
from math import sqrt

cnnnet = KerasResNetwork()
trainObj = KerasModelLoader(cnnnet, init_leanring_rate=1e-3)
trainObj.evaulateOnTest()