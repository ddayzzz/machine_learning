from homework.cnn_cifar.cifar10_buildNet2 import KerasResNetwork
from homework.cnn_cifar.cifar10_modelLoader import KerasModelLoader
from math import sqrt

cnnnet = KerasResNetwork(batch_size=32,
                         num_classes=10,
                         init_learning_rate=1e-3,
                         regularizer_ratio=1e-4,
                         learning_rate_decay=sqrt(0.1))
trainObj = KerasModelLoader(cnnnet)
trainObj.evaulateOnTest(verbose=1)