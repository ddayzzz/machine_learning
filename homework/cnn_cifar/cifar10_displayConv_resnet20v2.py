from homework.cnn_cifar.cifar10_buildNet2 import KerasResNetwork
from homework.cnn_cifar.cifar10_modelLoader import KerasModelLoader


cnnnet = KerasResNetwork(batch_size=32, num_classes=10, init_learning_rate=1e-3, steps_per_epoch=50000)
trainObj = KerasModelLoader(cnnnet)
trainObj.evaulateOnTest(verbose=1)