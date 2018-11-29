
from homework.cnn_cifar.cifar10_modelLoader import KerasModelLoader


trainObj = KerasModelLoader('models/KerasResNetwork20v2/checkpoints.h5')
trainObj.evaulateOnTest()