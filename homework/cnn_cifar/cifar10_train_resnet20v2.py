from homework.cnn_cifar.cifar10_buildNet2 import KerasResNetwork
from homework.cnn_cifar.cifar10_trainer import KerasTrainer
from math import sqrt


cnnnet  = KerasResNetwork()
trainObj = KerasTrainer(cnnnet,
                        max_epochs=200,
                        init_learning_rate=1e-3,
                        learning_rate_decay_rate=sqrt(0.1),
                        learning_rate_decay_per_epoch=None)  # 不定长的学习率递减
trainObj.train()