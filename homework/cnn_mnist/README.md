# CNN 的 MNIST 分类任务
使用了 Tensorflow 框架。
## 注意
- summary 没必要在 GPU 设备上跑。所以删除 `tf.device` 上下文。[参考](https://stackoverflow.com/questions/45876021/tensorflow-summary-ops-can-assign-to-gpu)
- 每次更新 `Summary`，需要删除之前的文件并重新启动 `tensorboard`。
## Bugs
- `Tensorboard` 中显示损失函数包括了 `validation` 和 `test` 的损失函数。
## MNIST 数据集
- [Lecun](http://yann.lecun.com/exdb/mnist/)
## 参考
- [(Document)Tensorboard Summary](https://www.tensorflow.org/guide/summaries_and_tensorboard)
