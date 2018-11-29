# Cifar-10 的 CNN
### 依赖
- Tensorflow
- keras(图像数据增广和 Resnet20 的实现)
## 数据集下载
- [cifar-10-python.tar.gz](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
- [官方数据说明](http://www.cs.toronto.edu/~kriz/cifar.html)
## 如何运行
1. 下载 Cifar10 数据集解压 **数据文件** 到 `cifar10_data` 目录下
2. 训练：
- `cifar10_train_vgg19_tf.py`: 使用 Tensorflow 训练 VGG19 模型
- `cifar10_train_resnet20v2.py`: 使用 Keras 训练 Resnet20v2 模型
3. 恢复：
### VGG19
运行 `cifar10_loadModel_vgg19_tf.py`：加载训练好的模型，可以:
- 卷积核过滤器的可视化：使用方法 `displayConvWeights`
- 卷积层可视化：使用方法 `displayConvLayers`
- 测试集上的准确率：使用方法 `evaulateOnTest`
### Resnet20
待完善
## 实现的功能
1. 支持从模型中恢复(用于继续训练和测试)
2. 定义训练器，专注于构建网络
3. 可以输出卷积核图像（仅 VGG19）
## 待完善
1. Resnet20 的测试集正确率不高
## 参考
- [CNN 中如何处理RGB图像](https://blog.csdn.net/sscc_learning/article/details/79814146)
- [可供参考的CNN模型-1](https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c)
- [可供参考的CNN模型-2](https://www.jianshu.com/p/4ed7f7b15736)
- [可视化卷积核](http://nooverfit.com/wp/%E7%94%A8tensorflow%E5%8F%AF%E8%A7%86%E5%8C%96%E5%8D%B7%E7%A7%AF%E5%B1%82%E7%9A%84%E6%96%B9%E6%B3%95/)
## P.S
如果提示找不到 homework 模块，可以将其删除（因为单独运行不带有子模块的性质）