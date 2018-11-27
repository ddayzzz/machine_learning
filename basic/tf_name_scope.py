"""
研究 tf 中 name_scope 和 variable_scope 的
"""
# 链接：https://www.zhihu.com/question/54513728/answer/181819324

import tensorflow as tf

with tf.name_scope('name_scope_x'):
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    var3 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var4 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var3.name, sess.run(var3))
    print(var4.name, sess.run(var4))
# 输出结果：
# var1:0 [-0.30036557]   可以看到前面不含有指定的'name_scope_x'
# name_scope_x/var2:0 [ 2.]
# name_scope_x/var2_1:0 [ 2.]  可以看到变量名自行变成了'var2_1'，避免了和'var2'冲突