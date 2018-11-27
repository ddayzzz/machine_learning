import tensorflow as tf
# 仅仅用来处理变量相关
state = tf.Variable(initial_value=0, name='counter')
one = tf.constant(1)  # 定义常量

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()  # 定义所有的变量
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))