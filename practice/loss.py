import tensorflow as tf
from numpy.random import RandomState

x = tf.placeholder(tf.float32,shape = (None,2),name = 'x_input')
y_ = tf.placeholder(tf.float32,shape = (None,1),name = 'y_input')

w1 = tf.Variable(tf.random_normal([2,1],stddev = 1, seed = 1))
y = tf.matmul(x, w1)

#define loss function
loss_less = 1
loss_more = 10
loss = tf.reduce_sum(tf.where(tf.greater(y_,y),loss_less*(y_-y),loss_more*(y-y_)))
training = tf.train.AdamOptimizer().minimize(loss)

dataset_size = 128
batch_size = 8
rdm = RandomState(1)#one dimension
#numpy.random.rand() 生成（0,1）区间数
X = rdm.rand(dataset_size,2)
Y = [[x1 + x2 + rdm.rand()/10.0 - 0.05] for x1,x2 in X]

initial = tf.global_variables_initializer()#tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(initial)
    for step in range(5000):
        start = (step*batch_size)%dataset_size
        end = min(batch_size + start,dataset_size)
        sess.run(training,feed_dict = {y_:Y[start:end], x:X[start:end]})
        if step%1000 == 0:
            print(step,sess.run(w1))