import tensorflow as tf

sess = tf.Session()
queue = tf.FIFOQueue(20,tf.uint8)
line_num = tf.Variable(0,dtype=tf.uint8)
increment_op = tf.assign_add(line_num,1)
example_op = queue.enqueue(line_num)

qr = tf.train.QueueRunner(queue,enqueue_ops=[increment_op,example_op]*5)
tf.train.add_queue_runner(qr)
out = queue.dequeue()

coord = tf.train.Coordinator()
sess.run(tf.initialize_all_variables())
threads = tf.train.start_queue_runners(sess = sess,coord = coord)

for i in range(10):
        if not coord.should_stop():
                print(sess.run(out),sess.run(line_num))

coord.request_stop()
coord.join(threads)
sess.close()

