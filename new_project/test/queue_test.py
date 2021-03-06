import tensorflow as tf
from preProcessing import Image

image = Image(r'/home/zhang/桌面/HIT-MW database/02 HIT-MW GT (binary)/黑白格式/b04010201.dgr')

sess = tf.Session()

coordinator = tf.train.Coordinator()
#定义队列大小和数据类型
queue = tf.FIFOQueue(100,tf.uint8)
#定义入队列的操作
step_num = image.get_lineNum()
line_num = tf.Variable(0,dtype=tf.uint8)
increment_op = tf.assign_add(line_num,1)
example = image.get_lineData(image.get_lineData(lineInd=line_num.value()[0]))

enqueue_op = queue.enqueue(example)
#定义出队操作
out_tensor = queue.dequeue()
input = out_tensor
#定义对输入的操作
train_op = image.priImg(out_tensor)

#申明一个qr，将传入队列参数和队列操作以及个数
qr = tf.train.QueueRunner(queue=queue, enqueue_ops=[increment_op,enqueue_op]*5)
#创建队列中的线程并运行
enqueue_threads = qr.create_threads(sess, coordinator, start = True)
for step in range(step_num):
    if coordinator.should_stop():
        break
    #线程操作
    sess.run(train_op)
    
#操作完成，申请停止线程
coordinator.request_stop()
#等待enqueue——threads中所有线程终止
coordinator.join(enqueue_threads)
sess.close()



    


