import tensorflow as tf

def inference(X):
    #推断x的输出，将结果返回
def loss(X,Y):
    #计算训练数据X及其期望输出Y计算损失
def inputs():

def train(total_loss):
    #依据计算的损失训练调整模型参数

def evaluate(sess, X, Y):
    #评估

saver = tf.train.Saver()

with tf.Session() as sess:
    #模型设置
    tf.global_variables_initializer().run()

    X, Y = inputs()
    total_loss = loss(X,Y)
    train_op = train(total_loss)

    coord = tf.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    initial_step = 0
    training_step = 1000
    #ckpt_dirname = ""

    #检验是否有检查点，如果有恢复参数
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    #ckpt = tf.train.get_checkpoint_state(ckpt_dirname)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-',1)[1])

    for step in range(initial_step, training_step):
        sess.run([train_op])
        if step%10 == 0:
            #储存参数
            saver.save(sess,'my_model', global_step = step)
            #saver.save(sess,ckpt_dirname + 'my_model', global_step = step)
            print("loss: ",sess.run([total_loss]))
    evaluate(sess,X,Y)
    saver.save(sess, 'my_model', global_step = training_step)

    coord.request_stop()
    coord.join(threads)