import tensorflow as tf

#建立一个lstm
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

rnn_layers = 5
stacked_rnn = tf.contrib.rnn.MultiRNNCell([lstm] * rnn_layers)

#初始化存储状态
state = stacked_rnn.zero_state(batch_size, tf.float32)
#初始化loss
loss = 0.0

num_steps = 10
words = tf.placeholder(tf.int32,[bacth_size,num_steps])


#截断反向传播
for i in range(len(num_steps)):
    #output, state = lstm(words[:,i],state)
    output, state = stacked_rnn(words[:,i],state)

    logits = tf.matmul(output * softmax_w) + softmax_b
    prodictions = tf.nn.softmax(logits)
    loss += loss_fun(prodictions, targets_words)
#每次输入num——steps长度后的state
final_state = state

num_state = stacked_rnn.zero_state(batch_size, tf.float32)
#每次处理一次batch更新状态和loss
for current_batch_of_words in words_in_dataset:

    num_state, num_loss = sess.run([final_state, loss],feed_dict = {state : num_state, words : current_batch_of_words})

    total_loss += num_loss
    
    #本次的输出和状态
    # output, state = lstm(current_batch_of_words,state)

    # #Lstm产生下一个词语的预测
    # logits = tf.matmul(output,softmax_w)+softmax_b
    # prodictions = tf.nn.softmax(logits)
    # loss += loss_fun(prodictions,targets_words)

embedding = tf.random_normal([vocabulary_size, embedding_size],-1,-1)
embed = tf.nn.embedding_lookup(embedding,words_id)

def _bidirectional_rnn(data, length):
    length_64 = tf.cast(length, tf.int64)
    


