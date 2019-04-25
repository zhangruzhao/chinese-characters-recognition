

# ctc_loss(
#     labels,
#     inputs,
#     sequence_length,
#     preprocess_collapse_repeated=False,
#     ctc_merge_repeated=True,
#     time_major=True
# )
# inputs: 输入（训练）数据，是一个三维float型的数据结构[max_time_step , batch_size , num_classes]，当修改time_major = False时，[batch_size,max_time_step,num_classes]。
# 总体的数据流：
# image_batch
# ->[batch_size,max_time_step,num_features]->lstm
# ->[batch_size,max_time_step,cell.output_size]->reshape
# ->[batch_sizemax_time_step,num_hidden]->affine projection AW+b
# ->[batch_size*max_time_step,num_classes]->reshape
# ->[batch_size,max_time_step,num_classes]->transpose
# ->[max_time_step,batch_size,num_classes]

# 本文输入图片大小是(32,256)，则num_features是32，max_time_step是256 代表最大划分序列，其中cell.output_size == num_hidden，num_hidden及num_classes的值见下文常量定义

# labels：OCR识别结果的标签，是一个稀疏矩阵，下文训练数据生成部分会有相关解释

# sequence_length: 一维数据，[max_time_step,…,max_time_step]长度为batch_size,值为max_time_step

# 因此我们需要做的就是将图片的标签label（需要OCR出的结果），图片数据，以及图片的长度转换为labels，inputs，和sequence_length

#定义一些常量
#图片大小，32 x 256
OUTPUT_SHAPE = (32,256)

#训练最大轮次
num_epochs = 10000
#LSTM
num_hidden = 64
num_layers = 1

obj = gen_id_card()
num_classes = obj.len + 1 + 1  # 10位数字 + blank + ctc blank

#初始化学习速率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9

DIGITS='0123456789'
BATCHES = 10
BATCH_SIZE = 64
TRAIN_SIZE = BATCHES * BATCH_SIZE

def gen_text(self, is_ran=False):
        text = ''
        vecs = np.zeros((self.max_size * self.len))
        
        //唯一变化，随机设定长度
        if is_ran == True:
            size = random.randint(1, self.max_size)
        else:
            size = self.max_size
            
        for i in range(size):
            c = random.choice(self.char_set)
            vec = self.char2vec(c)
            text = text + c
            vecs[i*self.len:(i+1)*self.len] = np.copy(vec)
        return text,vecs


# 生成一个训练batch
def get_next_batch(batch_size=128):
    obj = gen_id_card()
    #(batch_size,256,32)
    inputs = np.zeros([batch_size, OUTPUT_SHAPE[1],OUTPUT_SHAPE[0]])
    codes = []

    for i in range(batch_size):
        #生成不定长度的字串
        image, text, vec = obj.gen_image(True)
        #np.transpose 矩阵转置 (32*256,) => (32,256) => (256,32)
        inputs[i,:] = np.transpose(image.reshape((OUTPUT_SHAPE[0],OUTPUT_SHAPE[1])))
        #标签转成列表保存在codes
        codes.append(list(text))
    #比如batch_size=2，两条数据分别是"12"和"1"，则targets [['1','2'],['1']]
    targets = [np.asarray(i) for i in codes]
    #targets转成稀疏矩阵
    sparse_targets = sparse_tuple_from(targets)
    #(batch_size,) sequence_length值都是256，最大划分列数
    seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1]

    return inputs, sparse_targets, seq_len


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
 
    return indices, values, shape

def get_train_model():
    inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]])
    
    #定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)
    
    #1维向量 序列长度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])
    
    #定义LSTM网络
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)
    
    shape = tf.shape(inputs)
    #[batch_size,256]
    batch_s, max_timesteps = shape[0], shape[1]
    
    #[batch_size*max_time_step,num_hidden]
    outputs = tf.reshape(outputs, [-1, num_hidden])
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                          num_classes],
                                         stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
    #[batch_size*max_timesteps,num_classes]
    logits = tf.matmul(outputs, W) + b
    #[batch_size,max_timesteps,num_classes]
    logits = tf.reshape(logits, [batch_s, -1, num_classes])
    #转置矩阵，第0和第1列互换位置=>[max_timesteps,batch_size,num_classes]
    logits = tf.transpose(logits, (1, 0, 2))
    
    return logits, inputs, targets, seq_len, W, b


def train():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                global_step,
                                                DECAY_STEPS,
                                                LEARNING_RATE_DECAY_FACTOR,
                                                staircase=True)
    logits, inputs, targets, seq_len, W, b = get_train_model()
    
    #tragets是一个稀疏矩阵
    loss = tf.nn.ctc_loss(labels=targets,inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)
    
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    
    #前面说的划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
    #还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    
    init = tf.global_variables_initializer()

    def report_accuracy(decoded_list, test_targets):
        original_list = decode_sparse_tensor(test_targets)
        detected_list = decode_sparse_tensor(decoded_list)
        true_numer = 0
        
        if len(original_list) != len(detected_list):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
            return
        print("T/F: original(length) <-------> detectcted(length)")
        for idx, number in enumerate(original_list):
            detect_number = detected_list[idx]
            hit = (number == detect_number)
            print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
            if hit:
                true_numer = true_numer + 1
        print("Test Accuracy:", true_numer * 1.0 / len(original_list))

    def do_report():
        test_inputs,test_targets,test_seq_len = get_next_batch(BATCH_SIZE)
        test_feed = {inputs: test_inputs,
                     targets: test_targets,
                     seq_len: test_seq_len}
        dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
        report_accuracy(dd, test_targets)
 
    def do_batch():
        train_inputs, train_targets, train_seq_len = get_next_batch(BATCH_SIZE)
        
        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
        
        b_loss,b_targets, b_logits, b_seq_len,b_cost, steps, _ = session.run([loss, targets, logits, seq_len, cost, global_step, optimizer], feed)
        
        print b_cost, steps
        if steps > 0 and steps % REPORT_STEPS == 0:
            do_report()
            save_path = saver.save(session, "ocr.model", global_step=steps)
        return b_cost, steps
    
    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        for curr_epoch in xrange(num_epochs):
            print("Epoch.......", curr_epoch)
            train_cost = train_ler = 0
            for batch in xrange(BATCHES):
                start = time.time()
                c, steps = do_batch()
                train_cost += c * BATCH_SIZE
                seconds = time.time() - start
                print("Step:", steps, ", batch seconds:", seconds)
            
            train_cost /= TRAIN_SIZE
            
            train_inputs, train_targets, train_seq_len = get_next_batch(BATCH_SIZE)
            val_feed = {inputs: train_inputs,
                        targets: train_targets,
                        seq_len: train_seq_len}
 
            val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)
 
            log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
            print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler, time.time() - start, lr))