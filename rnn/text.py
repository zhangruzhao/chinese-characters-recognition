import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file).read()
print("Lenth of text is:{}".format(len(text)))

vocab = sorted(set(text))

char2idx = {u : i for i, u in enumerate(vocab)}
#print(char2idx)
idx2char = np.array(vocab)
#print(idx2char)

text_as_int = np.array([char2idx[c] for c in text])
# for char,_ in zip(char2idx,range(20)):
#     print("{:6s}---->{:4d}".format(char,char2idx[char]))

seq_size = 100
chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_size+1,drop_remainder = True)

# for item in chunks.take(5):
#     print(repr(''.join(idx2char[item])))

def split_input_target(chunks):
    input_text = chunks[:-1]
    target_text = chunks[1:]
    return input_text,target_text
dataset = chunks.map(split_input_target)
for input_example, target_example in dataset.take(1):
    print('input data:',''.join(idx2char[input_example]))
    print('target_data:',''.join(idx2char[target_example]))

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)


# 嵌入层：一个可训练的对照表，它会将每个字符的数字映射到具有 embedding_dim 个维度的高维度向量；
# GRU 层：一种层大小等于单位数的 RNN。（在此示例中，您也可以使用 LSTM 层。）
# 密集层：带有 vocab_size 个单元
class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Model, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(self.units,
                                                return_sequences = True,
                                                recurrent_initializer = 'glorot_uniform',
                                                stateful = True)
        else:
            self.gru = tf.keras.layers.GRU(self.units,
                                            return_sequences = True,
                                            recurrent_activation = 'sigmoid',
                                            recurrent_initializer = 'glorot_uniform',
                                            stateful = True)
        
        self.fc = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x):
        embedding = self.embedding(x)
        output = self.gru(embedding)
        prediction = self.fc(output)
        return prediction

vocab_size = len(vocab)
embedding_dim = 256
units = 1024
model = Model(vocab_size,embedding_dim,units)

optimizer = tf.train.AdamOptimizer()

def loss_function(real, prediction):
    return tf.losses.sparse_softmax_cross_entropy(labels = real, logits = prediction)

model.build(tf.TensorShape([BATCH_SIZE, seq_size]))
model.summary()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Epochs = 5
# for epoch in range(Epochs):
#     start = time.time()
#     #initializing the hidden state at the start of every epoch
#     #intially hidden is None
#     hidden = model.reset_states()

#     for (batch, (inp, target)) in enumerate(dataset):
#         with tf.GradientTape() as tape:
#             #feeding the hidden state back into the model
#             predictions = model(inp)
#             loss = loss_function(target, predictions)

#             grads = tape.gradient(loss, model.variables)
#             optimizer.apply_gradients(zip(grads, model.variables))
#         if batch % 100 == 0:
#             print('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1, batch, loss))

#     if (epoch+1)%5 == 0:
#         model.save_weights(checkpoint_prefix)
    
#     print('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
#     print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model = Model(vocab_size, embedding_dim, units)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1,None]))

num_generate = 1000
start_string = 'Q'

input_eval = [char2idx[i] for i in start_string]
input_eval = tf.expand_dims(input_eval, 0)

text_generate = []
temperature = 1.0

model.reset_states()
for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)

    predictions = predictions/temperature
    predictions_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
    #print(predictions_id)

    input_eval = tf.expand_dims([predictions_id], 0)
    text_generate.append(idx2char[predictions_id])
    
print(start_string,''.join(text_generate))