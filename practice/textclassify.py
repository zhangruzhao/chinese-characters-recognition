import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras


imdb = keras.datasets.imdb
(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words = 10000)#the most common 10000word
#train_data,labels,test_data,labels are lists
word_index = imdb.get_word_index()
#add four word into reviews text
for k,v in word_index.items():
    word_index[k] = v+3
word_index['<pad>'] = 0
word_index['<start>'] = 1
word_index['<unk>'] = 2
word_index['<unused>'] = 3

#reverse word and correspongding integer
reverse_word_index = dict([(value,key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'!') for i in text])
#print(decode_review(train_data[0]))

#data_preprocess
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value = word_index['<pad>'],
                                                        padding = 'post',
                                                        maxlen = 256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value = word_index['<pad>'],
                                                        padding = 'post',
                                                        maxlen = 256)

#build model
model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16))
model.add(keras.layers.GlobalAveragePooling1D())#???
model.add(keras.layers.Dense(16,activation = tf.nn.relu))
#model.add(keras.layers.Dropout(0.6))
model.add(keras.layers.Dense(1,activation = tf.nn.sigmoid))
model.summary()

model.compile(optimizer = tf.train.AdamOptimizer(),
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])

#create validation set
real_train_data = train_data[1000:]
real_train_labels = train_labels[1000:]

valid_data = train_data[:1000]
valid_labels = train_labels[:1000]


history = model.fit(real_train_data, real_train_labels, epochs = 20,
                    batch_size = 512, validation_data = (valid_data,valid_labels),
                    verbose = 1)
#history contains a dict

results = model.evaluate(test_data,test_labels)
print(results)

history_dict = history.history

valid_loss = history_dict['val_loss']
valid_acc = history_dict['val_acc']
acc = history_dict['acc']
loss = history_dict['loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, loss, 'bo', label = 'train_loss')
plt.plot(epochs, valid_loss, 'b', label = 'validation_loss')
#bo for blue pot and b for solid blue line
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

plt.clf()
plt.plot(epochs, acc, 'bo', label = 'train_accuracy')
plt.plot(epochs, valid_acc, 'b', label = 'validation_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()