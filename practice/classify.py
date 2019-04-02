import tensorflow as tf
import numpy as np 
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print(train_labels)
#preprocess
train_images = train_images/255.0
test_images = test_images/255.0

#set up model
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),#将二维压成一维数据
    keras.layers.Dense(128,#kernel_regularizer = keras.regularizers.l1(0.001),
                        activation = tf.nn.relu),
    #keras.layers.Dropout(0.1),
    keras.layers.Dense(10,activation = tf.nn.softmax)
])

model.compile(optimizer = tf.train.AdamOptimizer(),
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])
#train
model.fit(train_images,train_labels,epochs = 8)

#evaluate
test_loss,test_accuracy = model.evaluate(test_images,test_labels)
print('Test accuracy:',test_accuracy) #seems overfitting

predictions = model.predict(test_images)
print(predictions[0])
print(test_labels[0])

def plot_images(i,predictions_array,true_labels,img):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img,cmap = plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_name[predicted_label],
                                    100*np.max(predictions_array),
                                    class_name[true_label]),
                                    color = color)

def plot_bar(i,predictions_array,true_labels):
    predictions_array, true_label = predictions_array[i], true_labels[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10),predictions_array,color = '#666666')
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#plot figure
num_rows = 5
num_cols = 3
num_images = num_cols*num_rows
plt.figure(figsize=(2*2*num_cols, 2*num_rows)) #??

for i in range(num_images):
    plt.subplot(num_rows,2*num_cols,2*i+1)
    plot_images(i,predictions,test_labels,test_images)
    plt.subplot(num_rows,num_cols*2,2*i+2)
    plot_bar(i,predictions,test_labels)

plt.show()
