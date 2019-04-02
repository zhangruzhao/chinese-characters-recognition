import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#input data
x_data = np.random.rand(100).astype(np.float32)
noise = np.random.rand(100).astype(np.float32)*0.05
y_data = x_data*0.5 + 0.25 + noise

#plot input data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

#set up model
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

#optimize loss
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(400):
    sess.run(train)
    if step % 50 == 0:
        print(step,sess.run(Weights),sess.run(biases))

        #plot model line
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        lines = ax.plot(x_data,sess.run(y),'b-',lw=5)
        plt.pause(0.2)


