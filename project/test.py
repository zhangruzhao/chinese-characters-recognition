import tensorflow as tf
sess = tf.Session()
image_batch = tf.constant([
    [
        [[1],[1],[1]],
        [[2],[2],[2]],
        [[3],[3],[3]]
    ],
    [
        [[1],[1],[1]],
        [[2],[2],[2]],
        [[3],[3],[3]]
    ]
])
batch_size = 2
print(image_batch.get_shape())
image_list = tf.split(axis=0,value=image_batch,num_or_size_splits=batch_size)

image_slice_batch_list = []
for i in range(3):
    image_slice_expend_list = []
    for batch_ind in range(batch_size):
        image_slice = tf.slice(image_list[batch_ind][0],[0,i*1,0],[3,1,1])
        #print(sess.run(image_slice),image_slice.get_shape())
        image_slice_expend = tf.expand_dims(image_slice,0)#4d
        #print(sess.run(image_slice_expend),image_slice_expend.get_shape())

        image_slice_expend_list.append(image_slice_expend)
    image_slice_batch = tf.concat(axis=0,values=image_slice_expend_list)
    image_slice_batch_list.append(image_slice_batch)
    #print(image_slice_batch)
print(image_slice_batch_list)
    
sess.close()
