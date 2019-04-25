import numpy as np
import tensorflow as tf

indices = []
values = []
sequences = [
    ['1','2','3'],
    ['1','2'],
    ['1']
]

for n, seq in enumerate(sequences):
    indices.extend(zip([n] * len(seq), range(len(seq))))
    values.extend(seq)

indices = np.asarray(indices, dtype=np.int64)
print(indices.max(0))
values = np.asarray(values, dtype=np.int32)
shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
print(tf.SparseTensor(indices,values,shape))

dense_matrix = 5*np.ones(shape,dtype=np.int32)
dense_matrix[0,1] = 2
print("dm:",dense_matrix)


