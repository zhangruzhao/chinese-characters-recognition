# import numpy as np
# a = ['ab','cd']
# b = [x.encode() for x in a]
# print(b)
# a_bytes = ''.join(a).encode()
# print(a_bytes)

for i in range(10):
    if i%4 < 2 :
        t = i/4
    else:
        t = i/4+1
    print(t)