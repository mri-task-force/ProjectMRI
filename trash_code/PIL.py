import numpy as np
from PIL import Image

a = np.array(np.uint16([[500, 1000, 100], [123, 1, 13]]))
print(a)
im = Image.fromarray(a, mode='I;16')
print(list(im.getdata()))
# a = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 3, 0]], dtype=np.float)
# b = np.sum(a, axis=1)
# for i, x in enumerate(b):
#     b[i] = a[i, i] / x
# print(a)
# print(b)
# a[:,-1] = np.array([a[i, i] / np.sum(a[i,:]) for i in range(a.shape[0])])
# print(a)



# dict1 = {}
# dict1[7] = 1
# dict1[1] = 2
# dict1[1] += (1==1)
# dict1[3] = 5
# print(2 not in dict1)
# # items = dict1.items

# for key in sorted(dict1.keys()):
#     print(key)
# # for x, y in dict1.items():
# #     dict1[x] /= 100
# print(dict1)