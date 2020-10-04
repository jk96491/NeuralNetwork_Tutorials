import numpy as np

# scalar (0 - tensor)
a = np.array(1)
print("shape : {0}  value : {1}".format(a.shape, a))

# vector (1 - tensor)
a = np.array([1, 2])
print("shape : {0}  value : {1}".format(a.shape, a))

# matrix (2 - tensor)
a = np.array([[1, 2],
              [3, 4]])
print("shape : {0}  value : {1}".format(a.shape, a))

# 3 - Tensor
a = np.array([[[1, 2],
              [3, 4]]])
print("shape : {0}  value : {1}".format(a.shape, a))


# reshape
b = a.reshape(-1)
print("shape : {0}  value : {1}".format(b.shape, b))

b = a.reshape(1, -1)
print("shape : {0}  value : {1}".format(b.shape, b))

b = a.reshape(1, 1, -1)
print("shape : {0}  value : {1}".format(b.shape, b))

b = a.reshape(2, 2)
print("shape : {0}  value : {1}".format(b.shape, b))

# reshape and squeeze
b = a.reshape(1, -1).squeeze()
print("shape : {0}  value : {1}".format(b.shape, b))


# zero_like
c = np.zeros_like(a)
print("shape : {0}  value : {1}".format(c.shape, c))


# 긴 텐서 만들기
d = np.arange(256)
d = d.reshape((4, 4, -1))
d = np.zeros_like(d)

print(d)


# 긴 텐서 끼리의 곱셈

