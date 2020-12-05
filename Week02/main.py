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



a = np.array([[[1, 2],
              [3, 4]]])
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


a = np.array([[[1, 2],
              [3, 4]]])
# zero_like
c = np.zeros_like(a)
print("shape : {0}  value : {1}".format(c.shape, c))


# 긴 텐서 만들기
d = np.arange(32)
d = d.reshape(4, -1)

print(d)


# 텐서 값 가져오기
d = np.arange(32)
d = d.reshape(4, -1)

res = d[1:3, 1:3]

print(res)


# 텐서 곱셈
a = np.arange(256)
a = a.reshape(4, 4, 16)

b = np.arange(256)
b = b.reshape(4, 16, 4)

res = np.matmul(a, b)

print(res)

