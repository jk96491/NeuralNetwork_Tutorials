import numpy as np

mat_A = np.array([[1, 2],
                  [3, 4]])

mat_B = np.array([[3, 6],
                  [1, 2]])

result = np.matmul(mat_A, mat_B)


def mat_add(mat_A, mat_B):
    result = None
    result = mat_A + mat_B
    return result


def mat_subtract(mat_A, mat_B):
    result = None
    result = mat_A - mat_B
    return result


def mat_multiply(mat_A, mat_B):
    result = None
    result = np.matmul(mat_A, mat_B)
    return result

result = mat_multiply(mat_A, mat_B)

print(result)