import numpy as np


# 打印矩阵
np.set_printoptions(precision=2, suppress=True)  # 保留小数点2位，小数不用科学计数法
def print_matrix(mat):
    m, n = mat.shape
    for i in range(m):
        for j in range(n):
            print("%8.2f" % mat[i, j], end=' ')
        print()