import numpy as np
import os


# 打印矩阵
np.set_printoptions(precision=2, suppress=True)  # 保留小数点2位，小数不用科学计数法
def print_matrix(mat):
    if len(mat.shape) == 2:
        m, n = mat.shape
        for i in range(m):
            for j in range(n):
                print("%8.2f" % mat[i, j], end=' ')
            print()
    elif len(mat.shape) == 1:
        n = mat.shape[0]
        for i in range(n):
            print("%8.2f" % mat[i], end=' ')
        print()

def calculate_rank(A):
    # 计算矩阵A的秩
    m, n = A.shape
    B = A.copy()
    for j in range(min(m, n)):
        if B[j, j] == 0:   # 若主元为0，则寻找下面最大的主元行与之交换
            index = np.argmax(np.abs(B[j:, j]))   # 找到主元最大的行
            B[j, :], B[j+index, :] = B[j+index, :].copy(), B[j, :].copy()    # 交换行
        if B[j, j] == 0:
            continue
        for i in range(j+1, m):    # 对主元下面的行进行消元操作
            if B[i, j] == 0:
                continue
            a = B[i, j] / B[j, j]       # 计算消元系数
            B[i, j:] = B[i, j:] - a * B[j, j:]      # 消元
    r = m   # 得到矩阵的秩r
    for i in range(m):
        if np.all(B[i, :] == 0):
            r -= 1
    return r


def read_matrix(file_path):
    # 从文件读入矩阵
    if not os.path.exists(file_path):
        print("请给出正确的文件路径")
        return
    data = np.loadtxt(file_path, dtype=np.float64, delimiter=' ')
    return data