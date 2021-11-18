import numpy as np
import sys
sys.path.append('..')
from util import print_matrix, calculate_rank


def det(A):
    # 求矩阵A的行列式，使用PLU分解求，P的det根据交换次数来算，交换次数为偶数则是1，为奇数则是-1
    # L的det是1， U的det是主对角线元素的乘积
    # PA = LU, A = P^{-1}LU = PLU
    # det(A) = det(PLU) = det(P)det(U) = 正负1 * U的对角线元素乘积
    A = A.copy()
    if A is None or (A.shape[0] != A.shape[1]):  # 判断是否为方阵
        return "error"
    n = A.shape[0]   # 矩阵的维度n
    if calculate_rank(A) != n:  # 判断矩阵是否可逆
        return 0
    P = 0  # 交换次数
    for j in range(n):
        index = np.argmax(np.abs(A[j:n, j]))   # 找到主元最大的行
        if index != 0:
            A[j, :], A[j+index, :] = A[j+index, :].copy(), A[j, :].copy()    # 交换行
            P += 1   # 记录交换次数
        for i in range(j+1, n):    # 对主元下面的行进行消元操作
            a = A[i, j] / A[j, j]       # 计算消元系数
            A[i, j:] = A[i, j:] - a * A[j, j:]      # 消元
    # P是交换次数，交换奇数次是-1，交换偶数次是1，所以A的行列式值就是R的对角线元素乘符号
    Det = np.prod(np.diag(A)) * ((-1) ** P)
    return Det


def main():
    A = np.matrix([[2, 3], [2, 3]]).astype(float)
    d = det(A)
    if d == "error":
        print("矩阵不是方阵")
        return
    print("矩阵A：")
    print_matrix(A)
    print("矩阵A的行列式等于：", d)


if __name__ == '__main__':
    main()