import numpy as np
import sys
sys.path.append('..')
from util import print_matrix, calculate_rank


def PLU_Factorization(A):
    # 输入一个 n*n 的方阵，使用numpy格式
    # 输出PLU分解后的三个矩阵，分别是 P， L， U
    A = A.copy()
    if A is None or (A.shape[0] != A.shape[1]):  # 判断是否为方阵
        return 0, 0, 0    # 不满足条件，直接返回，返回数字为了后续判断
    n = A.shape[0]   # 矩阵的维度n
    if calculate_rank(A) != n:  # 判断矩阵是否可逆
        return 1, 1, 1    # 不满足条件，直接返回，返回数字为了后续判断
    P = np.eye(n)    # 交换矩阵P
    for j in range(n):
        index = np.argmax(np.abs(A[j:n, j]))   # 找到主元最大的行
        A[j, :], A[j+index, :] = A[j+index, :].copy(), A[j, :].copy()    # 交换行
        P[j, :], P[j+index, :] = P[j+index, :].copy(), P[j, :].copy()    # 同时交换矩阵P
        for i in range(j+1, n):    # 对主元下面的行进行消元操作
            a = A[i, j] / A[j, j]       # 计算消元系数
            A[i, j:] = A[i, j:] - a * A[j, j:]      # 消元
            A[i, j] = a     # 记录消元系数
    L = np.eye(n) + np.tril(A, -1)   # 最后L为A的下三角（去除主对角线）加上一个单位阵
    U = np.triu(A)  # U为A的上三角及其主对角线
    return P, L, U


def main():
    # 测试样例
    A = np.matrix([[1, 2, 4, 17], [3, 6, -12, 3], [2, 3, -3, 2], [0, 2, -2, 6]]).astype(float)
    # A = np.matrix([[1, 2, 4, 17], [3, 6, -12, 3], [2, 3, -3, 2], [1, 2, 4, 17]]).astype(float)
    P, L, U = PLU_Factorization(A.copy())   # 求P, L, U分解的结果（直接输入A会被修改）
    if type(P) is not np.ndarray  or type(L) is not np.ndarray or type(U) is not np.ndarray:
        if P == 0:
            print("输入必须为方阵！")
        elif P == 1:
            print("输入矩阵必须可逆！")
        return 
    print("A矩阵：")
    print_matrix(A)
    print("交换矩阵P：")
    print_matrix(P)
    print("L矩阵：")
    print_matrix(L)
    print("U矩阵：")
    print_matrix(U)
    print("验证正确性：(A=P^T*L*U)")
    print_matrix(P.T @ L @ U)


if __name__ == '__main__':
    main()