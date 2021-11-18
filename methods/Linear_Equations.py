import numpy as np
import sys
sys.path.append('..')
from util import print_matrix, calculate_rank
from methods.Householder_Reduction import house_holder


def linear_equations(A, b):
    # 使用QR分解求线性方程组Ax=b
    # 输入是一个 m*n 的系数矩阵A，以及m*1的向量b
    # 输出是 n*1的向量，表示n个未知数x的值
    # 先使用 householder约简求出Q，R
    Q, R = house_holder(A)
    # R可能存在全0行，将其去除掉，同时在Q中去掉对应的线性相关列
    mask = (np.abs(R) <= 1e-8)    # 判断每个元素是否是0
    p = np.where(mask.all(axis=1))[0]   # 找到全0行的index
    R = np.delete(R, p, 0)    # R矩阵删除全0的行
    Q = np.delete(Q, p, 1)    # Q矩阵删除全0的列
    # Ax = b  =>  QRx=b  =>  Rx = Q^T b 
    b = Q.T @ b   
    m, n = R.shape
    
    x = np.zeros(n)
    # 自下向上逐个求x_n, x_{n-1}, ... , x_2, x_1
    for i in range(m-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= x[j] * R[i, j]
        x[i] /= R[i, i]
    return x


def main():
    A = np.matrix([[1, 2, 4, 17], [3, 6, -12, 3], [2, 3, -3, 2], [0, 2, -2, 6]]).astype(float)
    b = np.array([17, 3, 3, 4]).astype(float)
    # A = np.matrix([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]).astype(float)
    # b = np.array([2, 7, 9, 12, 13, 14, 14, 13, 10, 8, 4]).astype(float)
    # A = np.matrix([[1, -5, 25], [1, -4, 16], [1, -3, 9], [1, -2, 4], [1, -1, 1], [1, 0, 0], [1, 1, 1], [1, 2, 4], [1, 3, 9], [1, 4, 16], [1, 5, 25]]).astype(float)
    x = linear_equations(A, b)
    print("矩阵A：")
    print_matrix(A)
    print("b: ")
    print_matrix(b)
    print("x: ")
    print_matrix(x)


if __name__ == '__main__':
    main()