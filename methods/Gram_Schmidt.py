import numpy as np
import sys
sys.path.append('..')
from util import print_matrix


def gram_schmidt(A):
    # 输入一个 n*m 的方阵，使用numpy格式
    # 输出二个矩阵，分别是 Q，R
    n, m = A.shape  #  矩阵的形状
    Q = np.zeros((n, m))    # 建立一个空的Q矩阵，形状为n*m
    R = np.zeros((m, m))    # 建立一个空的R矩阵，形状为m*m
    for i in range(m):      # 循环处理A的每一列
        q = np.array(A[:,i]).squeeze()   # 提出A的第i列
        for j in range(0, i):         # 循环对这一列减去他对前面已经计算出的正交向量的投影
            t = (Q[:,j]*q).sum()     # 求内积
            R[j, i] = t      # 内积赋到对应的R矩阵上
            q -= t * Q[:,j]    #  减去前面已经计算出的正交向量的投影
        norm = np.sqrt(np.power(q, 2).sum())    # 计算正交向量的模
        if norm != 0:     # 若模长不为0，则归一化
            q /= norm  
        R[i, i] = norm    # 模长赋到R矩阵对应的位置上
        Q[:,i] = q.ravel()   # 将计算好的正交向量赋到Q矩阵里
    return Q, R


def main():
    # 测试样例
    B = np.matrix([[0, -20, -14], [3, 27, -4], [4, 11, -2]]).astype(float)
    A = np.matrix([[4, -3, 4], [2, -14, -3], [-2, 14, 0], [1, -7, 15]]).astype(float)
    A = np.matrix([[1, 0, 0, 1, 0], [0, 1, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]).astype(float)
    Q, R = gram_schmidt(A)
    print("A矩阵：")
    print_matrix(A)
    print("Q矩阵：")
    print_matrix(Q)
    print("R矩阵：")
    print_matrix(R)
    print("验证正确性(Q*R)：")
    print_matrix( Q @ R)


if __name__ == "__main__":
    main()