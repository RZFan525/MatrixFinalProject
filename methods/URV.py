import numpy as np
import sys
sys.path.append('..')
from util import print_matrix, calculate_rank
from methods.Householder_Reduction import house_holder


def URV(A):
    # 输入一个m*n 的矩阵，使用numpy格式
    # 输出URV分解产生的R(m*m),R(m*n),V(n*n)矩阵
    m, n = A.shape  #  矩阵的形状
    # 首先求矩阵A的秩
    r = calculate_rank(A)
    print("矩阵的秩为：", r)

    A = A.copy()
    Q1, R1 = house_holder(A)   # 使用householder约简得到 PA = (B, 0)^T
    P = Q1.T
    B = R1[:r, :]    # B是r*n的矩阵
    Q2, R2 = house_holder(B.T)     # 然后对B^T做householder约简，得到QB^T = (T, 0)^T
    Q = Q2.T
    T = R2[:r, :]     # T是 r*r的矩阵
    # 最终可得到 A = P^T (T^T 0 \\ 0 0) Q
    # 对应A = U R V^T  可得   U = P.T , R =  (T^T 0 \\ 0 0) , V = Q.T
    U = P.T
    R = np.zeros((m, n))
    R[:r, :r] = T.T
    V = Q.T
    return U, R, V



def main():
    # 测试样例
    # A = np.matrix([[4, -3, 4], [2, -14, -3], [-2, 14, 0], [1, -7, 15]]).astype(float)
    # A = np.matrix([[4, -3], [2, -14], [-2, 14], [1, -7]]).astype(float)
    # A = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(float)
    A = np.matrix([[-4, -2, 4,  2], [2, -2, -2, -1], [-4, 1, 4, 2]]).astype(float)
    print("A矩阵：")
    print_matrix(A)
    U, R, V = URV(A)
    print("U矩阵：")
    print_matrix(U)
    print("R矩阵：")
    print_matrix(R)
    print("V矩阵：")
    print_matrix(V)
    print("验证正确性(U*R*V^T)：")
    print_matrix( U @ R @ V.T)


if __name__ == "__main__":
    main()