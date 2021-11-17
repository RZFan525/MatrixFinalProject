import numpy as np
import sys
sys.path.append('..')
from util import print_matrix


def givens(A):
    # 输入一个n*m 的矩阵，使用numpy格式
    # 输出Givens约减产生的Q(n*n),R(n*m)矩阵
    n, m = A.shape  #  矩阵的形状
    Q = np.identity(n)  # Q矩阵初始化为单位阵
    R = A.copy()  # R矩阵初始化为原始矩阵A
    for i in range(m):   #  对每一列做化简
        for j in range(i+1, n):     # 将主对角元素下面的每一个元素化为0
            if R[j, i] == 0:  # 若已经是0则跳过
                continue
            norm = np.sqrt(R[i, i] * R[i, i] + R[j, i] * R[j, i])  # 构造旋转矩阵，分别结算C和S
            c = R[i, i] / norm
            s = R[j, i] / norm
            P_ij = np.identity(n)    #  初始化旋转矩阵
            P_ij[i, i], P_ij[i, j], P_ij[j, i], P_ij[j, j] = c, s, -s, c  # 填入C和S
            R = P_ij @ R    # 对原矩阵进行约简
            Q = P_ij @ Q    # Q = (P23 P13 P12)^T
    Q = Q.T
    return Q, R


def main():
    # 测试样例
    A = np.matrix([[4, -3, 4], [2, -14, -3], [-2, 14, 0], [1, -7, 15]]).astype(float)
    # A = np.matrix([[4, -3], [2, -14], [-2, 14], [1, -7]]).astype(float)
    # A = np.matrix([[1, 0, 0, 1, 0], [0, 1, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]).astype(float)
    # A = np.matrix([[1, 19, -34], [-2, -5, 20], [2, 8, 37]]).astype(float)
    # A = np.matrix([[1, 0, 0, 1, 0], [0, 1, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]).astype(float)
    Q, R = givens(A)
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