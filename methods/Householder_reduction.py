import numpy as np
import sys
sys.path.append('..')
from util import print_matrix


def house_holder(A):
    # 输入一个n*m 的矩阵，使用numpy格式
    # 输出house_holder约减产生的Q(n*n),R(n*m)矩阵
    n, m = A.shape  #  矩阵的形状
    Q = np.identity(n)  # Q矩阵初始化为单位阵
    R = A.copy()  # R矩阵初始化为原始矩阵A
    for i in range(min(n-1, m)):   #  构造的反射矩阵的个数是 min(行数-1， 列数)
        u = R[i:, i].copy()        
        u[0] -= np.sqrt(np.power(u, 2).sum())   #  u = x - ||x||e_1
        r = np.identity(n)    # 将反射矩阵初始化为和R一样大的单位阵
        # 这里巧妙的将反射矩阵初始化为n*n，然后只对要进行约减的地方变换为反射矩阵，这样可以直接构造出n*n大小的反射矩阵
        norm = u.T @ u
        if norm != 0:
            r[i:, i:] -= 2 * (u @ u.T) / norm  # r = I - 2dd^T / d^Td
        # else:
            # r[i:, i:] -= 2 * (u @ u.T)
        R = r @ R    # 对原矩阵进行约减，最终会得到矩阵R
        Q = Q @ r    # 因为Q = (r_3 r_2 r_1)^T = r_1 r_2 r_3 ，所以只需要将按顺序将生成的反射矩阵相乘即可得到最终的Q
    return Q, R


def main():
    # 测试样例
    A = np.matrix([[4, -3, 4], [2, -14, -3], [-2, 14, 0], [1, -7, 15]]).astype(float)
    # A = np.matrix([[4, -3], [2, -14], [-2, 14], [1, -7]]).astype(float)
    # A = np.matrix([[1, 0, 0, 1, 0], [0, 1, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]).astype(float)
    Q, R = house_holder(A)
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