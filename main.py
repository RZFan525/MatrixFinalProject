import numpy as np
from util import print_matrix, read_matrix
from methods.LU import PLU_Factorization
from methods.Gram_Schmidt import gram_schmidt
from methods.Givens_Reduction import givens
from methods.URV import URV
from methods.Householder_Reduction import house_holder
from methods.Determinant import det
from methods.Linear_Equations import linear_equations


def choose_methods(index):
    if index == 0:
        return PLU_Factorization
    elif index == 1:
        return gram_schmidt
    elif index == 2:
        return house_holder
    elif index == 3:
        return givens
    elif index == 4:
        return URV
    elif index == 5:
        return det
    elif index == 6:
        return linear_equations


def main():
    print("请选择需要执行的方法编号:")
    print("0 : PLU分解(输入必须是可逆的方阵，输出分别是P,L,U矩阵)")
    print("1 : Gram_Schmidt正交化求QR分解(输入必须是列向量无关的矩阵,输出分别是Q,R矩阵)")
    print("2 : Householder约简求QR分解(输入为m*n的矩阵,输出分别是Q,R矩阵)")
    print("3 : Givens约简求QR分解(输入为m*n的矩阵,输出分别是Q,R矩阵)")
    print("4 : URV分解(输入为m*n的矩阵,输出分别是U,R,V矩阵)")
    print("5 : 求矩阵的行列式(输入必须为方阵,输出是行列式值)")
    print("6 : 求线性方程组Ax=b(输入必须为m*n的系数矩阵A和m*1的向量b,输出是x,若无解则输出最小二乘解)")

    try:
        index = int(input())
    except ValueError:
        print("请输入正确的方法编号!")
        return
    if index < 0 or index > 6:
        print("请输入正确的方法编号!")
        return
    print("请输入需要读取数据的文件名，直接回车默认为'data.txt':")
    file_path = input()
    if file_path == "":
        data = read_matrix("data.txt")
    else:
        data = read_matrix(file_path)
    
    if data is None:
        return
    data = np.matrix(data).astype(float)
    if index == 0:
        P, L, U = PLU_Factorization(data)
        if type(P) is not np.ndarray  or type(L) is not np.ndarray or type(U) is not np.ndarray:
            if P == 0:
                print("输入必须为方阵！")
            elif P == 1:
                print("输入矩阵必须可逆！")
            return 
        print("原矩阵：")
        print_matrix(data)
        print("交换矩阵P：")
        print_matrix(P)
        print("L矩阵：")
        print_matrix(L)
        print("U矩阵：")
        print_matrix(U)
        print("验证正确性：(A=P^T*L*U)")
        print_matrix(P.T @ L @ U)
    elif index == 1 or index == 2 or index == 3:
        method = choose_methods(index)
        Q, R = method(data)
        if type(Q) is not np.ndarray and type(Q) is not np.matrix:
            return 
        print("原矩阵：")
        print_matrix(data)
        print("Q矩阵：")
        print_matrix(Q)
        print("R矩阵：")
        print_matrix(R)
        print("验证正确性(Q*R)：")
        print_matrix( Q @ R)
    elif index == 4:
        U, R, V = URV(data)
        print("原矩阵：")
        print_matrix(data)
        print("U矩阵：")
        print_matrix(U)
        print("R矩阵：")
        print_matrix(R)
        print("V矩阵：")
        print_matrix(V)
        print("验证正确性(U*R*V^T)：")
        print_matrix( U @ R @ V.T)
    elif index == 5:
        D = det(data)
        if D == "error":
            print("矩阵不是方阵")
            return
        print("原矩阵：")
        print_matrix(data)
        print("矩阵A的行列式等于：", D)
    elif index == 6:      # 若求线性方程组则还需要输入b
        print("请输入需要读取Ax=b的b数据的文件名，直接回车默认为'b.txt':")
        b_path = input()
        if file_path == "":
            b = read_matrix("b.txt")
        else:
            b = read_matrix(b_path)
        if b is None:
            return
        # b = np.matrix(b).astype(float)
        # b = b.T
        x = linear_equations(data, b)
        print("系数矩阵：")
        print_matrix(data)
        print("b: ")
        print_matrix(b)
        print("x: ")
        print_matrix(x)
    


if __name__ == "__main__":
    main()