import numpy as np
from util import print_matrix, read_matrix
from methods.LU import PLU_Factorization
from methods.Gram_Schmidt import gram_schmidt
from methods.Givens_Reduction import givens
from methods.URV import URV
from methods.Householder_Reduction import house_holder


def main():
    print("请输入需要读取数据的文件名，直接回车默认为'data.txt':")
    file_path = input()
    if file_path == "":
        data = read_matrix("data.txt")
    else:
        data = read_matrix(file_path)
    print("请选择需要执行的方法")


if __name__ == "__main__":
    main()