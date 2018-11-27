"""
参照：https://blog.csdn.net/YE1215172385/article/details/79414702
验证SVD理解是否正确
"""
import numpy as np
from pprint import pprint

def main():
    A = np.array([[1,2],[3,4],[5,6],[0,0]])
    # 左奇异矩阵
    u = A.dot(A.T)
    mu_i, U = np.linalg.eig(u)
    # 右奇异矩阵
    v = A.T.dot(A)
    v_i, V = np.linalg.eig(v)
    pprint(v_i)
    pprint(V)
    # sigma，由V的特征值的算术平方根
    m = U.shape[0]
    n = V.shape[0]
    sigma = np.diag(np.zeros((m,n)))
    for i in range(V.)

main()