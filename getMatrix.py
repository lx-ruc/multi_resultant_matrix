import numpy as np
from sympy import *

# A_path = "/Users/lixin/sjm/multi_resultant_matrix/A.txt"
# B_path = "/Users/lixin/sjm/multi_resultant_matrix/B.txt"
# E_path = "/Users/lixin/sjm/multi_resultant_matrix/E.txt"
# F_path = "/Users/lixin/sjm/multi_resultant_matrix/F.txt"
# e_path = "/Users/lixin/sjm/multi_resultant_matrix/xe.txt"
# f_path = "/Users/lixin/sjm/multi_resultant_matrix/xf.txt"
A_path = "/Users/lixin/sjm/multi_resultant_matrix/matrix/A_P.txt"
B_path = "/Users/lixin/sjm/multi_resultant_matrix/matrix/B_P.txt"
E_path = "/Users/lixin/sjm/multi_resultant_matrix/matrix/E_P.txt"
F_path = "/Users/lixin/sjm/multi_resultant_matrix/matrix/F_P.txt"
e_path = "/Users/lixin/sjm/multi_resultant_matrix/matrix/xe_P.txt"
f_path = "/Users/lixin/sjm/multi_resultant_matrix/matrix/xf_P.txt"
def getMatrix():
    A = np.loadtxt(A_path)
    B = np.loadtxt(B_path)
    E = np.loadtxt(E_path)
    F = np.loadtxt(F_path)
    e = np.loadtxt(e_path)
    f = np.loadtxt(f_path)
    return A,B,E,F,e,f

# A,B,E,F,e,f = getMatrix()
# print("A.shape = ",A.shape)
# print("B.shape = ",B.shape)
# print("E.shape = ",E.shape)
# print("F.shape = ",F.shape)
# print("e,shape = ",e.shape)
# print("f.shape = ",f.shape)

# print("A.rows = ",A.shape[0])

def getBlocks():
    A,B,E,F,e,f = getMatrix()
    A = A.T
    B = B.T
    M = np.block([
        [np.zeros((A.shape[0],B.T.shape[1])),-A,E.T,-E.T,np.zeros((A.shape[0],F.T.shape[1])),np.zeros((A.shape[0],F.T.shape[1]))],
        [-B.T,np.zeros((B.T.shape[0],A.shape[1])),np.zeros((B.T.shape[0],E.T.shape[1])),np.zeros((B.T.shape[0],E.T.shape[1])),F.T,-F.T],
        [-E,np.zeros((E.shape[0],A.shape[1])),np.zeros((E.shape[0],E.T.shape[1])),np.zeros((E.shape[0],E.T.shape[1])),np.zeros((E.shape[0],F.T.shape[1])),np.zeros((E.shape[0],F.T.shape[1]))],
        [E,np.zeros((E.shape[0],A.shape[1])),np.zeros((E.shape[0],E.T.shape[1])),np.zeros((E.shape[0],E.T.shape[1])),np.zeros((E.shape[0],F.T.shape[1])),np.zeros((E.shape[0],F.T.shape[1]))],
        [np.zeros((F.shape[0],E.shape[1])),-F,np.zeros((F.shape[0],E.T.shape[1])),np.zeros((F.shape[0],E.T.shape[1])),np.zeros((F.shape[0],F.T.shape[1])),np.zeros((F.shape[0],F.T.shape[1]))],
        [np.zeros((F.shape[0],E.shape[1])),F,np.zeros((F.shape[0],E.T.shape[1])),np.zeros((F.shape[0],E.T.shape[1])),np.zeros((F.shape[0],F.T.shape[1])),np.zeros((F.shape[0],F.T.shape[1]))]

    ])
    b = np.concatenate(
        (np.zeros(A.shape[0]),np.zeros(B.T.shape[0]),e,-e,f,-f)
    )

    return M,b

M,b = getBlocks()
print("M.shape = ",M.shape)
print("b.shape = ",b.shape)
# b = Matrix(b)
# M_rows,M_cols = M.shape
# x = symbols('x1:%d'%(M_rows+1))
# x = Matrix(x)
# M = Matrix(M)
# x_rows,x_cols = x.shape
# print("M_rows = ",M_rows,"M_cols = ",M_cols,"x_rows = ",x_rows,"x_cols = ",x_cols)

# f = x.T * (M+M.T)/2 *x + b.T * x
# g = x.T @ (M+M.T)/2 * x + b.T @ x