import sympy as smp
from sympy import *
import numpy as np
import math
import copy
from decimal import Decimal
import numpy as np
import time
from mpmath import mp
from flint import acb_mat,ctx
import petsc4py
petsc4py.init()
from petsc4py import PETSc
from getMatrix import *
import concurrent.futures
start_time = time.perf_counter()
init_printing(use_unicode=True)
# 生成多项式的多重结果矩阵
# 输入多项式系统（n个变量n个方程），得到每个方程的degree：ri和系数、

# 2004年论文的第一个例子
# x,y,z = symbols('x y z')
# G1 = x**2 + y**2 + z**2 -1
# G2 = z - x**2 - y**2
# G3 = y - x**2 -z**2
# variables = [x,y,z]
# G = [
#     G1,
#     G2,
#     G3
# ] 
# x = Matrix([x,y,z])
# 2004年论文的第二个例子
# x,y,z = symbols('x y z')
# G1 = y**4 - 20/7*x**2
# G2 = x**2*z**4 + 7/10*x*z**4 + 7/48*z**4 -50/27*x**2 - 35/27*x - 49/216
# G3 = 3/5*x**6*y**2*z + x**5*y**3 + 3/7*x**5*y**2*z + 7/5*x**4*y**3 - 7/20*x**4*y*z**2 -3/20*x**4*z**3 + 609/1000*x**3*y**3 + 63/200*x**3*y**2*z - 77/125*x**3*y*z**2 -21/50*x**3*z**3 + 49/1250*x**2*y**3 +147/2000*x**2*y**2*z - 23863/60000*x**2*y*z**2 -91/400*x**2*z**3 -27391/800000*x*y**3 + 4137/800000*x*y**2*z -1078/9375*x*y*z**2 - 5887/200000*x*z**3 - 1029/160000*y**3 -24353/1920000*y*z**2 - 343/128000*z**3 
# variables = [x,y,z]

# G = [
#     G1,
#     G2,
#     G3
# ]

# 三层博弈树
# N = [
#     [0 ,0,0 ,0 ,0 ,0 ,0 ,1 ,-1, -1, 1, 0, 0, 0, 0],
#     [0 ,0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0],
#     [0 ,0 ,0 ,0 ,0 ,-10, 100, 0, 1, 0, -1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1],
#     [0, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1],
#     [0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1],
#     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
#     [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [-1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
# ]
# bb = [0,0,0,0,0,0,0,1,0,-1,0,1,0,-1,0]
# N = [
#     [0,0,0,0,0,1,-1,0,0],
#     [0,0,0,-10,100,1,-1,0,0],
#     [0,0,0,0,0,0,0,1,-1],
#     [0,-10,0,0,0,0,0,1,-1],
#     [0,-100,0,0,0,0,0,1,-1],
#     [-1,-1,0,0,0,0,0,0,0],
#     [1,1,0,0,0,0,0,0,0],
#     [0,0,-1,-1,-1,0,0,0,0],
#     [0,0,1,1,1,0,0,0,0]
# ]
# bb = [0,0,0,0,0,1,-1,1,-1]
# N = Matrix(N)
# bb = Matrix(bb)
# pprint(N)
# pprint(bb)

# N_rows,N_cols = N.shape 
# vars = symbols('x1:%d' % (N_rows+1))
# lamb = symbols('l1:%d' % (N_rows+1))
# lamb = Matrix(lamb)
# x = Matrix(vars)
# pprint(x)
# pprint(lamb)
# f = x.T * N * x + x.T * bb
# f = f[0]
# ST = N * x + bb
# pprint(ST)
# F = f + (lamb.T * ST)[0]
# variables = x.col_join(lamb)
# G = Matrix([F.diff(var) for var in variables])
# G_rows,G_cols = G.shape

# 四层博弈树
# NN = [
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,0,0,-1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,-1,-1,-1,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,6,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,8,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,10,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,12,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,14,0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,16,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,-1,1,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,-1,0,1,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,-1,0,-1,0,0,1,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,-1,-1,0,0,0,1],
#     [0,0,-16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0],
#     [0,0,-15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0],
#     [0,0,0,-14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0],
#     [0,0,0,-13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0],
#     [0,0,0,0,-12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0],
#     [0,0,0,0,-11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0],
#     [0,0,0,0,0,-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0],
#     [0,0,0,0,0,-9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0],
#     [0,0,0,0,0,0,-8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0],
#     [0,0,0,0,0,0,-7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0],
#     [0,0,0,0,0,0,0,-6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0],
#     [0,0,0,0,0,0,0,-5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0],
#     [0,0,0,0,0,0,0,0,-4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1],
#     [0,0,0,0,0,0,0,0,-3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1],
#     [0,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1],
#     [0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1],
#     [-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [1,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [1,0,0,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,1,0,0,0,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,1,0,0,0,0,0,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [-1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [-1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,-1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,-1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,1,0,0,0,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,-1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# ]
# bb = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,-1,0,0,0,0]

NN,bb = getBlocks()
N = Matrix(NN)
bb = Matrix(bb)
N_rows,N_cols = N.shape 
vars = symbols('x1:%d' % (N_rows+1))
lamb = symbols('l1:%d' % (N_rows+1))
lamb = Matrix(lamb)
x = Matrix(vars)
print("x = ",x)
print("x.shape =",x.shape)
f = x.T * N * x + x.T * bb
f = expand(f[0])
# f_with_values = f.subs({var: 0 for var in x if var not in [x[1],x[10],x[12],x[3],x[5],x[7],x[9],x[14],x[16],x[18],x[20],x[22],x[24],x[26],x[28]]})
# f_with_values = f.subs({var: 0 for var in x if var not in [x[0],x[2],x[11],x[12],x[14],x[4],x[6],x[8],x[10],x[14],x[16],x[18],x[20],x[22],x[24],x[26],x[28],x[30]]})
f_with_values = f.subs({var: 0 for var in x if var not in [x[0],x[2],x[8],x[3],x[6],x[10],x[12],x[14],x[16],x[7],x[16],x[18],x[20],x[22],x[24],x[26],x[28],x[30]]})
print("f_with_values = ",f_with_values)
ST = N * x + bb
F = f + (lamb.T * ST)[0]
variables = x.col_join(lamb)
G = Matrix([F.diff(var) for var in variables])
G_rows,G_cols = G.shape



# 测试M正确性的多项式1
# x1,x2,x3 = symbols('x1 x2 x3')
# G1 = x1**2 - x2**2
# G2 = x1**2 - x2**2 +3*x1*x3
# G3 = x2 - x1 + x3
# variables = [x1,x2,x3]
# G = [
#     G1,
#     G2,
#     G3
# ]

# 测试M正确性的多项式2
# variables = symbols('x y z')
# G1 = variables[0]**2 - variables[1]*variables[2] - 3*variables[1]**2
# G2 = variables[0]*variables[1] - 2*variables[2]**2
# G3 = variables[1]**2 + variables[1]*variables[2] - variables[0]*variables[2]
# G = [
#     G1,
#     G2,
#     G3
# ]


# 非齐次方程组1
# variables = symbols('x y z')
# G1 = variables[0]**2 + variables[1]**2 +variables[2]**2 + variables[0] -1
# G2 = variables[0]**2 + variables[1]**2 +variables[1] - variables[2]
# G3 = variables[0]**2 + variables[2]**2 -variables[1] + variables[2]
# G = [
#     G1,
#     G2,
#     G3
# ]

# # 无解的非齐次方程组
# variables = symbols('x y z')
# G1 = variables[0] + variables[2] -1
# G2 = variables[1] -1
# G3 = -variables[0] - variables[2] -1
# G = [
#     G1,
#     G2,
#     G3
# ]

# print("G = ",G)
# for g in G:
#     print(g)

# 定义辅助变量0
v0 = symbols('v0')

# 获得system的信息
# 系统中每个多项式的degree：r
# 系统的degree:sumr
# 齐次项的次数：d = 1-n+sum（ri）
# 矩阵M和逆序向量beta的维度：C_sum(ri)^(n-1)
# 多项式的项数:terms
def getInfo_rsdn(system,alter_variable):
    G = system
    r = []
    terms = 0
    for g in G:
        # 把输入的表达式拆分成每一项，生成一个列表
        g_degree = 0
        g_terms = g.args
        # print("g_terms = ",g_terms)
        coef = []
        powe = []
        # variables = expression.free_symbols
        for g_term in g_terms:
            # 把term的系数取出来，as_coeff_mul()返回一个元组
            # coefficient = term.as_coeff_mul()[0]
            origin,aux_var_term = g_term.as_independent(alter_variable)
            # print("origin = ",origin,"aux_var_term = ",aux_var_term)
            if total_degree(origin) > g_degree:
                g_degree = total_degree(origin)
            # print("g = ",g,"origin = ",origin,"g_total_degree = ",g_degree)
        r.append(g_degree)
    n = len(system)
    sumr = 0
    for ri in r:
        sumr += ri
    d = 1 - n + sumr
    # print("sumr = ",sumr,"n = ",n)
    # print("rn = ",r)

    s = (math.factorial(sumr))/((math.factorial(n-1))*math.factorial(sumr - n +1))
    for g in G:
        # print("g = ",g,"type of g = ",type(g))
        polynomial = sympify(g)
        # print("polynomial = ",polynomial,"type of polynomial =",type(polynomial))
        temp = polynomial.as_ordered_terms()
        # print("as ordered terms = ",temp)
        if len(temp) > terms:
            terms = len(temp)
    s = int(s)
    return sumr,s,d,n,terms,r


# 算法1：生成B、cG和dG
# 生成B
def find_integer_combinations(n, total_sum):
    combinations = []
    stack = [(total_sum, n, [])]

    while stack:
        curr_sum, curr_n, curr_combination = stack.pop()

        if curr_n == 0:
            if curr_sum == 0:
                combinations.append(curr_combination)
            continue

        for i in range(min(total_sum+1, curr_sum + 1)):
            stack.append((curr_sum - i, curr_n - 1, curr_combination + [i]))

    return combinations

def getB(n,d):
    B = find_integer_combinations(n, d)
    return B

# 生成G
# 合并同类项的函数，把当作系数的变量从变量列表里面除去，再把辅助变量添加到变量列表中
# 返回合并同类项后的方程组
def merge(system,alter_variable):
    rest_var_list = list()
    rest_var_list.append(v0)
    G_merge = list()
    for var in variables:
        if var == alter_variable:
            continue
        rest_var_list.append(var)
    # print("rest_var_list = ",rest_var_list)
    for g in system:
        merged_expression = collect(g,rest_var_list)
        G_merge.append(merged_expression)
    return G_merge,rest_var_list


def find_terms(expression,alter_variable,rest_var_list):
    # 把输入的表达式拆分成每一项，生成一个列表
    terms = expression.args
    # print("terms = ",terms)
    coef = []
    powe = []
    # variables = expression.free_symbols
    for term in terms:
        # 把term的系数取出来，as_coeff_mul()返回一个元组
        # coefficient = term.as_coeff_mul()[0]
        others,coefficient = term.as_independent(alter_variable)
        coeff_dict = others.as_coefficients_dict()
        first_value = next(iter(coeff_dict.values()))
        if first_value.is_negative:
            coefficient = -coefficient
        # print("term = ",term,"others = ",others,"coefficient = ",coefficient)
        powers = [others.as_powers_dict().get(var, 0) for var in rest_var_list]
        coef.append(coefficient)
        for pow in powers:
            powe.append(pow)
    return powe,coef


def getG(system,alter_variable,rest_var_list):
    G = system
    sumr,s,d,n,terms,rn = getInfo_rsdn(G,alter_variable)
    for i in range(len(G)):
        powers,coefficient = find_terms(smp.sympify(G[i]),alter_variable,rest_var_list)
        # print("powers1 = ",powers)
        if len(powers) < terms * n :
            powers = powers + [0] * (terms*n - len(powers))
        if len(coefficient) < terms:
            coefficient = coefficient + [0] * (terms - len(coefficient))
        # print("coefficient = ",coefficient)
        # print("Matrix coefficient = ",Matrix([coefficient]))
        # print("powers2 = ",powers)
        # print("cG[i,:] = ",cG[i,:])
        # print("cG[i,:][:] = ",cG[i,:][:])
        cG[i,:] = Matrix([coefficient])
        # print("cG = ",cG)
        dG[i,:] = powers
    return dG,cG


# 算法2：生成M
def getM(dG,cG):
    M = Matrix(np.zeros((s,s)))
    n,L = dG.shape
    r = L/n
    v = zeros(s,1)
    count = 0
    B = getB(n,d)
    B = B[::-1]
    # print("B = ",B)
    for i in range(n):
        for j in range(s):
            if v[j] == 0 and B[j][i] >= rn[i]:
                v[j] = 1
                # print("j = ",j)
                # print("rn[i] = ",rn[i])
                # print("B2 = ",B)
                c = copy.deepcopy(B[j])
                # print("B3 = ",B)
                c[i] = c[i] - rn[i]
                # print("B4 = ",B)
                for k in range(int(r)):
                    Q = dG[i][(k*n):((k+1)*n)] + c
                    Q = [int(x) for x in Q]
                    # print("Q = ",Q,"B[0] = ",B[0])
                    for jk in range(s):
                        if Q == B[jk]:
                            # print("dG = ",dG[i][(k*n):((k+1)*n)])
                            # print("c = ",c)
                            # print("Q = ",Q)
                            # print("shape M = ",M.shape,"shape cG =",cG.shape)
                            # print("conunt = ",count,"jk = ",jk,"i = ",i,"k = ",k)
                            # print("get M = ",M,"get M cG = ",cG[i,k])
                            # print("M[count][jk] = ",M[count,jk],"cG[i][k] = ",cG[i,k])
                            M[count,jk] = cG[i,k]
                count+=1
    return M

# 非齐次转化为齐次
# 输入一个多项式系统，把所有项的次数都用辅助变量补到最高次
# 选定一个变量xi作为常数，把这个变量用xi*v0代替，这时多项式还是齐次的
# 输入多项式系统，先转换为齐次，然后选定变量作为常数，以固定步长遍历变量的取值，记录变量的对应的det（M）和变量取值
def solutions(system,variables,begin,end,step):
    # print("system = ",system)
    G = homogeneous.homogeneous(system=system)
    # print("G = ",G)
    alllist = list()
    for xi in variables:
        xilist = list()
        G1 = homogeneous.alter(G,xi)
        # print("G1 = ",G1)
        for i in np.arange(begin,end+step,step):
            G2 = [g.subs(xi,i) for g in G1]
            DG,CG = getG(G2)
            M1 = getM(DG,CG)
            detval = np.linalg.det(M1)
            detval = Decimal(detval)
            xilist.append((detval,i))
        alllist.append(xilist)
    return alllist

# finallist = solutions(G,variables,-0.6,1.0,0.01)
# print("final list =",finallist)

# 割线法求一元函数的根
def secant(func,var,start,end,iterations):
    t= symbols('t')
    secant_func = (func.subs(var,end) - func.subs(var,start))/(end - start)*(t - start) + func.subs(var,start)
    iteration = 1
    while true:
        solution = solve(secant_func,t)
        func_val = func.subs(var,solution[0].evalf())
        if Abs(func_val) < 1e-6:
            break
        secant_func = (func.subs(var,end) - func.subs(var,solution[0].evalf()))/(end - solution[0].evalf())*(t - solution[0].evalf()) + func.subs(var,solution[0].evalf())
        iteration+=1
        # print("solution = ",solution[0].evalf(),"iteration = ",iteration)
        if iteration > iterations:
          break
    return solution

# 非齐次转化为齐次
# 输入一个多项式系统，把所有项的次数都用辅助变量补到最高次
# 选定一个变量xi作为常数，把这个变量用xi*v0代替，这时多项式还是齐次的
def homogeneous(system,alter_variable):
  # r_total 是去掉固定变量后的多项式的次数
    r_total = []
    for g in system:
        # 把输入的表达式拆分成每一项，生成一个列表
        g_degree = 0
        g_terms = g.args
        # print("g_terms = ",g_terms)
        # variables = expression.free_symbols
        for g_term in g_terms:
            # 把term的系数取出来，as_coeff_mul()返回一个元组
            # coefficient = term.as_coeff_mul()[0]
            rest,alter_var = g_term.as_independent(alter_variable)
            if total_degree(rest) > g_degree:
                g_degree = total_degree(rest)
        r_total.append(g_degree)

    # TCDlist存的是每个多项式的每一项的（项（Term）、系数（Coefficient）、次数（Degree））
    TCDlist = list()
    G1 = []
    for g in system:
        # 获取每一项的系数和次数
        coefficients_dict = g.as_coefficients_dict()
        temp_list = list()
        # 遍历并打印每一项的系数和次数
        for term, coefficient in coefficients_dict.items():
            degree = term.as_powers_dict().items()
            totaldegree = 0
            for term1,deg in degree:
                if term1 == alter_variable:
                  continue
                totaldegree += deg
            temp_list.append((term,coefficient,totaldegree))
        TCDlist.append(temp_list)
    # print("TCD_list = ",TCDlist)
    ri = 0

    for TCDs in TCDlist:
        poly = None
        maxdegree = r_total[ri]
        ri += 1
        i = 0
        for tcd in TCDs:
            t,c,d = tcd
            # sympy认为常数的次数为1,这里只要有一项是常数，就直接把次数置为0
            if isinstance(t,smp.core.numbers.Number):
                # print("------- t = ",t)
                temp = (tcd[0],tcd[1],0)
                TCDs[i] = temp
            i+=1
        poly = sum(tcd[1]*tcd[0]*v0**(maxdegree-tcd[2]) for tcd in TCDs)
        G1.append(poly)
    # print("G = ",G)
    # print("TCDlist = ",TCDlist)
    # print("typrof TCDlist[0][0][0] = ",type(TCDlist[0][0][0]))
    # print("G1 = ",G1)
    return G1

def split_variables(alter_variable):
    print("outer_thread_pool_variable = ",alter_variable)
    # 获得方程组的信息
    # 方程组中每个多项式的degree：r 存入一个列表rn中
    # 系统的degree:sumr
    # 齐次项的次数：d = 1-n+sum（ri）
    # 矩阵M和逆序向量beta的维度：C_sum(ri)^(n-1)
    # 多项式的项数:terms
    # 矩阵维数=变量个数=方程个数=n
    sumr,s,d,n,terms,rn = getInfo_rsdn(G,alter_variable)
    print("inner sumr = ",sumr)
    # 把输入的方程组齐次化
    print("pre GH")
    GH = homogeneous(G,alter_variable=alter_variable)
    print("GH")
    # 合并同类项，返回去除固定变量的变量列表（维度一致，新增了一个辅助变量）
    G_merge,rest_var_list = merge(GH,alter_variable=alter_variable)
    print("G_merge = ",G_merge,"rest_var_list = ",rest_var_list)
    var_min_eig_list = list()
    # 初始化所需要的矩阵，现在是数值矩阵，需要用Matrix强转为符号矩阵
    M = np.zeros((s,s))
    B = np.zeros((s,n))
    dG = np.zeros((n,terms*n))
    cG = Matrix(np.zeros((n,terms)))
    dG,cG = getG(G_merge,alter_variable=alter_variable,rest_var_list=rest_var_list)
    M = getM(dG,cG)
    print("inner_M.shape = ",M.shape)
    # thread_pool_inner = concurrent.futures.ThreadPoolExecutor(max_workers=20)
    print("inner thread pool created")
    results = []
    for val in np.arange(0.9,1.01,0.01):
        print("inner val = ",val)
        val,min_eig_val = get_val_eigval_task(M,alter_variable,val)
        var_min_eig_list.append((val,min_eig_val))
        # future = thread_pool_inner.submit(get_val_eigval_task,M,alter_variable,val)
        # results.append(future)
    # concurrent.futures.wait(results)
    # for future in results:
    #     var_min_eig_list.append(future.result())
    # thread_pool_inner.shutdown()
    return [alter_variable,var_min_eig_list]


def get_val_eigval_task(M,alter_variable,val):
    print("inner_thread_pool_variable = ",alter_variable)
    M_num = M.subs(alter_variable,val)
    M_num = np.array(M_num,dtype=np.float64)
    MM = M_num @ M_num.T
    eigenvalues, _ = np.linalg.eig(MM)
    positive_real_parts = [np.real(eig) for eig in eigenvalues if np.real(eig) > 0]
    min_eig_val = np.min(positive_real_parts)
    return val,min_eig_val

# 固定某个变量作为常数，用辅助变量v0替换它的位置
vr,vc = x.shape
# 创建线程池，并发计算特征值，填入列表对应位置，然后执行遍历操作
thread_pool_outer = concurrent.futures.ThreadPoolExecutor(max_workers=300)
all_var_eig_list = list()
results1 = []
for i in range(vr):
    alter_variable = x[i]
    print("var = ",alter_variable)
    future = thread_pool_outer.submit(split_variables,alter_variable)
    results1.append(future)

concurrent.futures.wait(results1)

all_var_eig_list = [item.result() for item in results1]

thread_pool_outer.shutdown()

print("all_var_eig_list = ")
for e in all_var_eig_list:
    print(e)


def get_report(all_var_eig_list,mutation):
    var_mutation_list = list()
    for term in all_var_eig_list:
        variable = term[0]
        val_eig = term[1]
        mut = false
        val_eig = [e for e in reversed(val_eig)]
        for i in range(len(val_eig)-1):
            # 相邻位置检测距离1最近的特征值突变，存入四元组（当前变量，突变处当前变量取值、突变处特征值，突变后一位置处特征值）
            if val_eig[i][1] / val_eig[i+1][1] <= mutation:
                var_mutation_list.append((variable,val_eig[i][0],val_eig[i+1][1],val_eig[i][1]))
                mut = true
                break
        if mut == false:
            var_mutation_list.append((variable,0,0,0))  
    return var_mutation_list

mutation = 1e-10
report_list = get_report(all_var_eig_list,mutation)
print("report_list = ")
for iterm in report_list:
    print(iterm)

all_var_eig_list_zero = list()
for i in range(vr):
    alter_variable = x[i]
    print("var = ",alter_variable)
    sumr,s,d,n,terms,rn = getInfo_rsdn(G,alter_variable)
    GH = homogeneous(G,alter_variable=alter_variable)
    ctx.prec = 8192
    G_merge,rest_var_list = merge(GH,alter_variable=alter_variable)
    var_min_eig_list = list()
    M = np.zeros((s,s))
    B = np.zeros((s,n))
    dG = np.zeros((n,terms*n))
    cG = Matrix(np.zeros((n,terms)))
    mp.prec = 8192

    dG,cG = getG(G_merge,alter_variable=alter_variable,rest_var_list=rest_var_list)


    M = getM(dG,cG)
    rows,cols = M.shape
    offset_matrix = np.diag([1e-5 for i in range(rows)])
    for val in np.arange(0,0.14,0.01):
        M_num = M.subs(alter_variable,val)
        M_num = np.array(M_num,dtype=np.float64)
        MM = M_num @ M_num.T
        k = 5  # 求5个最小特征值
        # numoy包算法
        eigenvalues, _ = np.linalg.eig(MM)
        positive_real_parts = [np.real(eig) for eig in eigenvalues if np.real(eig) > 0]
        min_eig_val = np.min(positive_real_parts)
        var_min_eig_list.append((val,min_eig_val))
    all_var_eig_list_zero.append([alter_variable,var_min_eig_list])

def get_report_zero(all_var_eig_list_zero,mutation):
    var_mutation_list = list()
    for term in all_var_eig_list_zero:
        variable = term[0]
        val_eig = term[1]
        mut = false
        for i in range(len(val_eig)-1):
            # 相邻位置检测距离1最近的特征值突变，存入四元组（当前变量，突变处当前变量取值、突变处特征值，突变后一位置处特征值）
            if val_eig[i+1][1] / val_eig[i][1] <= mutation:
                var_mutation_list.append((variable,val_eig[i+1][0],val_eig[i][1],val_eig[i+1][1]))
                mut = true
                break
        if mut == false:
            var_mutation_list.append((variable,0,0,0))  
    return var_mutation_list


print("all_var_eig_list_zero = ")
for e in all_var_eig_list_zero:
    print(e)


mutation = 1e-10
report_list = get_report_zero(all_var_eig_list_zero,mutation)
print("report_list_zero = ")
for iterm in report_list:
    print(iterm)

# 解线性方程组
# M = Matrix(M)
# print("rank of M  =",M.rank())
# nullspace_basis = M.nullspace()
# print("kernel of M = ")
# pprint(nullspace_basis)

# A = Matrix((ros+1,cols+1))
v = symbols('v')
A = zeros(rows+1,cols+1)
# print("type of A = ",type(A),"type of M = ",type(M))
# print("init A = ",A)

def generate_A(M):
    rows,cols = M.shape
    A = zeros(rows+1,cols+1)
    nullspace = M.nullspace()
    nullspaceT = (M.T).nullspace()
    if nullspace == []:
        c = 0.1
        # a = Matrix([round(random.uniform(-c, c),2) for _ in range(rows)])
        # b = Matrix([round(random.uniform(-c, c),2) for _ in range(cols)])
        # a = zeros(rows,1)
        # b = zeros(cols,1)
        # a[rows-1] =1
        # b[cols-1] =10
        a = ones(rows,1)
        b = ones(cols,1)
        print("a:")
        pprint(a)
        print("b:")
        pprint(b)
        A[0:rows,0:cols] = M[:,:]
        A[0:rows,cols:cols+1] = a
        A[rows:rows+1,0:cols] = b.T
    else:
        # 向量转矩阵
        M_nullspace = nullspace[0]
        rest_space = nullspace[1:]
        for col in rest_space:
            M_nullspace = M_nullspace.row_join(col)
        # print("M_nullspace =")
        # pprint(M_nullspace)
        # constants = zeros(M_nullspace.rows, 1)
        # ot_space = M_nullspace.gauss_jordan_solve(constants)
        # print("otspace:")
        # pprint(ot_space)
        # echelon = M_nullspace.echelon_form()
        # print("echelon = ")
        # pprint(echelon)
        othcom = M.rowspace()
        othcom = Matrix(othcom)
        print("othcom = ")
        pprint(othcom)
        o_rows,o_cols = othcom.shape
        print("o_rows = ",o_rows,"o_cols = ",o_cols)
        # print("rank of N = ",N.rank(),"rank of oth =",othcom.rank())
        # pprint(othcom * M_nullspace)
        o_vars = Matrix(symbols('o1:%d'%(o_cols+1)))
        # print("O_vars = ")
        # pprint(o_vars)
        b_Matrix = (othcom.T).row_join(o_vars)
        # print("b_Matrix =")
        # pprint(b_Matrix)
        b_echelon = b_Matrix.echelon_form()
        # print("b-echelon = ")
        # pprint(b_echelon)
        b1 = b_echelon[:,-1]
        # print("b = ")
        # pprint(b1)
        b1_last_nonzero = None
        for value in b1:
            if value != 0:
                b1_last_nonzero = value
        # print("last non zero =")
        # pprint(b1_last_nonzero)
        b_vars = b1_last_nonzero.free_symbols
        # print("b_vars = ")
        # pprint(b_vars)
        ob = b_vars.pop()
        # print("ob = ",ob)
        # print("type of ob = ",type(ob))
        b_final = Matrix([1 if var == ob else 0 for var in o_vars])
        print("b_final = ")
        pprint(b_final)
        b_final_matrix = (othcom.T).row_join(b_final)
        print("rank of b_Matrix = ",b_Matrix.rank(),"rank of oth =",othcom.rank())
        print("rank of b_final_matrix = ",b_final_matrix.rank())

        MT_nullspace = nullspaceT[0]
        rest_spaceT = nullspaceT[1:]
        for col in rest_spaceT:
            MT_nullspace = MT_nullspace.row_join(col)
        print("MT_nullspace =")
        pprint(MT_nullspace)
        constants = zeros(MT_nullspace.rows, 1)
        ot_spaceT = MT_nullspace.gauss_jordan_solve(constants)
        print("otspace:")
        pprint(ot_spaceT)
        echelonT = MT_nullspace.echelon_form()
        print("echelonT = ")
        pprint(echelonT)
        othcomT = (M.T).rowspace()
        othcomT = Matrix(othcomT)
        print("othcomT = ")
        pprint(othcomT)
        oT_rows,oT_cols = othcomT.shape
        print("rows = ",oT_rows,"cols = ",oT_cols)
        print("rank of N = ",N.rank(),"rank of oth =",othcomT.rank())
        pprint(othcomT * MT_nullspace)
        oT_vars = Matrix(symbols('oT1:%d'%(oT_cols+1)))
        print("OT_vars = ")
        pprint(oT_vars)
        a_Matrix = (othcomT.T).row_join(oT_vars)
        print("a_Matrix =")
        pprint(a_Matrix)
        a_echelon = a_Matrix.echelon_form()
        print("a_echelon = ")
        pprint(a_echelon)
        a1 = a_echelon[:,-1]
        print("a1 = ")
        pprint(a1)
        a1_last_nonzero = None
        for value in a1:
            if value != 0:
                a1_last_nonzero = value
        print("a1 last non zero =")
        pprint(a1_last_nonzero)
        a_vars = a1_last_nonzero.free_symbols
        print("a_vars = ")
        pprint(a_vars)
        oa = a_vars.pop()
        print("oa = ",oa)
        print("type of oa = ",type(oa))
        a_final = Matrix([1 if var == oa else 0 for var in oT_vars])
        print("a_final = ")
        pprint(a_final)

        A[0:rows,0:cols] = M[:,:]
        print("rank of A[0:rows,0:cols] = M[:,:] = ",A.rank())
        A[0:rows,cols:cols+1] = a_final
        print("A[0:rows,cols:cols+1] = a_final = ",A.rank())
        A[rows:rows+1,0:cols] = b_final.T
        print("A[rows:rows+1,0:cols] = b_final.T = ",A.rank())

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("矩阵的核不是零向量,找ab的函数还没有写")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return A


# 当M的核是零向量的时候，a和b都是随机生成的，这会导致解出的t每次都不一样
# 现在用的是全为1的列向量，但a和b要满足什么关系还不清楚
# 现在这一版的割线法有问题，不能逼近解，简单函数没有问题，应该是和函数的性质有关
# A = generate_A(M)
# b = zeros(rows+1,1)
# b[rows] = 1
# # pprint(b)
# pprint(A)
# print("rows of A =",rows+1)
# # print("rank of A = ",A.rank())
# print("rank of M = ",M.rank())
# sol = A.LUsolve(b)
# # pprint(sol)
# t = sol[rows]
# print("t =",t)
# # plot(t, (alter_variable, -1, 1))
# solu = list()
# for value in np.arange(-3,3,0.1):
#     result = t.subs(alter_variable,value)
#     # pre = result
#     # after = t.subs(alter_variable,value+0.05)
#     # print("Pre = ",pre,"after = ",after)
#     # if pre*after < 0:
#     #     solu.append(secant(t,x,pre,after,100))
#     print("value = ",value,"result = ",result)

# print("M rank after subs  = ",M.rank())
# print("M.subs = ",M.subs(alter_variable,1))
# print("rank of M.subs = ",(M.subs(alter_variable,1)).rank())
# print("solu = ",solu)