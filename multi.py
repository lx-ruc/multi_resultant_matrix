import sympy as smp
from sympy import *
import numpy as np
import pandas as pd
import math
import random
import copy
import homogeneous
from decimal import Decimal


init_printing(use_unicode=True)
# 生成多项式的多重结果矩阵
# 输入多项式系统（n个变量n个方程），得到每个方程的degree：ri和系数、

# 2004年论文的第一个例子
x,y,z = symbols('x y z')
G1 = x**2 + y**2 + z**2 -1
G2 = z - x**2 - y**2
G3 = y - x**2 -z**2
variables = [x,y,z]
G = [
    G1,
    G2,
    G3
] 

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
# 获得system的信息
# 系统中每个多项式的degree：r
# 系统的degree:sumr
# 齐次项的次数：d = 1-n+sum（ri）
# 矩阵M和逆序向量beta的维度：C_sum(ri)^(n-1)
# 多项式的项数:terms
def getInfo_rsdn(system):
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
            origin,aux_var_term = g_term.as_independent(homogeneous.v0)
            # print("origin = ",origin,"aux_var_term = ",aux_var_term)
            if total_degree(origin) > g_degree:
                g_degree = total_degree(origin)
        r.append(g_degree)
    n = len(system)
    sumr = 0
    for ri in r:
        sumr += ri
    d = 1 - n + sumr
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
    rest_var_list.append(homogeneous.v0)
    G_merge = list()
    for var in variables:
        if var == alter_variable:
            continue
        rest_var_list.append(var)
    print("rest_var_list = ",rest_var_list)
    for g in system:
        merged_expression = collect(g,rest_var_list)
        G_merge.append(merged_expression)
    return G_merge,rest_var_list


def find_terms(expression,alter_variable,rest_var_list):
    # 把输入的表达式拆分成每一项，生成一个列表
    terms = expression.args
    print("terms = ",terms)
    coef = []
    powe = []
    # variables = expression.free_symbols
    for term in terms:
        # 把term的系数取出来，as_coeff_mul()返回一个元组
        # coefficient = term.as_coeff_mul()[0]
        others,coefficient = term.as_independent(alter_variable)
        powers = [others.as_powers_dict().get(var, 0) for var in rest_var_list]
        coef.append(coefficient)
        for pow in powers:
            powe.append(pow)
    return powe,coef


def getG(system,alter_variable,rest_var_list):
    G = system
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
    print("B = ",B)
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
    print("system = ",system)
    G = homogeneous.homogeneous(system=system)
    print("G = ",G)
    alllist = list()
    for xi in variables:
        xilist = list()
        G1 = homogeneous.alter(G,xi)
        print("G1 = ",G1)
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


# G在最开始输入，可能是非齐次的方程组
# 把输入的方程组齐次化
GH = homogeneous.homogeneous(G)
print("GH = ",GH)

# 固定某个变量作为常数，用辅助变量v0替换它的位置
alter_variable = x
GH_aux = homogeneous.alter(GH,alter_variable)
print("GH_aux = ",GH_aux)

# 合并同类项，返回去除固定变量的变量列表（维度一致，新增了一个辅助变量）
G_merge,rest_var_list = merge(GH_aux,alter_variable=alter_variable)
print("G_merge = ",G_merge)

# 获得方程组的信息
# 方程组中每个多项式的degree：r 存入一个列表rn中
# 系统的degree:sumr
# 齐次项的次数：d = 1-n+sum（ri）
# 矩阵M和逆序向量beta的维度：C_sum(ri)^(n-1)
# 多项式的项数:terms
# 矩阵维数=变量个数=方程个数=n
sumr,s,d,n,terms,rn = getInfo_rsdn(G_merge)
print("rn =",rn,"terms = ",terms)
# 初始化所需要的矩阵，现在是数值矩阵，需要用Matrix强转为符号矩阵
M = np.zeros((s,s))
B = np.zeros((s,n))
dG = np.zeros((n,terms*n))
cG = Matrix(np.zeros((n,terms)))
print("prime cG = ",cG,"prime dG = ",dG)


dG,cG = getG(G_merge,alter_variable=alter_variable,rest_var_list=rest_var_list)
print("dG  = ",dG)
print("cG  = ",cG)

M = getM(dG,cG)
print("------------------------------M-------------------------------")
print("M =",M)
print("type of M = ",type(M))
# print("det(M) = ",np.linalg.det(M))

# 解线性方程组
# M = Matrix(M)
print("rank of M  =",M.rank())
nullspace_basis = M.nullspace()
print("kernel of M = ",nullspace_basis)
rows,cols = M.shape
print("rows = ",rows,"cols = ",cols)
# A = Matrix((ros+1,cols+1))
v = symbols('v')
A = zeros(rows+1,cols+1)
print("type of A = ",type(A),"type of M = ",type(M))
print("A = ",A)
def generate_A(M):
    rows,cols = M.shape
    A = zeros(rows+1,cols+1)
    nullspace_basis = M.nullspace()
    if nullspace_basis == []:
        c = 1
        # a = Matrix([round(random.uniform(-c, c),2) for _ in range(rows)])
        # b = Matrix([round(random.uniform(-c, c),2) for _ in range(cols)])
        a = ones(rows,1)
        b = ones(cols,1)
        A[0:rows,0:cols] = M[:,:]
        A[0:rows,cols:cols+1] = a
        A[rows:rows+1,0:cols] = b.T

    return A

# 当M的核是零向量的时候，a和b都是随机生成的，这会导致解出的t每次都不一样
# 现在用的是全为1的列向量，但a和b要满足什么关系还不清楚
# 现在这一版的割线法有问题，不能逼近解，简单函数没有问题，应该是和函数的性质有关
A = generate_A(M)
b = zeros(rows+1,1)
b[rows] = 1
pprint(b)
sol = A.solve(b)
# pprint(sol)
t = sol[rows]
print("t =",t)
solu = list()
for value in np.arange(-1,1,0.05):
    result = t.subs(x,value)
    pre = result
    after = t.subs(x,value+0.05)
    # print("Pre = ",pre,"after = ",after)
    # if pre*after < 0:
    #     solu.append(secant(t,x,pre,after,100))
    print("value = ",value,"result = ",result)

print("solu = ",solu)