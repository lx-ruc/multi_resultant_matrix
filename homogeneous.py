import sympy as smp
from sympy import *

# 输入多项式
# x1,x2,x3 = symbols('x1 x2 x3')
# G1 = x1**2 - x2**2
# G2 = x1**2 - x2**2 +3*x1*x3+ 4*x2
# G3 = x2 - x1 + x3
# variables = [x1,x2,x3]
# G = [
#     G1,
#     G2,
#     G3
# ]
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

# 定义辅助变量0
v0 = symbols('v0')


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

# 先判断每固定变量后的每个多项式最高次是多少，然后再补齐
def alter(system,variable):
    F = system
    t = variable
    G = []
    for f in F:
        # print("f = ",f)
        newpoly = f.subs(t,t*v0)
        # print("newpoly = ",newpoly)
        G.append(newpoly)
    return G
# print("G = ",G)
# G1 = homogeneous(G)
# print("G1 = ",G1)
# G2 = alter(G1,x)
# print("G2 = ",G2)
