# -*- coding: utf-8 -*-
"""

-------------------------------------------------
   File Name：        gea
   Description :
   Author :           hyx
   Create date：      2023/4/4
   Latest version:    v1.0.0
-------------------------------------------------
   Change Log:
#-----------------------------------------------#
    v1.0.0            hyx        2023/4/4
    1.
#-----------------------------------------------#
-------------------------------------------------

"""

def constraint(Vars):
    w_num = np.zeros((Vars.shape[0], 1))
    w_sum = np.zeros((Vars.shape[0], 1))
    
    for i in range(Vars.shape[0]):
        rows = Vars[i, :]
        num_rows = np.where(rows != 0, 1, 0)
        w_num[i] = np.sum(num_rows)
        w_sum[i] = np.sum(rows)
    
    print("====================================CONSTRAINTS====================================")
    print("------------------------------------Vars------------------------------------")
    print(Vars)
    print("------------------------------------NUM------------------------------------")
    print(w_num)
    print("------------------------------------SUM------------------------------------")
    print(w_sum)
    
    return w_num, w_sum

import numpy as np

import geatpy as ea

class MyProblem(ea.Problem):  # 继承Problem父类

    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 4  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0, 0, 0, 0]  # 决策变量下界
        ub = [0.3, 0.3, 0.8,0.8]  # 决策变量上界
        lbin = [1, 1, 1,1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)

    def evalVars(self, Vars):  # 目标函数
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        x4 = Vars[:, [3]]
        f = x1 * x2 * x3 * x4
        print(f)
        # 采用可行性法则处理约束
        # CV = np.hstack(
        #     [x1 + x2 - 0.3,0.2-x1-x2,x3 + x4 - 0.8,0.7-x3-x4,x2-x1,x4-x3, np.abs(x1 + x2 + x3 + x4 - 1)])
        CV = np.hstack(
            [np.abs(constraint(Vars)[0] - 3), np.abs(constraint(Vars)[1] - 1)])
        return f, CV

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值）
        referenceObjV = np.array([[0.035937]])
        return referenceObjV





if __name__ == '__main__':
    # 实例化问题对象
    problem = MyProblem()  # 生成问题对象
    # 快速构建算法
    algorithm = ea.soea_DE_currentToBest_1_bin_templet(
        problem,
        ea.Population(Encoding='RI', NIND=2000),
        MAXGEN=500,  # 最大进化代数。
        logTras=100)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    algorithm.mutOper.F = 0.7  # 差分进化中的参数F。
    algorithm.recOper.XOVR = 0.7  # 交叉概率。
    # 先验知识
    prophetVars = np.array([[0.8, 0.1, 0.1, 0	]])  # 假设已知[0.4, 0.2, 0.4]为一组比较优秀的变量。
    # 求解
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=True)
    print(res)


