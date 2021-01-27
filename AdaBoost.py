'''
作者：宋建军
时间：2021年1月22日
简介：本代码是知乎文章《手把手教Adaboost》的复现
AdaBoost算法是集成学习中常用的一种算法
其解决了两个问题
1.如何选择一组有不同优缺点的弱学习器，使得他们可以相互弥补不足
2.如何组合弱学习器的输出以获得整体的更好的决策表现
'''

# 引入本代码需要的两个函数库。numpy用来生产我们想要的数组，math用以写ln函数。
import numpy as np
import math
# 生成文章中所使用的数据集
xMat = np.arange(10)
yMat = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
w_1 = []
for x in range(10):
    w_1.append(0.1)

# 本函数输入一次学习过程中学习器的错误率，输出相对应的分类器权重。
def funcE(R):
    R_tmp = R
    # print(R_tmp)
    E = 0.5 * math.log((1 - (R_tmp)) / (R_tmp))
    return E

# 本函数输入一次学习过程中根据阈值判断的结果数组和权重数组，输出相对应的错误率。
def funcR_error(y_n, w_n):
    R_tmp = 0
    for i in range(10):
        if y_n[i] != yMat[i]:
            R_tmp += w_n[i]
    return R_tmp

# 本函数的目的是更新权重数组
def updateW(w_n, y_n):
    E = funcE(funcR_error(y_n,w_n))
    w_n_tmp = w_n
    for i in range(10):
        if y_n[i] != yMat[i]:
            # 《机器学习》上面实在写的太花里胡哨了，实在看不懂，但其实就是乘下面的e指数
            w_n_tmp[i] = w_n[i] * math.exp(E)
        else:
            w_n_tmp[i] = w_n[i] * math.exp(-E)
    # 这个D是为了将整个权重数组归一化
    D = sum(w_n_tmp)
    for i in range(10):
        w_n[i] = w_n_tmp[i] / D
    return w_n

# 第一个和第二个学习器的判断方法，就是通过划分阈值来找到不同阈值对应的判断数组
def JudgeValue(value):
    y_n = []
    for i in range(10):
        if xMat[i] < value:
            y_n.append(1)
        else:
            y_n.append(-1)
    return y_n
# 这里为什么要另写一个判断方法了，仔细观察文章里面第三次使用的判断方法和前两次都不一样，正好相反。不改的话，第三次的判断阈值就是2.5，不会到5.5上。
def JudgeValue2(value):
    y_n = []
    for i in range(10):
        if xMat[i] < value:
            y_n.append(-1)
        else:
            y_n.append(1)
    return y_n
# 初始化结果矩阵
Result_G = np.zeros(10)
w_n = w_1

# 进行第一次和第二次学习
for k in range(2):
    # 初始化
    value_right = 0.5
    i_right = 0
    R_right = 1
    # 循环内的内容是计算每一个阈值对应的错误率，之后找到最小的。
    for i in range(10):
        value = i + 0.5
        R_right_tmp = funcR_error(JudgeValue(value),w_n)
        if R_right_tmp < R_right:
            R_right = R_right_tmp
            value_right = value
            i_right = i
    print(value_right)

    E = funcE(funcR_error(JudgeValue(value_right), w_n))
    print(E)
    y_n = JudgeValue(value_right)
    print(y_n)
    w_n = updateW(w_1,y_n)
    print(w_n)
    for i in range(10):
        # 下面就是集成学习输出结果所使用的表达形式
        Result_G[i] = Result_G[i] + E*y_n[i]
    print(Result_G)
    print("------")

# 进行第三次学习
value_right = 0.5
i_right = 0
R_right = 1
for i in range(10):
    value = i + 0.5
    R_right_tmp = funcR_error(JudgeValue2(value),w_n)
    if R_right_tmp < R_right:
        R_right = R_right_tmp
        value_right = value
        i_right = i
print(value_right)

E = funcE(funcR_error(JudgeValue2(value_right), w_n))
print(E)
y_n = JudgeValue2(value_right)
print(y_n)
w_n = updateW(w_1,y_n)
print(w_n)
for i in range(10):
    Result_G[i] = Result_G[i] + E*y_n[i]
print(Result_G)
print("------")